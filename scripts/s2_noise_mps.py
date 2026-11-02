"""
s2_noise_mps.py
===============
Ruido del estimador MC de S2 = -log Tr(rho^2) y de su gradiente para ARViT + MPS ancilla, comparado con
el valor EXACTO (ED, N <= 10).  El estimador es el MISMO que usa src_renyi/mps_ancilla/training.py
(termino de pureza de make_mc_loss); aqui solo se varia como se emparejan las muestras:

    "all"   : todos los pares i != j del lote           (pair_block=None, coste ~ M^2 contracciones)
    "bB"    : todos los pares dentro de bloques de B    (pair_block=B,    coste ~ M*B)   b2 == pares disjuntos

PRESUPUESTO COMUN  C = numero de contracciones de pares <Phi_s|Phi_t> (lo que domina el coste a chi grande):
    all -> M = sqrt(C)      bB -> M = C / B
(el coste de las pasadas del ansatz es ~M, secundario).  Tambien se imprime el tiempo por estimacion.

Gradiente: con T=1 el gradiente de L_P es grad(log P) = -grad(S2); se compara -grad(L_P) con
grad(S2_exacto).  Metricas (como en s2_noise.py):
    cos  = <g, g_ex> / (|g||g_ex|)         direccion
    beta = <g, g_ex> / <g_ex, g_ex>         ganancia: con F = E - T S2 el estimador entrena a T_eff = beta*T
    |g|/|g_ex|                              modulo

Estados a evaluar:
    (por defecto) se entrenan con ED a cada T de --temps (chi fijo) y se cachean en --cache_dir;
    --load f1.msgpack f2.msgpack ... --chi 8   usa parametros guardados por training_mps_ladder.py.

Uso:
    python scripts/s2_noise_mps.py --N 8 --chi 4 --temps 0.5 1.5 2.5 4.0
    python scripts/s2_noise_mps.py --N 10 --chi 8 --load data/mps_N10/params/params_0010_chi8.msgpack ...
    python scripts/s2_noise_mps.py --replot          # rehace las figuras desde el JSON sin recalcular
Salida: ../plots/S2_noise_mps/  (pdf + png; .pgf si se pasa --tex)  y s2_noise_mps_N{N}_chi{chi}.json
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")      # antes de importar jax
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from src_renyi.mps_ancilla.models import mps_overlap_rect, mps_overlap_aligned
from src_renyi.mps_ancilla.training import (compute_rho, make_loss_fns, train, make_sampler, make_mc_loss)
from src_renyi.mps_ancilla.ladder import build_model, init_params, save_params, load_params

sg = jax.lax.stop_gradient
HX, JJ = -1.5, -1.0                      # mismo TFIM que en el resto del pipeline

# ── CLI ────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--N", type=int, default=8)
ap.add_argument("--chi", type=int, default=4)
ap.add_argument("--temps", type=float, nargs="+", default=[0.5, 1.5, 2.5, 4.0],
                help="T a las que se entrena un estado con ED (ignorado con --load).")
ap.add_argument("--train_steps", type=int, default=600)
ap.add_argument("--train_lr", type=float, default=5e-3)
ap.add_argument("--load", type=str, nargs="+", default=None, help="msgpack(s) de params de este chi.")
ap.add_argument("--cache_dir", type=str, default=None)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--blocks", type=int, nargs="+", default=[2, 8, 32], help="tamanos de bloque (pair_block).")
ap.add_argument("--no_all", action="store_true", help="no incluir el metodo 'todos los pares'.")
ap.add_argument("--nbr_blocks", type=int, nargs="+", default=[8],
                help="metodo 'nbr' (suma exacta de t=s y vecinos a un espin + parejas iid de d>=2 en bloques de este tamano).")
ap.add_argument("--no_nbr", action="store_true", help="no incluir el metodo 'nbr'.")
ap.add_argument("--budget_fixed", type=int, default=2 ** 14, help="C para la seccion 1 (contracciones de pares).")
ap.add_argument("--budgets", type=int, nargs="+", default=[2 ** 12, 2 ** 14, 2 ** 16])
ap.add_argument("--chunk", type=int, default=None,
                help="filas por trozo en el solape de 'all' (memoria). Para el ED se usa 128 si N>=10.")
ap.add_argument("--equal_M", action="store_true",
                help="interpreta --budget_fixed/--budgets como numero de MUESTRAS M (igual para todos los metodos) "
                     "en vez de contracciones de pares: aisla el efecto de emparejar mas o menos.")
ap.add_argument("--mem_gb", type=float, default=1.0,
                help="memoria aprox. (GB) por trozo de muestras en los metodos de bloques; si M no cabe, el gradiente "
                     "se calcula por trozos (exacto, mismo valor). Bajalo si hay OOM.")
ap.add_argument("--embedding", type=int, default=16)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--blocks_vit", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--replot", action="store_true")
ap.add_argument("--tex", action="store_true", help="matplotlib con pgf/usetex (necesita LaTeX).")
ap.add_argument("--out_dir", type=str, default=None)
args = ap.parse_args()

N, CHI = args.N, args.chi
assert N <= 10, "el valor exacto de referencia usa ED: N <= 10"
PLOTS_DIR = args.out_dir or os.path.join(script_dir, "..", "plots", "S2_noise_mps")
CACHE_DIR = args.cache_dir or os.path.join(script_dir, "..", "data", "s2_noise_mps")
os.makedirs(PLOTS_DIR, exist_ok=True)
JSON_PATH = os.path.join(PLOTS_DIR, f"s2_noise_mps_N{N}_chi{CHI}.json")

# spec de cada metodo: None = todos los pares | int b = bloques de b | ("nbr", b) = vecinos exactos + bloques de b
METHODS = (([] if args.no_all else [("all", None)]) + [(f"b{b}", b) for b in args.blocks]
           + ([] if args.no_nbr else [(f"nbr{b}", ("nbr", b)) for b in args.nbr_blocks]))
COLORS = {m: f"C{i}" for i, (m, _) in enumerate(METHODS)}
MARKERS = ["o", "s", "^", "d", "v", "<"]


# ══════════════════════════════════════════════════════════════════════════════
# Metricas
# ══════════════════════════════════════════════════════════════════════════════
def metrics(g_est, g_ref):
    num = float(np.dot(g_est, g_ref))
    ne, nr = float(np.linalg.norm(g_est)), float(np.linalg.norm(g_ref))
    return num / (ne * nr + 1e-30), num / (nr * nr + 1e-30), ne / (nr + 1e-30)   # cos, beta, |g|/|g_ex|


# ══════════════════════════════════════════════════════════════════════════════
# Calculo
# ══════════════════════════════════════════════════════════════════════════════
def compute():
    def tfim(N, hx, J):
        sx = np.array([[0, 1], [1, 0]], float); sz = np.diag([1., -1.]); I = np.eye(2)
        def op(o, i):
            m = np.array([[1.]])
            for k in range(N):
                m = np.kron(m, o if k == i else I)
            return m
        return sum(hx * op(sx, i) + J * op(sz, i) @ op(sz, (i + 1) % N) for i in range(N))

    model = build_model(N, CHI, args.embedding, args.heads, args.blocks_vit)
    all_s = jnp.asarray(np.array(np.meshgrid(*[[1., -1.]] * N, indexing="ij")).reshape(N, -1).T)
    ed_chunk = 128 if N >= 10 else None

    # ---- exacto: S2 y su gradiente ----
    def S2_exact(params):
        rho = compute_rho(model, params, all_s, ed_chunk)
        return -jnp.log(jnp.sum(jnp.abs(rho) ** 2))
    S2_and_grad = jax.jit(jax.value_and_grad(S2_exact))

    # ---- estados ----
    states = []                                              # (label, T o None, params)
    if args.load:
        for f in args.load:
            states.append((os.path.basename(f), None, load_params(f, model, N)))
    else:
        os.makedirs(CACHE_DIR, exist_ok=True)
        H = tfim(N, HX, JJ)
        loss_ed, _ = make_loss_fns(model, all_s, H, 1.0, chunk=ed_chunk)
        for T in args.temps:
            f = os.path.join(CACHE_DIR, f"params_N{N}_chi{CHI}_T{T:.3f}_steps{args.train_steps}.msgpack")
            if os.path.exists(f):
                p = load_params(f, model, N)
                print(f"[cache] T={T}: {f}", flush=True)
            else:
                print(f"[ED] entrenando estado a T={T} ({args.train_steps} pasos)...", flush=True)
                loss_T, _ = make_loss_fns(model, all_s, H, T, chunk=ed_chunk)
                out = train(loss_T, init_params(model, N, args.seed), args.train_steps, args.train_lr, T,
                            freq=max(1, args.train_steps // 5), warmup=min(50, args.train_steps // 5),
                            tag=f"[T={T}] ")
                p = out["params_best"]
                save_params(f, p)
            states.append((f"T={T:g}", T, p))

    ex = []
    for label, T, p in states:
        S2, g = S2_and_grad(p)
        ex.append((float(S2), np.asarray(ravel_pytree(g)[0], float)))
        print(f"  estado {label:>28s}:  S2 exacto = {ex[-1][0]:.4f}  (|g_ex| = {np.linalg.norm(ex[-1][1]):.3e})", flush=True)

    # ---- descomposicion EXACTA de P por distancia de Hamming (justifica el estimador 'nbr') ----
    # P = sum_{s,t} |rho(s,t)|^2 = sum_d P_d,   P_d = suma sobre pares con d(s,t)=d
    # Q_d = sum_{d(s,t)=d} p(s)p(t) = probabilidad de que una pareja iid de muestras caiga a distancia d
    # <f>_d = P_d / Q_d = sobrelapse medio de la ancilla a distancia d
    Dm = np.rint((N - np.asarray(all_s) @ np.asarray(all_s).T) / 2).astype(int)
    ham = []
    for label, T, p in states:
        rho = np.asarray(compute_rho(model, p, all_s, ed_chunk))
        pd_ = np.real(np.diag(rho))
        r2 = np.abs(rho) ** 2
        Pd = np.array([r2[Dm == d].sum() for d in range(N + 1)])
        Qd = np.array([np.outer(pd_, pd_)[Dm == d].sum() for d in range(N + 1)])
        ham.append({"label": label, "P_d": Pd.tolist(), "Q_d": Qd.tolist(), "P": float(Pd.sum())})
    print("\n=== 0. P descompuesta por distancia de Hamming (exacto) ===", flush=True)
    for h in ham:
        Pd, Qd = np.array(h["P_d"]), np.array(h["Q_d"])
        print(f"  {h['label']}:  P = {h['P']:.3e}")
        print("      d      " + " ".join(f"{d:8d}" for d in range(min(N, 6) + 1)))
        print("      P_d/P  " + " ".join(f"{x:8.4f}" for x in (Pd / Pd.sum())[:7]))
        print("      Q_d    " + " ".join(f"{x:8.1e}" for x in Qd[:7]))
        print("      <f>_d  " + " ".join(f"{x:8.1e}" for x in (Pd / np.maximum(Qd, 1e-300))[:7]))
        print(f"      fraccion de P con d<=1 (la que 'nbr' calcula exacta): {(Pd[:2].sum() / Pd.sum()):.3f}"
              f"   | prob. de que una pareja iid caiga en d<=1: {Qd[:2].sum():.2e}", flush=True)

    # ---- estimador MC de la pureza (mismo codigo que el bloque de pureza de make_mc_loss) ----
    def pair_terms(params, s, pair_block, chunk):
        s = sg(s)
        M = s.shape[0]
        logA, Mt = model.apply({"params": params}, s)
        n2 = jnp.real(mps_overlap_aligned(Mt, Mt, CHI))
        logp = 2.0 * jnp.real(logA)
        if pair_block is None:
            O = mps_overlap_rect(Mt, Mt, CHI, chunk)
            f = jnp.abs(O) ** 2 / (n2[:, None] * n2[None, :])
            mask = 1.0 - jnp.eye(M)
            lp = logp[:, None] + logp[None, :]
        else:
            b = pair_block
            nb = M // b
            Mb = Mt.reshape(nb, b, *Mt.shape[1:])
            O = jax.vmap(lambda m: mps_overlap_rect(m, m, CHI))(Mb)
            n2b = n2.reshape(nb, b)
            f = jnp.abs(O) ** 2 / (n2b[:, :, None] * n2b[:, None, :])
            mask = jnp.broadcast_to(1.0 - jnp.eye(b), (nb, b, b))
            lpb = logp.reshape(nb, b)
            lp = lpb[:, :, None] + lpb[:, None, :]
        return f, mask, lp

    def make_purity_loss(pair_block, chunk):
        def loss(params, s):
            f, mask, lp = pair_terms(params, s, pair_block, chunk)
            npairs = jnp.sum(mask)
            Pb = jnp.sum(f * mask) / npairs
            L = (Pb + jnp.sum(mask * sg(f - Pb) * lp) / npairs) / sg(Pb)      # T=1: grad L = grad log P = -grad S2
            return L, -jnp.log(sg(Pb))
        return loss

    def make_chunked_vg(pair_block, nc):
        """Gradiente de L_P por trozos de M/nc muestras (bloques completos): EXACTO, mismo valor que sin trocear.
        L = (sum_c [sum f*mask + sum mask*sg(f-Pb)*lp]) / (npairs * Pb) es una suma sobre bloques una vez fijado Pb,
        asi que: pasada 1 (sin gradiente) -> Pb global; pasada 2 -> suma de gradientes de cada trozo.
        La memoria baja ~nc veces (lo que pesa son los tensores MPS por muestra, no la red)."""
        def chunk_sum(params, sc):
            f, mask, _ = pair_terms(params, sc, pair_block, None)
            return jnp.sum(f * mask)

        def chunk_loss(params, sc, Pb, npairs):
            f, mask, lp = pair_terms(params, sc, pair_block, None)
            return (jnp.sum(f * mask) + jnp.sum(mask * sg(f - Pb) * lp)) / (npairs * Pb)

        @jax.jit
        def vg(params, s):
            M = s.shape[0]
            sch = s.reshape(nc, M // nc, s.shape[1])
            npairs = M * (pair_block - 1.0)
            Pb = sg(jnp.sum(jax.lax.map(lambda sc: chunk_sum(params, sc), sch)) / npairs)

            def body(acc, sc):
                g = jax.grad(chunk_loss)(params, sc, Pb, npairs)
                return jax.tree_util.tree_map(jnp.add, acc, g), None
            g, _ = jax.lax.scan(body, jax.tree_util.tree_map(jnp.zeros_like, params), sch)
            return (0.0, -jnp.log(Pb)), g
        return vg

    def nbr_terms(params, s, pair_block):
        """Para cada muestra: g(s) = p(s) + sum_k p(s^k) f(s,s^k)  (t=s y vecinos a un espin, EXACTO en t),
        y los terminos de parejas dentro de bloques (f, mask, lp) con m2 = 1 si d(s_i,s_j) >= 2."""
        s = sg(s)
        M, N_ = s.shape
        logA, Mt = model.apply({"params": params}, s)
        n2 = jnp.real(mps_overlap_aligned(Mt, Mt, CHI))
        logp = 2.0 * jnp.real(logA)
        flip = 1.0 - 2.0 * jnp.eye(N_)
        sf = (s[:, None, :] * flip[None]).reshape(M * N_, N_)                 # s con el espin k invertido
        logAf, Mf = model.apply({"params": params}, sf)
        Of = mps_overlap_aligned(Mf, jnp.repeat(Mt, N_, axis=0), CHI)
        n2f = jnp.real(mps_overlap_aligned(Mf, Mf, CHI))
        Gf = Of / jnp.sqrt(n2f * jnp.repeat(n2, N_))                          # G(s^k, s) normalizado
        g = jnp.exp(logp) + jnp.sum((jnp.exp(2.0 * jnp.real(logAf)) * jnp.abs(Gf) ** 2).reshape(M, N_), axis=1)
        b = pair_block
        nb = M // b
        Mb = Mt.reshape(nb, b, *Mt.shape[1:])
        O = jax.vmap(lambda m: mps_overlap_rect(m, m, CHI))(Mb)
        n2b = n2.reshape(nb, b)
        f = jnp.abs(O) ** 2 / (n2b[:, :, None] * n2b[:, None, :])
        mask = jnp.broadcast_to(1.0 - jnp.eye(b), (nb, b, b))
        lpb = logp.reshape(nb, b)
        lp = lpb[:, :, None] + lpb[:, None, :]
        sb = s.reshape(nb, b, N_)
        m2 = (jnp.einsum("nbk,nck->nbc", sb, sb) <= N_ - 4).astype(f.dtype)
        return g, logp, f, mask, lp, m2

    def make_nbr_loss(pair_block):
        def loss(params, s):
            g, logp, f, mask, lp, m2 = nbr_terms(params, s, pair_block)
            npairs = jnp.sum(mask)
            P_near = jnp.mean(g)
            L_near = jnp.mean(g + sg(g - P_near) * logp)
            h = f * m2
            P_far = jnp.sum(mask * h) / npairs
            L_far = (jnp.sum(mask * h) + jnp.sum(mask * sg(h - P_far) * lp)) / npairs
            Pb = P_near + P_far
            return (L_near + L_far) / sg(Pb), -jnp.log(sg(Pb))               # T=1: grad = grad log P = -grad S2
        return loss

    def make_chunked_vg_nbr(pair_block, nc):
        """Igual que make_chunked_vg pero para 'nbr': P_near, P_far y sus baselines son globales (pasada 1,
        sin gradiente); la pasada 2 suma los gradientes de cada trozo.  EXACTO."""
        def chunk_sums(params, sc):
            g, logp, f, mask, lp, m2 = nbr_terms(params, sc, pair_block)
            return jnp.sum(g), jnp.sum(mask * f * m2)

        def chunk_loss(params, sc, Pn, Pf, Pb, M, npairs):
            g, logp, f, mask, lp, m2 = nbr_terms(params, sc, pair_block)
            h = f * m2
            Ln = jnp.sum(g + sg(g - Pn) * logp) / M
            Lf = (jnp.sum(mask * h) + jnp.sum(mask * sg(h - Pf) * lp)) / npairs
            return (Ln + Lf) / Pb

        @jax.jit
        def vg(params, s):
            M = s.shape[0]
            sch = s.reshape(nc, M // nc, s.shape[1])
            npairs = M * (pair_block - 1.0)
            sg_, sf_ = jax.lax.map(lambda sc: chunk_sums(params, sc), sch)
            Pn = sg(jnp.sum(sg_) / M)
            Pf = sg(jnp.sum(sf_) / npairs)
            Pb = Pn + Pf

            def body(acc, sc):
                gr = jax.grad(chunk_loss)(params, sc, Pn, Pf, Pb, M, npairs)
                return jax.tree_util.tree_map(jnp.add, acc, gr), None
            gr, _ = jax.lax.scan(body, jax.tree_util.tree_map(jnp.zeros_like, params), sch)
            return (0.0, -jnp.log(Pb)), gr
        return vg

    # self-check: coincide con make_mc_loss(terms="P") de training.py (si cambias training.py, esto avisa)
    p0 = states[0][2]
    s_chk = make_sampler(model, N, 64)(p0, jax.random.PRNGKey(123))
    mc = make_mc_loss(model, N, 1.0, HX, JJ, None, 8, terms="P")
    g_ref = ravel_pytree(jax.grad(lambda p: mc(p, s_chk)[0])(p0))[0]
    g_mine = ravel_pytree(jax.grad(lambda p: make_purity_loss(8, None)(p, s_chk)[0])(p0))[0]
    print(f"[self-check] |grad(mi L_P) - grad(make_mc_loss P)|_max = {float(jnp.max(jnp.abs(g_ref - g_mine))):.2e}  "
          f"(debe ser ~0)", flush=True)

    gc_ = ravel_pytree(make_chunked_vg(8, 2)(p0, s_chk)[1])[0]
    gu_ = ravel_pytree(jax.grad(lambda p: make_purity_loss(8, None)(p, s_chk)[0])(p0))[0]
    print(f"[self-check] |grad troceado (2 trozos) - grad sin trocear|_max = {float(jnp.max(jnp.abs(gc_ - gu_))):.2e}  "
          f"(debe ser ~1e-15)", flush=True)

    mc_n = make_mc_loss(model, N, 1.0, HX, JJ, None, 8, terms="P", purity="nbr")
    gn_ref = ravel_pytree(jax.grad(lambda p: mc_n(p, s_chk)[0])(p0))[0]
    gn_mine = ravel_pytree(jax.grad(lambda p: make_nbr_loss(8)(p, s_chk)[0])(p0))[0]
    gn_chk = ravel_pytree(make_chunked_vg_nbr(8, 2)(p0, s_chk)[1])[0]
    print(f"[self-check nbr] |mi grad - make_mc_loss(purity='nbr')|_max = {float(jnp.max(jnp.abs(gn_ref - gn_mine))):.2e}"
          f"   |troceado - sin trocear|_max = {float(jnp.max(jnp.abs(gn_chk - gn_mine))):.2e}   (deben ser ~1e-15)", flush=True)

    grad_fns, samplers = {}, {}

    def parse(spec):
        return ("nbr", spec[1]) if isinstance(spec, tuple) else ("iid", spec)

    def sizes(spec, C):
        """(M, chunk_de_filas_para_all, n_trozos_de_muestras).  C = contracciones de pares (o M con --equal_M).
        Coste por muestra: iid-bloques b -> b ; nbr -> N (vecinos) + b-1 (pares)."""
        kind, pb = parse(spec)
        if pb is None:                                                  # todos los pares
            M = max(C if args.equal_M else int(np.sqrt(C)), 2)
            ch = None
            if args.chunk and M > args.chunk:
                M = (M // args.chunk) * args.chunk
                ch = args.chunk
            return M, ch, 1
        per_cost = pb if kind == "iid" else (N + pb - 1)
        M = C if args.equal_M else C // per_cost
        M = max(M // pb * pb, 2 * pb)
        per = N * 2 * CHI * CHI * 16 * 8 * (1 if kind == "iid" else (N + 1))   # bytes/muestra (tensores MPS x copias de autodiff)
        lim = max(pb, int(args.mem_gb * 2 ** 30 / per))
        nc = 1
        if M > lim:
            nc = -(-M // lim)
            M = max((M // (nc * pb)) * nc * pb, nc * pb)
        return M, None, nc

    def estimate(spec, C, params, key):
        kind, pb = parse(spec)
        M, ch, nc = sizes(spec, C)
        k = (spec, ch, nc)
        if k not in grad_fns:
            if kind == "nbr":
                grad_fns[k] = (make_chunked_vg_nbr(pb, nc) if nc > 1 else
                               jax.jit(jax.value_and_grad(make_nbr_loss(pb), has_aux=True)))
            else:
                grad_fns[k] = (make_chunked_vg(pb, nc) if nc > 1 else
                               jax.jit(jax.value_and_grad(make_purity_loss(pb, ch), has_aux=True)))
        if M not in samplers:
            samplers[M] = make_sampler(model, N, M)
        s = samplers[M](params, key)
        (_, S2), g = grad_fns[k](params, s)
        return float(S2), -np.asarray(ravel_pytree(g)[0], float), M

    def run(iS, spec, C):
        """reps estimaciones del estado iS con el metodo `spec` a presupuesto C."""
        kind, pb = parse(spec)
        params = states[iS][2]
        S2_ex, g_ex = ex[iS]
        keys = jax.random.split(jax.random.PRNGKey(args.seed + 17 * iS), args.reps + 1)
        estimate(spec, C, params, keys[0])                                 # calentamiento (compila)
        out = {"S2": [], "cos": [], "beta": [], "nrm": []}
        t0 = time.perf_counter()
        for r in range(args.reps):
            S2, g, M = estimate(spec, C, params, keys[r + 1])
            c, b, n = metrics(g, g_ex)
            out["S2"].append(S2); out["cos"].append(c); out["beta"].append(b); out["nrm"].append(n)
        out["ms"] = 1e3 * (time.perf_counter() - t0) / args.reps
        out["M"] = M
        out["nc"] = sizes(spec, C)[2]
        out["pairs"] = M * (M - 1) if pb is None else (M * (pb - 1) if kind == "iid" else M * (N + pb - 1))
        return out

    res = {"meta": {"N": N, "chi": CHI, "reps": args.reps, "budget_fixed": args.budget_fixed,
                    "budgets": args.budgets, "equal_M": args.equal_M, "methods": [m for m, _ in METHODS]},
           "states": [{"label": l, "T": T, "S2_ex": e[0]} for (l, T, _), e in zip(states, ex)],
           "hamming": ham,
           "sec1": {m: [] for m, _ in METHODS}, "sec2": {}}

    # ---- Seccion 1: error vs entropia a presupuesto fijo ----
    print(f"\n=== 1. Error vs entropia ({'M = ' + str(args.budget_fixed) + ' muestras' if args.equal_M else 'C = ' + str(args.budget_fixed) + ' contracciones de pares'}) ===", flush=True)
    for iS, (label, T, _) in enumerate(states):
        print(f"\n  {label}   S2_ex = {ex[iS][0]:.4f}", flush=True)
        for m, pb in METHODS:
            r = run(iS, pb, args.budget_fixed)
            res["sec1"][m].append(r)
            d = np.array(r["S2"]) - ex[iS][0]
            print(f"    {m:>4s} M={r['M']:6d}{'/' + str(r['nc']) if r['nc'] > 1 else '':<3s} |dS2|={np.abs(d).mean():.4f} sesgo={d.mean():+.4f}  "
                  f"cos={np.mean(r['cos']):.3f}  beta={np.mean(r['beta']):.3f}+-{np.std(r['beta']):.3f}  "
                  f"|g|/|g_ex|={np.mean(r['nrm']):.2f}  {r['ms']:.0f} ms/est", flush=True)

    # ---- Seccion 2: error vs presupuesto (S2 baja y alta) ----
    order = np.argsort([e[0] for e in ex])
    for lab, iS in (("low", int(order[0])), ("high", int(order[-1]))):
        if len(states) == 1 and lab == "high":
            break
        print(f"\n=== 2. Error vs presupuesto ({lab} S2: {states[iS][0]}, S2_ex={ex[iS][0]:.3f}) ===", flush=True)
        sec = {"label": states[iS][0], "S2_ex": ex[iS][0], "by_method": {m: [] for m, _ in METHODS}}
        for C in args.budgets:
            for m, pb in METHODS:
                r = run(iS, pb, C)
                r["C"] = C
                sec["by_method"][m].append(r)
                d = np.array(r["S2"]) - ex[iS][0]
                print(f"    C={C:7d} {m:>4s} M={r['M']:6d}{'/' + str(r['nc']) if r['nc'] > 1 else '':<3s} |dS2|={np.abs(d).mean():.4f} sesgo={d.mean():+.4f} "
                      f"cos={np.mean(r['cos']):.3f} beta={np.mean(r['beta']):.3f}  {r['ms']:.0f} ms", flush=True)
        res["sec2"][lab] = sec

    with open(JSON_PATH, "w") as f:
        json.dump(res, f)
    print(f"\nresultados crudos: {JSON_PATH}", flush=True)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Figuras
# ══════════════════════════════════════════════════════════════════════════════
def plot(res):
    import matplotlib
    if args.tex:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({"pgf.texsystem": "pdflatex", "font.family": "serif",
                                    "text.usetex": True, "pgf.rcfonts": False})
    else:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        path = os.path.join(PLOTS_DIR, name)
        fig.savefig(path + ".pdf", bbox_inches="tight")
        fig.savefig(path + ".png", bbox_inches="tight", dpi=150)
        if args.tex:
            fig.savefig(path + ".pgf", bbox_inches="tight")
        print(f"  guardado: {path}.pdf")
        plt.close(fig)

    meths = res["meta"]["methods"]
    col = {m: f"C{i}" for i, m in enumerate(meths)}
    mk = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(meths)}
    lab_m = {m: ("all pairs" if m == "all" else (f"neighbors + blocks of {m[3:]}" if m.startswith("nbr") else f"blocks of {m[1:]}"))
             for m in meths}
    S2 = np.array([s["S2_ex"] for s in res["states"]])
    o = np.argsort(S2)
    S2 = S2[o]
    C0 = res["meta"]["budget_fixed"]
    eqM = res["meta"].get("equal_M", False)
    bud_lab = "samples M" if eqM else "budget C (pair contractions)"
    
    def per_state(m, key, fn):
        return np.array([fn(np.array(res["sec1"][m][i][key])) for i in range(len(o))])[o]

    # --- descomposicion de P por distancia de Hamming (por que funciona 'nbr') ---
    if res.get("hamming"):
        hm = sorted(res["hamming"], key=lambda h: -np.log(h["P"]))      # S2 creciente
        nd = min(len(hm[0]["P_d"]), 9)
        ds = np.arange(nd)
        cs = plt.cm.viridis(np.linspace(0.0, 0.85, len(hm)))
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
        w = 0.8 / len(hm)
        for i, h in enumerate(hm):
            Pd, Qd = np.array(h["P_d"]), np.array(h["Q_d"])
            lab = rf"{h['label']}  ($S_2={-np.log(h['P']):.2f}$)"
            ax[0].bar(ds + (i - (len(hm) - 1) / 2) * w, (Pd / Pd.sum())[:nd], w, color=cs[i], label=lab)
            ax[1].semilogy(ds, np.maximum(Qd[:nd], 1e-300), "o-", color=cs[i], label=lab)
            ax[2].semilogy(ds, np.maximum((Pd / np.maximum(Qd, 1e-300))[:nd], 1e-300), "o-", color=cs[i], label=lab)
        ax[0].set_title(r"Share of $P$ by Hamming distance: $P_d/P$")
        ax[1].set_title(r"Prob. an iid pair is at distance $d$: $Q_d$")
        ax[2].set_title(r"Mean ancilla overlap at distance $d$: $\langle f\rangle_d=P_d/Q_d$")
        ax[1].set_ylim(1e-8, 2); ax[2].set_ylim(1e-8, 2)
        for a in ax:
            a.set_xlabel(r"Hamming distance $d(s,t)$"); a.grid(True, alpha=0.3, which="both"); a.legend(fontsize=7)
            a.axvspan(-0.5, 1.5, color="gray", alpha=0.12)
        fig.tight_layout(); save(fig, "hamming_decomposition")

    # --- error de S2 vs entropia ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for m in meths:
        err = np.array([np.abs(np.array(res["sec1"][m][i]["S2"]) - res["states"][i]["S2_ex"]) for i in range(len(o))])
        err = err[o]
        ax[0].plot(S2, err.mean(1), mk[m] + "-", color=col[m], label=lab_m[m])
        ax[0].fill_between(S2, np.percentile(err, 25, 1), np.percentile(err, 75, 1), color=col[m], alpha=0.2)
        ax[1].plot(S2, err.mean(1) / S2, mk[m] + "-", color=col[m], label=lab_m[m])
    ref = np.linspace(S2.min(), S2.max(), 100)
    ax[0].plot(ref, np.exp(ref) / np.sqrt(C0), "k--", label=r"$e^{S_2}/\sqrt{C}$")
    for a, t in zip(ax, (rf"$|\Delta S_2|$  ({'M = ' + str(C0) + ' samples' if eqM else 'C = ' + str(C0) + ' pair contractions'})", r"$|\Delta S_2|/S_2$")):
        a.set_yscale("log"); a.set_xlabel(r"$S_2$ (exact)"); a.set_title(t); a.grid(True, alpha=0.3, which="both")
        a.legend(fontsize=7, ncol=2)
    fig.tight_layout(); save(fig, "s2_error_vs_entropy")

    # --- gradiente vs entropia ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for m in meths:
        for a, key in zip(ax, ("cos", "beta", "nrm")):
            mu = per_state(m, key, np.mean); sd = per_state(m, key, np.std)
            a.errorbar(S2, mu, yerr=sd, fmt=mk[m] + "-", color=col[m], capsize=3, label=lab_m[m])
    ax[0].set_ylabel(r"$\cos(\hat g, g_\mathrm{ex})$"); ax[0].set_title("Direction")
    ax[1].set_ylabel(r"$\beta$"); ax[1].set_title(r"Gain $\beta$ ($T_\mathrm{eff}=\beta T$)")
    ax[1].axhline(1.0, color="k", ls="--", alpha=0.6)
    ax[2].set_ylabel(r"$\|\hat g\|/\|g_\mathrm{ex}\|$"); ax[2].set_title("Norm ratio")
    ax[2].axhline(1.0, color="k", ls="--", alpha=0.6)
    for a in ax:
        a.set_xlabel(r"$S_2$ (exact)"); a.grid(True, alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout(); save(fig, "grad_quality_vs_entropy")

    # --- vs presupuesto (S2 baja / alta) ---
    for lab, sec in res["sec2"].items():
        S2e = sec["S2_ex"]
        fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
        for m in meths:
            rr = sec["by_method"][m]
            C = np.array([r["C"] for r in rr], float)
            err = np.array([np.abs(np.array(r["S2"]) - S2e).mean() for r in rr])
            bias = np.array([np.mean(r["S2"]) - S2e for r in rr])
            ax[0].loglog(C, err, mk[m] + "-", color=col[m], label=lab_m[m])
            ax[1].semilogx(C, bias, mk[m] + "-", color=col[m], label=lab_m[m])
            ax[2].semilogx(C, [np.mean(r["cos"]) for r in rr], mk[m] + "-", color=col[m], label=lab_m[m])
            ax[3].errorbar(C, [np.mean(r["beta"]) for r in rr], yerr=[np.std(r["beta"]) for r in rr],
                           fmt=mk[m] + "-", color=col[m], capsize=3, label=lab_m[m])
        Cr = np.geomspace(C.min(), C.max(), 50)
        ax[0].loglog(Cr, ax[0].lines[0].get_ydata()[0] * np.sqrt(C.min() / Cr), "k--", alpha=0.6, label=r"$\propto C^{-1/2}$")
        ax[1].axhline(0, color="k", ls="--", alpha=0.6); ax[3].axhline(1, color="k", ls="--", alpha=0.6)
        for a, t in zip(ax, (r"$|\Delta S_2|$", r"bias of $\hat S_2$", r"$\cos$", r"$\beta$")):
            a.set_xscale("log"); a.set_xlabel(bud_lab); a.set_title(f"{t} — {lab} $S_2$ ({S2e:.2f})")
            a.grid(True, alpha=0.3, which="both"); a.legend(fontsize=7)
        fig.tight_layout(); save(fig, f"s2_vs_budget_{lab}")


if args.replot:
    with open(JSON_PATH) as f:
        res = json.load(f)
else:
    res = compute()
plot(res)
print("\nDone. Figuras en", os.path.abspath(PLOTS_DIR))