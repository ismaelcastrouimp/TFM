"""
gap_mc_hybrid.py — copia de gap_mc_generic.py con --exact_part {E,S2}: sustituye el gradiente MC de la energía
(E) o de la pureza (S2) por el exacto (Ψ densa, N <= 10), para separar qué estimador limita el entrenamiento.

gap_mc_generic.py
=================
Entrenamiento MC de ARViT + ancilla MPS (F = E − T S2) para CUALQUIER hamiltoniano de espines de
src_renyi/mps_ancilla/generic_eloc.py (tfim, tfim_hz, tfim_dis, xxz, j1j2).
  energía : E_loc genérica (conexiones de NetKet, suma exacta sobre la ancilla), score + camino como mc_loss
  pureza  : estimador inclinado con mezcla de κ (make_tilted_purity + make_mix_partners), como gap_mc_tilted
Referencia: MRE exacto. xxz/j1j2 con PBC por bloques Sz + k (block_ed.py, hasta N ~ 20); resto por H densa (N <= 14).
            N <= 10: además F exacta del estado (ρ completa).
Uso:
  python scripts/gap_mc_generic.py --model j1j2 --J2 0.5 --N 10 --T 1 --chi 8 --arch vit:mlp --steps 3000 \
      --lr 1e-3 --label j1j2_test --out_dir data/gap_arch/generic
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
sys.path.insert(0, script_dir)
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["tfim", "tfim_hz", "xxz", "j1j2"])
ap.add_argument("--N", type=int, required=True); ap.add_argument("--T", type=float, required=True)
ap.add_argument("--obc", action="store_true"); ap.add_argument("--J", type=float, default=None)
ap.add_argument("--h", type=float, default=-1.5); ap.add_argument("--hz", type=float, default=-0.5)
ap.add_argument("--Delta", type=float, default=1.0); ap.add_argument("--J2", type=float, default=0.5)
ap.add_argument("--chi", type=int, default=8); ap.add_argument("--arch", default="vit:conv:r=1")
ap.add_argument("--d", type=int, default=16); ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--init", default="scratch"); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--init_chi", type=int, default=None,
                help="chi de --init; si es menor que --chi se amplía con pad_chi_mlp (ancilla global, estado idéntico)")
ap.add_argument("--pad_noise", type=float, default=1e-3)
ap.add_argument("--M", type=int, default=1024, help="muestras de la energía")
ap.add_argument("--Mp", type=int, default=256); ap.add_argument("--K", type=int, default=5)
ap.add_argument("--kappas", type=float, nargs="*", default=[0.05, 0.1, 0.2, 0.4, 1.0])
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--end_frac", type=float, default=1e-2)
ap.add_argument("--clip", type=float, default=10.0); ap.add_argument("--ema", type=float, default=0.995)
ap.add_argument("--e_chunk", type=int, default=128); ap.add_argument("--log_every", type=int, default=50)
ap.add_argument("--eval_M", type=int, default=4096); ap.add_argument("--eval_batches", type=int, default=16)
ap.add_argument("--max_hours", type=float, default=None)
ap.add_argument("--spectra_dir", default="data/gap_arch/generic/spectra", help="caché de espectros por bloques")
ap.add_argument("--exact_part", default="none", choices=["none", "E", "S2"])
ap.add_argument("--p_chunks", type=int, default=1, help="pureza por trozos de s (mismo estimador: P̂ global fijo como den y base)")
ap.add_argument("--pest", default="tilted", choices=["tilted", "u1", "u1shell", "loc1", "ham2", "loc2", "u1c"],
                help="estimador de pureza: tilted (make_tilted_purity) | u1 (parejas en el sector de Sz de s, "
                     "purity_lowvar) | u1shell (u1 + d=0 y d=2 exactos) | loc1/ham2/loc2 (sector + cascarón por máscaras)")
ap.add_argument("--psym", type=int, default=0, help="forma simétrica de la pureza (todo el score en s; purity_lowvar, sym)")
ap.add_argument("--eform", default="std", choices=["std", "vmc"],
                help="gradiente de la energía: std (score + camino, make_energy_terms) | vmc (vmc_energy, forma de estado puro)")
ap.add_argument("--cmu", type=float, default=0.3, help="pest u1c: μ de la carga transportada (purity_charge.py)")
ap.add_argument("--ckap", type=float, default=0.5, help="pest u1c: κ de las componentes c = -1, 0, +1")
ap.add_argument("--sz_label", type=int, default=0, help="1: etiqueta clásica de Sz en la ancilla (sz_label.py; ρ diagonal por bloques; solo con --pest u1)")
ap.add_argument("--cv", type=int, default=0, help="variable de control en el score de t (purity_lowvar)")
ap.add_argument("--label", required=True); ap.add_argument("--out_dir", default="data/gap_arch/generic")
a = ap.parse_args()

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from src_renyi.mps_ancilla.training import make_sampler, make_loss_fns
from src_renyi.mps_ancilla.tilted_loss import make_mix_partners, make_tilted_purity
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.models import count_params
from src_renyi.mps_ancilla.ladder import init_params, load_params, save_params
from src_renyi.mps_ancilla.generic_eloc import build_hamiltonian, make_energy_terms

assert "/data/0" not in os.path.abspath(a.out_dir), "no escribir en data/0"
os.makedirs(a.out_dir, exist_ok=True)
N, T, chi, tag = a.N, a.T, a.chi, f"[{a.label}] "
H = build_hamiltonian(a.model, N, obc=a.obc, J=a.J, h=a.h, hz=a.hz, Delta=a.Delta, J2=a.J2)
model = build_alt_model(N, chi, a.arch, a.d, a.heads, a.blocks)
if a.sz_label:
    from src_renyi.mps_ancilla.sz_label import SzLabelModel
    assert a.pest != "tilted" and a.exact_part == "none", "--sz_label no vale con el inclinado (parejas fuera del sector: log 0)"
    model = SzLabelModel(model, N); print(f"{tag}etiqueta de Sz en la ancilla: {model.n_label} sitios extra", flush=True)
if a.init == "scratch":
    params = init_params(model, N, a.seed)
elif a.init_chi and a.init_chi != chi:
    from pad_chi_mlp import pad_chi_mlp
    assert a.init_chi < chi
    p_small = load_params(a.init, build_alt_model(N, a.init_chi, a.arch, a.d, a.heads, a.blocks), N)
    params = pad_chi_mlp(p_small, init_params(model, N, a.seed), N, a.init_chi, chi, a.pad_noise, a.seed)
    print(f"{tag}init: {a.init} ampliado chi {a.init_chi} -> {chi} (ruido {a.pad_noise:g})", flush=True)
else:
    params = load_params(a.init, model, N)
if a.eform == "vmc":
    from src_renyi.mps_ancilla.vmc_energy import make_energy_terms_vmc
    terms = make_energy_terms_vmc(model, chi, H)
else:
    terms = make_energy_terms(model, chi, H)
if a.pest == "tilted":
    purity = make_tilted_purity(model, N, chi, T)
    mix = make_mix_partners(model, N, a.kappas, a.K)
else:
    from src_renyi.mps_ancilla.purity_lowvar import make_u1_mix_partners, make_u1_lowvar_purity, make_u1_mask_purity
    if a.pest == "u1c":
        from src_renyi.mps_ancilla.purity_charge import make_charge_mix_partners, charge_comps
        assert not a.cv, "u1c: la variable de control usa la densidad u1"
        purity = make_u1_lowvar_purity(model, N, chi, T, a.kappas, a.K, shell=False, cv_t=False, sym=bool(a.psym))
    elif a.pest in ("u1", "u1shell"):
        purity = make_u1_lowvar_purity(model, N, chi, T, a.kappas, a.K, shell=a.pest == "u1shell", cv_t=bool(a.cv),
                                       sym=bool(a.psym))
    else:
        purity = make_u1_mask_purity(model, N, chi, T, a.kappas, a.K, kind=a.pest, cv_t=bool(a.cv), sym=bool(a.psym))
    mix = (make_charge_mix_partners(model, N, charge_comps(a.ckap), a.cmu, a.K) if a.pest == "u1c"
           else make_u1_mix_partners(model, N, a.kappas, a.K))
assert not (a.psym and a.pest == "tilted"), "--psym solo con los estimadores de purity_lowvar"
sampler = make_sampler(model, N, a.M)
ec = max(1, min(a.e_chunk, a.M)); assert a.M % ec == 0

# ── referencia exacta ──
# MRE: xxz / j1j2 con PBC -> espectro por bloques Sz + k (block_ed, cualquier N hasta ~20; caché en --spectra_dir);
#      resto -> H densa (N <= 14).  N <= 10: además F exacta del estado (ρ completa).
ref, exact, ev = None, None, None
D = 2 ** N
use_blocks = a.model in ("xxz", "j1j2") and not a.obc
if use_blocks:
    from src_renyi.mps_ancilla.block_ed import spectrum
    Jr = 1.0 if a.J is None else a.J
    stag = (f"{a.model}_J2{a.J2}_N{N}" if a.model == "j1j2" else f"{a.model}_D{a.Delta}_N{N}") + ("" if Jr == 1.0 else f"_J{Jr}")
    fsp = os.path.join(a.spectra_dir, stag + ".npy")
    if os.path.exists(fsp):
        ev = np.load(fsp); print(f"{tag}espectro leído de {fsp}", flush=True)
    else:
        t0 = time.time(); ev = spectrum(a.model, N, J=Jr, J2=a.J2, Delta=a.Delta)
        print(f"{tag}espectro por bloques Sz+k: {time.time() - t0:.1f} s", flush=True)
        if not os.path.abspath(a.spectra_dir).count("/data/0"):
            os.makedirs(a.spectra_dir, exist_ok=True); np.save(fsp, ev)
    assert len(ev) == D
if N <= 10 or (ev is None and N <= 14):
    idx = np.arange(D); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
    sp, mels = H.to_jax_operator().get_conn_padded(jnp.asarray(all_s))
    sp, mels = np.asarray(sp), np.asarray(mels).real
    col = (((sp + 1) / 2).astype(np.int64) << np.arange(N)).sum(-1)
    Hd = np.zeros((D, D)); np.add.at(Hd, (np.repeat(idx, sp.shape[1]), col.ravel()), mels.ravel())
    assert np.allclose(Hd, Hd.T)
    if ev is None:
        ev = np.linalg.eigvalsh(Hd)
    else:
        assert np.abs(np.linalg.eigvalsh(Hd) - ev).max() < 1e-8, "espectro por bloques != H densa"
    if N <= 10:
        lossx = jax.jit(make_loss_fns(model, all_s, Hd, T, chunk=64)[0])
        exact = lambda p: (lambda r: (float(r[0]), float(r[1][0]), float(r[1][1])))(lossx(p))
if ev is not None:
    from src_renyi.mps_ancilla.block_ed import mre as mre_fn      # raíz de mínima F (puede haber varias)
    ref = {k: v for k, v in mre_fn(ev, T).items() if k in ("E", "S2", "F")}
    print(f"{tag}MRE exacto: F={ref['F']:.6f} E={ref['E']:.6f} S2={ref['S2']:.5f}  (E0/N={ev[0] / N:.6f})", flush=True)


assert a.exact_part == "none" or N <= 10
if a.exact_part != "none":
    Sx = jnp.asarray(all_s); Hx = jnp.asarray(Hd)

    def psi_dense(p):
        lA, Mt = model.apply({"params": p}, Sx)
        v = Mt[:, 0, :, 0, :]
        for k in range(1, N):
            v = jnp.einsum("sbi,saij->sbaj", v, Mt[:, k]).reshape(D, -1, Mt.shape[-1])
        phi = v[..., 0]; phi = phi / jnp.linalg.norm(phi, axis=1, keepdims=True)
        lA = lA - jnp.max(jnp.real(lA)); return jnp.exp(lA)[:, None] * phi

    def E_exact(p):
        Ps = psi_dense(p); tr = jnp.real(jnp.sum(jnp.abs(Ps) ** 2))
        return jnp.real(jnp.sum(Ps.conj() * (Hx @ Ps))) / tr

    def S2_exact(p):
        Ps = psi_dense(p); tr = jnp.real(jnp.sum(jnp.abs(Ps) ** 2)); G = Ps.conj().T @ Ps
        return -jnp.log(jnp.sum(jnp.abs(G) ** 2) / tr ** 2)


@jax.jit
def grads(p, key):
    k1, k2 = jax.random.split(key)
    s = sampler(p, k1)
    sc = s.reshape(-1, ec, N)
    Ebar = jnp.sum(jax.lax.map(lambda c: terms(p, c, 0.0, a.M)[1], sc))
    gE = jax.lax.map(lambda c: jax.grad(lambda q: terms(q, c, Ebar, a.M)[0])(p), sc)
    gE = jax.tree_util.tree_map(lambda x: jnp.sum(x, 0), gE)
    sp_ = s[: a.Mp]
    t, lq = mix(p, sp_, k2)
    if a.p_chunks == 1:
        (_, (P, _, _, _)), gP = jax.value_and_grad(purity, has_aux=True)(p, sp_, t, lq, a.K)
    else:
        nc = a.p_chunks; mc = a.Mp // nc
        sc_ = sp_.reshape(nc, mc, N); tc_ = t.reshape(nc, mc * a.K, N); lc_ = lq.reshape(nc, mc * a.K)
        P = jnp.mean(jax.lax.map(lambda c: purity(p, c[0], c[1], c[2], a.K)[1][0], (sc_, tc_, lc_)))
        Pf = jax.lax.stop_gradient(P)
        gs = jax.lax.map(lambda c: jax.grad(lambda q: purity(q, c[0], c[1], c[2], a.K, Pf, Pf)[0])(p), (sc_, tc_, lc_))
        gP = jax.tree_util.tree_map(lambda x: jnp.mean(x, 0), gs)
    if a.exact_part == "E":
        Ebar, gE = jax.value_and_grad(E_exact)(p)
    elif a.exact_part == "S2":
        S2x, gS = jax.value_and_grad(S2_exact)(p); gP = jax.tree_util.tree_map(lambda x: -T * x, gS); P = jnp.exp(-S2x)
    return Ebar, -jnp.log(P), jax.tree_util.tree_map(jnp.add, gE, gP)


ev_sampler = make_sampler(model, N, a.eval_M)
ecv = max(1, min(a.e_chunk, a.eval_M))


@jax.jit
def eval_batch(p, key):
    k1, k2 = jax.random.split(key)
    s = ev_sampler(p, k1)
    E = jnp.sum(jax.lax.map(lambda c: terms(p, c, 0.0, a.eval_M)[1], s.reshape(-1, ecv, N)))
    t, lq = mix(p, s[: a.Mp], k2)
    if a.p_chunks == 1:
        _, (P, _, _, _) = purity(p, s[: a.Mp], t, lq, a.K)
    else:
        nc = a.p_chunks; mc = a.Mp // nc
        P = jnp.mean(jax.lax.map(lambda c: purity(p, c[0], c[1], c[2], a.K)[1][0],
                                 (s[: a.Mp].reshape(nc, mc, N), t.reshape(nc, mc * a.K, N), lq.reshape(nc, mc * a.K))))
    return E, P


def evaluate(p, seed):
    key = jax.random.PRNGKey(seed); Es, Ps = [], []
    for _ in range(a.eval_batches):
        key, k = jax.random.split(key); E, P = eval_batch(p, k); Es.append(float(E)); Ps.append(float(P))
    Es, Ps = np.array(Es), np.array(Ps); S2 = -np.log(Ps.mean())
    nb = a.eval_batches
    return dict(E=Es.mean(), E_err=Es.std(ddof=1) / np.sqrt(nb), S2=S2, S2_err=Ps.std(ddof=1) / np.sqrt(nb) / Ps.mean(),
                F=Es.mean() - T * S2)


print(f"{tag}{a.model} N={N} T={T} chi={chi} arch={a.arch} n_params={count_params(params)} M={a.M} Mp={a.Mp} K={a.K} "
      f"steps={a.steps} lr={a.lr}  {jax.devices()}", flush=True)
sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, min(a.warmup, max(a.steps // 5, 1)), a.steps, a.lr * a.end_frac)
tx = optax.chain(optax.clip_by_global_norm(a.clip), optax.adam(sched))
opt = tx.init(params); ema = params; params0 = params


@jax.jit
def update(p, opt, ema, g):
    u, opt = tx.update(g, opt, p); p = optax.apply_updates(p, u)
    return p, opt, jax.tree_util.tree_map(lambda e, x: a.ema * e + (1 - a.ema) * x, ema, p)


key = jax.random.PRNGKey(a.seed + 100); hist = []
t0 = t_last = time.time()
for it in range(1, a.steps + 1):
    key, k = jax.random.split(key)
    E, S2, g = grads(params, k)
    params, opt, ema = update(params, opt, ema, g)
    F = float(E) - T * float(S2); hist.append((float(E), float(S2), F))
    if it % a.log_every == 0 or it == 1:
        msg = f"{tag}{it:6d} {(time.time() - t_last) / (a.log_every if it > 1 else 1):.3f}s/paso | MC E={float(E):.4f} S2={float(S2):.4f} F={F:.4f}"
        if ref:
            msg += f" dF/N={(F - ref['F']) / N:+.5f}"
        if exact and it % (5 * a.log_every) == 0:
            Fx, Ex, Sx = exact(params); msg += f" | exacto F={Fx:.5f} dF/N={(Fx - ref['F']) / N:+.6f}"
        print(msg, flush=True); t_last = time.time()
    if a.max_hours and time.time() - t0 > 3600 * a.max_hours:
        print(f"{tag}max_hours en el paso {it}", flush=True); break

ema_hat = jax.tree_util.tree_map(lambda e, p0: (e - a.ema ** it * p0) / (1 - a.ema ** it), ema, params0)
res = dict(args=vars(a), n_params=count_params(params), steps_done=it, time_s=time.time() - t0, ref=ref)
for name, p in (("last", params), ("ema", ema_hat)):
    r = evaluate(p, a.seed + 999); res[name] = r
    msg = f"{tag}FINAL {name}: E={r['E']:.5f}±{r['E_err']:.5f} S2={r['S2']:.5f}±{r['S2_err']:.5f} F={r['F']:.5f}"
    if ref:
        r["dF_per_site"] = (r["F"] - ref["F"]) / N; msg += f" dF/N={r['dF_per_site']:+.5f}"
    if exact:
        Fx, Ex, Sx = exact(p); r.update(F_ex=Fx, E_ex=Ex, S2_ex=Sx, dF_ex_per_site=(Fx - ref["F"]) / N)
        msg += f" | exacto F={Fx:.5f} E={Ex:.5f} S2={Sx:.5f} dF/N={(Fx - ref['F']) / N:+.6f}"
    print(msg, flush=True)
    save_params(os.path.join(a.out_dir, f"{a.label}_params{'' if name == 'last' else '_ema'}.msgpack"), p)
np.save(os.path.join(a.out_dir, f"{a.label}_hist.npy"), np.array(hist))
json.dump(res, open(os.path.join(a.out_dir, f"{a.label}.json"), "w"), indent=1, default=float)
print(f"{tag}guardado {a.out_dir}/{a.label}.json  ({(time.time() - t0) / 60:.1f} min)", flush=True)
