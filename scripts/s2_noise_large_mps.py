"""
s2_noise_large_mps.py
=====================
Ruido del estimador de S2 y de su gradiente para N GRANDE (sin ED), a COSTE IGUALADO, sobre parametros
ya entrenados (ARViT + MPS ancilla).  Equivalente a s2_noise_large.py (swap/DRUT/Wang) para esta arquitectura.

Estimadores (todos dan una estimacion de grad S2 = -grad log P):
  iid : parejas iid dentro de bloques de pair_block                 (coste  M evaluaciones)
  nbr : t=s y vecinos a un espin exactos + parejas iid con d>=2     (coste  M (1+N))
  swap: cadenas de Metropolis sobre pi = |rho|^2/P (swap_mc.py)     (coste  C (3 + evals/paso * n_steps))
        (swap solo da el gradiente, no un valor de S2)

Para cada (chi, T) y cada presupuesto B (evaluaciones del ansatz), repite n_rep veces y mide:
  * std de S2 entre repeticiones (iid, nbr)
  * cos de autoconsistencia: media y std del coseno entre los gradientes de pares de repeticiones
  * cos con un gradiente de REFERENCIA = media de los gradientes nbr del presupuesto mayor
    (no es el exacto: a N grande no existe; sirve para ver sesgo, beta = g.ref/|ref|^2, y |g|/|ref|)
  * tiempo por llamada

Parametros: lee <data_dir>/params/params_{idx:04d}_chi{chi}.msgpack y <data_dir>/results_mps_N{N}_vs_T.json
(para pasar de T a idx), igual que training_mps_ladder.py.

Uso:
  python scripts/s2_noise_large_mps.py --data_dir data/0/mps_N25 --chi 4 --T 1.3 3.0 3.6 \
      --budgets 20000 80000 320000 --n_rep 8 --pair_block 32 --mem_gb 1.5
"""
import argparse, json, os, sys, time, itertools
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from src_renyi.mps_ancilla.training import make_sampler
from src_renyi.mps_ancilla.mc_loss import make_mc_vg, auto_chunks
from src_renyi.mps_ancilla.ladder import build_model, load_params
from src_renyi.mps_ancilla.swap_mc import make_swap_chains, make_swap_loss, MOVES

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data_dir", required=True, help="p.ej. data/0/mps_N25 (con params/ y results_mps_N*_vs_T.json)")
ap.add_argument("--N", type=int, default=None, help="por defecto, el del JSON de resultados")
ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--T", type=float, nargs="+", default=None, help="temperaturas (se toma la mas cercana del JSON)")
ap.add_argument("--idx", type=int, nargs="+", default=None, help="indices directos en vez de --T")
ap.add_argument("--params_file", type=str, default=None, help="un fichero concreto (requiere un unico --T)")
ap.add_argument("--h_x", type=float, default=-1.5)
ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--embedding", type=int, default=16)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--blocks_vit", type=int, default=2)
ap.add_argument("--budgets", type=int, nargs="+", default=[20000, 80000, 320000],
                help="presupuestos B en evaluaciones del ansatz")
ap.add_argument("--n_rep", type=int, default=8)
ap.add_argument("--pair_block", type=int, default=32)
ap.add_argument("--methods", nargs="+", default=["iid", "nbr", "swap"], choices=["iid", "nbr", "swap"])
ap.add_argument("--swap_steps", type=int, nargs="+", default=[10, 40])
ap.add_argument("--swap_moves", nargs="+", default=list(MOVES), choices=list(MOVES))
ap.add_argument("--swap_group", type=int, default=256, help="cadenas por grupo (acota la memoria)")
ap.add_argument("--mem_gb", type=float, default=None)
ap.add_argument("--sample_chunk", type=int, default=None)
ap.add_argument("--out", type=str, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_plots", action="store_true")
args = ap.parse_args()

data_dir = args.data_dir
res_files = [f for f in os.listdir(data_dir) if f.startswith("results_mps_N") and f.endswith("_vs_T.json")]
Tlist, N = None, args.N
if res_files:
    with open(os.path.join(data_dir, res_files[0])) as f:
        rj = json.load(f)
    Tlist = rj["T"]
    N = N or rj.get("N")
assert N, "no se pudo deducir N: pasa --N"
hx, J, chi = args.h_x, args.J_ZZ, args.chi

# ---- seleccion de (idx, T) ----
if args.params_file:
    assert args.T and len(args.T) == 1, "--params_file requiere un unico --T"
    jobs = [(None, args.T[0], args.params_file)]
else:
    if args.idx is not None:
        jobs_idx = args.idx
    else:
        assert args.T and Tlist, "necesito --T (y el JSON de resultados) o --idx"
        jobs_idx = [int(np.argmin([abs(t - T) if t is not None else 1e9 for t in Tlist])) for T in args.T]
    jobs = [(i, Tlist[i] if Tlist else float("nan"),
             os.path.join(data_dir, "params", f"params_{i:04d}_chi{chi}.msgpack")) for i in jobs_idx]

out_dir = args.out or os.path.join(data_dir, "s2_noise")
os.makedirs(out_dir, exist_ok=True)

model = build_model(N, chi, args.embedding, args.heads, args.blocks_vit)
print(f"N={N} chi={chi} dispositivos={jax.devices()} pair_block={args.pair_block} n_rep={args.n_rep}", flush=True)


# ── costes (evaluaciones del ansatz) ──────────────────────────────────────────
MOVE_EVALS = {"flip_t": 1, "flip_s": 1, "flip_both": 2, "indep_t": 0, "indep_s": 0}


def swap_cost(C, n_steps, moves):
    ev = sum(MOVE_EVALS[moves[i % len(moves)]] for i in range(n_steps))
    return C * (1 + 1 + ev + 2)          # pool + arranque, pasos, 2 forward del gradiente (s y t)


def sizes(B, method, n_steps=None):
    pb = args.pair_block
    q = pb * 16                                   # multiplos "redondos": el troceado necesita divisores de M/pair_block

    def rnd(M):
        return M // q * q if M >= q else max(pb, M // pb * pb)
    if method == "iid":
        return rnd(B)
    if method == "nbr":
        return rnd(B // (N + 1))
    g = args.swap_group
    per = swap_cost(1, n_steps, args.swap_moves)
    C = max(1, B // per)
    return C if C < g else C // g * g


# ── generadores de (S2, grad S2) ──────────────────────────────────────────────
def make_runner(params, T, method, B, n_steps=None):
    """devuelve (fn(key)->(S2 o nan, flat grad S2), coste, descripcion)"""
    if method in ("iid", "nbr"):
        M = sizes(B, method)
        nch = auto_chunks(M, N, chi, args.pair_block, args.sample_chunk, args.mem_gb, grad=True)
        vg = jax.jit(make_mc_vg(model, N, 1.0, hx, J, None, args.pair_block, terms="P", purity=method,
                                n_chunks=nch))
        gsz = args.pair_block * 16 if M >= args.pair_block * 16 else M   # el muestreador NO trocea: lo hago aqui
        samp = make_sampler(model, N, gsz)
        cost = M * (1 + (N if method == "nbr" else 0))

        def sampler(params, key):
            return jnp.concatenate([samp(params, jax.random.fold_in(key, k)) for k in range(M // gsz)], axis=0)

        def fn(key):
            s = sampler(params, key)
            (_, (_, S2, _)), g = vg(params, s)
            gf = -ravel_pytree(g)[0]                      # grad L_P = grad log P = -grad S2
            return float(S2), np.asarray(gf)
        return fn, cost, f"M={M} trozos={nch}"
    C = sizes(B, "swap", n_steps)
    g_ = min(C, args.swap_group)
    ngr = C // g_
    run = jax.jit(make_swap_chains(model, N, n_steps, tuple(args.swap_moves)))
    gfun = jax.jit(jax.grad(make_swap_loss(model, N, 1.0)))
    sampler = make_sampler(model, N, g_)

    def fn(key):
        acc_g, accs = None, []
        for k in range(ngr):
            kk = jax.random.fold_in(key, k)
            pool = sampler(params, kk)
            s, t, acc = run(params, jax.random.fold_in(kk, 1), pool)
            gr = ravel_pytree(gfun(params, s, t))[0]
            acc_g = gr if acc_g is None else acc_g + gr
            accs.append({m: float(v) for m, v in acc.items()})
        fn.acc = accs[0]
        return float("nan"), -np.asarray(acc_g) / ngr     # grad L = grad log P -> grad S2 = -grad L
    fn.acc = {}
    return fn, swap_cost(g_, n_steps, args.swap_moves) * ngr, f"C={g_ * ngr} ({ngr}x{g_}) pasos={n_steps}", fn


def cosm(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


all_out = {}
for idx, T, pfile in jobs:
    print("\n" + "=" * 96 + f"\nT = {T:.3f}   idx={idx}   {pfile}\n" + "=" * 96, flush=True)
    params = load_params(pfile, model, N)
    rows = {}                                                   # (label, B) -> dict
    for B in args.budgets:
        for method in args.methods:
            for n_steps in (args.swap_steps if method == "swap" else [None]):
                r = make_runner(params, T, method, B, n_steps)
                fn, cost, desc = r[0], r[1], r[2]
                if method == "swap":
                    fn = r[3]
                label = method if method != "swap" else f"swap{n_steps}"
                key0 = jax.random.PRNGKey(args.seed + 7919 * B % 100003)
                fn(key0)                                         # compilacion
                S2s, gs, ts = [], [], []
                for rep in range(args.n_rep):
                    t0 = time.time()
                    S2, g = fn(jax.random.fold_in(key0, rep + 1))
                    ts.append(time.time() - t0)
                    S2s.append(S2); gs.append(g)
                pair = [cosm(gs[i], gs[j]) for i, j in itertools.combinations(range(len(gs)), 2)]
                rows[(label, B)] = dict(method=method, label=label, B=B, cost=int(cost), desc=desc, S2=S2s,
                                        gs=gs, cos_self_mean=float(np.mean(pair)), cos_self_std=float(np.std(pair)),
                                        time=float(np.mean(ts)), acc=getattr(fn, "acc", {}))
                std = np.std(S2s) if method != "swap" else float("nan")
                mS2 = np.mean(S2s) if method != "swap" else float("nan")
                print(f"  B={B:>9,}  {label:>7s} {desc:<34s} coste={int(cost):>9,} | S2={mS2:7.3f} std={std:6.3f} | "
                      f"cos_self={np.mean(pair):.3f}±{np.std(pair):.3f} | {np.mean(ts):6.2f}s"
                      + (f" | acc " + " ".join(f"{k[:6]}:{v:.2f}" for k, v in fn.acc.items()) if method == "swap" else ""),
                      flush=True)

    # referencia = media de gradientes nbr (o iid) del presupuesto mayor
    refm = "nbr" if "nbr" in args.methods else args.methods[0]
    Bmax = max(args.budgets)
    ref = np.mean(rows[(refm, Bmax)]["gs"], axis=0) if refm != "swap" else None
    if ref is not None:
        print(f"\n  referencia: media de {args.n_rep} gradientes {refm} con B={Bmax:,}   |ref|={np.linalg.norm(ref):.4f}")
        print(f"  {'B':>10s} {'est':>7s} | {'cos_ref':>7s} {'beta':>6s} {'|g|/|ref|':>9s}")
        for (label, B), r in rows.items():
            cs = [cosm(g, ref) for g in r["gs"]]
            be = [g @ ref / (ref @ ref) for g in r["gs"]]
            nr = [np.linalg.norm(g) / np.linalg.norm(ref) for g in r["gs"]]
            r.update(cos_ref=float(np.mean(cs)), beta=float(np.mean(be)), norm_ratio=float(np.mean(nr)))
            print(f"  {B:>10,} {label:>7s} | {np.mean(cs):7.3f} {np.mean(be):6.3f} {np.mean(nr):9.3f}")
        print("  (la referencia comparte muestras con las filas nbr de B maximo: sus cos_ref estan inflados)")

    all_out[f"T={T:.4f}"] = [{k: v for k, v in r.items() if k != "gs"} for r in rows.values()]

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = sorted({r["label"] for r in rows.values()})
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for lab in labels:
            rr = sorted([r for r in rows.values() if r["label"] == lab], key=lambda r: r["cost"])
            c = [r["cost"] for r in rr]
            if not lab.startswith("swap"):
                ax[0].loglog(c, [np.std(r["S2"]) for r in rr], "o-", label=lab)
            ax[1].semilogx(c, [r["cos_self_mean"] for r in rr], "o-", label=lab)
        ax[0].set_xlabel("coste (evaluaciones del ansatz)"); ax[0].set_ylabel("std(S2) entre repeticiones")
        ax[0].set_title(f"ruido de S2   N={N} chi={chi} T={T:.2f}"); ax[0].grid(alpha=.3, which="both"); ax[0].legend()
        ax[1].set_xlabel("coste (evaluaciones del ansatz)"); ax[1].set_ylabel("<cos> entre gradientes (pares)")
        ax[1].set_title("autoconsistencia del gradiente"); ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=.3); ax[1].legend()
        fig.tight_layout()
        fn_png = os.path.join(out_dir, f"noise_N{N}_chi{chi}_T{T:.2f}.png")
        fig.savefig(fn_png, dpi=140); fig.savefig(fn_png[:-4] + ".pdf"); plt.close(fig)
        print(f"  figura: {fn_png}", flush=True)

with open(os.path.join(out_dir, f"noise_N{N}_chi{chi}.json"), "w") as f:
    json.dump(all_out, f, indent=1)
print(f"\nJSON: {os.path.join(out_dir, f'noise_N{N}_chi{chi}.json')}")