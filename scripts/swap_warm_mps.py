"""
swap_warm_mps.py
================
Test de CADENAS CALIENTES para el estimador swap-MCMC del gradiente de S2 (N grande, sin ED).

Pregunta: el sesgo beta<1 de swap es solo de arranque (cadenas que empiezan en (s,s) y no han mezclado)?
En entrenamiento las cadenas son persistentes, asi que lo que importa es el gradiente de cadenas YA calentadas.

Para cada repeticion se lanzan C cadenas independientes (arranque (s,s), s~p exacto), se avanzan en tramos
de --stride pasos y se toma el gradiente en los checkpoints L = stride, 2 stride, ... (--lengths).
Se mide, por checkpoint (sobre n_rep repeticiones independientes):
  * beta   = g.ref/|ref|^2 con ref = media de gradientes nbr de M grande (muestras INDEPENDIENTES de las de swap)
  * |g|/|ref|, cos con ref, cos_self (entre repeticiones)
  * cos(media_L, media_Lmax): si deja de cambiar al subir L, el gradiente ha convergido
  * coste amortizado por llamada si las cadenas persisten y cada paso de optimizador avanza --stride pasos.
Lectura: si beta ~ 1 a partir de L=L*, con cadenas persistentes basta avanzar --stride pasos por iteracion y el
sesgo de arranque desaparece; si beta se queda < 1 aunque L sea grande, el problema es de mezcla de los movimientos.

Uso:
  python scripts/swap_warm_mps.py --data_dir data/0/mps_N25 --chi 4 --T 3.0 3.6 --C 256 \
      --lengths 20 60 120 300 600 --n_rep 6 --ref_M 2048 --ref_reps 6 --pair_block 32 --mem_gb 1.5
"""
import argparse, os, sys, time, json, itertools
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
ap.add_argument("--data_dir", required=True)
ap.add_argument("--N", type=int, default=None)
ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--T", type=float, nargs="+", required=True)
ap.add_argument("--h_x", type=float, default=-1.5)
ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--embedding", type=int, default=16)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--blocks_vit", type=int, default=2)
ap.add_argument("--C", type=int, default=256, help="cadenas por repeticion (un grupo, acota la memoria)")
ap.add_argument("--lengths", type=int, nargs="+", default=[20, 60, 120, 300, 600])
ap.add_argument("--stride", type=int, default=20, help="pasos por tramo (tambien los pasos/iteracion amortizados)")
ap.add_argument("--n_rep", type=int, default=6)
ap.add_argument("--moves", nargs="+", default=list(MOVES), choices=list(MOVES))
ap.add_argument("--ref_M", type=int, default=2048)
ap.add_argument("--ref_reps", type=int, default=6)
ap.add_argument("--pair_block", type=int, default=32)
ap.add_argument("--mem_gb", type=float, default=None)
ap.add_argument("--sample_chunk", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

res_files = [f for f in os.listdir(args.data_dir) if f.startswith("results_mps_N") and f.endswith("_vs_T.json")]
with open(os.path.join(args.data_dir, res_files[0])) as f:
    rj = json.load(f)
Tlist, N = rj["T"], args.N or rj["N"]
chi, hx, J = args.chi, args.h_x, args.J_ZZ
assert all(L % args.stride == 0 for L in args.lengths), "--lengths deben ser multiplos de --stride"
model = build_model(N, chi, args.embedding, args.heads, args.blocks_vit)
MOVE_EVALS = {"flip_t": 1, "flip_s": 1, "flip_both": 2, "indep_t": 0, "indep_s": 0}
ev_stride = sum(MOVE_EVALS[args.moves[i % len(args.moves)]] for i in range(args.stride))
cosm = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))

run = jax.jit(make_swap_chains(model, N, args.stride, tuple(args.moves)))
gfun = jax.jit(jax.grad(make_swap_loss(model, N, 1.0)))
samp_C = make_sampler(model, N, args.C)
print(f"N={N} chi={chi} C={args.C} stride={args.stride} dispositivos={jax.devices()}", flush=True)

for T in args.T:
    i = int(np.argmin([abs(t - T) for t in Tlist]))
    pfile = os.path.join(args.data_dir, "params", f"params_{i:04d}_chi{chi}.msgpack")
    params = load_params(pfile, model, N)
    print("\n" + "=" * 90 + f"\nT = {Tlist[i]:.3f}  idx={i}  {pfile}\n" + "=" * 90, flush=True)

    # ---- referencia: nbr con M grande (independiente de las cadenas) ----
    M = args.ref_M // args.pair_block * args.pair_block
    nch = auto_chunks(M, N, chi, args.pair_block, args.sample_chunk, args.mem_gb, grad=True)
    vg = jax.jit(make_mc_vg(model, N, 1.0, hx, J, None, args.pair_block, terms="P", purity="nbr", n_chunks=nch))
    gsz = min(M, args.pair_block * 16)
    samp = make_sampler(model, N, gsz)
    refs, S2r = [], []
    t0 = time.time()
    for r in range(args.ref_reps):
        k = jax.random.PRNGKey(args.seed + 100 + r)
        s = jnp.concatenate([samp(params, jax.random.fold_in(k, j)) for j in range(M // gsz)], 0)
        (_, (_, S2, _)), g = vg(params, s)
        refs.append(-np.asarray(ravel_pytree(g)[0])); S2r.append(float(S2))
    ref = np.mean(refs, axis=0)
    # |g_verdadero|^2 SIN sesgo de ruido: producto escalar entre referencias DISTINTAS (el ruido es independiente)
    ref_sq = float(np.mean([refs[a] @ refs[b] for a, b in itertools.combinations(range(len(refs)), 2)]))
    print(f"|ref|^2={ref @ ref:.3f}  |g_verd|^2 (pares distintos)={ref_sq:.3f}  -> |ref| inflado por ruido x{np.sqrt(ref @ ref / ref_sq):.3f}")
    cs_ref = np.mean([cosm(refs[a], refs[b]) for a, b in itertools.combinations(range(len(refs)), 2)])
    print(f"referencia nbr M={M} x{args.ref_reps}: S2={np.mean(S2r):.3f}±{np.std(S2r):.3f}  |ref|={np.linalg.norm(ref):.3f}  "
          f"cos entre refs={cs_ref:.3f} (calidad de la referencia)  [{time.time()-t0:.0f}s]", flush=True)

    # ---- cadenas: gradiente en cada checkpoint ----
    G = {L: [] for L in args.lengths}
    accs = {}
    t0 = time.time()
    for r in range(args.n_rep):
        k = jax.random.PRNGKey(args.seed + 5000 + r)
        pool = samp_C(params, k)
        s, t = pool, pool
        done = 0
        first = True
        while done < max(args.lengths):
            s, t, acc = run(params, jax.random.fold_in(k, done + 1), pool, None if first else s, None if first else t)
            first = False
            done += args.stride
            if done in G:
                G[done].append(-np.asarray(ravel_pytree(gfun(params, s, t))[0]))
                accs[done] = {m: float(v) for m, v in acc.items()}
    print(f"cadenas: {args.n_rep} reps x {args.C} cadenas hasta L={max(args.lengths)}  [{time.time()-t0:.0f}s]\n", flush=True)

    Lmax = max(args.lengths)
    gmax = np.mean(G[Lmax], axis=0)
    cost_am = args.C * (2 + ev_stride + 2)               # por llamada, cadenas persistentes avanzando `stride` pasos
    print(f"coste amortizado por iteracion (cadenas persistentes, {args.stride} pasos): {cost_am:,} evaluaciones")
    print(f"{'L':>5s} | {'beta':>6s} {'beta_corr':>9s} {'|g|/|ref|':>9s} {'cos_ref':>7s} | {'cos_self':>8s} | {'cos(media_L, media_Lmax)':>24s} | "
          f"{'|media_L|/|ref|':>15s} | acc flip_both/s/t")
    for L in args.lengths:
        gs = G[L]
        be = np.mean([g @ ref / (ref @ ref) for g in gs])
        nr = np.mean([np.linalg.norm(g) / np.linalg.norm(ref) for g in gs])
        cr = np.mean([cosm(g, ref) for g in gs])
        cself = np.mean([cosm(gs[a], gs[b]) for a, b in itertools.combinations(range(len(gs)), 2)])
        gm = np.mean(gs, axis=0)
        a_ = accs[L]
        bc = np.mean([g @ ref for g in gs]) / ref_sq
        print(f"{L:5d} | {be:6.3f} {bc:9.3f} {nr:9.3f} {cr:7.3f} | {cself:8.3f} | {cosm(gm, gmax):24.3f} | "
              f"{np.linalg.norm(gm)/np.linalg.norm(ref):15.3f} | {a_['flip_both']:.2f}/{a_['flip_s']:.2f}/{a_['flip_t']:.2f}",
              flush=True)
    print("\nbeta_corr usa |g_verd|^2 estimado sin ruido (la referencia con ruido infla |ref| y hace beta artificialmente < 1).")
    print("beta ~ 1 y estable al subir L => el sesgo era de arranque; cos(media_L, media_Lmax) ~ 1 => convergido.")