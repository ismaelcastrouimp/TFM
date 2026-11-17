"""
s2_bias_check.py
================
Prueba 6 (notas_gap.md): sesgo de S2 = -log(media de P) a N grande.

El estimador nbr calcula EXACTA la parte de la pureza con d(s,t) <= 1 y estima con parejas iid la de d >= 2.
P es insesgado, pero si la parte lejana tiene colas pesadas la media muestral se queda corta con
probabilidad alta -> S2 inflada y F aparentemente mejor.  Aqui se comparan, sobre EL MISMO estado:

  nbr1 : exacto d<=1  + iid d>=2          (el de siempre)
  nbr2 : exacto d<=2  + iid d>=3          (N(N-1)/2 configuraciones mas por muestra)
y para cada uno la S2 acumulada al ir anadiendo lotes (si sigue bajando al anadir muestras -> sesgo de colas),
mas la fraccion de P en d=0, d=1, d=2 y d>=3.

Si nbr2 da una S2 menor que nbr1 de forma significativa, nbr1 tenia sesgo (S2 real menor, hueco real mayor).

Uso:
  python scripts/s2_bias_check.py --params data/0/mps_N25/params/params_0030_chi4.msgpack --N 25 --chi 4 \
      --T 3.0 --M 1024 --batches 64 --pair_block 32 --out data/gap_arch/s2_bias_data0_chi4.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src_renyi.mps_ancilla.ladder import build_model, load_params
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.models import mps_overlap_aligned, mps_overlap_rect
from src_renyi.mps_ancilla.training import make_sampler

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--N", type=int, required=True)
ap.add_argument("--chi", type=int, required=True); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--d", type=int, default=16); ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--arch", default=None)
ap.add_argument("--M", type=int, default=1024, help="muestras por lote")
ap.add_argument("--batches", type=int, default=64)
ap.add_argument("--pair_block", type=int, default=32)
ap.add_argument("--cfg_chunk", type=int, default=8192, help="configuraciones por trozo en las evaluaciones del modelo")
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--out", required=True)
args = ap.parse_args()
assert not os.path.abspath(args.out).startswith(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "0")) + os.sep), "no escribir en data/0"
N, chi = args.N, args.chi
model = (build_alt_model(N, chi, args.arch, args.d, 4, args.blocks) if args.arch
         else build_model(N, chi, args.d, 4, args.blocks))
params = load_params(args.params, model, N)
sampler = make_sampler(model, N, args.M)

pairs = np.array([(i, j) for i in range(N) for j in range(i + 1, N)])                 # (N(N-1)/2, 2)
F1 = 1.0 - 2.0 * np.eye(N)                                                              # flips de 1 espin
F2 = np.ones((len(pairs), N)); F2[np.arange(len(pairs)), pairs[:, 0]] = -1; F2[np.arange(len(pairs)), pairs[:, 1]] = -1
FL = jnp.asarray(np.concatenate([F1, F2]))                                              # (N + N(N-1)/2, N)
nF1, nF = N, FL.shape[0]


def fwd(s):
    """log p, M para un lote grande, en trozos."""
    n = s.shape[0]; c = args.cfg_chunk
    pad = (-n) % c
    sp = jnp.concatenate([s, jnp.ones((pad, N))]) if pad else s
    def one(x):
        logA, M = model.apply({"params": params}, x)
        return 2.0 * jnp.real(logA), M
    lp, M = jax.lax.map(one, sp.reshape(-1, c, N))
    return lp.reshape(-1)[:n], M.reshape(-1, *M.shape[2:])[:n]


@jax.jit
def batch_stats(s):
    Ms = s.shape[0]
    lp, Mt = fwd(s)
    n2 = jnp.real(mps_overlap_aligned(Mt, Mt, chi))
    sf = (s[:, None, :] * FL[None]).reshape(Ms * nF, N)
    lpf, Mf = fwd(sf)
    Mrep = jnp.repeat(Mt, nF, axis=0)
    O = mps_overlap_aligned(Mf, Mrep, chi)
    n2f = jnp.real(mps_overlap_aligned(Mf, Mf, chi))
    w = (jnp.exp(lpf) * jnp.abs(O) ** 2 / (n2f * jnp.repeat(n2, nF))).reshape(Ms, nF)
    g0 = jnp.exp(lp)                                                                    # d=0: p(s) f(s,s)=p(s)
    g1 = w[:, :nF1].sum(1)
    g2 = w[:, nF1:].sum(1)
    # parte lejana con parejas iid dentro de bloques
    b = args.pair_block; nb = Ms // b
    Mb = Mt.reshape(nb, b, *Mt.shape[1:])
    Ob = jax.vmap(lambda m: mps_overlap_rect(m, m, chi))(Mb)
    n2b = n2.reshape(nb, b)
    f = jnp.abs(Ob) ** 2 / (n2b[:, :, None] * n2b[:, None, :])
    sb = s.reshape(nb, b, N)
    dist = (N - jnp.einsum("nbk,nck->nbc", sb, sb)) / 2
    off = 1.0 - jnp.eye(b)
    npairs = Ms * (b - 1)
    far2 = jnp.sum(off * f * (dist >= 2)) / npairs
    far3 = jnp.sum(off * f * (dist >= 3)) / npairs
    return jnp.array([g0.mean(), g1.mean(), g2.mean(), far2, far3])


t0 = time.time()
rows = []
for k in range(args.batches):
    s = sampler(params, jax.random.PRNGKey(args.seed + k))
    rows.append(np.asarray(batch_stats(s)))
    if k == 0:
        print(f"primer lote {time.time() - t0:.1f} s", flush=True)
R = np.array(rows)                                  # (batches, 5): P_d0, P_d1, P_d2, far(d>=2), far(d>=3)
P1 = R[:, 0] + R[:, 1] + R[:, 3]                    # nbr1
P2 = R[:, 0] + R[:, 1] + R[:, 2] + R[:, 4]          # nbr2
out = {"params": args.params, "N": N, "chi": chi, "M": args.M, "batches": args.batches, "time_s": time.time() - t0}
for name, P in (("nbr1", P1), ("nbr2", P2)):
    S2 = -np.log(P.mean())
    # error de S2 por propagacion: std(P)/(sqrt(n) mean P)
    err = P.std(ddof=1) / np.sqrt(len(P)) / P.mean()
    cum = [-np.log(P[:n].mean()) for n in (1, 2, 4, 8, 16, 32, 64, 128, 256) if n <= len(P)]
    out[name] = {"S2": S2, "S2_err": err, "S2_acumulada_n=1,2,4,...": cum, "cv_lote": P.std(ddof=1) / P.mean()}
    print(f"{name}: S2 = {S2:.4f} +- {err:.4f}   acumulada (1,2,4,..lotes) = {' '.join(f'{c:.3f}' for c in cum)}  "
          f"CV por lote {P.std(ddof=1) / P.mean():.2f}", flush=True)
Pm = R.mean(0)
tot = Pm[0] + Pm[1] + Pm[2] + Pm[4]
out["fracciones_P(nbr2)"] = {"d0": Pm[0] / tot, "d1": Pm[1] / tot, "d2": Pm[2] / tot, "d>=3": Pm[4] / tot}
out["cv_por_parte"] = {k: float(R[:, i].std(ddof=1) / max(R[:, i].mean(), 1e-300))
                       for i, k in enumerate(["d0", "d1", "d2", "far>=2", "far>=3"])}
print("fracciones de P (nbr2):", {k: round(v, 3) for k, v in out["fracciones_P(nbr2)"].items()})
print("CV por lote de cada parte:", {k: round(v, 2) for k, v in out["cv_por_parte"].items()})
print(f"dS2 (nbr1 - nbr2) = {out['nbr1']['S2'] - out['nbr2']['S2']:+.4f}   ({time.time() - t0:.0f} s)")
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1, default=float)
