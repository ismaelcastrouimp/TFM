"""
grad_bias_test.py
=================
Medida DIRECTA del sesgo del gradiente de S2 que usa el entrenamiento (mc_loss.py, pureza nbr), sin entrenar.
Para un modelo FIJO, con K lotes independientes k de M muestras y pair_block pb:
    G_k = grad del termino de pureza del lote  = T * gradP_k / P_k     (lo que ve Adam en cada paso)
    P_k = pureza estimada del lote
  verdadero (ratio de medias):   R = <G_k P_k> / <P_k>          (= T grad P / P, consistente para K grande)
  lo que da el entrenamiento:    B = <G_k>                      (media del cociente por lote)
  sesgo:                         b = B - R,  error por linealizacion (delta method) sobre los K lotes.
Imprime |b|/|R|, cos(b, R), la proyeccion de b sobre R/|R| en sigmas, y lo compara con el gradiente de la
energia (mismos lotes) para ver si el sesgo pesa frente a la parte de E.  Guarda un json.
Uso:  python scripts/grad_bias_test.py --params X.msgpack --chi 8 --N 25 --M 1024 --pair_block 8 --K 400 --out f.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from src_renyi.mps_ancilla.training import make_sampler
from src_renyi.mps_ancilla.mc_loss import make_mc_vg, auto_chunks
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import load_params

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--N", type=int, default=25); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--h_x", type=float, default=-1.5); ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--M", type=int, default=1024); ap.add_argument("--pair_block", type=int, default=8)
ap.add_argument("--K", type=int, default=400); ap.add_argument("--mem_gb", type=float, default=4.0)
ap.add_argument("--seed", type=int, default=31337); ap.add_argument("--out", required=True)
a = ap.parse_args()
assert "/data/0/" not in os.path.abspath(a.out)
N, T = a.N, a.T
model = build_alt_model(N, a.chi, a.arch, a.d, 4, a.blocks); params = load_params(a.params, model, N)
nch = auto_chunks(a.M, N, a.chi, a.pair_block, None, a.mem_gb, grad=True)
sampler = jax.jit(make_sampler(model, N, a.M))
vgP = jax.jit(make_mc_vg(model, N, T, a.h_x, a.J_ZZ, None, a.pair_block, terms="P", purity="nbr", n_chunks=nch))
vgE = jax.jit(make_mc_vg(model, N, T, a.h_x, a.J_ZZ, None, a.pair_block, terms="E", purity="nbr", n_chunks=nch))
flat0, _ = ravel_pytree(params); D = flat0.size
print(f"N={N} M={a.M} pb={a.pair_block} K={a.K} chunks={nch} n_params={D}", flush=True)

key = jax.random.PRNGKey(a.seed); G, P, GE = [], [], []; t0 = time.time()
for k in range(a.K):
    key, sub = jax.random.split(key)
    s = sampler(params, sub)
    (_, (E, S2, _)), g = vgP(params, s)
    (_, _), gE = vgE(params, s)
    G.append(np.asarray(ravel_pytree(g)[0])); P.append(float(np.exp(-S2))); GE.append(np.asarray(ravel_pytree(gE)[0]))
    if k % max(1, a.K // 10) == 0:
        print(f"  lote {k}  S2_lote={float(S2):.4f}  ({time.time() - t0:.0f} s)", flush=True)
G, P, GE = np.array(G), np.array(P), np.array(GE)
Pb = P.mean(); R = (G * P[:, None]).mean(0) / Pb; B = G.mean(0); b = B - R
psi = (G - B) - ((G * P[:, None]) - R[None, :] * P[:, None] - (G * P[:, None]).mean(0) + R * Pb) / Pb   # influencia de b
se_b = psi.std(0, ddof=1) / np.sqrt(a.K)
gEm = GE.mean(0); se_gE = GE.std(0, ddof=1) / np.sqrt(a.K)
nR = np.linalg.norm(R); u = R / nR
proj = b @ u; proj_se = (psi @ u).std(ddof=1) / np.sqrt(a.K)
gtot_true = gEm + R
res = dict(params=a.params, N=N, M=a.M, pair_block=a.pair_block, K=a.K, n_params=int(D),
           S2_ratio_of_means=float(-np.log(Pb)), S2_mean_of_logs=float(np.mean(-np.log(P))),
           norm_R=float(nR), norm_b=float(np.linalg.norm(b)), norm_b_noise=float(np.sqrt((se_b ** 2).sum())),
           cos_b_R=float(b @ R / (np.linalg.norm(b) * nR)), proj_b_on_R=float(proj), proj_se=float(proj_se),
           chi2_b=float(((b[se_b > 0] / se_b[se_b > 0]) ** 2).sum()), norm_gE=float(np.linalg.norm(gEm)),
           cos_b_gtrue=float(b @ gtot_true / (np.linalg.norm(b) * np.linalg.norm(gtot_true))),
           proj_b_on_gtrue=float(b @ gtot_true / np.linalg.norm(gtot_true)),
           norm_gtrue=float(np.linalg.norm(gtot_true)), secs=time.time() - t0)
print(f"S2: -log<P>={res['S2_ratio_of_means']:.4f}   <-log P_lote>={res['S2_mean_of_logs']:.4f}", flush=True)
print(f"|R| (grad S2 verdadero*T)={nR:.4g}  |b| (sesgo)={res['norm_b']:.4g}  (ruido esperado de |b| ~{res['norm_b_noise']:.4g})"
      f"  chi2(b)={res['chi2_b']:.0f} / {D}", flush=True)
print(f"cos(b,R)={res['cos_b_R']:+.3f}  proyeccion de b sobre R/|R| = {proj:+.4g} +- {proj_se:.2g} ({proj / proj_se:+.1f} sigma)", flush=True)
print(f"|grad E|={res['norm_gE']:.4g}  |grad F verdadero|={res['norm_gtrue']:.4g}  cos(b, gradF)={res['cos_b_gtrue']:+.3f}"
      f"  proyeccion b sobre gradF = {res['proj_b_on_gtrue']:+.4g}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
