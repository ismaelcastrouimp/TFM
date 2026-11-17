"""
tilted_grad_check.py
====================
Validacion (N <= 10) del gradiente de pureza del AR inclinado (src_renyi/mps_ancilla/tilted_loss.py) frente al
EXACTO por ED:  g_ex = grad( -T S2 ) = T grad P / P  (autodiff de make_loss_fns).
Con den = base = P_exacta, la media sobre lotes de grad L_P es insesgada para g_ex.  Imprime el error relativo
|<g> - g_ex| / |g_ex| (con su ruido esperado por lotes), el coseno, y lo mismo para el gradiente de mc_loss (nbr, pb)
como referencia.  Tambien compara P_hat medio con P exacta.
Uso: python scripts/tilted_grad_check.py --params X.msgpack --chi 4 --N 10 --M 128 --K 4 --kappa 0.1 --batches 200
"""
import argparse, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from src_renyi.mps_ancilla.training import make_sampler, make_loss_fns
from src_renyi.mps_ancilla.mc_loss import make_mc_vg
from src_renyi.mps_ancilla.tilted_loss import make_tilt_sampler, make_tilted_purity, make_mix_partners
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import load_params

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--N", type=int, required=True); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--M", type=int, default=128)
ap.add_argument("--K", type=int, default=4); ap.add_argument("--kappa", type=float, default=0.1)
ap.add_argument("--batches", type=int, default=200); ap.add_argument("--pb", type=int, default=8)
ap.add_argument("--kappas", type=float, nargs="*", default=None)
a = ap.parse_args()
N, T = a.N, a.T
assert N <= 10
model = build_alt_model(N, a.chi, a.arch, 16, 4, 2); params = load_params(a.params, model, N)
idx = np.arange(2 ** N); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
H = np.zeros((2 ** N, 2 ** N)); H[idx, idx] = -1.0 * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
for i in range(N):
    H[idx, idx ^ (1 << i)] += -1.5
loss, _ = make_loss_fns(model, all_s, H, T, chunk=64)
S2x = float(jax.jit(loss)(params)[1][1]); Px = np.exp(-S2x)
gex = np.asarray(ravel_pytree(jax.grad(lambda p: -T * loss(p)[1][1])(params))[0])
print(f"N={N}: S2 exacta {S2x:.5f}  |g_ex| = {np.linalg.norm(gex):.4g}", flush=True)

tilt = jax.jit(make_tilt_sampler(model, N)); pl = make_tilted_purity(model, N, a.chi, T)
gfun = jax.jit(jax.grad(lambda p, s, t, lq: pl(p, s, t, lq, a.K, Px, Px)[0]))
pfun = jax.jit(lambda p, s, t, lq: pl(p, s, t, lq, a.K, Px, Px)[1][0])
mix = jax.jit(make_mix_partners(model, N, a.kappas, a.K)) if a.kappas else None
smp = make_sampler(model, N, a.M); key = jax.random.PRNGKey(7); G, P = [], []
for _ in range(a.batches):
    key, k1, k2 = jax.random.split(key, 3)
    s = smp(params, k1)
    t, lq = mix(params, s, k2) if mix is not None else tilt(params, jnp.repeat(s, a.K, axis=0), k2, np.log(a.kappa))
    G.append(np.asarray(ravel_pytree(gfun(params, s, t, lq))[0])); P.append(float(pfun(params, s, t, lq)))
G = np.array(G); gm = G.mean(0); noise = np.sqrt((G.var(0, ddof=1) / a.batches).sum())
print(f"INCLINADO kappa={a.kappas or a.kappa} M={a.M} K={a.K}: P_hat/P_ex = {np.mean(P) / Px:.5f} +- {np.std(P) / np.sqrt(len(P)) / Px:.5f}"
      f"   |<g>-g_ex|/|g_ex| = {np.linalg.norm(gm - gex) / np.linalg.norm(gex):.4f} (ruido esperado {noise / np.linalg.norm(gex):.4f})"
      f"   cos = {gm @ gex / np.linalg.norm(gm) / np.linalg.norm(gex):.5f}", flush=True)

vgP = jax.jit(make_mc_vg(model, N, T, -1.5, -1.0, None, a.pb, terms="P", purity="nbr"))
smp2 = make_sampler(model, N, a.M * a.K); G2 = []
for _ in range(a.batches):
    key, k1 = jax.random.split(key)
    _, g = vgP(params, smp2(params, k1)); G2.append(np.asarray(ravel_pytree(g)[0]))
G2 = np.array(G2); gm2 = G2.mean(0); noise2 = np.sqrt((G2.var(0, ddof=1) / a.batches).sum())
print(f"NBR (mc_loss) pb={a.pb} M={a.M * a.K}: |<g>-g_ex|/|g_ex| = {np.linalg.norm(gm2 - gex) / np.linalg.norm(gex):.4f}"
      f" (ruido esperado {noise2 / np.linalg.norm(gex):.4f})   cos = {gm2 @ gex / np.linalg.norm(gm2) / np.linalg.norm(gex):.5f}", flush=True)
