"""
filter_anneal.py
================
¿El escalon de ancilla (astep) es solo una GUIA de optimizacion?  Exacto, N <= 10, ruta de ancilla explicita.
  Fase 1: red (ARViT conv + ancilla MPS chi) + astep entrenados juntos desde cero (como filter_joint_exact.py).
  Fase 2: se congela el escalon y se quita poco a poco:  W_lam = (1-lam) W_fit + lam * 1,  lam = 1/K, ..., 1,
          reentrenando la red en cada etapa (warm start).  Con lam=1 el modelo es un modelo SIN filtro normal.
Si el dF final queda muy por debajo del entrenamiento sin filtro desde cero (N=10 chi=16: +0.0073), el escalon
funciona como curriculum y el obstaculo es de optimizacion, no de capacidad.

Uso: python scripts/filter_anneal.py --N 10 --chi 16 --out data/gap_arch/filter/anneal_N10_chi16.json
"""
import argparse, json, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "..")); sys.path.insert(0, script_dir)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import brentq

from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla import training
from mre_teff import MRE

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--chi", type=int, default=16)
ap.add_argument("--d", type=int, default=16); ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--stages", type=int, default=6); ap.add_argument("--stage_steps", type=int, default=600)
ap.add_argument("--stage_lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--step_off", type=float, default=0.5)
ap.add_argument("--out", required=True)
a = ap.parse_args()
N, T = a.N, a.T; assert N <= 10
assert "/data/0/" not in os.path.abspath(a.out)

idx = np.arange(2 ** N)
all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
H = np.zeros((2 ** N, 2 ** N))
H[idx, idx] = -1.0 * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
for i in range(N):
    H[idx, idx ^ (1 << i)] += -1.5
ev, V = np.linalg.eigh(H)
mre = MRE(ev); Ex, Sx, Fx = mre(T)
c_m = brentq(lambda c: c - mre._sums(c)[1] / mre._sums(c)[0] - 2 * T, ev.min() + 1e-9, ev.max() + 70)
p = np.maximum(c_m - ev, 0); p /= p.sum(); s_ref = float(np.sqrt(p @ ev ** 2 - (p @ ev) ** 2))
print(f"N={N} chi={a.chi} MRE F={Fx:.5f} corte={c_m:.4f}", flush=True)
Hj, evj, Vj, all_sj = jnp.asarray(H), jnp.asarray(ev), jnp.asarray(V), jnp.asarray(all_s)
I = jnp.eye(2 ** N)


def ancilla_vectors(M):
    S = M.shape[0]
    v = jnp.zeros((S, 1, M.shape[-1]), M.dtype).at[:, 0, 0].set(1.0)
    for k in range(M.shape[1]):
        v = jnp.einsum("sai,sbij->sbaj", v, M[:, k]).reshape(S, -1, M.shape[-1])
    return v[:, :, 0]


model = build_alt_model(N, a.chi, a.arch, a.d, 4, a.blocks)


def FES(net, W):
    logA, M = model.apply({"params": net}, all_sj)
    A = jnp.exp(logA - jnp.max(jnp.real(logA))); A = A / jnp.linalg.norm(A)
    Phi = ancilla_vectors(M)
    n = jnp.real(jnp.sum(jnp.abs(Phi) ** 2, axis=1))
    O = Phi @ W.astype(Phi.dtype) @ jnp.conj(Phi).T
    rho = A[:, None] * jnp.conj(A)[None, :] * O / jnp.sqrt(n[:, None] * n[None, :])
    Z = jnp.real(jnp.trace(rho))
    E = jnp.real(jnp.sum(Hj.T * rho)) / Z
    S2 = -jnp.log(jnp.sum(jnp.abs(rho) ** 2) / Z ** 2)
    return E - T * S2, (E, S2)


def Wstep(g):
    return (Vj * jax.nn.sigmoid((g[0] - evj) / jnp.exp(g[1]))[None, :]) @ Vj.T


res = []
def log(r):
    res.append(r); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"  >> {r['stage']}: dF={r['dF']:+.5f} dE={r['dE']:+.4f} dS2={r['dS2']:+.4f}", flush=True)


params = {"net": init_params(model, N, a.seed), "g": jnp.asarray([c_m + a.step_off * s_ref, 0.0])}
out = training.train(lambda pp: FES(pp["net"], Wstep(pp["g"])), params, a.steps, a.lr, T,
                     freq=a.steps // 10, tag=f"[astep chi={a.chi}] ")
net, g = out["params_best"]["net"], out["params_best"]["g"]
gl = np.asarray(g).tolist()
log(dict(stage="astep", lam=0.0, c=gl[0], tau=float(np.exp(gl[1])), dF=out["F"] - Fx, dE=out["E"] - Ex,
         dS2=out["S2"] - Sx, F=out["F"]))
Wfit = Wstep(g)
F0, (E0, S0) = FES(net, I)
log(dict(stage="astep_net_sin_filtro", lam=1.0, dF=float(F0) - Fx, dE=float(E0) - Ex, dS2=float(S0) - Sx, F=float(F0)))

for k in range(1, a.stages + 1):
    lam = k / a.stages
    W = (1 - lam) * Wfit + lam * I
    o = training.train(lambda nt, W=W: FES(nt, W), net, a.stage_steps, a.stage_lr, T,
                       freq=a.stage_steps // 3, tag=f"[lam={lam:.3f}] ")
    net = o["params_best"]
    log(dict(stage=f"lam={lam:.3f}", lam=lam, dF=o["F"] - Fx, dE=o["E"] - Ex, dS2=o["S2"] - Sx, F=o["F"]))
print("FIN", flush=True)
