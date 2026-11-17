"""
p_opt_fixed_anc.py
==================
Prueba (0) de la hipotesis de p(s) (notas_gap.md, 10-11): con las ancillas de un modelo CONGELADAS, optimizar
exactamente la reponderacion de p(s) y ver cuanto baja F.

rho(s,t) = q_s q_t^* G(s,t),  G(s,t) = <phi_t|W|phi_s> / (||psi_s|| ||psi_t||),  psi_s = sqrt(W) phi_s
(W = 1 sin filtro).  Se fija la fase de q_s a la del modelo, arg A_s, y se optimizan los modulos |q_s| = exp(u_s)
(2^N parametros libres, es decir, p(s) ARBITRARIA) con L-BFGS sobre el F exacto.
  - si F_opt ~ F_modelo: A(s) ya da la p(s) optima para sus ancillas -> el gap NO es la reponderacion,
    esta en las ancillas M(s).
  - si F_opt << F_modelo: la red no aprende la reponderacion (hipotesis de p(s) confirmada).
Variante --free_phase: tambien se optimizan las fases (cota aun mas generosa).

Uso: python scripts/p_opt_fixed_anc.py --N 10 --out data/gap_arch/filter/p_opt_N10.json \
        --run anone15k:data/gap_arch/filter/joint_anc_N10_chi16_15k_anone_s1.msgpack:16:anone \
              astep:data/gap_arch/filter/joint_anc_N10_save_astep_s1.msgpack:4:astep
     (etiqueta:params:chi:filtro[:arch]; filtro en anone | astep | achebK | astep_off = params astep sin filtro)
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "..")); sys.path.insert(0, script_dir)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import serialization
from scipy.optimize import minimize

from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from mre_teff import MRE

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--d", type=int, default=16); ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--run", nargs="+", required=True)
ap.add_argument("--maxiter", type=int, default=3000)
ap.add_argument("--free_phase", action="store_true")
ap.add_argument("--out", required=True)
a = ap.parse_args()
N, T = a.N, a.T; assert N <= 10 and "/data/0/" not in os.path.abspath(a.out)

idx = np.arange(2 ** N)
all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
H = np.zeros((2 ** N, 2 ** N))
H[idx, idx] = -1.0 * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
for i in range(N):
    H[idx, idx ^ (1 << i)] += -1.5
ev, V = np.linalg.eigh(H)
Ex, Sx, Fx = MRE(ev)(T)
print(f"N={N} T={T} F_MRE={Fx:.5f}", flush=True)
Y = (H - 0.5 * (ev[0] + ev[-1]) * np.eye(2 ** N)) / (0.5 * (ev[-1] - ev[0]) * 1.0001)
all_sj = jnp.asarray(all_s)


def ancilla_vectors(M):
    S = M.shape[0]
    v = jnp.zeros((S, 1, M.shape[-1]), M.dtype).at[:, 0, 0].set(1.0)
    for k in range(M.shape[1]):
        v = jnp.einsum("sai,sbij->sbaj", v, M[:, k]).reshape(S, -1, M.shape[-1])
    return v[:, :, 0]


def W_of(kind, g):
    if kind in ("anone", "astep_off"): return None
    if kind == "astep":
        w = 1 / (1 + np.exp(-(g[0] - ev) / np.exp(g[1]))); return (V * w[None, :]) @ V.T
    if kind.startswith("acheb"):
        K = int(kind[5:]); Tm, Tk = np.eye(2 ** N), Y; P = g[0] * Tm + g[1] * Tk
        for k in range(2, K + 1):
            Tm, Tk = Tk, 2 * Y @ Tk - Tm; P = P + g[k] * Tk
        return P @ P
    raise ValueError(kind)


res = json.load(open(a.out)) if os.path.exists(a.out) else []
for spec in a.run:
    parts = spec.split(":")
    lab, pth, chi, kind = parts[0], parts[1], int(parts[2]), parts[3]
    arch = ":".join(parts[4:]) if len(parts) > 4 else "vit:conv:r=1"
    model = build_alt_model(N, chi, arch, a.d, 4, a.blocks)
    raw = serialization.msgpack_restore(open(pth, "rb").read())
    ng = len(raw["g"]) if "g" in raw else 1
    pp = serialization.from_bytes({"net": init_params(model, N, 1), "g": jnp.zeros(ng)}, open(pth, "rb").read())
    g = np.asarray(pp["g"])
    logA, M = model.apply({"params": pp["net"]}, all_sj)
    logA = np.asarray(logA); Phi = np.asarray(ancilla_vectors(M))
    W = W_of(kind, g)
    O = Phi @ (W if W is not None else np.eye(2 ** N)).astype(Phi.dtype) @ np.conj(Phi).T     # O(s,t)=<phi_t|W|phi_s>
    nW = np.real(np.diag(O))                                                                     # ||psi_s||^2
    G = O / np.sqrt(nW[:, None] * nW[None, :])
    n0 = np.real(np.sum(np.abs(Phi) ** 2, 1))
    # q del modelo:  |q_s|^2 = |A_s|^2 ||psi_s||^2 / n_s  (con filtro, la reponderacion que hace el filtro)
    lq0 = np.real(logA) + 0.5 * np.log(nW / n0); lq0 -= lq0.max()
    ph0 = np.imag(logA)
    Gj, Hj = jnp.asarray(G), jnp.asarray(H)
    A2G = jnp.abs(Gj) ** 2

    def FES(u, ph):
        q = jnp.exp(u + 1j * ph)
        Z = jnp.sum(jnp.exp(2 * u))
        rho = q[:, None] * jnp.conj(q)[None, :] * Gj
        E = jnp.real(jnp.sum(Hj.T * rho)) / Z
        m2 = jnp.exp(2 * u)
        S2 = -jnp.log(m2 @ A2G @ m2 / Z ** 2)
        return E - T * S2, E, S2

    F0, E0, S0 = map(float, FES(jnp.asarray(lq0), jnp.asarray(ph0)))
    nfree = 2 ** N * (2 if a.free_phase else 1)

    def split(x):
        return (x[:2 ** N], x[2 ** N:]) if a.free_phase else (x, jnp.asarray(ph0))
    fg = jax.jit(jax.value_and_grad(lambda x: FES(*split(x))[0]))
    x0 = np.concatenate([lq0, ph0]) if a.free_phase else lq0.copy()
    t0 = time.time()
    r = minimize(lambda x: tuple(np.asarray(v, np.float64) for v in fg(jnp.asarray(x))), x0, jac=True,
                 method="L-BFGS-B", options=dict(maxiter=a.maxiter, maxcor=50, ftol=1e-15, gtol=1e-10))
    F1, E1, S1 = map(float, FES(*split(jnp.asarray(r.x))))
    u1 = split(jnp.asarray(r.x))[0]
    p0 = np.exp(2 * lq0); p0 /= p0.sum(); p1 = np.exp(2 * np.asarray(u1)); p1 /= p1.sum()
    tv = 0.5 * np.abs(p0 - p1).sum()
    kl = float(np.sum(p1 * (np.log(p1 + 1e-300) - np.log(p0 + 1e-300))))
    rr = dict(label=lab, params=pth, chi=chi, filter=kind, arch=arch, N=N, T=T, F_MRE=Fx, free_phase=a.free_phase,
              dF_model=F0 - Fx, dE_model=E0 - Ex, dS2_model=S0 - Sx, dF_popt=F1 - Fx, dE_popt=E1 - Ex, dS2_popt=S1 - Sx,
              TV_p=tv, KL_popt_model=kl, lbfgs_nit=int(r.nit), lbfgs_msg=str(r.message), secs=time.time() - t0)
    print(f"[{lab}] chi={chi} {kind}: dF modelo={F0 - Fx:+.5f} -> p(s) optima dF={F1 - Fx:+.5f} "
          f"(dE {E1 - Ex:+.4f} dS2 {S1 - Sx:+.4f})  TV(p_mod,p_opt)={tv:.3e} KL={kl:.2e}  nit={r.nit} {r.message}",
          flush=True)
    res.append(rr); json.dump(res, open(a.out, "w"), indent=1)
print("FIN", flush=True)
