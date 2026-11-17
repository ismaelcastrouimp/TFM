"""
reweight_tests.py
=================
Pruebas (a) y (b) de la hipotesis de p(s) (notas_gap.md, 10-11).  Exacto, N <= 10, CPU.
Profesor: modelo astep guardado (filter_joint_exact.py --save_params).  Su filtro reescala p(s) por
r(s) = ||sqrt(W) phi_s||^2 / n_s  y cambia las ancillas a  psi_s = sqrt(W) phi_s.

  --mode frozen_anc  (b): ancillas = las psi_s normalizadas del profesor, CONGELADAS (Gram fijo).  Se entrena
                     solo A(s) (red nueva desde cero).  ¿Aprende A(s) la reponderacion que el filtro hace gratis?
                     (cota: p optima con esas ancillas = +0.00102, p_opt_fixed_anc.py)
  --mode oracle      (a): red nueva SIN filtro, pero p(s) = |A(s)|^2 r(s) con r(s) del profesor fijo.
                     ¿Baja de +0.005-0.007 si la reponderacion viene dada?
  --mode none        control: misma ruta, sin nada (= anone).

Uso: python scripts/reweight_tests.py --teacher data/gap_arch/filter/joint_anc_N10_save_astep_s1.msgpack \
        --N 10 --chi 16 --mode oracle --out data/gap_arch/filter/reweight_N10.json
"""
import argparse, json, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "..")); sys.path.insert(0, script_dir)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import serialization

from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla import training
from mre_teff import MRE

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True); ap.add_argument("--N", type=int, required=True)
ap.add_argument("--chi_t", type=int, default=4); ap.add_argument("--chi", type=int, default=16)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--blocks", type=int, default=2); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--mode", required=True, choices=["frozen_anc", "oracle", "none"])
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
Hj, all_sj = jnp.asarray(H), jnp.asarray(all_s)


def ancilla_vectors(M):
    S = M.shape[0]
    v = jnp.zeros((S, 1, M.shape[-1]), M.dtype).at[:, 0, 0].set(1.0)
    for k in range(M.shape[1]):
        v = jnp.einsum("sai,sbij->sbaj", v, M[:, k]).reshape(S, -1, M.shape[-1])
    return v[:, :, 0]


# ---- profesor: Gram filtrado normalizado y r(s) ----
mt = build_alt_model(N, a.chi_t, "vit:conv:r=1", a.d, 4, a.blocks)
pp = serialization.from_bytes({"net": init_params(mt, N, 1), "g": jnp.zeros(2)}, open(a.teacher, "rb").read())
g = np.asarray(pp["g"])
W = (V * (1 / (1 + np.exp(-(g[0] - ev) / np.exp(g[1]))))[None, :]) @ V.T
logAt, Mt = mt.apply({"params": pp["net"]}, all_sj)
Phi_t = np.asarray(ancilla_vectors(Mt))
O = Phi_t @ W.astype(Phi_t.dtype) @ np.conj(Phi_t).T
nW = np.real(np.diag(O)); n0 = np.real(np.sum(np.abs(Phi_t) ** 2, 1))
G_t = jnp.asarray(O / np.sqrt(nW[:, None] * nW[None, :]))
log_r = jnp.asarray(np.log(nW / n0))
print(f"N={N} T={T} F_MRE={Fx:.5f}  mode={a.mode}  r(s): max/min = {np.exp(np.ptp(np.asarray(log_r))):.1f}", flush=True)


def FES(rho):
    Z = jnp.real(jnp.trace(rho))
    E = jnp.real(jnp.sum(Hj.T * rho)) / Z
    S2 = -jnp.log(jnp.sum(jnp.abs(rho) ** 2) / Z ** 2)
    return E - T * S2, E, S2


model = build_alt_model(N, a.chi, a.arch, a.d, 4, a.blocks)


def rho_of(net):
    logA, M = model.apply({"params": net}, all_sj)
    if a.mode == "oracle":
        logA = logA + 0.5 * log_r
    A = jnp.exp(logA - jnp.max(jnp.real(logA)))
    if a.mode == "frozen_anc":
        G = G_t
    else:
        Phi = ancilla_vectors(M); n = jnp.real(jnp.sum(jnp.abs(Phi) ** 2, 1))
        G = (Phi @ jnp.conj(Phi).T) / jnp.sqrt(n[:, None] * n[None, :])
    return A[:, None] * jnp.conj(A)[None, :] * G


def loss(net):
    F, E, S2 = FES(rho_of(net)); return F, (E, S2)


net = init_params(model, N, a.seed)
out = training.train(loss, net, a.steps, a.lr, T, freq=max(1, a.steps // 10), tag=f"[{a.mode} chi={a.chi}] ")
F, E, S2 = map(float, FES(rho_of(out["params_best"])))
r = dict(mode=a.mode, N=N, T=T, chi=a.chi, arch=a.arch, seed=a.seed, steps=a.steps, lr=a.lr, teacher=a.teacher,
         F=F, E=E, S2=S2, F_MRE=Fx, dF=F - Fx, dE=E - Ex, dS2=S2 - Sx, s_per_step=out["s_per_step"])
print(f"[{a.mode} chi={a.chi}] FINAL dF={F - Fx:+.5f} dE={E - Ex:+.4f} dS2={S2 - Sx:+.4f}", flush=True)
res = json.load(open(a.out)) if os.path.exists(a.out) else []
res.append(r); json.dump(res, open(a.out, "w"), indent=1)
