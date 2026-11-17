"""
anc_filtered_bond.py
====================
Cuanta bond dimension necesitan los estados de ancilla FILTRADOS  psi_s = sqrt(W) phi_s  de un modelo entrenado con
filtro de ancilla (filter_joint_exact.py --save_params, variantes astep / achebK).
   astep  : sqrt(W) = sqrt(sigmoid((c - H)/tau))
   achebK : W = P(H)^2  ->  psi_s = P(H) phi_s
rho'(s,t) = A(s) A*(t) <psi_t|psi_s> / sqrt(n_s n_t),  n_s = <phi_s|phi_s>  (igual que en el entrenamiento).
Se comprime cada psi_s a un MPS de bond chi' (TT-SVD, OBC, corte en todos los enlaces) y se recalcula el F exacto con
los psi truncados.  Si con chi' moderado se conserva casi toda la ganancia, a N=25 basta una ancilla de bond chi'
(p.ej. parametrizada directamente, o P(H) aplicado con compresion), sin MPO exacto del escalon.

Uso: python scripts/anc_filtered_bond.py --params data/gap_arch/filter/joint_anc_N10_save_astep_s1.msgpack \
        --kind astep --N 10 --chi 4 --chis 4 8 16 32
"""
import argparse, os, sys
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
from mre_teff import MRE

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--kind", required=True)
ap.add_argument("--N", type=int, required=True); ap.add_argument("--chi", type=int, default=4)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--blocks", type=int, default=2); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--chis", type=int, nargs="+", default=[4, 8, 16, 32])
a = ap.parse_args()
N, T = a.N, a.T

idx = np.arange(2 ** N)
all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
H = np.zeros((2 ** N, 2 ** N))
H[idx, idx] = -1.0 * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
for i in range(N):
    H[idx, idx ^ (1 << i)] += -1.5
ev, V = np.linalg.eigh(H)
Ex, Sx, Fx = MRE(ev)(T)

model = build_alt_model(N, a.chi, a.arch, a.d, 4, a.blocks)
ng = 2 if a.kind == "astep" else int(a.kind[5:]) + 1
target = {"net": init_params(model, N, 1), "g": jnp.zeros(ng)}
pp = serialization.from_bytes(target, open(a.params, "rb").read())
g = np.asarray(pp["g"])

if a.kind == "astep":
    w = 1 / (1 + np.exp(-(g[0] - ev) / np.exp(g[1])))
    R = (V * np.sqrt(w)) @ V.T
else:
    Y = (H - 0.5 * (ev[0] + ev[-1]) * np.eye(2 ** N)) / (0.5 * (ev[-1] - ev[0]) * 1.0001)
    Tm, Tk = np.eye(2 ** N), Y; R = g[0] * Tm + g[1] * Tk
    for k in range(2, len(g)):
        Tm, Tk = Tk, 2 * Y @ Tk - Tm; R = R + g[k] * Tk

logA, M = model.apply({"params": pp["net"]}, jnp.asarray(all_s))
logA, M = np.asarray(logA), np.asarray(M)
A = np.exp(logA - logA.real.max()); A /= np.linalg.norm(A)
S = 2 ** N
v = np.zeros((S, 1, a.chi), complex); v[:, 0, 0] = 1
for k in range(N):
    v = np.einsum("sai,sbij->sbaj", v, M[:, k]).reshape(S, -1, a.chi)
Phi = v[:, :, 0]                                   # Phi[s, a], a = sum_k a_k 2^k
n = np.sum(np.abs(Phi) ** 2, axis=1)
Psi = Phi @ R.T                                    # psi_s = R phi_s  (R real simetrica)


def F_of(Ps):
    O = Ps @ np.conj(Ps).T
    rho = A[:, None] * np.conj(A)[None, :] * O / np.sqrt(n[:, None] * n[None, :])
    Z = np.real(np.trace(rho))
    E = np.real(np.sum(H.T * rho)) / Z
    S2 = -np.log(np.sum(np.abs(rho) ** 2) / Z ** 2)
    return E - T * S2, E, S2


def tt_truncate(psi, chi):
    """psi: vector 2^N con a = sum_k a_k 2^k  ->  sitio 0 = bit menos significativo.  TT-SVD izquierda->derecha."""
    t = psi.reshape([2] * N)                               # C-order: eje 0 = bit N-1
    t = np.transpose(t, list(range(N))[::-1])               # eje k = bit k
    cores, rest, r = [], t.reshape(1, -1), 1
    for k in range(N - 1):
        rest = rest.reshape(r * 2, -1)
        U, s, Vh = np.linalg.svd(rest, full_matrices=False)
        r_new = min(chi, len(s))
        cores.append(U[:, :r_new].reshape(r, 2, r_new))
        rest = s[:r_new, None] * Vh[:r_new]; r = r_new
    cores.append(rest.reshape(r, 2, 1))
    out = cores[0]
    for c in cores[1:]:
        out = np.tensordot(out, c, axes=([-1], [0]))
    out = out.reshape([2] * N)
    return np.transpose(out, list(range(N))[::-1]).reshape(-1)


F_phi = F_of(Phi)[0]
F_psi, E_psi, S_psi = F_of(Psi)
print(f"[{a.kind} N={N} chi={a.chi}] sin filtro (mismos params): dF={F_phi - Fx:+.5f} | con filtro exacto: dF={F_psi - Fx:+.5f}"
      f"  (E={E_psi:.4f} S2={S_psi:.4f})")
# entrelazamiento en el corte central, ponderado con p(s) = rho'_ss
p = np.abs(A) ** 2 * np.sum(np.abs(Psi) ** 2, axis=1) / n; p /= p.sum()
h = N // 2
Sent = []
for s_ in range(S):
    sv = np.linalg.svd(Psi[s_].reshape(2 ** (N - h), 2 ** h), compute_uv=False)
    q = sv ** 2 / np.sum(sv ** 2); q = q[q > 1e-30]; Sent.append(-(q * np.log(q)).sum())
Sent = np.array(Sent)
print(f"  entropia de entrelazamiento de psi_s (corte central): media p-ponderada {p @ Sent:.3f}, max {Sent.max():.3f}"
      f"   (log chi = {np.log(a.chi):.3f}, log 2^{h} = {h * np.log(2):.3f})")
for chi in a.chis:
    Pt = np.array([tt_truncate(Psi[s_], chi) for s_ in range(S)])
    Ft = F_of(Pt)[0]
    print(f"  chi'={chi:>3}: dF={Ft - Fx:+.5f}" + ("   (sin truncar: debe coincidir con el filtro exacto)"
          if chi >= 2 ** h else ""), flush=True)
