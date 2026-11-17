"""
step_mpo_entanglement.py
========================
Cuanto cuesta como MPO el filtro de ancilla W = sigmoid((c - H)/tau) (TFIM PBC) que, entrenado junto con la red,
cierra ~97% del gap a N=8 (filter_joint_exact.py, astep).  Se mide la entropia de operador / espectro de Schmidt de
W en el corte central (sitios 0..N/2-1 | N/2..N-1) y el bond dimension D necesario para un error relativo de
Frobenius eps.  Referencias: exp(-H/T) (Gibbs, MPO barato conocido) y el propio H (D=3 en OBC, 4-5 en PBC).
c = corte del MRE a T + delta (delta ~ -0.2..-0.4 en los entrenamientos).

Uso: python scripts/step_mpo_entanglement.py --Ns 8 10 12 --taus 0.1 0.2 0.5 1.0
"""
import argparse, os, sys
script_dir = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, script_dir)
import numpy as np
from scipy.optimize import brentq
from mre_teff import MRE

ap = argparse.ArgumentParser()
ap.add_argument("--Ns", type=int, nargs="+", default=[8, 10, 12])
ap.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0])
ap.add_argument("--delta", type=float, default=-0.3)
ap.add_argument("--T", type=float, default=3.0)
a = ap.parse_args()


def op_schmidt(W, N):
    h = N // 2; dl, dr = 2 ** h, 2 ** (N - h)
    # indice = L + 2^h R  ->  reshape C-order (R, L, R', L')
    M = W.reshape(dr, dl, dr, dl).transpose(1, 3, 0, 2).reshape(dl * dl, dr * dr)
    s = np.linalg.svd(M, compute_uv=False)
    return s / np.linalg.norm(s)


def D_for(s, eps):
    tail = np.sqrt(np.maximum(1 - np.cumsum(s ** 2), 0))     # error de Frobenius relativo al truncar a D
    return int(np.argmax(tail < eps) + 1) if np.any(tail < eps) else len(s)


print(f"{'N':>3} {'operador':<22} {'S_op':>6} {'D(1e-2)':>8} {'D(1e-3)':>8} {'D(1e-4)':>8}")
for N in a.Ns:
    idx = np.arange(2 ** N)
    s_ = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(float)
    H = np.zeros((2 ** N, 2 ** N))
    H[idx, idx] = -1.0 * sum(s_[:, i] * s_[:, (i + 1) % N] for i in range(N))
    for i in range(N):
        H[idx, idx ^ (1 << i)] += -1.5
    ev, V = np.linalg.eigh(H)
    mre = MRE(ev)
    c_m = brentq(lambda c: c - mre._sums(c)[1] / mre._sums(c)[0] - 2 * a.T, ev.min() + 1e-9, ev.max() + 70)
    ops = {"H": H, f"exp(-H/T)": (V * np.exp(-(ev - ev.min()) / a.T)) @ V.T}
    for tau in a.taus:
        w = 1 / (1 + np.exp(-(c_m + a.delta - ev) / tau))
        ops[f"step tau={tau}"] = (V * w) @ V.T
    ops["(c-H)_+ (MRE)"] = (V * np.maximum(c_m - ev, 0)) @ V.T
    for name, W in ops.items():
        s = op_schmidt(W, N)
        p = s ** 2; p = p[p > 1e-300]
        print(f"{N:>3} {name:<22} {-(p * np.log(p)).sum():6.3f} {D_for(s, 1e-2):>8} {D_for(s, 1e-3):>8} {D_for(s, 1e-4):>8}",
              flush=True)
