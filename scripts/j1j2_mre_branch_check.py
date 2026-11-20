import sys, json; sys.path.insert(0, '.')
import numpy as np
from scipy.optimize import brentq
from src_renyi.mps_ancilla.block_ed import mre
N = 10; D = 2 ** N; idx = np.arange(D); s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(float)
def ops(r):
    zz = np.zeros(D); xy = np.zeros((D, D))
    for i in range(N):
        j = (i + r) % N; f = s[:, i] != s[:, j]; zz += s[:, i] * s[:, j] / N
        xy[idx[f], (idx ^ (1 << i) ^ (1 << j))[f]] += 2.0 / N
    return zz, xy
O = {r: ops(r) for r in range(1, 6)}
def Hof(J2): return sum(c * N * (np.diag(O[r][0]) + O[r][1]) for r, c in ((1, 1.0), (2, J2)))
def C(rho): pd = np.diag(rho); return np.array([np.sum(O[r][0] * pd) + np.sum(O[r][1] * rho) for r in O])
H05 = Hof(0.5)
net = json.load(open('data/gap_arch/generic/obs/obs_N10_T1_mc_lr1e-3.json'))['nets']['mc_lr1e-3_ema']
Cn, S2n = np.array(net['C']), net['S2']
print(f"red MC (T=1, J2=0.5): S2={S2n:.3f} C={np.round(Cn, 3)}")
for J2 in (0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
    ev, V = np.linalg.eigh(Hof(J2))
    def S2T(T):
        m = mre(ev, T); return m['S2']
    T = brentq(lambda T: S2T(T) - S2n, 0.05, 20)
    m = mre(ev, T); w = np.maximum(m['c'] - ev, 0); w /= w.sum(); rho = (V * w) @ V.T
    Cr = C(rho); E05 = np.sum(H05 * rho)
    print(f"MRE J2'={J2:.1f} T'={T:.3f} (misma S2): C={np.round(Cr, 3)}  |ΔC|={np.linalg.norm(Cr - Cn):.3f}  E(H_J2=0.5)={E05:.3f} (red {net['E']:.3f})")
print("\nMRE J2=0.5 N=10 frente a T:")
ev, V = np.linalg.eigh(H05)
for T in (0.90, 0.94, 0.96, 0.97, 0.98, 0.99, 1.0, 1.02, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0):
    m = mre(ev, T); w = np.maximum(m['c'] - ev, 0); n = (w > 0).sum(); w /= w.sum(); rho = (V * w) @ V.T; Cr = C(rho)
    EJ2 = 0.5 * N * Cr[1]
    print(f"T={T:.2f}: S2={m['S2']:.3f} E={m['E']:.3f} niveles en el soporte={n:4d}  c={m['c']:.3f}  C(2)={Cr[1]:+.3f} E_J2={EJ2:+.3f}")
gaps = np.unique(np.round(ev, 6))[:12]; print("primeros niveles distintos:", np.round(gaps, 3))
