"""
test_block_ed_obs.py
====================
Comprueba block_ed.spectrum_obs / mre_obs: C(r) del MRE por bloques = C(r) con la ρ del MRE densa (como
local_obs_heis.py), para j1j2 (J2 = 0.5, 0.0) y xxz (Δ = 0.7), N = 8 y 10, T = 1 y 3; y que el espectro coincide.
Uso: python scripts/test_block_ed_obs.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from src_renyi.mps_ancilla.block_ed import spectrum_obs, mre_obs, mre, spectrum

ok = True
for N in (8, 10):
    D = 2 ** N; idx = np.arange(D); s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(float)

    def ops(r):
        zz = np.zeros(D); xy = np.zeros((D, D))
        for i in range(N):
            j = (i + r) % N; f = s[:, i] != s[:, j]; zz += s[:, i] * s[:, j] / N
            xy[idx[f], (idx ^ (1 << i) ^ (1 << j))[f]] += 2.0 / N
        return np.diag(zz) + xy
    O = {r: ops(r) for r in range(1, N // 2 + 1)}
    for name, kw, Hd in (("j1j2", dict(J2=0.5), N * (O[1] + 0.5 * O[2])), ("j1j2", dict(J2=0.0), N * O[1]),
                         ("xxz", dict(Delta=0.7), None)):
        if Hd is None:
            zz1 = np.diag(np.diag(O[1])); Hd = N * (O[1] - zz1 + 0.7 * zz1)
        ev, Ob, rs = spectrum_obs(name, N, **kw)
        evd, V = np.linalg.eigh(Hd)
        e1 = np.abs(ev - evd).max(); e2 = np.abs(ev - spectrum(name, N, **kw)).max()
        for T in (1.0, 3.0):
            m = mre_obs(ev, Ob, T)
            md = mre(evd, T); p = np.maximum(md["c"] - evd, 0); p /= p.sum(); rho = (V * p) @ V.T
            Cd = np.array([np.sum(O[r] * rho.T) for r in rs])
            err = np.abs(np.array(m["C"]) - Cd).max()
            ok &= err < 1e-10 and e1 < 1e-9 and e2 < 1e-9
            print(f"N={N} {name} {kw} T={T}: max|Δλ| {e1:.1e}/{e2:.1e}   max|ΔC(r)| = {err:.1e}   C = {np.round(m['C'], 4)}",
                  flush=True)
print("TODO OK" if ok else "FALLO", flush=True)
