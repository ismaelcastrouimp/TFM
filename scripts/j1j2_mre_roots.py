import sys; sys.path.insert(0, '.')
import numpy as np
from src_renyi.mps_ancilla.block_ed import spectrum
for N in (10, 12, 14, 16, 18, 20):
    ev = np.load(f'data/gap_arch/generic/spectra/j1j2_J20.5_N{N}.npy') if N in (10,16,18,20) else spectrum('j1j2', N, J2=0.5)
    T = 1.0; D = len(ev); S1c, S2c = np.cumsum(ev), np.cumsum(ev ** 2); sols = []
    for k in range(2, D + 1):
        S1, S2_ = S1c[k - 1], S2c[k - 1]; disc = (S1 + T * k) ** 2 - k * (S2_ + 2 * T * S1)
        if disc < 0: continue
        hi = ev[k] if k < D else np.inf
        for c in ((S1 + T * k) + np.sqrt(disc)) / k, ((S1 + T * k) - np.sqrt(disc)) / k:
            if ev[k - 1] < c <= hi + 1e-12:
                w = np.maximum(c - ev, 0); p = w / w.sum(); E = (p * ev).sum(); S2 = -np.log((p ** 2).sum())
                sols.append((E - T * S2, E, S2, k))
    sols = sorted(set((round(a, 6), round(b, 4), round(c_, 4), k) for a, b, c_, k in sols))
    print(f"N={N}: soluciones autoconsistentes a T=1 (F, E, S2, n niveles):", sols[:4])
