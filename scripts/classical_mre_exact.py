"""
classical_mre_exact.py
======================
Rényi-2 (MRE) EXACTO del Ising clasico en anillo, H = J sum s_i s_{i+1} (J=-1), N espines PBC.
E solo depende del numero de paredes k (par): E_k = -J*... = J*(N-2k), degeneracion 2*C(N,k).
MRE: p(s) ∝ (c - E(s))_+ con c - <E> = 2T.  Devuelve E/N, S2, F, <s_i s_{i+r}> (r=1..R) por combinatoria.
<s_i s_{i+r}> a k fijo: (-1)^{#paredes entre i e i+r}; con k paredes repartidas uniformemente entre N enlaces:
  E[(-1)^m] con m ~ hipergeometrica(N, k, r).
Uso: python scripts/classical_mre_exact.py --N 25 --T 1 2 3 [--out ...json]
"""
import argparse, json
import numpy as np
from math import comb, log
from scipy.optimize import brentq
from scipy.stats import hypergeom

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=25); ap.add_argument("--J", type=float, default=-1.0)
ap.add_argument("--T", type=float, nargs="+", default=[3.0]); ap.add_argument("--R", type=int, default=5)
ap.add_argument("--out", default=None)
a = ap.parse_args()
N = a.N
ks = np.arange(0, N + 1, 2)
g = np.array([2.0 * comb(N, int(k)) for k in ks]); Ek = a.J * (N - 2 * ks)
assert abs(g.sum() - 2 ** N) < 1e-6 * 2 ** N
rows = []
for T in a.T:
    def w(c): return g * np.maximum(c - Ek, 0)
    def Emean(c): W = w(c); return (W * Ek).sum() / W.sum()
    c = brentq(lambda c: c - Emean(c) - 2 * T, Ek.min() + 1e-12, Ek.max() + 100 * T + 100)
    W = w(c); Pk = W / W.sum(); ps = np.maximum(c - Ek, 0) / W.sum()      # p de UNA configuracion del sector k
    E = (Pk * Ek).sum(); P = (g * ps ** 2).sum(); S2 = -log(P)
    zz = {f"szz{r}": float(sum(Pk[i] * sum(((-1) ** m) * hypergeom.pmf(m, N, int(k), r) for m in range(0, min(r, int(k)) + 1))
                                for i, k in enumerate(ks))) for r in range(1, a.R + 1)}
    r = dict(N=N, T=T, c=c, E=E, E_per_site=E / N, S2=S2, F=E - T * S2, frac_cut=float((g * (ps > 0)).sum() / 2 ** N), **zz)
    rows.append(r)
    print(f"T={T}: E/N={E/N:+.4f} S2={S2:.4f} F={r['F']:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in zz.items())
          + f" frac_cut={r['frac_cut']:.3e}")
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
