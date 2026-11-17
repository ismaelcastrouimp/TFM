"""
mre_teff.py
===========
MRE exacto del TFIM con PBC (H = J_ZZ sum ZZ + h_x sum X, J_ZZ=-1, h_x=-1.5) a cualquier N via fermiones libres
(espectro completo de 2^N niveles, sectores de paridad NS/R), y temperaturas efectivas de un estado variacional:
  T_E  : T a la que el MRE exacto tiene la energia E del modelo
  T_S2 : T a la que el MRE exacto tiene la S2 del modelo
  S2_MRE(E) - S2 : entropia que le falta al modelo frente al MRE con su MISMA energia (el MRE maximiza S2 a E fija).

Validado: espectro = ED a N=8 (error 4e-14); reproduce F(N=8,16,25; T=3) = -19.5966, -39.6865, -62.4769.

Uso:  python scripts/mre_teff.py --N 25 --T 3.0 --runs etiqueta:E:S2 [etiqueta:E:S2 ...]
"""
import argparse
import numpy as np
from scipy.optimize import brentq

J, H = 1.0, 1.5   # H = -J sum ZZ - h sum X : mismo espectro que J_ZZ=-1, h_x=-1.5


def spectrum(N):
    out = []
    for ks, par in (((2 * np.arange(N) + 1) * np.pi / N, 0), ((2 * np.arange(N)) * np.pi / N, 1)):
        eps = np.array([2 * (H - J * np.cos(k)) if np.isclose(np.sin(k), 0)
                        else 2 * np.sqrt(J ** 2 + H ** 2 - 2 * J * H * np.cos(k)) for k in ks])
        E = np.array([-0.5 * eps.sum()]); n = np.array([0], dtype=np.int8)
        for e in eps:
            E = np.concatenate([E, E + e]); n = np.concatenate([n, (n + 1) % 2])
        out.append(E[n == par])
    return np.sort(np.concatenate(out))


class MRE:
    """p_i = max(c - E_i, 0)/Z con c - <E> = 2T.  Sumas acumuladas -> cada evaluacion es O(log 2^N)."""
    def __init__(self, Ev):
        self.E = np.sort(Ev)
        self.c1 = np.concatenate([[0.0], np.cumsum(self.E)])
        self.c2 = np.concatenate([[0.0], np.cumsum(self.E ** 2)])

    def _sums(self, c):
        k = np.searchsorted(self.E, c)
        s1, s2 = self.c1[k], self.c2[k]
        Z = k * c - s1                       # sum w
        WE = c * s1 - s2                     # sum w E
        W2 = k * c * c - 2 * c * s1 + s2     # sum w^2
        return Z, WE, W2

    def __call__(self, T):
        f = lambda c: c - self._sums(c)[1] / self._sums(c)[0] - 2 * T
        c = brentq(f, self.E[0] + 1e-9, self.E[-1] + 20 * T + 10, xtol=1e-13)
        Z, WE, W2 = self._sums(c)
        E, S2 = WE / Z, -np.log(W2 / Z ** 2)
        return E, S2, E - T * S2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=25)
    ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args()
    m = MRE(spectrum(a.N))
    Ex, Sx, Fx = m(a.T)
    print(f"MRE exacto N={a.N} T={a.T}: E={Ex:.5f} S2={Sx:.5f} F={Fx:.5f}")
    print("| run | E | S2 | T_E | T_S2 | S2_MRE(E) | S2_MRE(E) - S2 | F | dF |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in a.runs:
        lab, E, S = r.split(":"); E, S = float(E), float(S)
        TE = brentq(lambda t: m(t)[0] - E, 0.05, 50)
        TS = brentq(lambda t: m(t)[1] - S, 0.05, 50)
        SE = m(TE)[1]; F = E - a.T * S
        print(f"| {lab} | {E:.3f} | {S:.3f} | {TE:.3f} | {TS:.3f} | {SE:.3f} | {SE - S:+.3f} | {F:.3f} | {F - Fx:+.2f} |")


if __name__ == "__main__":
    main()
