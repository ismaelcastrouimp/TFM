"""
mre_dos.py
==========
MRE exacto del TFIM periodico (H = -J sum ZZ - h sum X; J=1, h=1.5 <=> J_ZZ=-1, h_x=-1.5) a N GRANDE (N=50, 100, ...)
sin enumerar los 2^N niveles: densidad de estados por CONVOLUCION modo a modo (fermiones libres), en cada sector de
paridad (NS / R, como mre_teff.py), sobre una rejilla de energia de paso --dE.  Cada modo eps_k desplaza el histograma
(reparto lineal entre las dos celdas vecinas: conserva cuentas y energia media) y cambia la paridad.
Con la DOS g(E):  p ∝ (c - E)_+ ,  c - <E> = 2T  ->  E, S2 = -log(sum w^2 / Z^2), F = E - T S2.
Observables por Hellmann-Feynman (F_MRE es el minimo de F_2):  dF/dh = -sum <X>,  dF/dJ = -sum <ZZ>
  -> sx = -(1/N) dF/dh,  szz1 = -(1/N) dF/dJ  (diferencias centrales).
T = 0: estado fundamental (E0 exacta del sector fisico; S2 = 0).
Validacion (--validate): N=16/25 frente a mre_teff.py (espectro completo) y a ff_N25 (szz1, sx).
Uso: python scripts/mre_dos.py --N 50 --T 0 0.1 ... 4.0 --out data/gap_arch/n50/mre_N50.json
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import brentq


def sector_eps(N, J, h):
    out = []
    for ks, par in (((2 * np.arange(N) + 1) * np.pi / N, 0), ((2 * np.arange(N)) * np.pi / N, 1)):
        eps = np.array([2 * (h - J * np.cos(k)) if np.isclose(np.sin(k), 0)
                        else 2 * np.sqrt(J ** 2 + h ** 2 - 2 * J * h * np.cos(k)) for k in ks])
        out.append((eps, par))
    return out


def ground_energy(N, J, h):
    """E0 del sector fisico: NS (par 0) -> vacio; R (par 1) -> vacio + modo mas barato si hace falta paridad impar."""
    best = np.inf
    for eps, par in sector_eps(N, J, h):
        ev = -0.5 * eps.sum()
        if par == 0:
            best = min(best, ev)
        else:
            # paridad impar: ocupar un numero impar de modos; el minimo ocupa los negativos (si los hay) y ajusta paridad
            neg = np.sort(eps[eps < 0]); pos = np.sort(eps[eps >= 0])
            base = ev + neg.sum(); nocc = len(neg)
            if nocc % 2 == 1:
                best = min(best, base)
            else:
                cands = []
                if len(pos): cands.append(base + pos[0])
                if len(neg): cands.append(base - neg[-1])
                best = min(best, min(cands))
    return best


def dos(N, J, h, dE):
    """Devuelve (Egrid, g) con g = numero de estados fisicos por celda."""
    secs = sector_eps(N, J, h)
    lo = min(-0.5 * e.sum() + np.minimum(e, 0).sum() for e, _ in secs) - 1.0
    hi = max(-0.5 * e.sum() + np.maximum(e, 0).sum() for e, _ in secs) + 1.0
    nb = int(np.ceil((hi - lo) / dE)) + 2
    G = np.zeros(nb)
    for eps, par in secs:
        ev = -0.5 * eps.sum()
        cnt = [np.zeros(nb), np.zeros(nb)]                 # paridad 0 / 1
        x = (ev - lo) / dE; i0 = int(np.floor(x)); f = x - i0
        cnt[0][i0] += 1 - f; cnt[0][i0 + 1] += f
        for e in eps:
            s = e / dE; k = int(np.floor(s)); fr = s - k
            new0 = cnt[0].copy(); new1 = cnt[1].copy()
            for src, dst in ((cnt[1], new0), (cnt[0], new1)):
                sh = np.zeros(nb)
                if k >= 0:
                    sh[k:] += (1 - fr) * src[:nb - k]; sh[k + 1:] += fr * src[:nb - k - 1]
                else:
                    sh[:nb + k] += (1 - fr) * src[-k:]; sh[:nb + k + 1] += fr * src[-k - 1:]
                dst += sh
            cnt = [new0, new1]
        G += cnt[par]
    return lo + dE * np.arange(nb), G


class MREdos:
    def __init__(self, E, g):
        m = g > 0
        self.E, self.g = E[m], g[m]
        self.c0 = np.concatenate([[0.0], np.cumsum(self.g)])
        self.c1 = np.concatenate([[0.0], np.cumsum(self.g * self.E)])
        self.c2 = np.concatenate([[0.0], np.cumsum(self.g * self.E ** 2)])

    def _sums(self, c):
        k = np.searchsorted(self.E, c)
        n, s1, s2 = self.c0[k], self.c1[k], self.c2[k]
        return n * c - s1, c * s1 - s2, n * c * c - 2 * c * s1 + s2

    def _at(self, c, T):
        Z, WE, W2 = self._sums(c)
        E, S2 = WE / Z, -np.log(W2 / Z ** 2)
        return E, S2, E - T * S2

    def __call__(self, T):
        """Con huecos en el espectro, c - <E>_c = 2T puede tener VARIAS raices (puntos estacionarios); el MRE es la de
        MINIMA F.  Se buscan todos los cambios de signo en una rejilla fina de c y se refina cada uno."""
        f = lambda c: c - self._sums(c)[1] / self._sums(c)[0] - 2 * T
        cs = np.linspace(self.E[0] + 1e-9, self.E[-1] + 2 * T + 1, 200001)  # toda la banda (antes E0+40T+5: fallaba a N>=200)
        fv = np.array([f(c) for c in cs])
        best = None
        for i in np.nonzero(np.sign(fv[:-1]) != np.sign(fv[1:]))[0]:
            c = brentq(f, cs[i], cs[i + 1], xtol=1e-13)
            r = self._at(c, T)
            if best is None or r[2] < best[2]:
                best = r
        return best


def mre_point(N, T, J=1.0, h=1.5, dE=2e-3):
    if T == 0:
        E0 = ground_energy(N, J, h); return E0, 0.0, E0
    return MREdos(*dos(N, J, h, dE))(T)


def observables(N, T, dE, d=2e-3):
    E, S2, F = mre_point(N, T, dE=dE)
    Fh = [mre_point(N, T, 1.0, 1.5 + s * d, dE)[2] for s in (+1, -1)]
    FJ = [mre_point(N, T, 1.0 + s * d, 1.5, dE)[2] for s in (+1, -1)]
    sx = -(Fh[0] - Fh[1]) / (2 * d) / N
    szz1 = -(FJ[0] - FJ[1]) / (2 * d) / N
    return dict(T=T, E=E, S2=S2, F=F, E_per_site=E / N, sx=sx, szz1=szz1, check_E=-(szz1 + 1.5 * sx))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=50); ap.add_argument("--T", type=float, nargs="*", default=[3.0])
    ap.add_argument("--dE", type=float, default=2e-3); ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.validate:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from mre_teff import MRE, spectrum
        for N in (16, 25):
            ex = MRE(spectrum(N))
            for T in (0.3, 0.5, 1.0, 3.0):
                r = observables(N, T, a.dE); Ex, Sx, Fx = MREdos(np.sort(spectrum(N)), np.ones(2 ** N))(T)
                print(f"N={N} T={T}: DOS E={r['E']:.5f} S2={r['S2']:.5f} F={r['F']:.5f} | exacto(espectro, todas las raices) E={Ex:.5f} S2={Sx:.5f} F={Fx:.5f}"
                      f" | szz1={r['szz1']:.4f} sx={r['sx']:.4f} (E/N check {r['check_E']:.4f} vs {r['E_per_site']:.4f})", flush=True)
            print(f"N={N} E0: {ground_energy(N, 1.0, 1.5):.6f} vs espectro {spectrum(N)[0]:.6f}", flush=True)
        sys.exit()
    d = 2e-3
    M0 = MREdos(*dos(a.N, 1.0, 1.5, a.dE))
    Mh = [MREdos(*dos(a.N, 1.0, 1.5 + sg * d, a.dE)) for sg in (+1, -1)]
    MJ = [MREdos(*dos(a.N, 1.0 + sg * d, 1.5, a.dE)) for sg in (+1, -1)]
    E0 = ground_energy(a.N, 1.0, 1.5)
    res = []
    for T in a.T:
        if T == 0:
            E, S2, F = E0, 0.0, E0
            Fh = [ground_energy(a.N, 1.0, 1.5 + sg * d) for sg in (+1, -1)]
            FJ = [ground_energy(a.N, 1.0 + sg * d, 1.5) for sg in (+1, -1)]
        else:
            E, S2, F = M0(T); Fh = [m(T)[2] for m in Mh]; FJ = [m(T)[2] for m in MJ]
        sx = -(Fh[0] - Fh[1]) / (2 * d) / a.N; szz1 = -(FJ[0] - FJ[1]) / (2 * d) / a.N
        r = dict(T=T, E=E, S2=S2, F=F, E_per_site=E / a.N, sx=sx, szz1=szz1, check_E=-(szz1 + 1.5 * sx)); res.append(r)
        print(f"N={a.N} T={T:.2f}: E={r['E']:.4f} (E/N {r['E_per_site']:.5f}) S2={r['S2']:.4f} F={r['F']:.4f} szz1={r['szz1']:.4f} sx={r['sx']:.4f}", flush=True)
    if a.out:
        assert "/data/0/" not in os.path.abspath(a.out)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(dict(N=a.N, dE=a.dE, results=res), open(a.out, "w"), indent=1)
