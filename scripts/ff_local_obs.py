"""
ff_local_obs.py
===============
Observables locales EXACTOS de Gibbs y del MRE del TFIM periodico (H = -J sum ZZ - h sum X, J=1, h=1.5), a N finito,
via fermiones libres.  Ambos ensembles son funcion de H, rho = f(H), asi que se diagonalizan en la base de estados de
Fock de los modos BdG de cada sector de paridad P = prod X:
    rho = sum_n f(E_n) |n><n|,   n recorre los estados de Fock FISICOS (paridad del estado = P del sector).
Cada |n> es gaussiano -> cualquier cadena de Majoranas por Wick = Pfaffiano de la covarianza C_pq = <g_p g_q>.
Se MUESTREAN estados n con peso f(E_n) (MRE: (c - E)_+ ; Gibbs: exp(-beta E)) y se promedia el valor de cada estado.

JW (campo en x):  a_j, b_j Majoranas,  g_{2j} = a_j, g_{2j+1} = b_j
    X_j = i g_{2j} g_{2j+1};   Z_j Z_{j+1} = i g_{2j+1} g_{2j+2}  (enlace frontera: factor -P)
    Z_j Z_{j+r} = prod_{l=j}^{j+r-1} (i g_{2l+1} g_{2l+2}) = i^r Pf(C[2j+1 : 2j+2r+1])   (sin cruzar la frontera)
    X_j X_{j+1} = - Pf(C[2j : 2j+4])
Observables (por sitio, promediando posiciones que no cruzan la frontera): <X>, <Z_i Z_{i+r}> r=1..R,
<X_i X_{i+1}>, y la conexa <X_i X_{i+1}> - <X>^2.

Validacion: --validate (N<=10) compara con ED exacta enumerando TODOS los estados (sin muestreo).
Uso: python scripts/ff_local_obs.py --N 25 --E -21.52 -21.60 --T_ref 3.0 --samples 40000 --out data/gap_arch/obs/ff_N25.json
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.linalg import schur
from scipy.optimize import brentq

J, Hf = 1.0, 1.5


def majorana_A(N, P):
    """H = (i/4) sum_pq A_pq g_p g_q.  Un termino c * i g_p g_q -> A_pq += 2c, A_qp -= 2c."""
    A = np.zeros((2 * N, 2 * N))
    def add(c, p, q):
        A[p, q] += 2 * c; A[q, p] -= 2 * c
    for j in range(N):
        add(-Hf, 2 * j, 2 * j + 1)                        # -h X_j
    for j in range(N - 1):
        add(-J, 2 * j + 1, 2 * j + 2)                     # -J Z_j Z_{j+1}
    add(-J * (-P), 2 * N - 1, 0)                          # enlace frontera, Z_{N-1} Z_0 = -P i b_{N-1} a_0
    return A


class Sector:
    def __init__(self, N, P):
        self.N, self.P = N, P
        A = majorana_A(N, P)
        S, Z = schur(A, output="real")
        s = np.array([S[2 * k, 2 * k + 1] for k in range(N)])
        assert np.allclose(S, np.kron(np.diag(s), np.array([[0, 1], [-1, 0]])), atol=1e-9), "Schur no 2x2"
        self.O = Z.T                                       # g' = O g
        self.s = s; self.eps = np.abs(s)
        self.detO = np.linalg.det(self.O)
        self.Evac = -0.5 * self.eps.sum()
        # paridad del vacio: P = det(O) prod_k x_k,  x_k = <i g'_{2k} g'_{2k+1}> = -sign(s_k) (1 - 2 n_k)
        self.Pvac = np.sign(self.detO) * np.prod(-np.sign(s))

    def enumerate(self):
        """energias y bits de los estados de Fock FISICOS (paridad = P)."""
        N = self.N
        E = np.array([self.Evac]); par = np.zeros(1, np.int8)
        for e in self.eps:
            E = np.concatenate([E, E + e]); par = np.concatenate([par, 1 - par])
        idx = np.arange(2 ** N, dtype=np.int64)
        Pn = self.Pvac * (1 - 2 * par.astype(np.int64))
        keep = Pn == self.P
        return E[keep], idx[keep]

    def covariance(self, occ):
        """occ (B, N) en {0,1} -> C (B, 2N, 2N), C_pq = <g_p g_q>."""
        B, N = occ.shape
        x = -np.sign(self.s)[None, :] * (1 - 2 * occ)          # (B,N)
        Cp = np.zeros((B, 2 * N, 2 * N), complex)
        k = np.arange(N)
        Cp[:, 2 * k, 2 * k + 1] = -1j * x; Cp[:, 2 * k + 1, 2 * k] = 1j * x
        C = self.O.T @ Cp @ self.O
        C = C + np.eye(2 * N)[None]
        return C


def pfaffian_batch(K):
    """Pfaffiano de matrices antisimetricas (B, 2m, 2m) por desarrollo de la primera fila (2m <= 10)."""
    n = K.shape[-1]
    if n == 0: return np.ones(K.shape[0], complex)
    if n == 2: return K[:, 0, 1]
    out = np.zeros(K.shape[0], complex)
    for j in range(1, n):
        rest = [i for i in range(1, n) if i != j]
        out += (-1) ** (j + 1) * K[:, 0, j] * pfaffian_batch(K[:, rest][:, :, rest])
    return out


def antisym(C, sl):
    K = C[:, sl][:, :, sl]
    return 0.5 * (K - np.swapaxes(K, 1, 2))     # parte antisimetrica = <g_p g_q> para p != q


def state_obs(C, N, R):
    """valores por estado (promedio sobre posiciones sin cruzar frontera)."""
    out = {}
    j = np.arange(N)
    out["sx"] = np.real(1j * C[:, 2 * j, 2 * j + 1]).mean(1)
    for r in range(1, R + 1):
        vals = [np.real(1j ** r * pfaffian_batch(antisym(C, list(range(2 * jj + 1, 2 * jj + 2 * r + 1)))))
                for jj in range(N - r)]
        out[f"szz{r}"] = np.mean(vals, 0)
    vals = [np.real(-pfaffian_batch(antisym(C, list(range(2 * jj, 2 * jj + 4))))) for jj in range(N - 1)]
    out["sxx"] = np.mean(vals, 0)
    # producto X_j X_{j+1} exacto por estado (para la conexa se usa <X>^2 del ensemble, abajo)
    return out


def occ_of(idx, N):
    return ((idx[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int64)


def build(N):
    secs = [Sector(N, +1), Sector(N, -1)]
    Es, ids, sid = [], [], []
    for k, sc in enumerate(secs):
        E, idx = sc.enumerate(); Es.append(E); ids.append(idx); sid.append(np.full(len(E), k, np.int8))
    return secs, np.concatenate(Es), np.concatenate(ids), np.concatenate(sid)


def weights(kind, Eall, Etarget=None, T=None):
    """MRE: w = (c - E)_+ ;  Gibbs: w = exp(-beta (E - Emin)).  Ajusta c / beta a <E> = Etarget (o a T)."""
    E0 = Eall.min()
    if kind == "mre":
        def mean(c):
            w = np.maximum(c - Eall, 0); return (w * Eall).sum() / w.sum(), c
        if Etarget is not None:
            c = brentq(lambda c: mean(c)[0] - Etarget, E0 + 1e-9, Eall.max() + 1e3, xtol=1e-11)
        else:
            c = brentq(lambda c: c - mean(c)[0] - 2 * T, E0 + 1e-9, Eall.max() + 1e3, xtol=1e-11)
        return np.maximum(c - Eall, 0), dict(c=c)
    if kind == "gibbs":
        def mean(b):
            w = np.exp(-b * (Eall - E0)); return (w * Eall).sum() / w.sum()
        b = 1 / T if Etarget is None else brentq(lambda b: mean(b) - Etarget, 1e-4, 50, xtol=1e-12)
        return np.exp(-b * (Eall - E0)), dict(beta=b, T=1 / b)


def ensemble_obs(secs, Eall, ids, sid, w, N, R, samples, rng, chunk=2048):
    p = w / w.sum()
    Emean = float(p @ Eall)
    if samples is None:                       # exacto: todos los estados con peso
        pick = np.nonzero(p > 0)[0]; pw = p[pick]
    else:
        pick = rng.choice(len(p), size=samples, p=p); pw = np.full(samples, 1.0 / samples)
    acc = {}; acc2 = {}
    for k, sc in enumerate(secs):
        m = sid[pick] == k
        pk, wk = pick[m], pw[m]
        for st in range(0, len(pk), chunk):
            C = sc.covariance(occ_of(ids[pk[st:st + chunk]], N))
            o = state_obs(C, N, R); ww = wk[st:st + chunk]
            for key, v in o.items():
                acc[key] = acc.get(key, 0.0) + float(ww @ v)
                acc2[key] = acc2.get(key, 0.0) + float(ww @ v ** 2)
    res = {"E_per_site": Emean / N}
    for key in acc:
        res[key] = acc[key]
        if samples is not None:
            res[key + "_err"] = float(np.sqrt(max(acc2[key] - acc[key] ** 2, 0) / samples))
    res["sxx_conn"] = res["sxx"] - res["sx"] ** 2
    res["E_check"] = -J * res["szz1"] - Hf * res["sx"]     # = E/N salvo el enlace frontera (ruido/1/N)
    return res


def ed_check(N, R):
    """ED densa: Gibbs a T=3 y MRE a T=3, observables exactos."""
    idx = np.arange(2 ** N); s = 2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1
    H = np.zeros((2 ** N, 2 ** N)); H[idx, idx] = -J * sum(s[:, i] * s[:, (i + 1) % N] for i in range(N))
    X = []
    for i in range(N):
        Xi = np.zeros((2 ** N, 2 ** N)); Xi[idx, idx ^ (1 << i)] = 1; X.append(Xi); H -= Hf * Xi
    ev, V = np.linalg.eigh(H)
    out = {}
    for kind in ("gibbs", "mre"):
        w, info = weights(kind, ev, T=3.0)
        rho = (V * (w / w.sum())[None, :]) @ V.T
        o = {"E_per_site": float(np.trace(rho @ H)) / N, "sx": float(np.trace(rho @ X[0]))}
        for r in range(1, R + 1):
            o[f"szz{r}"] = float(np.real(np.diag(rho)) @ (s[:, 0] * s[:, r]))
        o["sxx"] = float(np.trace(rho @ X[0] @ X[1]))
        out[kind] = o
    return ev, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=25); ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--E", type=float, nargs="*", default=[], help="energias TOTALES objetivo (modelo)")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--T_ref", type=float, default=3.0)
    ap.add_argument("--samples", type=int, default=40000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    N, R = a.N, a.R
    t0 = time.time()
    secs, Eall, ids, sid = build(N)
    print(f"N={N}: {len(Eall)} estados fisicos (2^N={2 ** N}), Pvac NS/R = {secs[0].Pvac:+.0f}/{secs[1].Pvac:+.0f}, "
          f"E0={Eall.min():.6f}  ({time.time() - t0:.1f} s)", flush=True)
    rng = np.random.default_rng(a.seed)
    if a.validate:
        assert N <= 10
        ev, ed = ed_check(N, R)
        print(f"espectro: max|E_ff - E_ED| = {np.max(np.abs(np.sort(Eall) - ev)):.2e}", flush=True)
        for kind in ("gibbs", "mre"):
            w, _ = weights(kind, Eall, T=3.0)
            ff = ensemble_obs(secs, Eall, ids, sid, w, N, R, None, rng)
            print(f"[{kind} T=3] " + "  ".join(f"{k}: ff={ff[k]:+.10f} ED={ed[kind][k]:+.10f}" for k in ed[kind]), flush=True)
            ffs = ensemble_obs(secs, Eall, ids, sid, w, N, R, a.samples, rng)
            print(f"[{kind} T=3, {a.samples} muestras] " + "  ".join(f"{k}={ffs[k]:+.5f}+-{ffs[k + '_err']:.5f}"
                  for k in ed[kind] if k != "E_per_site"), flush=True)
        return
    targets = [("T_ref", None)] + [(lab, E) for lab, E in zip(a.labels or [f"E{i}" for i in range(len(a.E))], a.E)]
    res = []
    for lab, Et in targets:
        for kind in ("mre", "gibbs"):
            w, info = weights(kind, Eall, Etarget=Et, T=a.T_ref)
            o = ensemble_obs(secs, Eall, ids, sid, w, N, R, a.samples, rng)
            o.update(kind=kind, target=lab, E_target=Et, N=N, samples=a.samples, **info)
            res.append(o)
            print(f"[{lab} {kind}] E/N={o['E_per_site']:+.5f} " + " ".join(
                f"{k}={o[k]:+.5f}" for k in ["sx", "sxx", "sxx_conn"] + [f"szz{r}" for r in range(1, R + 1)])
                + f"  {info}", flush=True)
            if a.out:
                os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
                json.dump(res, open(a.out, "w"), indent=1)
    print(f"FIN ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
