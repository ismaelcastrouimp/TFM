"""
svn_extrap_theory.py
====================
Limite IDEAL (ansatz y estimadores perfectos) de minimizar  F~ = E - T * S~,  con S~ = extrapolacion a n -> 1 de las
Renyi S_n (n en un conjunto de nodos).  Cualquier funcional de la forma E - T*f(espectro) se minimiza con rho diagonal en
la base de H y poblaciones funcion de la energia -> optimizamos las poblaciones por celda de energia de la DOS exacta
(scripts/mre_dos.py; N grande) o del espectro completo (N<=12).  Se compara con el Gibbs exacto del mismo N.

Esquemas (S~ = sum_n c_n S_n, lineales en S_n, c_n = pesos de Lagrange en x=1):
  poly_n:2,3,4     polinomio en n por los nodos
  poly_inv:2,3,4   polinomio en 1/n
  sn:k             solo S_k (minimizador = "MRE_k", p ~ (c-E)_+^(1/(k-1)))
  s2               solo S2 (=MRE: control)      vn  S_vN exacta (=Gibbs: control)
Uso: .venv/bin/python scripts/svn_extrap_theory.py --N 50 --T 0.5 1 2 3 4 --schemes s2 vn poly_n:2,3 poly_n:2,3,4 ...
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from scipy.optimize import minimize


def lagrange_at(x_nodes, x0):
    x = np.asarray(x_nodes, float)
    return np.array([np.prod([(x0 - x[j]) / (x[i] - x[j]) for j in range(len(x)) if j != i]) for i in range(len(x))])


def scheme_coefs(s):
    if s in ("s2", "vn"): return None
    if s.startswith("sn:"): return [int(s[3:])], np.array([1.0])          # solo S_k (MRE_k)
    kind, nodes = s.split(":"); n = [int(v) for v in nodes.split(",")]
    c = lagrange_at(n, 1.0) if kind == "poly_n" else lagrange_at([1 / v for v in n], 1.0)
    return n, c


def make_fns(E, lg, T, scheme):
    E, lg = jnp.asarray(E), jnp.asarray(lg)
    sc = scheme_coefs(scheme)

    def parts(th):
        lZ = logsumexp(th + lg); lp = th - lZ                         # log p por estado de la celda
        p_cell = jnp.exp(lp + lg)                                     # probabilidad total de la celda
        Em = jnp.sum(p_cell * E)
        Svn = -jnp.sum(p_cell * lp)
        Sn = lambda n: (logsumexp(n * lp + lg)) / (1 - n)
        return Em, Svn, Sn, p_cell

    def ftilde(th):
        Em, Svn, Sn, _ = parts(th)
        if scheme == "vn": S = Svn
        elif scheme == "s2": S = Sn(2.0)
        else: S = sum(c * Sn(float(n)) for n, c in zip(*sc))
        return Em - T * S
    return jax.jit(ftilde), jax.jit(jax.value_and_grad(ftilde)), parts


def solve(E, lg, T, scheme, inits):
    f, vg, parts = make_fns(E, lg, T, scheme)
    best = None
    for name, th0 in inits:
        r = minimize(lambda x: tuple(np.asarray(v) for v in vg(jnp.asarray(x))), th0, jac=True, method="L-BFGS-B",
                     options=dict(maxiter=20000, maxfun=40000, ftol=1e-15, gtol=1e-10))
        if best is None or r.fun < best[0]: best = (r.fun, r.x, name, r.nit, r.success)
    th = jnp.asarray(best[1]); Em, Svn, Sn, _ = parts(th)
    return dict(scheme=scheme, Ft=float(best[0]), E=float(Em), Svn=float(Svn), S2=float(Sn(2.0)), S3=float(Sn(3.0)),
                F_vn=float(Em - T * Svn), init=best[2], nit=int(best[3]), ok=bool(best[4])), th


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=50); ap.add_argument("--T", type=float, nargs="+", default=[3.0])
    ap.add_argument("--dE", type=float, default=0.02)
    ap.add_argument("--schemes", nargs="+", default=["s2", "vn", "poly_n:2,3", "poly_n:2,3,4", "poly_n:2,3,4,5",
                                                     "poly_inv:2,3", "poly_inv:2,3,4", "poly_inv:2,3,4,5"])
    ap.add_argument("--hdelta", type=float, default=0.0, help="desplaza h (para Hellmann-Feynman externo)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    J, h = 1.0, 1.5 + a.hdelta
    if a.N <= 12:
        from mre_teff import spectrum
        ev = np.sort(spectrum(a.N)); Ec, cnt = np.unique(np.round(ev, 10), return_counts=True)
    else:
        from mre_dos import dos
        Eg, g = dos(a.N, J, h, a.dE); m = g > 1e-12; Ec, cnt = Eg[m], g[m]
    lg = np.log(cnt)
    res = []
    for T in a.T:
        lZ = np.logaddexp.reduce(-Ec / T + lg); pg = np.exp(-Ec / T + lg - lZ)
        EG = (pg * Ec).sum(); FG = -T * lZ; SG = (EG - FG) / T
        S2G = -(np.logaddexp.reduce(-2 * Ec / T + lg) - 2 * lZ)
        print(f"N={a.N} T={T}: GIBBS E/N={EG / a.N:.5f} Svn/N={SG / a.N:.5f} S2/N={S2G / a.N:.5f} F/N={FG / a.N:.5f}", flush=True)
        for s in a.schemes:
            inits = [("gibbs", -Ec / T), ("gibbs2T", -Ec / (2 * T)), ("gibbsT/2", -2 * Ec / T)]
            r, _ = solve(Ec, lg, T, s, inits)
            r.update(N=a.N, T=T, E_G=EG, Svn_G=SG, F_G=FG, S2_G=S2G, dF_vn=r["F_vn"] - FG)
            res.append(r)
            print(f"   {s:18s} E/N={r['E'] / a.N:.5f} (G {EG / a.N:.5f})  Svn/N={r['Svn'] / a.N:.5f}  S~/N={(r['E'] - r['Ft']) / T / a.N:.5f}"
                  f"  F_vn - F_G = {r['dF_vn']:+.4f} ({r['dF_vn'] / a.N:+.5f}/sitio)  [{r['init']}, nit {r['nit']}, ok {r['ok']}]", flush=True)
    if a.out:
        assert "/data/0/" not in os.path.abspath(a.out)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
