"""
filter_gH_exact.py
==================
Prueba EXACTA (N <= 12, sin reentrenar) del filtro espectral sobre un modelo ya entrenado:
    rho' = g(H) rho g(H) / Tr(...)
En la base propia de H (|n>, E_n): rho'_nm = g_n g_m rho_nm / Z,  Z = sum_n g_n^2 q_n,  q_n = rho_nn.
    E'      = sum_n g_n^2 q_n E_n / Z
    Tr rho'^2 = w^T A w / Z^2,   w_n = g_n^2,  A_nm = |rho_nm|^2
Se minimiza F' = E' - T S2' respecto a los coeficientes de g:
  - polinomios de grado 1, 2, 3 en x = (E - E_modelo)/std (g = 1 + sum_k a_k x^k), con varios arranques;
  - "techo": g_n arbitraria por capa degenerada (funcion arbitraria de H), L-BFGS con gradiente analitico.
    Es el maximo que puede dar CUALQUIER filtro g(H) sobre ese rho.
Sale cuanto del gap dF = F - F_MRE cierra cada grado.  Memoria: igual que s2_deficit_split (rho por bloques), ~2.5 GB a N=12.

Uso: python scripts/filter_gH_exact.py --params P --arch vit:conv:r=1 --N 12 --chi 4 --label L --out data/gap_arch/filter/L.json
"""
import argparse, json, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "..")); sys.path.insert(0, script_dir)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize

from src_renyi.mps_ancilla.ladder import build_model, load_params
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.training import compute_rho
from mre_teff import MRE

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--arch", default=None)
ap.add_argument("--N", type=int, required=True); ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--d", type=int, default=16); ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--h_x", type=float, default=-1.5); ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--label", default=None); ap.add_argument("--out", default=None)
a = ap.parse_args()
N, T = a.N, a.T; assert N <= 12
if a.out: assert "/data/0/" not in os.path.abspath(a.out)

model = (build_alt_model(N, a.chi, a.arch, a.d, 4, a.blocks) if a.arch else build_model(N, a.chi, a.d, 4, a.blocks))
params = load_params(a.params, model, N)
idx = np.arange(2 ** N)
all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
H = np.zeros((2 ** N, 2 ** N))
H[idx, idx] = a.J_ZZ * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
for i in range(N):
    H[idx, idx ^ (1 << i)] += a.h_x
ev, V = np.linalg.eigh(H); del H
chunk = min(a.chunk, 2 ** N)
rho = np.asarray(jax.jit(lambda p, s: compute_rho(model, p, s, chunk))(params, jnp.asarray(all_s)))
rho = rho / np.trace(rho).real
rH = V.T @ (rho @ V); del rho, V
A = np.abs(rH) ** 2; q = np.real(np.diag(rH)).copy(); del rH

mre = MRE(ev); Ex, Sx, Fx = mre(T)


def FES(w):
    Z = w @ q; E = (w * q) @ ev / Z; P = w @ (A @ w) / Z ** 2
    return E - T * (-np.log(P)), E, -np.log(P)


F0, E0, S0 = FES(np.ones_like(q))
std = np.sqrt(q @ ev ** 2 - E0 ** 2); x = (ev - E0) / std
print(f"[{a.label}] N={N}: F_modelo={F0:.5f} (E={E0:.4f} S2={S0:.4f})  F_MRE={Fx:.5f}  dF={F0 - Fx:+.5f}")
res = dict(label=a.label, N=N, chi=a.chi, T=T, F_MRE=Fx, E_MRE=Ex, S2_MRE=Sx, F0=F0, E0=E0, S2_0=S0, dF0=F0 - Fx)

rng = np.random.default_rng(0)
for deg in (1, 2, 3):
    f = lambda c: FES((1 + sum(c[k] * x ** (k + 1) for k in range(deg))) ** 2)[0]
    best = None
    for trial in range(12):
        c0 = np.zeros(deg) if trial == 0 else rng.normal(0, 0.3, deg)
        r = minimize(f, c0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))
        if best is None or r.fun < best.fun: best = r
    g = 1 + sum(best.x[k] * x ** (k + 1) for k in range(deg))
    F, E, S = FES(g ** 2)
    roots = [float(E0 + std * rt.real) for rt in np.roots(np.r_[best.x[::-1], 1.0]) if abs(rt.imag) < 1e-9]
    print(f"  grado {deg}: F={F:.5f} E={E:.4f} S2={S:.4f}  dF={F - Fx:+.5f}  ({1 - (F - Fx) / (F0 - Fx):.0%} del gap cerrado)"
          f"  coef={np.round(best.x, 4).tolist()}  ceros de g en E={np.round(roots, 3).tolist()}")
    res[f"deg{deg}"] = dict(F=F, E=E, S2=S, dF=F - Fx, closed=1 - (F - Fx) / (F0 - Fx), coef=best.x.tolist(), zeros_E=roots)

# corte aprendible: w = g^2 = sigmoid((c - E)/tau) [* (1 + b x)^2]; p = (c, log tau[, b])
from scipy.special import expit
fam = {"step": lambda p: expit((p[0] - ev) / np.exp(p[1])),
       "step_lin": lambda p: expit((p[0] - ev) / np.exp(p[1])) * (1 + p[2] * x) ** 2}
for name, wf in fam.items():
    best = None
    for c0 in np.linspace(E0 + 0.5 * std, E0 + 3 * std, 6):
        for lt in (np.log(0.1), np.log(0.5), np.log(2.0)):
            p0 = [c0, lt] + ([0.0] if name == "step_lin" else [])
            r = minimize(lambda p: FES(wf(p))[0], p0, method="Nelder-Mead",
                         options=dict(xatol=1e-9, fatol=1e-12, maxiter=20000))
            if best is None or r.fun < best.fun: best = r
    F, E, S = FES(wf(best.x))
    print(f"  {name}: F={F:.5f} E={E:.4f} S2={S:.4f}  dF={F - Fx:+.5f}  ({1 - (F - Fx) / (F0 - Fx):.0%} del gap cerrado)"
          f"  c={best.x[0]:.3f} tau={np.exp(best.x[1]):.3f}" + (f" b={best.x[2]:.4f}" if name == "step_lin" else ""))
    res[name] = dict(F=F, E=E, S2=S, dF=F - Fx, closed=1 - (F - Fx) / (F0 - Fx), c=best.x[0], tau=float(np.exp(best.x[1])),
                     extra=best.x[2:].tolist())

# techo: g arbitraria por capa degenerada, w = exp(u)
order = np.argsort(ev); shell = np.empty(len(ev), int)
shell[order] = np.concatenate([[0], np.cumsum(np.diff(ev[order]) > a.tol)]); nsh = shell.max() + 1


def obj(u):
    w = np.exp(u[shell]); Aw = A @ w; Z = w @ q; Eq = (w * q) @ ev; P2 = w @ Aw
    E = Eq / Z; F = E + T * np.log(P2 / Z ** 2)
    dw = q * ev / Z - Eq * q / Z ** 2 + T * (2 * Aw / P2 - 2 * q / Z)
    return F, np.bincount(shell, dw * w, nsh)


r = minimize(obj, np.zeros(nsh), jac=True, method="L-BFGS-B", options=dict(maxiter=20000, maxfun=50000, gtol=1e-12))
F, E, S = FES(np.exp(r.x[shell]))
print(f"  TECHO g arbitraria(H): F={F:.5f} E={E:.4f} S2={S:.4f}  dF={F - Fx:+.5f}  ({1 - (F - Fx) / (F0 - Fx):.0%} del gap cerrado)"
      f"  [{r.message}]")
res["ceiling"] = dict(F=F, E=E, S2=S, dF=F - Fx, closed=1 - (F - Fx) / (F0 - Fx))
if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    np.savez(os.path.splitext(a.out)[0] + "_g.npz", ev=ev, q=q, w_ceiling=np.exp(r.x[shell]))
