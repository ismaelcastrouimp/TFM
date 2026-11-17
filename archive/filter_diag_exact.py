"""
filter_diag_exact.py
====================
Prueba B (exacta, N <= 10): techo de un filtro DIAGONAL en la base computacional sobre un modelo ya entrenado,
    rho' = D rho D^dag / Tr,   D = diag(d(s)),  d(s) complejo arbitrario (una componente por configuracion).
Multiplicar por d(s) es lo mismo que cambiar la amplitud A(s) -> d(s) A(s) del ARViT dejando la ancilla igual,
asi que esto mide cuanto del gap se podria cerrar SOLO tocando A(s).  Si es mucho, el corte "cabe" en A(s) y el
problema es de optimizacion/inductive bias; si es poco, la cola esta en las coherencias (ancilla).
    Z = sum_s |d_s|^2 rho_ss,  E Z = d^dag K d con K_ts = H_ts rho_st,  Tr rho'^2 Z^2 = sum_st |d_s|^2 |d_t|^2 |rho_st|^2
Se dan dos versiones: d real positivo (solo modulo) y d complejo (modulo + fase).  L-BFGS con gradiente de JAX,
arranque d = 1.  Comparar con el techo de g(H) de filter_gH_exact.py (87%).

Uso: python scripts/filter_diag_exact.py --params P --arch vit:conv:r=1 --N 10 --chi 4 --label L --out data/gap_arch/filter/diag_L.json
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
ap.add_argument("--chunk", type=int, default=128)
ap.add_argument("--label", default=None); ap.add_argument("--out", default=None)
a = ap.parse_args()
N, T = a.N, a.T; assert N <= 10
if a.out: assert "/data/0/" not in os.path.abspath(a.out)

model = (build_alt_model(N, a.chi, a.arch, a.d, 4, a.blocks) if a.arch else build_model(N, a.chi, a.d, 4, a.blocks))
params = load_params(a.params, model, N)
idx = np.arange(2 ** N)
all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
H = np.zeros((2 ** N, 2 ** N))
H[idx, idx] = a.J_ZZ * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
for i in range(N):
    H[idx, idx ^ (1 << i)] += a.h_x
ev = np.linalg.eigvalsh(H)
chunk = min(a.chunk, 2 ** N)
rho = np.asarray(jax.jit(lambda p, s: compute_rho(model, p, s, chunk))(params, jnp.asarray(all_s)))
rho = rho / np.trace(rho).real
Ex, Sx, Fx = MRE(ev)(T)

K = jnp.asarray(H * rho.T)            # K_ts = H_ts rho_st
A = jnp.asarray(np.abs(rho) ** 2); r0 = jnp.asarray(np.real(np.diag(rho)))


def FES(d):
    w = jnp.abs(d) ** 2
    Z = w @ r0
    E = jnp.real(jnp.conj(d) @ (K @ d)) / Z
    S2 = -jnp.log(w @ (A @ w) / Z ** 2)
    return E - T * S2, E, S2


F0, E0, S0 = [float(v) for v in FES(jnp.ones(2 ** N, complex))]
print(f"[{a.label}] N={N}: F_modelo={F0:.5f} (E={E0:.4f} S2={S0:.4f})  F_MRE={Fx:.5f}  dF={F0 - Fx:+.5f}")
res = dict(label=a.label, N=N, F_MRE=Fx, F0=F0, E0=E0, S2_0=S0, dF0=F0 - Fx)

M = 2 ** N
variants = {"real": lambda u: jnp.exp(u).astype(complex),                       # u = log|d|
            "complejo": lambda u: jnp.exp(u[:M] + 1j * u[M:])}                   # modulo y fase
for name, dfun in variants.items():
    f = jax.jit(jax.value_and_grad(lambda u: FES(dfun(u))[0]))
    fun = lambda u: tuple(np.asarray(v, dtype=np.float64) for v in f(jnp.asarray(u)))
    u0 = np.zeros(M if name == "real" else 2 * M)
    r = minimize(fun, u0, jac=True, method="L-BFGS-B", options=dict(maxiter=50000, maxfun=100000, gtol=1e-12, ftol=1e-16))
    for seed in range(3):                                   # arranques aleatorios (no convexo)
        rs = minimize(fun, np.random.default_rng(seed).normal(0, 0.5, len(u0)), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=50000, maxfun=100000, gtol=1e-12, ftol=1e-16))
        if rs.fun < r.fun: r = rs
    gn = float(np.abs(fun(r.x)[1]).max())
    F, E, S = [float(v) for v in FES(dfun(jnp.asarray(r.x)))]
    print(f"  diag {name}: F={F:.5f} E={E:.4f} S2={S:.4f}  dF={F - Fx:+.5f}  ({1 - (F - Fx) / (F0 - Fx):.0%} del gap cerrado)"
          f"  [{r.nit} it, max grad {gn:.1e}]")
    res[name] = dict(F=F, E=E, S2=S, dF=F - Fx, closed=1 - (F - Fx) / (F0 - Fx), nit=int(r.nit))
if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
