"""
ite_ed_ansatz.py
================
Prototipo EXACTO (N<=8) de la via alternativa al Gibbs SIN entropias: evolucion en tiempo imaginario de la purificacion
(termocampo).  |psi(beta)> ∝ (e^{-beta H/2} (x) 1) |Phi+>  =>  rho_sys = e^{-beta H}/Z.
Con el ansatz ARViT + ancilla MPS:  psi(s,a) = A_s Phi_s(a)/||Phi_s||  (A_s del ARViT, Phi_s la MPS condicionada).
El init de siempre (copy init) es ya beta=0 (rho = 1/D).  Cada paso: objetivo psi_t = e^{-dbeta H/2} psi_prev (exacto),
y se ajusta theta minimizando la infidelidad 1 - |<psi_theta|psi_t>|^2 con Adam (arranque en caliente).
A N grande el mismo paso es p-tVMC: infidelidad estimada por muestreo de (s,a) ~ |psi|^2 (sin estimador de pureza).
Reporta, a lo largo de la trayectoria, F_vN - F_Gibbs y distancia de traza al Gibbs del mismo beta: UNA trayectoria
da toda la curva en T.
Uso: python scripts/ite_ed_ansatz.py --N 8 --chi 4 --dbeta 0.02 --beta_max 2 --inner 150 --out data/gap_arch/svn/ite_N8.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sd = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(sd, ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=8); ap.add_argument("--chi", type=int, default=4)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--dbeta", type=float, default=0.02)
ap.add_argument("--beta_max", type=float, default=2.0); ap.add_argument("--inner", type=int, default=150)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--report", type=float, nargs="*", default=[0.25, 1 / 3, 0.5, 1.0, 2.0])
ap.add_argument("--out", required=True)
a = ap.parse_args(); N = a.N; D = 2 ** N
idx = np.arange(D); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
H = np.zeros((D, D)); H[idx, idx] = -1.0 * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
for i in range(N): H[idx, idx ^ (1 << i)] += -1.5
ev, V = np.linalg.eigh(H)
Ustep = jnp.asarray((V * np.exp(-a.dbeta * ev / 2)) @ V.T)          # e^{-dbeta H/2}
model = build_alt_model(N, a.chi, a.arch, 16, 4, 2); all_sj = jnp.asarray(all_s)


def psi_mat(p):
    """psi[s, a] normalizada (D x D)."""
    logA, M = model.apply({"params": p}, all_sj)                      # M: (D, K, 2, chi, chi)
    A = jnp.exp(logA - jnp.max(jnp.real(logA))); A = A / jnp.linalg.norm(A)
    v = jnp.zeros((D, 1, a.chi), jnp.complex128).at[:, 0, 0].set(1.0)
    for k in range(M.shape[1]):
        v = jnp.einsum("bxi,bcij->bxcj", v, M[:, k]).reshape(D, -1, a.chi)
    Phi = v[:, :, 0]                                                  # (D, 2^K)
    Phi = Phi / jnp.linalg.norm(Phi, axis=1, keepdims=True)
    return A[:, None] * Phi


def infid(p, tgt):
    return 1.0 - jnp.abs(jnp.vdot(psi_mat(p), tgt)) ** 2


def thermo(psi, beta):
    rho = psi @ psi.conj().T; rho = 0.5 * (rho + rho.conj().T)
    lam = np.clip(np.linalg.eigvalsh(rho), 1e-300, None)
    E = float(np.real(np.sum(H * rho.T))); S = float(-(lam * np.log(lam)).sum())
    pG = np.exp(-beta * (ev - ev[0])); pG /= pG.sum(); rhoG = (V * pG) @ V.T
    EG = (pG * ev).sum(); SG = -(pG * np.log(pG + 1e-300)).sum(); T = 1 / beta
    return dict(beta=beta, E=E, Svn=S, E_G=EG, Svn_G=SG, dF_vn=(E - T * S) - (EG - T * SG),
                trace_dist=float(0.5 * np.abs(np.linalg.eigvalsh(rho - rhoG)).sum()))


params = init_params(model, N, a.seed)
psi0 = np.asarray(psi_mat(params)); r0 = psi0 @ psi0.conj().T
print(f"beta=0 (init): S_vN={-(lambda l: (l * np.log(l)).sum())(np.clip(np.linalg.eigvalsh(r0), 1e-300, None)):.4f} (max {N * np.log(2):.4f})", flush=True)
tx = optax.adam(a.lr); st = tx.init(params)
vg = jax.jit(jax.value_and_grad(infid))
upd = jax.jit(lambda g, st, p: tx.update(g, st, p))
res, beta, t0 = [], 0.0, time.time()
nsteps = int(round(a.beta_max / a.dbeta)); rep = sorted(a.report)
for k in range(1, nsteps + 1):
    tgt = Ustep @ psi_mat(params); tgt = tgt / jnp.linalg.norm(tgt)  # objetivo desde el estado AJUSTADO (como en p-tVMC)
    for _ in range(a.inner):
        f, g = vg(params, tgt); u, st = upd(g, st, params); params = optax.apply_updates(params, u)
    beta = k * a.dbeta
    if rep and beta >= rep[0] - 1e-9:
        r = thermo(np.asarray(psi_mat(params)), beta); r.update(last_infid=float(f), N=N, chi=a.chi, dbeta=a.dbeta,
                                                                inner=a.inner, secs=time.time() - t0)
        res.append(r); rep.pop(0)
        print(f"beta={beta:.3f} (T={1 / beta:.3f}): E={r['E']:.4f} (G {r['E_G']:.4f})  Svn={r['Svn']:.4f} (G {r['Svn_G']:.4f})  "
              f"F_vN-F_G={r['dF_vn']:+.5f}  D(rho,rhoG)={r['trace_dist']:.4f}  infid_paso={float(f):.2e}  [{time.time() - t0:.0f}s]", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
