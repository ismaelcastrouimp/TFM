"""
filter_joint_exact.py
=====================
Prueba A (exacta, N <= 10): entrenar JUNTOS la red (ARViT conv + ancilla MPS) y un filtro espectral,
    rho' = B rho B / Tr,
con gradiente exacto (ED), misma receta que scripts/gap_ed_small.py (scratch, seed, 3000 pasos, lr 5e-3).
Variantes (--filters):
  none   : control, sin filtro (= gap_ed_small)
  poly1  : B = 1 + a1 x                    x = (H - E_ref)/s_ref   (implementable a N=25: suma de N+1 MPS)
  poly2  : B = 1 + a1 x + a2 x^2           (N=25: ~N^2/2 vecinos, caro)
  step   : B = sqrt(sigmoid((c - H)/tau))  (NO implementable a N=25; referencia: corte ideal entrenado junto)
E_ref, s_ref = energia y desviacion del MRE exacto a T (constantes, solo para condicionar).
Compara con el filtro aplicado a posteriori (filter_gH_exact.py: poly1 0%, poly2 2-3%, step 74%).

Uso: python scripts/filter_joint_exact.py --N 8 --filters none poly1 poly2 step --out data/gap_arch/filter/joint_N8.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "..")); sys.path.insert(0, script_dir)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla import training
from src_renyi.mps_ancilla.training import compute_rho
from mre_teff import MRE

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--h_x", type=float, default=-1.5); ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--chi", type=int, default=4)
ap.add_argument("--d", type=int, default=16); ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--step_off", type=float, default=0.5, help="c inicial del escalon = corte MRE + step_off*std")
ap.add_argument("--step_tau0", type=float, default=1.0, help="tau inicial del escalon (grande = casi lineal, gradiente vivo)")
ap.add_argument("--filters", nargs="+", default=["none", "poly1", "poly2", "step"])
ap.add_argument("--out", required=True)
ap.add_argument("--save_params", action="store_true", help="guarda params (red + filtro) en <out>_<filtro>_s<seed>.msgpack")
a = ap.parse_args()
N, T = a.N, a.T; assert N <= 10
assert "/data/0/" not in os.path.abspath(a.out)

idx = np.arange(2 ** N)
all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
H = np.zeros((2 ** N, 2 ** N))
H[idx, idx] = a.J_ZZ * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
for i in range(N):
    H[idx, idx ^ (1 << i)] += a.h_x
ev, V = np.linalg.eigh(H)
mre = MRE(ev); Ex, Sx, Fx = mre(T)
from scipy.optimize import brentq
c_m = brentq(lambda c: c - mre._sums(c)[1] / mre._sums(c)[0] - 2 * T, ev.min() + 1e-9, ev.max() + 70)
p = np.maximum(c_m - ev, 0); p /= p.sum(); s_ref = float(np.sqrt(p @ ev ** 2 - (p @ ev) ** 2))
print(f"N={N} T={T}  MRE: E={Ex:.5f} S2={Sx:.5f} F={Fx:.5f}  corte c={c_m:.4f}  std={s_ref:.4f}  {jax.devices()}", flush=True)

Hj = jnp.asarray(H); X = (Hj - Ex * jnp.eye(2 ** N)) / s_ref; X2 = X @ X
evj, Vj = jnp.asarray(ev), jnp.asarray(V)
all_sj = jnp.asarray(all_s)


def B_of(kind, g):
    if kind == "none":  return None
    if kind == "poly1": return jnp.eye(2 ** N) + g[0] * X
    if kind == "poly2": return jnp.eye(2 ** N) + g[0] * X + g[1] * X2
    if kind.startswith("poly") and kind not in ("poly1", "poly2"):      # polyK generico
        K = int(kind[4:]); B = jnp.eye(2 ** N); P = jnp.eye(2 ** N)
        for k in range(K):
            P = P @ X; B = B + g[k] * P
        return B
    if kind == "step":
        w = jax.nn.sigmoid((g[0] - evj) / jnp.exp(g[1]))
        return (Vj * jnp.sqrt(w)[None, :]) @ Vj.T
    raise ValueError(kind)


Y = (Hj - 0.5 * (ev[0] + ev[-1]) * jnp.eye(2 ** N)) / (0.5 * (ev[-1] - ev[0]) * 1.0001)
Dzz = jnp.asarray(np.diag(H).copy())
Dperm = jnp.asarray(np.random.default_rng(12345).permutation(ev))


def W_anc(kind, g):
    """Filtro en el lado de la ANCILLA: G'(s,t) = <phi_t| W |phi_s> / sqrt(n_s n_t), W = f(H_anc) con H_anc = el mismo
    TFIM sobre los K=N sitios de la ancilla (bit k de a <-> sitio k).  Barato a N=25 (contraccion MPS-MPO-MPS)."""
    if kind == "astep":
        w = jax.nn.sigmoid((g[0] - evj) / jnp.exp(g[1]))
        return (Vj * w[None, :]) @ Vj.T
    if kind == "anone":                         # control: misma ruta (vectores de ancilla explicitos), sin filtro
        return None
    if kind == "astepZZ":                       # control barato a N=25: escalon de la energia CLASICA zz (diagonal)
        return jnp.diag(jax.nn.sigmoid((g[0] - Dzz) / jnp.exp(g[1])))
    if kind == "astepRand":                     # control: diagonal con el espectro de H permutado al azar
        return jnp.diag(jax.nn.sigmoid((g[0] - Dperm) / jnp.exp(g[1])))
    if kind.startswith("acheb"):                # W = P(H)^2, P = sum_k g_k T_k(Y) (Chebyshev), Y = H escalado a [-1,1]
        K = int(kind[5:]); Tm, Tk = jnp.eye(2 ** N), Y; P = g[0] * Tm + g[1] * Tk
        for k in range(2, K + 1):
            Tm, Tk = Tk, 2 * Y @ Tk - Tm; P = P + g[k] * Tk
        return P @ P
    if kind == "apoly1":
        B = jnp.eye(2 ** N) + g[0] * X; return B @ B
    if kind == "apoly2":
        B = jnp.eye(2 ** N) + g[0] * X + g[1] * X2; return B @ B
    raise ValueError(kind)


def ancilla_vectors(M):
    """M: (S,K,2,chi,chi) -> Phi (S, 2^K), Phi_s(a) = e0^T M_1^{a_1} ... M_K^{a_K} e0, a = sum_k a_k 2^k."""
    S = M.shape[0]
    v = jnp.zeros((S, 1, M.shape[-1]), M.dtype).at[:, 0, 0].set(1.0)
    for k in range(M.shape[1]):
        v = jnp.einsum("sai,sbij->sbaj", v, M[:, k]).reshape(S, -1, M.shape[-1])
    return v[:, :, 0]


def rho_anc(model, net, W):
    logA, M = model.apply({"params": net}, all_sj)
    A = jnp.exp(logA - jnp.max(jnp.real(logA))); A = A / jnp.linalg.norm(A)
    Phi = ancilla_vectors(M)
    n = jnp.real(jnp.sum(jnp.abs(Phi) ** 2, axis=1))
    O = Phi @ W.astype(Phi.dtype) @ jnp.conj(Phi).T if W is not None else Phi @ jnp.conj(Phi).T
    return A[:, None] * jnp.conj(A)[None, :] * O / jnp.sqrt(n[:, None] * n[None, :])


g0 = {"anone": [0.0], "astep": [c_m + a.step_off * s_ref, np.log(a.step_tau0)],
      "astepZZ": [c_m + a.step_off * s_ref, np.log(a.step_tau0)], "astepRand": [c_m + a.step_off * s_ref, np.log(a.step_tau0)], "apoly1": [0.0], "apoly2": [0.0, 0.0],
      "none": [0.0], "poly1": [0.0], "poly2": [0.0, 0.0], "step": [c_m + a.step_off * s_ref, np.log(a.step_tau0)]}

for kind in a.filters:
    if kind.startswith("poly") and kind not in g0: g0[kind] = [0.0] * int(kind[4:])
for kind in a.filters:
    if kind.startswith("acheb"): g0[kind] = [1.0] + [0.0] * int(kind[5:])
results = json.load(open(a.out)) if os.path.exists(a.out) else []
for kind in a.filters:
    model = build_alt_model(N, a.chi, a.arch, a.d, 4, a.blocks)
    params = {"net": init_params(model, N, a.seed), "g": jnp.asarray(g0[kind], jnp.float64)}

    if kind.startswith("a"):                    # comprobacion: con W = 1 se recupera compute_rho
        r1 = rho_anc(model, params["net"], None); r0 = compute_rho(model, params["net"], all_sj, a.chunk)
        err = float(jnp.max(jnp.abs(r1 - r0))); print(f"[check ancilla] max|rho_anc - compute_rho| = {err:.1e}", flush=True)
        assert err < 1e-10

    def loss(pp, kind=kind, model=model):
        if kind.startswith("a"):
            rho = rho_anc(model, pp["net"], W_anc(kind, pp["g"]))
        else:
            rho = compute_rho(model, pp["net"], all_sj, a.chunk)
            B = B_of(kind, pp["g"])
            if B is not None:
                rho = B @ rho @ B
        Z = jnp.real(jnp.trace(rho))
        E = jnp.real(jnp.sum(Hj.T * rho)) / Z
        S2 = -jnp.log(jnp.sum(jnp.abs(rho) ** 2) / Z ** 2)
        return E - T * S2, (E, S2)

    tag = f"[N={N} {kind}] "
    out = training.train(loss, params, a.steps, a.lr, T, freq=max(1, a.steps // 10), tag=tag)
    gfin = np.asarray(out["params_best"]["g"]).tolist()
    r = dict(N=N, T=T, filter=kind, step_off=a.step_off if kind == "step" else None, step_tau0=a.step_tau0 if kind == "step" else None, arch=a.arch, chi=a.chi, seed=a.seed, steps=a.steps, lr=a.lr,
             F=out["F"], E=out["E"], S2=out["S2"], F_MRE=Fx, dF=out["F"] - Fx, dE=out["E"] - Ex, dS2=out["S2"] - Sx,
             g=gfin, s_per_step=out["s_per_step"])
    extra = (f" c={gfin[0]:.3f} tau={np.exp(gfin[1]):.3f}" if kind in ("step", "astep", "astepZZ", "astepRand") else f" g={np.round(gfin, 4).tolist()}")
    print(f"{tag}FINAL F={r['F']:.5f} dF={r['dF']:+.5f} dE={r['dE']:+.4f} dS2={r['dS2']:+.4f}{extra}"
          f"  {1e3 * r['s_per_step']:.0f} ms/paso", flush=True)
    if a.save_params:
        from flax import serialization
        pth = f"{os.path.splitext(a.out)[0]}_{kind}_s{a.seed}.msgpack"
        open(pth, "wb").write(serialization.to_bytes(out["params_best"])); r["params_file"] = pth
    results.append(r)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)
