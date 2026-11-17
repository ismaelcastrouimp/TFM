"""
gauge_drift.py  (Claude 17-11-2026)
===================================
Libertad gauge de la ancilla física: ψ -> (I⊗U)ψ deja ρ_S (y F) igual, pero la varianza de los estimadores de S₂ cambia
×100-1000 (scripts/s2_physical_bridge.py --gauge).  El gauge canónico ψ = ρ^{1/2} (thermofield double) es hermítico como
matriz ψ(s,a) y es de los mejores.  Medida: H = Re Σ ψ(s,a) ψ(a,s) ∈ [-1, 1] (= 1 sii ψ hermítica).

Preguntas:  (1) ¿el entrenamiento se queda cerca del gauge hermítico o deriva?  (2) ¿el ruido del gradiente lo empuja
fuera (difusión en las direcciones planas de F)?  (3) ¿una penalización λ(1 - H) lo arregla sin cambiar F?
Red: ARNNDense (sistema primero) sobre 2N sitios, N pequeño, todo EXACTO por enumeración salvo el gradiente "ruidoso"
(el de S₂ con renyi2_bridge, M pequeño).  Al final: η del puente (Var(P̂)/P² · coste) medido con el estimador real.
Uso (CPU): JAX_PLATFORMS=cpu .venv/bin/python -u scripts/gauge_drift.py --N 4 --T 2 3
"""
import argparse, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax
from src_renyi.physical_ancilla.bridge import renyi2_bridge

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=4); ap.add_argument("--T", type=float, nargs="+", default=[2.0, 3.0])
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--lam", type=float, default=0.5); ap.add_argument("--M_noisy", type=int, default=32)
ap.add_argument("--features", type=int, default=16); ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--eta_rep", type=int, default=40); ap.add_argument("--eta_M", type=int, default=256)
ap.add_argument("--out", default="data/gap_arch/physical_tilted/gauge_drift.json")
a = ap.parse_args()
N = a.N; D = 2 ** N; hi = nk.hilbert.Spin(1 / 2, 2 * N); A = list(range(N))

# ── H del sistema (TFIM PBC, J=-1, h=-1.5) en la base de netket (s = ±1; índice: states_to_numbers) ─────────────
hs = nk.hilbert.Spin(1 / 2, N)
Hs = sum(-1.0 * nk.operator.spin.sigmaz(hs, i) @ nk.operator.spin.sigmaz(hs, (i + 1) % N) for i in range(N)) + \
     sum(-1.5 * nk.operator.spin.sigmax(hs, i) for i in range(N))
Hm = jnp.asarray(Hs.to_dense())
cfg = jnp.asarray(hi.all_states())
iS = np.asarray(hs.states_to_numbers(np.asarray(cfg[:, :N]))); iA = np.asarray(hs.states_to_numbers(np.asarray(cfg[:, N:])))
ev = np.linalg.eigvalsh(np.asarray(Hm))


def mre_F(T):
    best = np.inf; S1c, S2c = np.cumsum(ev), np.cumsum(ev ** 2)
    best = ev[0]
    for k in range(2, D + 1):
        S1, S2_ = S1c[k - 1], S2c[k - 1]; disc = (S1 + T * k) ** 2 - k * (S2_ + 2 * T * S1)
        if disc < 0: continue
        hi_ = ev[k] if k < D else np.inf
        for c in ((S1 + T * k) + np.sqrt(disc)) / k, ((S1 + T * k) - np.sqrt(disc)) / k:
            if not (ev[k - 1] < c <= hi_ + 1e-12): continue
            w = np.maximum(c - ev, 0); p = w / w.sum(); best = min(best, (p * ev).sum() + T * np.log((p ** 2).sum()))
    return best


model = nk.models.ARNNDense(hilbert=hi, layers=a.layers, features=a.features, param_dtype=jnp.float64)


def psi_mat(p):
    lp = model.apply({"params": p}, cfg)
    psi = jnp.exp(lp - jnp.max(jnp.real(lp))); psi = psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2))
    return jnp.zeros((D, D), psi.dtype).at[iS, iA].set(psi)


def quantities(p, T):
    Psi = psi_mat(p); rho = Psi @ Psi.conj().T
    E = jnp.real(jnp.trace(Hm @ rho)); S2 = -jnp.log(jnp.real(jnp.sum(jnp.abs(rho) ** 2)))
    herm = jnp.real(jnp.sum(Psi * Psi.T))
    return E - T * S2, E, S2, herm


def train(T, mode, lam, seed=0):
    vs = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model, n_samples=16, seed=seed)
    p = vs.parameters; tx = optax.adam(a.lr); st = tx.init(p)
    fexact = jax.jit(lambda p: quantities(p, T))
    gE = jax.jit(jax.grad(lambda p: quantities(p, T)[1]))
    gF = jax.jit(jax.grad(lambda p: quantities(p, T)[0] - lam * quantities(p, T)[3]))
    gH = jax.jit(jax.grad(lambda p: -lam * quantities(p, T)[3]))
    hist = []
    for t in range(a.steps):
        if mode == "exact":
            g = gF(p)
        else:                                       # E exacto + T∇S₂ ruidoso (puente, M pequeño) + penalización exacta
            vs.parameters = p
            _, gS = renyi2_bridge(vs, A, a.M_noisy, K=5, n_a=2, key=seed * 100000 + t, check_order=(t == 0))
            g = jax.tree_util.tree_map(lambda x, y, z: x - T * y + z, gE(p), gS, gH(p))
        upd, st = tx.update(g, st, p); p = optax.apply_updates(p, upd)
        if t % (a.steps // 10) == 0 or t == a.steps - 1:
            F, E, S2, h = map(float, fexact(p)); hist.append(dict(step=t, F=F, E=E, S2=S2, herm=h))
    vs.parameters = p
    Ps = np.array([renyi2_bridge(vs, A, a.eta_M, K=5, n_a=2, key=10 ** 6 + r, compute_grad=False, return_info=True)[2]["P"]
                   for r in range(a.eta_rep)])
    cost = N + 5 * (N + 1 + 2 * (2 * N + 2))
    return hist, float(Ps.var() * a.eta_M / Ps.mean() ** 2 * cost)


res = []
for T in a.T:
    FM = mre_F(T)
    for mode in ("exact", "noisy"):
        for lam in (0.0, a.lam):
            hist, eta = train(T, mode, lam)
            h = hist[-1]
            r = dict(N=N, T=T, mode=mode, lam=lam, F_mre=FM, dF_per_site=(h["F"] - FM) / N, herm=h["herm"], eta_bridge=eta,
                     hist=hist)
            res.append(r)
            print(f"N={N} T={T} {mode:5s} λ={lam:.2f}: dF/N={r['dF_per_site']:+.5f}  H={h['herm']:+.3f}  "
                  f"η_puente={eta:.3g}  | H(t): " + " ".join(f"{x['herm']:+.2f}" for x in hist), flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
