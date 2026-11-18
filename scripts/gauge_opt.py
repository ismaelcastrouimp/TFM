"""
gauge_opt.py
============
Techo de un "gauge aprendido" para la ancilla física: dada una Ψ(s,a) entrenada (N pequeño, matriz 2^N x 2^N), busca la
unitaria U de la ancilla (Ψ -> Ψ U^T; ρ y F no cambian) que minimiza el segundo momento del solapamiento estimado por el
puente (n_a = 1):
    S(s1,s2) = E_r |Ô|^2 = Σ_a |φ(a|s1) φ(a|s2)|^2 / r(a|s1,s2),   r = AR "puente" ∝ sqrt(p(a_k|s1,a_<k) p(a_k|s2,a_<k))
    J = Σ_{s1,s2} p(s1) p(s2) S(s1,s2)^2 / P - P      (P = Tr ρ^2; J = 0 si el estimador interno no tiene ruido)
La parte de s (p(s), propuesta de s2) no depende del gauge.  U = exp(K), K antihermítica (completa, 4^N parámetros).
Al final guarda Ψ U^T para medir la η real con s2_physical_bridge.py --psi_file.
Uso: JAX_PLATFORMS=cpu .venv/bin/python scripts/gauge_opt.py --psi data/gap_arch/phys_bridge/ex_N6_T3_free_psi_psi.npy --label free
"""
import argparse, json, os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

ap = argparse.ArgumentParser()
ap.add_argument("--psi", required=True); ap.add_argument("--label", required=True)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--out_dir", default="data/gap_arch/phys_bridge/gauge_opt")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

Psi0 = np.load(a.psi).astype(complex); Psi0 /= np.linalg.norm(Psi0)
D = Psi0.shape[0]; N = int(round(np.log2(D)))
rho = Psi0 @ Psi0.conj().T; P = float(np.sum(np.abs(rho) ** 2))
ev, V = np.linalg.eigh(rho); canon = (V * np.sqrt(np.clip(ev, 0, None))) @ V.conj().T


def J(Psi):
    w = jnp.abs(Psi) ** 2; p = w.sum(1)                                  # p(s)
    ph2 = w / p[:, None]                                                 # |φ(a|s)|^2
    lr = jnp.zeros((D, D, D))                                            # log r(a|s1,s2)
    for k in range(N):                                                   # bit k de a (MSB primero)
        Mk = ph2.reshape(D, 2 ** k, -1).sum(-1)                          # (s, prefijo k)
        Mk1 = ph2.reshape(D, 2 ** (k + 1), -1).sum(-1)
        q1 = Mk1[:, 1::2] / jnp.maximum(Mk, 1e-300)                      # p(bit=1 | s, prefijo)
        g1 = jnp.sqrt(q1[:, None] * q1[None]); g0 = jnp.sqrt((1 - q1)[:, None] * (1 - q1)[None])
        Z = g1 + g0
        pre = jnp.arange(D) >> (N - k); bit = (jnp.arange(D) >> (N - k - 1)) & 1
        g = jnp.where(bit[None, None] == 1, g1[:, :, pre], g0[:, :, pre]) / Z[:, :, pre]
        lr = lr + jnp.log(jnp.maximum(g, 1e-300))
    S = jnp.sum(ph2[:, None] * ph2[None] * jnp.exp(-lr), -1)             # (s1, s2)
    return jnp.sum(p[:, None] * p[None] * S ** 2) / P - P


def unitary(K):
    A = K - K.conj().T
    return jax.scipy.linalg.expm(A)


loss = jax.jit(lambda K: J(jnp.asarray(Psi0) @ unitary(K).T))
J0, Jc = float(J(jnp.asarray(Psi0))), float(J(jnp.asarray(canon)))
# comprobación: U aleatoria no cambia ρ
U = np.asarray(unitary(jnp.asarray(0.3 * (np.random.default_rng(0).normal(size=(D, D)) + 1j * np.random.default_rng(1).normal(size=(D, D))))))
assert np.abs((Psi0 @ U.T) @ (Psi0 @ U.T).conj().T - rho).max() < 1e-12
print(f"[{a.label}] N={N} P={P:.5f}  J inicial={J0:.4g}  J canónico ρ^1/2={Jc:.4g}", flush=True)

K = jnp.zeros((D, D), complex)
tx = optax.adam(optax.cosine_decay_schedule(a.lr, a.steps, 1e-2)); st = tx.init(K)
vg = jax.jit(jax.value_and_grad(loss))
best, Kb, hist = J0, K, []
for t in range(1, a.steps + 1):
    v, g = vg(K)
    upd, st = tx.update(jnp.conj(g), st, K); K = optax.apply_updates(K, upd)
    if float(v) < best: best, Kb = float(v), K
    if t % 100 == 0 or t == 1:
        hist.append((t, float(v))); print(f"[{a.label}] {t:5d} J={float(v):.4g}", flush=True)
Psi = np.asarray(jnp.asarray(Psi0) @ unitary(Kb).T)
herm = float(np.real(np.sum(Psi * Psi.T)))
np.save(os.path.join(a.out_dir, f"{a.label}_psi.npy"), Psi)
json.dump(dict(psi=a.psi, N=N, P=P, J0=J0, J_canon=Jc, J_best=best, herm_final=herm, hist=hist),
          open(os.path.join(a.out_dir, f"{a.label}.json"), "w"), indent=1)
print(f"[{a.label}] J: inicial {J0:.4g} -> optimizado {best:.4g} (canónico {Jc:.4g});  H final {herm:+.3f}", flush=True)
