"""test_hermiticity.py (17-11-2026): bridge.hermiticity (MC) frente al H exacto y su gradiente, N=3+3.
Uso: JAX_PLATFORMS=cpu .venv/bin/python -u scripts/test_hermiticity.py"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, netket as nk, numpy as np
from jax.flatten_util import ravel_pytree
from src_renyi.physical_ancilla import models as pam
from src_renyi.physical_ancilla.bridge import hermiticity

N = 3; hi = nk.hilbert.Spin(1 / 2, 2 * N); cfg = jnp.asarray(hi.all_states())
perm = np.r_[np.arange(N, 2 * N), np.arange(N)]
idx_sw = np.asarray(hi.states_to_numbers(np.asarray(cfg[:, perm])))
ok = True
for kind in ("ARNNDense", "ARSpinViT_Causal"):
    m = (nk.models.ARNNDense(hilbert=hi, layers=2, features=8, param_dtype=jnp.float64) if kind == "ARNNDense" else
         pam.ARSpinViT_Causal(hilbert=hi, embedding_d=8, n_heads=2, n_blocks=1, n_ffn_layers=1, param_dtype=jnp.float64))
    vs = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), m, n_samples=16, seed=0)
    leaves, td = jax.tree_util.tree_flatten(vs.parameters)
    ks = jax.random.split(jax.random.PRNGKey(3), len(leaves))
    vs.parameters = jax.tree_util.tree_unflatten(td, [x + 0.6 * jax.random.normal(k, x.shape) for k, x in zip(ks, leaves)])

    def Hex(p):
        psi = jnp.exp(m.apply({"params": p}, cfg).astype(jnp.complex128)); psi = psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2))
        return jnp.real(jnp.sum(psi * psi[idx_sw]))
    H0, g0 = float(Hex(vs.parameters)), np.asarray(ravel_pytree(jax.grad(Hex)(vs.parameters))[0])
    Hs, gs = [], []
    for r in range(200):
        H, g = hermiticity(vs, range(N), range(N, 2 * N), 256, key=r); Hs.append(float(H)); gs.append(np.asarray(ravel_pytree(g)[0]))
    Hs, gs = np.array(Hs), np.array(gs); gm = gs.mean(0); noise = np.linalg.norm(gs.std(0)) / np.sqrt(len(gs))
    z = (Hs.mean() - H0) / (Hs.std() / np.sqrt(len(Hs)))
    good = abs(z) < 4 and np.linalg.norm(gm - g0) < 4 * noise + 1e-3 * np.linalg.norm(g0); ok &= good
    print(f"{kind:18s} H exacto {H0:+.5f}  MC {Hs.mean():+.5f} ({z:+.1f}σ) | ∇H |dif|/|ex| {np.linalg.norm(gm - g0) / np.linalg.norm(g0):.3f} "
          f"(ruido {noise / np.linalg.norm(g0):.3f}) cos {gm @ g0 / np.linalg.norm(gm) / np.linalg.norm(g0):.4f} | {'OK' if good else 'FALLA'}")
print("TODO OK" if ok else "HAY FALLOS")
