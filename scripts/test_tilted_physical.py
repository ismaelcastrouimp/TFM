"""
test_tilted_physical.py  (Claude 16-11-2026)
============================================
Valida `entropy.renyi2_tilted` (AR inclinado, ancilla física) frente al cálculo EXACTO de S₂ y ∇S₂ para varias redes
autoregresivas: ARNNDense, InterleavedARNNDense, ARSpinViT_Causal (complejo), ARSpinViT_Causal_Interleaved (complejo).
Para cada una: media de P̂ en n_rep repeticiones vs P exacta (P̂ es insesgado) y media de ∇P̂ vs ∇P exacto.
Además compara la varianza de P̂ (CV por lote) frente al swap iid (kappas=(1,)) con el mismo número de pares.
Uso (CPU):  JAX_PLATFORMS=cpu .venv/bin/python -u scripts/test_tilted_physical.py [--N 3] [--rep 200]
"""
import argparse, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import netket as nk
import numpy as np
from jax.flatten_util import ravel_pytree
from src_renyi.physical_ancilla import models as pam
from src_renyi.physical_ancilla.entropy import renyi2_tilted

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=3); ap.add_argument("--rep", type=int, default=200)
ap.add_argument("--M", type=int, default=64); ap.add_argument("--K", type=int, default=5)
a = ap.parse_args()
N = a.N; hi = nk.hilbert.Spin(s=1 / 2, N=2 * N); A = list(range(N))


def perturb(params, seed, scale):
    leaves, tdef = jax.tree_util.tree_flatten(params)
    ks = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    return jax.tree_util.tree_unflatten(tdef, [x + scale * jax.random.normal(k, x.shape) for k, x in zip(ks, leaves)])


def make(kind, seed=0):
    if kind == "ARNNDense":
        m = nk.models.ARNNDense(hilbert=hi, layers=2, features=8, param_dtype=jnp.float64); sc = 0.5
    elif kind == "InterleavedARNNDense":
        m = pam.InterleavedARNNDense(hilbert=hi, layers=2, features=8, param_dtype=jnp.float64); sc = 0.5
    elif kind == "ARSpinViT_Causal":
        m = pam.ARSpinViT_Causal(hilbert=hi, embedding_d=8, n_heads=2, n_blocks=1, n_ffn_layers=1,
                                 param_dtype=jnp.float64); sc = 0.7
    else:
        m = pam.ARSpinViT_Causal_Interleaved(hilbert=hi, embedding_d=8, n_heads=2, n_blocks=1, n_ffn_layers=1,
                                             param_dtype=jnp.float64); sc = 0.7
    vs = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), m, n_samples=16, seed=seed)
    vs.parameters = perturb(vs.parameters, seed + 11, sc)                    # estado no trivial, fase O(1)
    return vs


def exact_P(vs):
    cfg = jnp.asarray(hi.all_states())
    iA = np.asarray(nk.hilbert.Spin(1 / 2, N).states_to_numbers(np.asarray(cfg[:, :N])))
    iB = np.asarray(nk.hilbert.Spin(1 / 2, N).states_to_numbers(np.asarray(cfg[:, N:])))

    def P(p):
        psi = jnp.exp(vs.model.apply({"params": p, **vs.model_state}, cfg).astype(jnp.complex128))
        psi = psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2))
        Mt = jnp.zeros((2 ** N, 2 ** N), complex).at[iA, iB].set(psi)
        rho = Mt @ Mt.conj().T
        return jnp.real(jnp.trace(rho @ rho))
    return float(P(vs.parameters)), np.asarray(ravel_pytree(jax.grad(P)(vs.parameters))[0])


ok = True
for kind in ("ARNNDense", "InterleavedARNNDense", "ARSpinViT_Causal", "ARSpinViT_Causal_Interleaved"):
    vs = make(kind); t0 = time.time()
    try:
        renyi2_tilted(vs, A, 8, K=a.K, key=0, compute_grad=False)
    except ValueError as e:                                                 # red no causal en su orden declarado
        print(f"{kind:30s} RECHAZADA (esperado si la red ignora su reorder): {e}", flush=True); continue
    P_ex, gP_ex = exact_P(vs)
    lp = vs.model.apply({"params": vs.parameters, **vs.model_state}, jnp.asarray(hi.all_states()))
    ph = float(jnp.std(jnp.imag(lp))) if jnp.iscomplexobj(lp) else 0.0
    Ps, gPs, ess = [], [], []
    for r in range(a.rep):
        S2, g, info = renyi2_tilted(vs, A, a.M, K=a.K, key=r, return_info=True)
        P = float(np.exp(-S2)); Ps.append(P); ess.append(info["ESS_over_K"])
        gPs.append(-P * np.asarray(ravel_pytree(g)[0]))                          # ∇P̂ = -P̂ ∇S₂
    Ps, gPs = np.array(Ps), np.array(gPs)
    zP = (Ps.mean() - P_ex) / (Ps.std() / np.sqrt(a.rep))
    gm = gPs.mean(0); noise = np.linalg.norm(gPs.std(0)) / np.sqrt(a.rep)
    rel = np.linalg.norm(gm - gP_ex) / np.linalg.norm(gP_ex); cos = gm @ gP_ex / np.linalg.norm(gm) / np.linalg.norm(gP_ex)
    # varianza frente al swap iid con el mismo número de pares (kappas=(1,))
    Pi = np.array([np.exp(-float(renyi2_tilted(vs, A, a.M, K=a.K, kappas=(1.0,), key=1000 + r,
                                               compute_grad=False)[0])) for r in range(50)])
    good = abs(zP) < 4 and np.linalg.norm(gm - gP_ex) < 4 * noise + 1e-3 * np.linalg.norm(gP_ex)
    ok &= good
    print(f"{kind:30s} fase std {ph:.2f} | P exacta {P_ex:.6f}  media P̂ {Ps.mean():.6f} ({zP:+.1f}σ) | "
          f"∇P: |dif|/|ex| {rel:.3f} (ruido {noise / np.linalg.norm(gP_ex):.3f}) cos {cos:.5f} | "
          f"CV(P̂) inclinado {Ps.std() / Ps.mean():.3f} vs swap iid {Pi.std() / Pi.mean():.3f} | ESS/K {np.mean(ess):.2f}"
          f" | {'OK' if good else 'FALLA'}  ({time.time() - t0:.0f} s)", flush=True)
print("TODO OK" if ok else "HAY FALLOS")
