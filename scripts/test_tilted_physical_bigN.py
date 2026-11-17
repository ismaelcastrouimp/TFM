"""
test_tilted_physical_bigN.py  (Claude 16-11-2026)
=================================================
Pruebas de `renyi2_tilted` a N grande (sin referencia exacta):
 (FastARNNDense no se prueba: con esta versión de jax el propio ARDirectSampler de NetKet falla con ella.)
 1. Troceado: con chunk_size la memoria no crece con M; se mide tiempo por llamada y pico de memoria.
Uso: JAX_PLATFORMS=cpu .venv/bin/python -u scripts/test_tilted_physical_bigN.py [--Ns 50 100]
"""
import argparse, os, resource, sys, time
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
ap.add_argument("--Ns", type=int, nargs="+", default=[50, 100]); ap.add_argument("--M", type=int, default=256)
ap.add_argument("--K", type=int, default=5); ap.add_argument("--chunk", type=int, default=64)
a = ap.parse_args()


def pert(p, s=0.3):
    l, t = jax.tree_util.tree_flatten(p); ks = jax.random.split(jax.random.PRNGKey(5), len(l))
    return jax.tree_util.tree_unflatten(t, [x + s * jax.random.normal(k, x.shape) for k, x in zip(ks, l)])


def run(vs, A, **kw):
    t0 = time.time(); S2, g, info = renyi2_tilted(vs, A, a.M, K=a.K, key=3, return_info=True, **kw)
    jax.block_until_ready(g); return float(S2), np.asarray(ravel_pytree(g)[0]), info, time.time() - t0


for N in a.Ns:                                        # N = sitios de sistema; total 2N
    hi = nk.hilbert.Spin(1 / 2, 2 * N); A = list(range(N))
    m_slow = nk.models.ARNNDense(hilbert=hi, layers=2, features=8, param_dtype=jnp.float64)
    vs_s = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), m_slow, n_samples=16, seed=0)
    vs_s.parameters = pert(vs_s.parameters)
    run(vs_s, A, chunk_size=a.chunk); S2, g, info, t = run(vs_s, A, chunk_size=a.chunk)
    print(f"N_total={2 * N:4d} {'ARNNDense':14s} S2={S2:.4f} ±{info['S2_err']:.3f}  ESS/K={info['ESS_over_K']:.2f} "
          f"<d>={info['mean_d']:.1f}  {t:.1f} s  RSS pico {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
    vit = pam.ARSpinViT_Causal_Interleaved(hilbert=hi, embedding_d=8, n_heads=2, n_blocks=1, n_ffn_layers=1,
                                           param_dtype=jnp.float64)
    vs_v = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), vit, n_samples=16, seed=0); vs_v.parameters = pert(vs_v.parameters)
    run(vs_v, A, chunk_size=a.chunk); S2, g, info, t = run(vs_v, A, chunk_size=a.chunk)
    print(f"N_total={2 * N:4d} {'ViT interleaved':14s} S2={S2:.4f} ±{info['S2_err']:.3f}  ESS/K={info['ESS_over_K']:.2f}  "
          f"{t:.1f} s  RSS pico {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
