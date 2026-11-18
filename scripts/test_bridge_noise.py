"""
test_bridge_noise.py
====================
Comprueba el estimador MC de Q = E[S²] (bridge_noise.py) y su gradiente frente al cálculo exacto por enumeración,
en un estado encdec entrenado (N pequeño).  Imprime Q exacto vs MC (± error), ln r del puente vs recalculado, y el
coseno / cociente de normas entre ∇Q exacto y la media de R estimaciones MC.
Uso: JAX_PLATFORMS=cpu .venv/bin/python scripts/test_bridge_noise.py --params data/gap_arch/phys_bridge/ex_N6_T3_encdec_phmlp_c0_params.msgpack
"""
import argparse, json, os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import netket as nk
import flax.serialization as ser
from src_renyi.physical_ancilla.models_encdec import ARPhysEncDec
from src_renyi.physical_ancilla.bridge_noise import bridge_noise

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--n_pairs", type=int, default=4096)
ap.add_argument("--R", type=int, default=20); ap.add_argument("--chunk", type=int, default=1024)
a = ap.parse_args()
cfg_json = json.load(open(a.params.replace("_params.msgpack", ".json")))["args"]
N = cfg_json["N"]
hi = nk.hilbert.Spin(1 / 2, 2 * N)
model = ARPhysEncDec(hilbert=hi, embedding_d=cfg_json["d"], n_heads=cfg_json["heads"], sys_blocks=cfg_json["sys_blocks"],
                     enc_blocks=cfg_json["enc_blocks"], dec_blocks=cfg_json["dec_blocks"], copy_init=cfg_json["copy_init"],
                     param_dtype=jnp.float64, phase_net=cfg_json["phase_net"])
vs = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model, n_samples=16, seed=0)
params = ser.from_bytes(vs.parameters, open(a.params, "rb").read())
vs.parameters = params
cfg = jnp.asarray(hi.all_states())
D = 2 ** N


def Q_exact(p):
    psi = jnp.exp(model.apply({"params": p}, cfg).astype(jnp.complex128))
    Psi = psi.reshape(D, D) / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2))
    w = jnp.abs(Psi) ** 2; ps = w.sum(1); ph2 = w / ps[:, None]
    lr = jnp.zeros((D, D, D))
    for k in range(N):
        Mk = ph2.reshape(D, 2 ** k, -1).sum(-1); Mk1 = ph2.reshape(D, 2 ** (k + 1), -1).sum(-1)
        q1 = Mk1[:, 1::2] / jnp.maximum(Mk, 1e-300)
        g1 = jnp.sqrt(q1[:, None] * q1[None]); g0 = jnp.sqrt((1 - q1)[:, None] * (1 - q1)[None]); Z = g1 + g0
        pre = jnp.arange(D) >> (N - k); bit = (jnp.arange(D) >> (N - k - 1)) & 1
        g = jnp.where(bit[None, None] == 1, g1[:, :, pre], g0[:, :, pre]) / Z[:, :, pre]
        lr = lr + jnp.log(jnp.maximum(g, 1e-300))
    S = jnp.sum(ph2[:, None] * ph2[None] * jnp.exp(-lr), -1)
    return jnp.sum(ps[:, None] * ps[None] * S ** 2)


Qe, gQe = jax.value_and_grad(Q_exact)(params)
flat = lambda g: np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(g)])
ge = flat(gQe)
Qs, gs = [], []
for r in range(a.R):
    Q, Qerr, g = bridge_noise(vs, list(range(N)), a.n_pairs, key=r, chunk_size=a.chunk)
    Qs.append(Q); gs.append(flat(g))
Qs, gs = np.array(Qs), np.array(gs)
gm = gs.mean(0)
cos = lambda u, v: float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
print(f"N={N}  Q exacto = {float(Qe):.5f}   MC = {Qs.mean():.5f} ± {Qs.std() / np.sqrt(a.R):.5f}  "
      f"(desv. {(Qs.mean() - float(Qe)) / (Qs.std() / np.sqrt(a.R)):+.1f}σ)")
print(f"∇Q: cos(exacto, media de {a.R}) = {cos(ge, gm):.4f}   |media|/|exacto| = {np.linalg.norm(gm) / np.linalg.norm(ge):.3f}"
      f"   cos medio de una sola estimación = {np.mean([cos(ge, x) for x in gs]):.3f}")
ok = abs(Qs.mean() - float(Qe)) < 4 * Qs.std() / np.sqrt(a.R) + 1e-12 and cos(ge, gm) > 0.9
print("OK" if ok else "FALLO")
