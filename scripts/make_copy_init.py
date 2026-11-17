"""
make_copy_init.py
=================
Parametros de vit:conv:r=1 con la ancilla = COPIA EXACTA de s (anc_out = 0, mps_copy = copia): G(s,s') = delta_{ss'},
rho = diag(p).  Con --freeze anc_ mps_copy en gap_mc.py y --h_x 0 se entrena SOLO p(s) en el Ising clasico
(test de capacidad/optimizacion del ARViT por MC; objetivo exacto en scripts/classical_mre_exact.py).
--sys_from params.msgpack:chi0 copia ademas los pesos del sistema (todo lo que no es anc_*/mps_copy) de otro modelo conv.
Uso: python scripts/make_copy_init.py --N 25 --chi 1 --out data/gap_arch/classical/init_copy_N25.msgpack [--sys_from X:8]
"""
import argparse, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax.serialization as serialization
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import load_params, init_params
from src_renyi.mps_ancilla.models import mps_overlap_aligned

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True); ap.add_argument("--d", type=int, default=16); ap.add_argument("--blocks", type=int, default=2); ap.add_argument("--chi", type=int, default=1)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--sys_from", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert "/data/0/" not in os.path.abspath(a.out)
N, chi = a.N, a.chi
model = build_alt_model(N, chi, "vit:conv:r=1", a.d, 4, a.blocks)
pl = jax.tree_util.tree_map(np.asarray, init_params(model, N, a.seed))
wrapped = "params" in pl and isinstance(pl["params"], dict)
dst = pl["params"] if wrapped else pl
dst["anc_out"] = {k: np.zeros_like(v) for k, v in dst["anc_out"].items()}
if a.sys_from:
    path, chi0 = a.sys_from.rsplit(":", 1)
    src = jax.tree_util.tree_map(np.asarray, load_params(path, build_alt_model(N, int(chi0), "vit:conv:r=1"), N))
    src = src.get("params", src)
    for k in src:
        if not (k.startswith("anc_") or k == "mps_copy"):
            dst[k] = src[k]
s = jnp.asarray(np.random.default_rng(0).choice([-1.0, 1.0], (64, N)))
_, M = model.apply({"params": dst}, s)
G = np.asarray(mps_overlap_aligned(jnp.repeat(M, 64, 0), jnp.tile(M, (64, 1, 1, 1, 1)), chi)).reshape(64, 64)
same = np.all(np.asarray(s)[:, None] == np.asarray(s)[None], -1)
assert np.allclose(G, same.astype(float)), "la ancilla no es una copia exacta"
print("comprobado: G(s,s') = delta en 64x64 configuraciones")
with open(a.out, "wb") as f:
    f.write(serialization.to_bytes(pl))
print("guardado", a.out)
