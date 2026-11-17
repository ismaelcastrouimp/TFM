"""
make_hybrid_init.py
===================
Construye los parametros iniciales de vit:convmlp (models_hybrid.py) a partir de un vit:conv entrenado al MISMO N y chi:
copia todo lo comun y deja el residuo con su init (res_out = 0).  Comprueba que (logA, M) coinciden con el conv.
Uso:
  python scripts/make_hybrid_init.py --params data/gap_arch/params_X5_conv_chi8_from_N16.msgpack --N 25 --chi 8 \
      --out data/gap_arch/init_hyb_N25_chi8_from_X5.msgpack
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

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--N", type=int, required=True)
ap.add_argument("--chi", type=int, required=True); ap.add_argument("--conv_arch", default="vit:conv:r=1")
ap.add_argument("--arch", default="vit:convmlp:r=1"); ap.add_argument("--out", required=True)
a = ap.parse_args()
assert "/data/0/" not in os.path.abspath(a.out)
mc, mh = build_alt_model(a.N, a.chi, a.conv_arch), build_alt_model(a.N, a.chi, a.arch)
pc = jax.tree_util.tree_map(np.asarray, load_params(a.params, mc, a.N))
ph = jax.tree_util.tree_map(np.asarray, init_params(mh, a.N))
wrapped = "params" in ph and isinstance(ph["params"], dict)
pc = pc.get("params", pc); dst = ph["params"] if wrapped else ph
for k in pc:
    assert k in dst, k
    dst[k] = pc[k]
print("residuo nuevo:", [k for k in dst if k not in pc])
s = jnp.asarray(np.random.default_rng(0).choice([-1.0, 1.0], (64, a.N)))
lc, Mc = mc.apply({"params": pc}, s); lh, Mh = mh.apply({"params": dst}, s)
assert np.allclose(lc, lh) and np.allclose(Mc, Mh), "el hibrido no reproduce al conv"
print("comprobado: hibrido == conv en 64 configuraciones")
with open(a.out, "wb") as f:
    f.write(serialization.to_bytes(ph))
print("guardado", a.out)
