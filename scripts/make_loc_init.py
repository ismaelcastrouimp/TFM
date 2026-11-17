"""
make_loc_init.py
================
Convierte un vit:conv:r=R entrenado en vit:convloc:r=R (models_hybrid.AblationModel): copia los pesos COMPARTIDOS de la
cabeza conv en cada sitio (loc_*[k] = anc_*), con lo que el modelo arranca exactamente en el conv.  Comprueba la igualdad.
Uso:
  python scripts/make_loc_init.py --params data/gap_arch/mcN10/params_mc_conv_s1.msgpack --N 10 --chi 16 \
      --out data/gap_arch/mcN10/init_loc_from_mc_conv_s1.msgpack
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
ap.add_argument("--chi", type=int, required=True); ap.add_argument("--r", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert "/data/0/" not in os.path.abspath(a.out)
N = a.N
mc, ml = build_alt_model(N, a.chi, f"vit:conv:r={a.r}"), build_alt_model(N, a.chi, f"vit:convloc:r={a.r}")
pc = jax.tree_util.tree_map(np.asarray, load_params(a.params, mc, N)); pc = pc.get("params", pc)
pl = jax.tree_util.tree_map(np.asarray, init_params(ml, N))
wrapped = "params" in pl and isinstance(pl["params"], dict)
dst = pl["params"] if wrapped else pl
tile = lambda x: np.broadcast_to(x, (N,) + x.shape).copy()
for k in pc:
    if k.startswith("anc_h"):
        li = k[5:]
        dst[f"loc_h{li}"] = tile(pc[k]["kernel"]); dst[f"loc_b{li}"] = tile(pc[k]["bias"])
    elif k == "anc_out":
        dst["loc_out"] = tile(pc[k]["kernel"]); dst["loc_bout"] = tile(pc[k]["bias"])
    elif k == "mps_copy":
        dst[k] = tile(pc[k])
    else:
        assert k in dst, k
        dst[k] = pc[k]
s = jnp.asarray(np.random.default_rng(0).choice([-1.0, 1.0], (64, N)))
lc, Mc = mc.apply({"params": pc}, s); ll, Ml = ml.apply({"params": dst}, s)
assert np.allclose(lc, ll) and np.allclose(Mc, Ml), "convloc no reproduce al conv"
print("comprobado: convloc == conv en 64 configuraciones")
with open(a.out, "wb") as f:
    f.write(serialization.to_bytes(pl))
print("guardado", a.out)
