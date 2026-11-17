"""
transfer_N.py
=============
Transfiere parametros de un modelo models_alt (sys=vit/vitp, ancilla local conv/convg/causal) entrenado a N_from a
otro N_to.  Todo es independiente de N salvo la tabla de posiciones 'pos' (N+1, d) del ViT (y rel_bias de vitp),
que se interpola linealmente en los sitios 1..N (el token BOS, fila 0, se copia).
Uso:
  python scripts/transfer_N.py --params data/gap_arch/scaling/params_S_mc_N16.msgpack --arch vit:conv:r=1 --chi 4 \
      --N_from 16 --N_to 25 --out data/gap_arch/scaling/init_N25_from_N16.msgpack
"""
import argparse, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import flax.serialization as serialization
from src_renyi.mps_ancilla.ladder import load_params, init_params
from src_renyi.mps_ancilla.models_alt import build_alt_model

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--arch", required=True)
ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--N_from", type=int, required=True); ap.add_argument("--N_to", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert "/data/0/" not in os.path.abspath(a.out), "no escribir en data/0"
assert a.arch.split(":")[0] == "vit", "solo implementado para sys=vit"

p0 = load_params(a.params, build_alt_model(a.N_from, a.chi, a.arch), a.N_from)
p1 = init_params(build_alt_model(a.N_to, a.chi, a.arch), a.N_to)
p0 = jax.tree_util.tree_map(np.asarray, p0)
p1 = jax.tree_util.tree_map(np.asarray, p1)
p0 = p0.get("params", p0) if isinstance(p0, dict) else p0

def walk(src, dst, path=""):
    out = {}
    for k in dst:
        if isinstance(dst[k], dict):
            out[k] = walk(src[k], dst[k], path + "/" + k)
        elif k == "pos":
            s = src[k]; nf, nt = s.shape[0] - 1, dst[k].shape[0] - 1
            xf = np.linspace(0, 1, nf); xt = np.linspace(0, 1, nt)
            body = np.stack([np.interp(xt, xf, s[1:, j]) for j in range(s.shape[1])], 1)
            out[k] = np.concatenate([s[:1], body], 0)
            print(f"{path}/{k}: {s.shape} -> {out[k].shape} (interpolado)")
        else:
            assert src[k].shape == dst[k].shape, f"{path}/{k}: {src[k].shape} vs {dst[k].shape}"
            out[k] = src[k]
    return out

wrapped = "params" in p1 and isinstance(p1["params"], dict)
src = p0 if not wrapped or "params" not in p0 else p0["params"]
new = walk(src, p1["params"] if wrapped else p1)
new = {"params": new} if wrapped else new
with open(a.out, "wb") as f:
    f.write(serialization.to_bytes(new))
print("guardado", a.out)
