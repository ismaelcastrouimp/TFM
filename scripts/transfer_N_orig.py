"""
transfer_N_orig.py
==================
Transfiere la arquitectura ORIGINAL (models.py / ladder.build_model: ARViT + ancilla MPS con contexto) de N_from a N_to.
Dependen de N:
  pos (N+1, d)                     -> interpolacion lineal en los sitios (fila 0 = BOS se copia), como transfer_N.py
  mps_head kernel (d, N*P), bias   -> bloque P = 2*chi*chi por sitio de salida k: se copia el del sitio viejo m(k) (nearest)
  mps_copy (N, N*P)                -> C'[j, k] = C[(m(k)+o) mod N_from, m(k)], o = desfase circular j-k en N_to
                                      (se conservan |o| <= N_from//2 - 1; el resto a 0).  Asi la copia diagonal y la
                                      estructura local, incluida la que cruza la frontera PBC, se trasladan sin emborronar.
Uso:
  python scripts/transfer_N_orig.py --params data/gap_arch/scaling/params_S_old_N16_chi4.msgpack --chi 4 \
      --N_from 16 --N_to 20 --out data/gap_arch/scaling/init_old_N20_from_N16_chi4.msgpack
"""
import argparse, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import flax.serialization as serialization
from src_renyi.mps_ancilla.ladder import load_params, init_params, build_model

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--N_from", type=int, required=True); ap.add_argument("--N_to", type=int, required=True)
ap.add_argument("--d", type=int, default=16); ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert "/data/0/" not in os.path.abspath(a.out), "no escribir en data/0"
Nf, Nt, P = a.N_from, a.N_to, 2 * a.chi * a.chi

p0 = jax.tree_util.tree_map(np.asarray, load_params(a.params, build_model(Nf, a.chi, a.d, 4, a.blocks), Nf))
p1 = jax.tree_util.tree_map(np.asarray, init_params(build_model(Nt, a.chi, a.d, 4, a.blocks), Nt))
m = np.rint(np.arange(Nt) * (Nf - 1) / (Nt - 1)).astype(int)          # sitio nuevo k -> sitio viejo m(k)
cols = (m[:, None] * P + np.arange(P)[None]).reshape(-1)              # columnas viejas para cada columna nueva


def circ(o, n):  # desfase circular en (-n/2, n/2]
    o = o % n
    return np.where(o > n // 2, o - n, o)


def walk(src, dst, path=""):
    out = {}
    for k in dst:
        if isinstance(dst[k], dict):
            out[k] = walk(src[k], dst[k], path + "/" + k); continue
        s, t = src[k], dst[k]
        if s.shape == t.shape:
            out[k] = s; continue
        q = path + "/" + k
        if k == "pos":
            xf, xt = np.linspace(0, 1, Nf), np.linspace(0, 1, Nt)
            body = np.stack([np.interp(xt, xf, s[1:, j]) for j in range(s.shape[1])], 1)
            out[k] = np.concatenate([s[:1], body], 0)
        elif path.endswith("mps_head"):
            out[k] = s[..., cols]
        elif k == "mps_copy":
            new = np.zeros(t.shape)
            for kk in range(Nt):
                for j in range(Nt):
                    o = circ(j - kk, Nt)
                    if abs(o) <= Nf // 2 - 1:
                        new[j, kk * P:(kk + 1) * P] = s[(m[kk] + o) % Nf, m[kk] * P:(m[kk] + 1) * P]
            out[k] = new
        else:
            raise ValueError(f"{q}: {s.shape} -> {t.shape} sin regla")
        print(f"{q}: {s.shape} -> {out[k].shape}")
    return out


new = walk(p0, p1)
# autocomprobacion: con N_to == N_from la regla de mps_copy debe reproducir la matriz salvo desfases |o| = N/2
if Nt == Nf:
    o = circ(np.arange(Nf)[:, None] - np.arange(Nf)[None], Nf)
    keep = np.repeat(np.abs(o) <= Nf // 2 - 1, P, axis=1)
    assert np.allclose(new["mps_copy"][keep], p0["mps_copy"][keep])
with open(a.out, "wb") as f:
    f.write(serialization.to_bytes(new))
print("guardado", a.out)
