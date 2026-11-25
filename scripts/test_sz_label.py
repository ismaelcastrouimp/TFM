"""
test_sz_label.py
================
Comprobación exacta (N pequeño) de la etiqueta de Sz en la ancilla (sz_label.py), en parámetros aleatorios perturbados
(lejos de cualquier óptimo) y opcionalmente en unos dados:
  - ρ completa con y sin etiqueta: fracción de P fuera del sector de Sz (con etiqueta debe ser 0 exactamente) y
    max |ρ(s,t)| con Sz(s) != Sz(t);
  - con etiqueta: media MC (nb lotes) de P̂ del estimador u1 simétrico (comb) frente a la P total exacta.
Uso:  python scripts/test_sz_label.py --N 8 --chi 4 [--params X.msgpack]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=8); ap.add_argument("--chi", type=int, default=4); ap.add_argument("--arch", default="vit:mlp")
ap.add_argument("--T", type=float, default=3.0); ap.add_argument("--params", default=None); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
ap.add_argument("--nb", type=int, default=200); ap.add_argument("--M", type=int, default=256); ap.add_argument("--K", type=int, default=10)
a = ap.parse_args()
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import serialization
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla.training import compute_rho, make_sampler
from src_renyi.mps_ancilla.sz_label import SzLabelModel
from src_renyi.mps_ancilla.purity_lowvar import make_u1_mix_partners, make_u1_lowvar_purity

N, chi, T = a.N, a.chi, a.T; D = 2 ** N
base = build_alt_model(N, chi, a.arch, 16); lab = SzLabelModel(base, N)
idx = np.arange(D); z = (2 * ((idx[:, None] >> np.arange(N)) & 1) - 1); mag = z.sum(1); same = mag[:, None] == mag[None, :]
Sx = jnp.asarray(z.astype(np.float64))
ks = [0.05, 0.1, 0.2, 0.4, 1.0]
mix = jax.jit(make_u1_mix_partners(lab, N, ks, a.K))
pur = make_u1_lowvar_purity(lab, N, chi, T, ks, a.K, shell=False, cv_t=False, sym=True)
Phat = jax.jit(lambda p, s, t, lq: pur(p, s, t, lq, a.K)[1][0])
sampler = make_sampler(lab, N, a.M)

cases = []
if a.params:
    cases.append(("params", serialization.from_bytes(init_params(base, N), open(a.params, "rb").read())))
for sd in a.seeds:
    q = init_params(base, N, sd)
    q = jax.tree_util.tree_map(lambda x: x + 0.05 * jax.random.normal(jax.random.PRNGKey(sd), x.shape, x.dtype), q)
    cases.append((f"aleatorio s{sd}", q))
for nm, q in cases:
    out = {}
    for tag, m in (("sin etiqueta", base), ("con etiqueta", lab)):
        r = np.asarray(compute_rho(m, q, Sx, 64)); a2 = np.abs(r) ** 2
        out[tag] = (a2.sum(), a2[same].sum(), np.abs(r[~same]).max(), np.trace(r).real)
    key = jax.random.PRNGKey(0); vals = []
    for i in range(a.nb):
        key, k1, k2 = jax.random.split(key, 3); s = sampler(q, k1); t, lq = mix(q, s, k2); vals.append(float(Phat(q, s, t, lq)))
    vals = np.asarray(vals); Pl = out["con etiqueta"][0]
    for tag, (Pt, Pin, mx, tr) in out.items():
        print(f"{nm} {tag}: Tr ρ = {tr:.6f}  P total = {Pt:.6g}  P en el sector = {Pin:.6g}  fracción fuera = {(Pt - Pin) / Pt:.3e}  "
              f"max|ρ fuera| = {mx:.2e}", flush=True)
    print(f"{nm} con etiqueta: P̂ u1 (media de {a.nb} lotes) = {vals.mean():.6g} ± {vals.std(ddof=1) / np.sqrt(a.nb):.2g}  "
          f"frente a P exacta {Pl:.6g}  → (media − exacta)/error = {(vals.mean() - Pl) / (vals.std(ddof=1) / np.sqrt(a.nb)):+.2f}", flush=True)
