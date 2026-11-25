"""
test_label_estimators.py
========================
Con etiqueta de Sz (N pequeño, parámetros aleatorios perturbados): para cada estimador de la pureza en el sector
(u1, u1shell, loc1, ham2, u1c) comprueba que el gradiente de la pérdida es finito (sin nan por log 0 en los huecos no
válidos del cascarón) y que la media de P̂ en nb lotes coincide con la P exacta (ρ completa).
Uso:  python scripts/test_label_estimators.py --N 8 --chi 4
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=8); ap.add_argument("--chi", type=int, default=4); ap.add_argument("--nb", type=int, default=100)
ap.add_argument("--M", type=int, default=128); ap.add_argument("--K", type=int, default=6); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla.training import compute_rho, make_sampler
from src_renyi.mps_ancilla.sz_label import SzLabelModel
from src_renyi.mps_ancilla.purity_lowvar import make_u1_mix_partners, make_u1_lowvar_purity, make_u1_mask_purity
from src_renyi.mps_ancilla.purity_charge import make_charge_mix_partners, charge_comps

N, chi, T, K = a.N, a.chi, 3.0, a.K; D = 2 ** N
base = build_alt_model(N, chi, "vit:mlp", 16); lab = SzLabelModel(base, N)
q = init_params(base, N, a.seed)
q = jax.tree_util.tree_map(lambda x: x + 0.05 * jax.random.normal(jax.random.PRNGKey(a.seed), x.shape, x.dtype), q)
idx = np.arange(D); Sx = jnp.asarray((2 * ((idx[:, None] >> np.arange(N)) & 1) - 1).astype(np.float64))
Px = float(np.sum(np.abs(np.asarray(compute_rho(lab, q, Sx, 64))) ** 2))
ks = [0.05, 0.1, 0.2, 0.4, 1.0][:min(5, K)]
sampler = make_sampler(lab, N, a.M); flat = lambda g: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(g)])
for nm in ("u1", "u1shell", "loc1", "ham2", "u1c"):
    if nm in ("u1", "u1shell"):
        pur = make_u1_lowvar_purity(lab, N, chi, T, ks, K, shell=nm == "u1shell", cv_t=False, sym=True); mix = make_u1_mix_partners(lab, N, ks, K)
    elif nm == "u1c":
        pur = make_u1_lowvar_purity(lab, N, chi, T, ks, K, shell=False, cv_t=False, sym=True)
        mix = make_charge_mix_partners(lab, N, charge_comps(0.5), 0.3, K)
    else:
        pur = make_u1_mask_purity(lab, N, chi, T, ks, K, kind=nm, cv_t=False, sym=True); mix = make_u1_mix_partners(lab, N, ks, K)
    mix = jax.jit(mix)
    f = jax.jit(lambda p, s, t, lq: (pur(p, s, t, lq, K)[1][0], flat(jax.grad(lambda r: pur(r, s, t, lq, K)[0])(p))))
    key = jax.random.PRNGKey(0); vals = []; fin = True
    for i in range(a.nb):
        key, k1, k2 = jax.random.split(key, 3); s = sampler(q, k1); t, lq = mix(q, s, k2); P, g = f(q, s, t, lq)
        vals.append(float(P)); fin &= bool(np.all(np.isfinite(np.asarray(g))))
    v = np.asarray(vals); se = v.std(ddof=1) / np.sqrt(a.nb)
    print(f"{nm:8s}: gradiente finito en todos los lotes: {fin}   P̂ = {v.mean():.6g} ± {se:.2g}  frente a exacta {Px:.6g}  "
          f"→ {(v.mean() - Px) / se:+.2f} errores   (ruido relativo de P̂ por lote {v.std(ddof=1) / v.mean():.3f})", flush=True)
