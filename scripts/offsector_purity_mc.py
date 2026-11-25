"""
offsector_purity_mc.py
======================
Fracción de la pureza FUERA del sector de Sz, P_fuera/P = Σ_{Sz(t)≠Sz(s)} |ρ(s,t)|² / Σ |ρ(s,t)|², por MC a cualquier N,
con las parejas inclinadas SIN restricción de sector (tilted_loss.make_mix_partners):  w = p(t) f(s,t) / q(t|s),
P̂ = media de w,  P̂_fuera = media de w·[Sz(t) ≠ Sz(s)].  Error del cociente por lotes (nb lotes independientes).
Uso:  python scripts/offsector_purity_mc.py --N 20 --arch vit:mlp --params X.msgpack --nb 40 [--out X.json]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=20); ap.add_argument("--chi", type=int, default=16); ap.add_argument("--arch", default="vit:mlp")
ap.add_argument("--params", required=True); ap.add_argument("--nb", type=int, default=40); ap.add_argument("--M", type=int, default=256)
ap.add_argument("--K", type=int, default=40); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
a = ap.parse_args()
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import serialization
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla.training import make_sampler
from src_renyi.mps_ancilla.tilted_loss import make_mix_partners, make_pair_logs

N, chi, K = a.N, a.chi, a.K
model = build_alt_model(N, chi, a.arch, 16)
p = serialization.from_bytes(init_params(model, N), open(a.params, "rb").read())
sampler = make_sampler(model, N, a.M)
mix = jax.jit(make_mix_partners(model, N, [0.05, 0.1, 0.2, 0.4, 1.0], K))
pl = make_pair_logs(model, chi)


@jax.jit
def batch(s, t, lq):
    _, lpt, logf = pl(p, s, t, K)
    w = jnp.exp(lpt + logf - lq)
    out = jnp.sum(t, 1) != jnp.repeat(jnp.sum(s, 1), K)
    return jnp.mean(w), jnp.mean(jnp.where(out, w, 0.0)), jnp.mean(out)


key = jax.random.PRNGKey(a.seed); P, Po, fo = [], [], []
for i in range(a.nb):
    key, k1, k2 = jax.random.split(key, 3); s = sampler(p, k1); t, lq = mix(p, s, k2)
    x, y, z = batch(s, t, lq); P.append(float(x)); Po.append(float(y)); fo.append(float(z))
P, Po = np.asarray(P), np.asarray(Po); r = Po.sum() / P.sum()
# error del cociente (método delta, lotes independientes)
n = a.nb; mP, mo = P.mean(), Po.mean(); C = np.cov(np.stack([Po, P]), ddof=1) / n
err = abs(r) * np.sqrt(max(C[0, 0] / mo ** 2 + C[1, 1] / mP ** 2 - 2 * C[0, 1] / (mo * mP), 0.0)) if mo != 0 else np.sqrt(C[0, 0]) / mP
res = dict(params=a.params, N=N, arch=a.arch, nb=n, M=a.M, K=K, P=mP, P_err=float(P.std(ddof=1) / np.sqrt(n)), S2=float(-np.log(mP)),
           frac_out=float(r), frac_out_err=float(err), frac_pairs_out=float(np.mean(fo)))
print(f"{os.path.basename(a.params)}: P = {mP:.4g} ± {res['P_err']:.2g} (S2 = {res['S2']:.4f})   fracción de P fuera del sector = "
      f"{r:.2e} ± {err:.1e}   (parejas fuera del sector: {np.mean(fo):.2f})", flush=True)
if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
