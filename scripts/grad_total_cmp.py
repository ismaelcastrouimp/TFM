"""
grad_total_cmp.py
=================
Ruido y SEÑAL del gradiente total de F2 = E − T S2 por lote (energía + pureza, como gap_mc_hybrid), sin referencia exacta
(sirve a cualquier N), en unos parámetros fijos.  Una configuración por proceso (memoria pico propia); con la misma
--seed las s son las mismas en todas.
  config = "nombre:eform:pest:psym:Mp:K:kappas:p_chunks"   (eform std|vmc; pest tilted|u1|loc1|ham2; psym 0|1)
Por configuración (nb lotes):  ruido = sqrt(tr Σ) de un lote;  señal |g| con |g|² = |m̂|² − tr Σ/nb (sin sesgo) y su error;
S/R = |g|/ruido;  (S/R)²/t = eficiencia por segundo (nº de "lotes equivalentes con S/R = 1" por segundo);  memoria pico.
Guarda <out_dir>/<nombre>.npz (media, varianza) y <nombre>.json.
--analyse: tabla de todas las configuraciones de out_dir y consistencia de las medias (todas insesgadas: |m_i − m_j| debe
ser del orden de sqrt(tr Σ_i/nb + tr Σ_j/nb)).
Uso:
  python scripts/grad_total_cmp.py --N 20 --arch vit:mlp --params X.msgpack --nb 60 --out_dir D --config comb:vmc:u1:1:1024:40:0.05,0.1,0.2,0.4,1.0:16
  python scripts/grad_total_cmp.py --analyse --out_dir D
"""
import argparse, json, os, sys, time, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--J2", type=float, default=0.5); ap.add_argument("--N", type=int, default=20); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--chi", type=int, default=16); ap.add_argument("--arch", default="vit:mlp")
ap.add_argument("--params", default=None); ap.add_argument("--nb", type=int, default=60); ap.add_argument("--M", type=int, default=1024)
ap.add_argument("--e_chunk", type=int, default=64); ap.add_argument("--config", default=None); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out_dir", required=True); ap.add_argument("--analyse", action="store_true")
ap.add_argument("--sz_label", type=int, default=0, help="1: etiqueta de Sz en la ancilla (solo pest u1)")
a = ap.parse_args()
assert "/data/0" not in os.path.abspath(a.out_dir)
import numpy as np

if a.analyse:
    R = {}
    for f in sorted(glob.glob(os.path.join(a.out_dir, "*.json"))):
        r = json.load(open(f)); z = np.load(f[:-5] + ".npz"); R[r["name"]] = (r, z["mean"], z["var"])
    print(f"{'config':12s} {'parejas':>7s} {'casc.':>5s} {'ruido':>7s} {'señal |g|':>14s} {'S/R':>6s} {'s/lote':>7s} {'(S/R)²/s':>9s} {'mem. pico':>9s}")
    for nm, (r, m, v) in R.items():
        print(f"{nm:12s} {r['pairs']:7d} {r['n_shell']:5d} {r['noise']:7.3f} {r['signal']:7.4f} ± {r['signal_err']:.4f} {r['snr']:6.4f} "
              f"{r['sec_per_batch']:7.2f} {r['snr2_per_s']:9.2e} {r['peak_GB']:7.2f} GB")
    names = list(R); print("consistencia de las medias: |m_i − m_j| / sqrt(trΣ_i/nb + trΣ_j/nb)  (~1 si las dos son insesgadas)")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            (ri, mi, vi), (rj, mj, vj) = R[names[i]], R[names[j]]
            print(f"  {names[i]:12s} vs {names[j]:12s}: {np.linalg.norm(mi - mj) / np.sqrt(vi.sum() / ri['nb'] + vj.sum() / rj['nb']):.2f}")
    sys.exit(0)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import serialization
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla.training import make_sampler
from src_renyi.mps_ancilla.tilted_loss import make_mix_partners, make_tilted_purity
from src_renyi.mps_ancilla.generic_eloc import build_hamiltonian, make_energy_terms
from src_renyi.mps_ancilla.vmc_energy import make_energy_terms_vmc
from src_renyi.mps_ancilla.purity_lowvar import make_u1_mix_partners, make_u1_lowvar_purity, make_u1_mask_purity

N, T, chi = a.N, a.T, a.chi; sg = jax.lax.stop_gradient
model = build_alt_model(N, chi, a.arch, 16)
p = serialization.from_bytes(init_params(model, N), open(a.params, "rb").read())
if a.sz_label:
    from src_renyi.mps_ancilla.sz_label import SzLabelModel
    model = SzLabelModel(model, N)
H = build_hamiltonian("j1j2", N, J2=a.J2)
sampler = jax.jit(make_sampler(model, N, a.M))
flat = lambda g: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(g)])
nm, eform, pest, psym, Mp, K, ks, pc = a.config.split(":")
psym, Mp, K, pc = bool(int(psym)), int(Mp), int(K), int(pc); ks = [float(x) for x in ks.split(",")]
assert Mp <= a.M and Mp % pc == 0 and (not a.sz_label or pest != "tilted")
terms = make_energy_terms_vmc(model, chi, H) if eform == "vmc" else make_energy_terms(model, chi, H)
if pest == "tilted":
    pur, mix = make_tilted_purity(model, N, chi, T), make_mix_partners(model, N, ks, K)
elif pest == "u1c":                            # kappas = "κ,μ" (componentes c = -1, 0, +1)
    from src_renyi.mps_ancilla.purity_charge import make_charge_mix_partners, charge_comps
    pur = make_u1_lowvar_purity(model, N, chi, T, ks, K, shell=False, cv_t=False, sym=psym)
    mix = make_charge_mix_partners(model, N, charge_comps(ks[0]), ks[1], K)
elif pest == "u1":
    pur, mix = make_u1_lowvar_purity(model, N, chi, T, ks, K, shell=False, cv_t=False, sym=psym), make_u1_mix_partners(model, N, ks, K)
else:
    pur, mix = make_u1_mask_purity(model, N, chi, T, ks, K, kind=pest, cv_t=False, sym=psym), make_u1_mix_partners(model, N, ks, K)
nsh = getattr(pur, "n_eval", 0); mc = Mp // pc; ec = min(a.e_chunk, a.M)


@jax.jit
def g(s, key):
    sc = s.reshape(-1, ec, N)
    Ebar = jnp.sum(jax.lax.map(lambda c: terms(p, c, 0.0, a.M)[1], sc))
    gE = jax.lax.map(lambda c: flat(jax.grad(lambda q: terms(q, c, Ebar, a.M)[0])(p)), sc).sum(0)
    s_ = s[:Mp]; t, lq = mix(p, s_, key)
    cs = (s_.reshape(pc, mc, N), t.reshape(pc, mc * K, N), lq.reshape(pc, mc * K))
    P = jnp.mean(jax.lax.map(lambda c: pur(p, c[0], c[1], c[2], K)[1][0], cs)); Pf = sg(P)
    gP = jax.lax.map(lambda c: flat(jax.grad(lambda q: pur(q, c[0], c[1], c[2], K, Pf, Pf)[0])(p)), cs).mean(0)
    return gE + gP, Ebar, P


key = jax.random.PRNGKey(a.seed); s1 = s2 = 0.0; tw = 0.0; Es, Ps = [], []; t0 = time.time()
print(f"{nm}: N={N} arch={a.arch} eform={eform} pest={pest} psym={psym} M={a.M} Mp={Mp} K={K} cascarón={nsh} p_chunks={pc}  "
      f"{jax.devices()}", flush=True)
for i in range(a.nb):
    key, k = jax.random.split(key); k1, k2 = jax.random.split(k); s = sampler(p, k1)
    tt = time.time(); v, E, P = g(s, k2); v = np.asarray(v)
    if i > 0:
        tw += time.time() - tt
    s1 = s1 + v; s2 = s2 + v * v; Es.append(float(E)); Ps.append(float(P))
    if i in (0, 9) or (i + 1) % 20 == 0:
        print(f"  lote {i + 1}/{a.nb} ({time.time() - t0:.0f} s)", flush=True)
n = a.nb; m = s1 / n; var = np.maximum((s2 - n * m * m) / (n - 1), 0); trS = var.sum()
g2 = m @ m - trS / n; sig = float(np.sqrt(max(g2, 0.0)))
g2_err = float(np.sqrt(4 * np.sum(m * m * var) / n + 2 * np.sum(var * var) / n ** 2))     # error de |g|² (Σ diagonal)
sig_err = float(g2_err / (2 * sig)) if sig > 0 else float(np.sqrt(g2_err))
noise = float(np.sqrt(trS)); tpb = tw / max(n - 1, 1)
try:
    peak = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
except Exception:
    peak = float("nan")
r = dict(g2=float(g2), g2_err=g2_err, name=nm, config=a.config, arch=a.arch, params=a.params, nb=n, pairs=Mp * K, n_shell=nsh, noise=noise, signal=sig,
         signal_err=sig_err, snr=sig / noise, sec_per_batch=tpb, snr2_per_s=(sig / noise) ** 2 / tpb, peak_GB=peak,
         E_mean=float(np.mean(Es)), S2_mean=float(-np.log(np.mean(Ps))))
print(f"{nm}: ruido {noise:.3f}  señal |g| = {sig:.4f} ± {sig_err:.4f} (|m̂| = {np.linalg.norm(m):.4f})  S/R = {r['snr']:.4f}  "
      f"{tpb:.2f} s/lote  (S/R)²/s = {r['snr2_per_s']:.2e}  memoria pico {peak:.2f} GB  E = {r['E_mean']:.4f}  S2 = {r['S2_mean']:.4f}", flush=True)
os.makedirs(a.out_dir, exist_ok=True)
np.savez(os.path.join(a.out_dir, nm + ".npz"), mean=m, var=var)
json.dump(r, open(os.path.join(a.out_dir, nm + ".json"), "w"), indent=1)
