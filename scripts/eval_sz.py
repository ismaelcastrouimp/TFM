"""
eval_sz.py
==========
Evaluación MC de F = E − T S2 de unos parámetros, CON etiqueta de Sz en la ancilla (= quitando la pureza fuera del sector;
pureza u1 simétrica, exacta para el modelo con etiqueta) o SIN ella (pureza inclinada, como antes).  E no depende de la
etiqueta (H conserva Sz).  dF/N frente al MRE (espectro por bloques en caché).  Lotes independientes: E_err, S2_err (como
gap_mc_hybrid.evaluate).
Uso:  python scripts/eval_sz.py --N 20 --arch vit:mlp --params X.msgpack --sz_label 1 --nb 128 --out X.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--J2", type=float, default=0.5); ap.add_argument("--N", type=int, default=20); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--chi", type=int, default=16); ap.add_argument("--arch", default="vit:mlp"); ap.add_argument("--params", required=True)
ap.add_argument("--sz_label", type=int, default=1); ap.add_argument("--nb", type=int, default=128)
ap.add_argument("--M", type=int, default=1024); ap.add_argument("--Mp", type=int, default=1024); ap.add_argument("--K", type=int, default=40)
ap.add_argument("--p_chunks", type=int, default=16); ap.add_argument("--e_chunk", type=int, default=64); ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--spectra_dir", default="data/gap_arch/generic/spectra"); ap.add_argument("--out", default=None)
a = ap.parse_args()
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import serialization
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla.training import make_sampler
from src_renyi.mps_ancilla.generic_eloc import build_hamiltonian, make_energy_terms
from src_renyi.mps_ancilla.tilted_loss import make_mix_partners, make_tilted_purity
from src_renyi.mps_ancilla.purity_lowvar import make_u1_mix_partners, make_u1_lowvar_purity
from src_renyi.mps_ancilla.sz_label import SzLabelModel
from src_renyi.mps_ancilla.block_ed import mre

N, T, chi, K = a.N, a.T, a.chi, a.K
model = build_alt_model(N, chi, a.arch, 16)
p = serialization.from_bytes(init_params(model, N), open(a.params, "rb").read())
ks = [0.05, 0.1, 0.2, 0.4, 1.0]
if a.sz_label:
    model = SzLabelModel(model, N)
    pur, mix = make_u1_lowvar_purity(model, N, chi, T, ks, K, shell=False, cv_t=False, sym=True), make_u1_mix_partners(model, N, ks, K)
else:
    pur, mix = make_tilted_purity(model, N, chi, T), make_mix_partners(model, N, ks, K)
H = build_hamiltonian("j1j2", N, J2=a.J2)
terms = make_energy_terms(model, chi, H)
sampler = make_sampler(model, N, a.M); ec = min(a.e_chunk, a.M); pc = a.p_chunks; mc = a.Mp // pc
ev = np.load(os.path.join(a.spectra_dir, f"j1j2_J2{a.J2}_N{N}.npy")); ref = mre(ev, T)


@jax.jit
def batch(key):
    k1, k2 = jax.random.split(key)
    s = sampler(p, k1)
    E = jnp.sum(jax.lax.map(lambda c: terms(p, c, 0.0, a.M)[1], s.reshape(-1, ec, N)))
    s_ = s[: a.Mp]; t, lq = mix(p, s_, k2)
    P = jnp.mean(jax.lax.map(lambda c: pur(p, c[0], c[1], c[2], K)[1][0],
                             (s_.reshape(pc, mc, N), t.reshape(pc, mc * K, N), lq.reshape(pc, mc * K))))
    return E, P


key = jax.random.PRNGKey(a.seed); Es, Ps = [], []; t0 = time.time()
for i in range(a.nb):
    key, k = jax.random.split(key); E, P = batch(k); Es.append(float(E)); Ps.append(float(P))
    if i in (0, 9) or (i + 1) % 32 == 0:
        print(f"  lote {i + 1}/{a.nb} ({time.time() - t0:.0f} s)", flush=True)
Es, Ps = np.array(Es), np.array(Ps); n = a.nb
E, S2 = Es.mean(), -np.log(Ps.mean()); Ee, Se = Es.std(ddof=1) / np.sqrt(n), Ps.std(ddof=1) / np.sqrt(n) / Ps.mean()
F = E - T * S2; dF = (F - ref["F"]) / N; dFe = np.hypot(Ee, T * Se) / N
r = dict(params=a.params, arch=a.arch, N=N, T=T, sz_label=a.sz_label, nb=n, M=a.M, Mp=a.Mp, K=K, E=E, E_err=Ee, S2=S2, S2_err=Se, F=F,
         dF_per_site=dF, dF_err=dFe, F_MRE=ref["F"], E_MRE=ref["E"], S2_MRE=ref["S2"])
print(f"{os.path.basename(a.params)} etiqueta={a.sz_label}: E={E:.5f}±{Ee:.5f} S2={S2:.5f}±{Se:.5f} F={F:.5f}  dF/N={dF:+.5f}±{dFe:.5f}  "
      f"(MRE: E={ref['E']:.5f} S2={ref['S2']:.5f})", flush=True)
if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(r, open(a.out, "w"), indent=1, default=float)
