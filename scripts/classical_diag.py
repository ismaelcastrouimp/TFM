"""
classical_diag.py
=================
Diagnosticos del ARViT entrenado en el Ising clasico (h_x=0, ancilla copia; ver gap_arch_queue30.sh) frente al MRE exacto.
Por muestras s ~ p_modelo (log p exacta del autorregresivo):
  - pesos por capa k (= numero de paredes de dominio) frente a P_MRE(k) = g_k (c-E_k)_+ / Z
  - media y dispersion de log p dentro de cada k (MRE: log p constante en cada k -> dispersion 0)
  - Var(E) del modelo frente al MRE
  - descomposicion exacta de F: F_mod - F_MRE = [parte de capas: F con p_k del modelo y p uniforme en cada capa]
                                              + [parte intra-capa: T*(S2_uniforme_en_capa - S2_mod) >= 0]
    (P_mod = sum_s p(s)^2 = E_s[p(s)];  a p_k fijos, la uniforme en cada capa maximiza S2)
Uso: python scripts/classical_diag.py --params data/gap_arch/classical/params_C0_scratch.msgpack --N 25 --T 3
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from math import comb
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import load_params
from src_renyi.mps_ancilla.training import make_sampler

ap = argparse.ArgumentParser()
ap.add_argument("--params", nargs="+", required=True); ap.add_argument("--N", type=int, default=25); ap.add_argument("--d", type=int, default=16); ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--T", type=float, default=3.0); ap.add_argument("--J", type=float, default=-1.0)
ap.add_argument("--samples", type=int, default=32768); ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--out", default=None)
a = ap.parse_args()
N, T = a.N, a.T
ks = np.arange(0, N + 1, 2); g = np.array([2.0 * comb(N, int(k)) for k in ks]); Ek = a.J * (N - 2 * ks)
from scipy.optimize import brentq
w = lambda c: g * np.maximum(c - Ek, 0)
c = brentq(lambda c: c - (w(c) * Ek).sum() / w(c).sum() - 2 * T, Ek.min() + 1e-12, Ek.max() + 100)
Pk_ex = w(c) / w(c).sum(); E_ex = (Pk_ex * Ek).sum(); S2_ex = -np.log((Pk_ex ** 2 / g).sum()); F_ex = E_ex - T * S2_ex
varE_ex = (Pk_ex * Ek ** 2).sum() - E_ex ** 2
print(f"MRE: E={E_ex:.4f} S2={S2_ex:.4f} F={F_ex:.4f} Var(E)={varE_ex:.3f}")
out = []
for pth in a.params:
    model = build_alt_model(N, 1, "vit:conv:r=1", a.d, 4, a.blocks); params = load_params(pth, model, N)
    sampler = make_sampler(model, N, a.batch)
    logp_f = jax.jit(lambda p, s: 2 * jnp.real(model.apply({"params": p}, s)[0]))
    key = jax.random.PRNGKey(3); S, LP = [], []
    for _ in range(a.samples // a.batch):
        key, sub = jax.random.split(key); s = sampler(params, sub)
        S.append(np.asarray(s)); LP.append(np.asarray(logp_f(params, s)))
    S, LP = np.concatenate(S), np.concatenate(LP)
    k = (S != np.roll(S, -1, 1)).sum(1); E = a.J * (N - 2 * k)
    Pk = np.array([(k == kk).mean() for kk in ks])
    P_mod = np.exp(LP).mean(); S2_mod = -np.log(P_mod); E_mod = E.mean(); F_mod = E_mod - T * S2_mod
    S2_shell = -np.log((Pk ** 2 / g).sum()); F_shell = E_mod - T * S2_shell     # p_k del modelo, uniforme en cada capa
    print(f"\n{os.path.basename(pth)}: E={E_mod:.4f} (zz1={-E_mod / N / a.J * -1:.4f}) S2={S2_mod:.4f} F={F_mod:.4f} "
          f"dF={F_mod - F_ex:+.4f}  Var(E)={E.var():.3f}")
    print(f"  reparto dF: capas (p_k) {F_shell - F_ex:+.4f}  +  intra-capa (no uniforme) {F_mod - F_shell:+.4f}")
    print("   k   P_MRE(k)  P_mod(k)   <log p>_mod  log p_MRE   sd(log p)_mod")
    rows = []
    for i, kk in enumerate(ks):
        m = k == kk
        lpe = np.log((c - Ek[i]) / w(c).sum()) if c > Ek[i] else -np.inf
        if Pk[i] > 0 or Pk_ex[i] > 0:
            mu = LP[m].mean() if m.any() else np.nan; sd = LP[m].std() if m.sum() > 1 else np.nan
            print(f"  {kk:2d}   {Pk_ex[i]:.4f}    {Pk[i]:.4f}    {mu:9.3f}   {lpe:9.3f}   {sd:.3f}")
            rows.append(dict(k=int(kk), P_MRE=float(Pk_ex[i]), P_mod=float(Pk[i]), logp_mean=float(mu), logp_MRE=float(lpe), logp_sd=float(sd)))
    out.append(dict(params=pth, E=E_mod, S2=S2_mod, F=F_mod, dF=F_mod - F_ex, varE=float(E.var()), dF_shell=F_shell - F_ex,
                    dF_intra=F_mod - F_shell, shells=rows))
if a.out:
    json.dump(dict(MRE=dict(E=E_ex, S2=S2_ex, F=F_ex, varE=varE_ex), models=out), open(a.out, "w"), indent=1)
