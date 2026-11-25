"""
test_charge_q.py
================
Comprobación exacta (N pequeño, enumeración) de purity_charge: para varias s,
  (a) Σ_{t en el sector} q_mix(t|s) = 1;
  (b) log q_mix de JAX == referencia numpy (condicionales exactas de la p del modelo, como charge_B_exact.logq_all);
  (c) μ = 1, c = 0  ==  propuesta u1 de purity_lowvar (misma κ);
  (d) muestreador: frecuencias de n_draw parejas frente a q_mix (distancia de variación total y χ² reducido).
Uso:  python scripts/test_charge_q.py --N 8 --chi 4
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=8); ap.add_argument("--chi", type=int, default=4); ap.add_argument("--arch", default="vit:mlp")
ap.add_argument("--mu", type=float, default=0.3); ap.add_argument("--kappa", type=float, default=0.5)
ap.add_argument("--n_s", type=int, default=4); ap.add_argument("--n_draw", type=int, default=60000); ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args()
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla.purity_charge import make_charge_mix_logq, make_charge_mix_partners, charge_comps
from src_renyi.mps_ancilla.purity_lowvar import make_u1_mix_logq

N = a.N; D = 2 ** N
model = build_alt_model(N, a.chi, a.arch, 16)
q0 = init_params(model, N, a.seed)
p = jax.tree_util.tree_map(lambda x: x + 0.3 * jax.random.normal(jax.random.PRNGKey(a.seed), x.shape, x.dtype), q0)
idx = np.arange(D); z = (2 * ((idx[:, None] >> np.arange(N)) & 1) - 1).astype(np.float64); mag = z.sum(1)
lg_all = np.asarray(jax.nn.log_softmax(model.apply({"params": p}, jnp.asarray(z), only_logits=True), -1))
pt = np.exp(np.take_along_axis(lg_all, ((z > 0).astype(int))[..., None], -1)[..., 0].sum(1)); print(f"Σ p = {pt.sum():.12f}")
marg = [np.bincount(idx & ((1 << i) - 1), weights=pt, minlength=1 << i) for i in range(N + 1)]
comps = charge_comps(a.kappa); J = len(comps); K = J


def ref_logq(s, T_sec, comps, mu):                       # = charge_B_exact.logq_all con las marginales del modelo
    tb = ((T_sec + 1) // 2).astype(np.int64); tidx = (tb << np.arange(N)).sum(1); n = len(T_sec); m_s = s.sum(); lqs = []
    for kap, c in comps:
        lq = np.zeros(n); Q = np.zeros(n); cs = np.zeros(n)
        for i in range(N):
            pre = tidx & ((1 << i) - 1); den = marg[i][pre]; num = []
            for v in (-1, 1):
                pv = marg[i + 1][pre | (((v + 1) // 2) << i)] / np.maximum(den, 1e-300)
                Qv = Q + (v - s[i]) / 2; fe = np.abs((m_s - cs) - v) <= (N - i - 1)
                num.append(np.where(fe, pv * kap ** float(v != s[i]) * mu ** np.abs(Qv - c), 0.0))
            numv = np.stack(num, 1); ti = (T_sec[:, i] > 0).astype(int)
            lq += np.log(np.maximum(numv[np.arange(n), ti], 1e-300)) - np.log(np.maximum(numv.sum(1), 1e-300))
            Q = Q + (T_sec[:, i] - s[i]) / 2; cs = cs + T_sec[:, i]
        lqs.append(lq)
    return np.logaddexp.reduce(np.stack(lqs), 0) - np.log(len(comps))


mixq = jax.jit(make_charge_mix_logq(model, comps, a.mu, K))
mixq_u1c = jax.jit(make_charge_mix_logq(model, [(a.kappa, 0)], 1.0, 1))
u1 = make_u1_mix_logq(model)
rng = np.random.default_rng(0)
cand = np.where(np.bincount(mag.astype(int) + N)[mag.astype(int) + N] > 1)[0]          # s en sectores con más de un t
S_list = rng.choice(cand, size=a.n_s, replace=False)
ok = True
for si in S_list:
    s = z[si]; sec = np.where(mag == mag[si])[0]; Ts = z[sec]; Sr = np.repeat(s[None], len(sec), 0)
    lq = np.asarray(mixq(p, jnp.asarray(Sr), jnp.asarray(Ts))); q = np.exp(lq)
    lr = ref_logq(s, Ts, comps, a.mu)
    l1 = np.asarray(mixq_u1c(p, jnp.asarray(Sr), jnp.asarray(Ts)))
    lu, _ = u1(p, jnp.asarray(Sr), jnp.asarray(Ts), np.log([a.kappa]), np.log([1.0])); lu = np.asarray(lu)
    # (d) muestreo
    nrep = a.n_draw // K
    part = jax.jit(make_charge_mix_partners(model, N, comps, a.mu, K))
    t, lqs = part(p, jnp.asarray(np.repeat(s[None], nrep, 0)), jax.random.PRNGKey(int(si)))
    t = np.asarray(t); tid = (((t + 1) // 2).astype(np.int64) << np.arange(N)).sum(1)
    insec = np.all(mag[tid] == mag[si])
    cnt = np.bincount(np.searchsorted(sec, tid), minlength=len(sec)) / len(tid)
    tv = 0.5 * np.abs(cnt - q).sum(); nd = len(tid); chi2 = np.sum((cnt - q) ** 2 * nd / np.maximum(q, 1e-300)) / (len(sec) - 1)
    lq_s = np.asarray(lqs); lq_ref_s = lq[np.searchsorted(sec, tid)]
    print(f"s={si:4d} (Sz={int(mag[si]):+d}, sector {len(sec)}):  Σq = {q.sum():.12f}  |log q − ref|max = {np.abs(lq - lr).max():.1e}  "
          f"μ=1,c=0 vs u1: {np.abs(l1 - lu).max():.1e}  muestreo: en sector {insec}, VT = {tv:.4f}, χ²/gl = {chi2:.2f}, "
          f"|lq devuelto − lq|max = {np.abs(lq_s - lq_ref_s).max():.1e}", flush=True)
    ok &= abs(q.sum() - 1) < 1e-9 and np.abs(lq - lr).max() < 1e-8 and np.abs(l1 - lu).max() < 1e-9 and insec and chi2 < 1 + 4 * np.sqrt(2 / (len(sec) - 1))
print("TODO OK" if ok else "FALLO", flush=True)
