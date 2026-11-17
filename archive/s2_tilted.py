"""
s2_tilted.py
============
Estimador de la pureza P = sum_{s,t} p(s) p(t) f(s,t) (ARViT + ancilla MPS, f = |<Phi_t|Phi_s>|^2 normalizada) con
MUESTREO CONDICIONADO DEL COMPANERO (idea de Wang & Davis 2020 adaptada; opcion (a) de notas_gap.md, 12-11):
    P = E_{s~p}[ Z(s) ],   Z(s) = sum_t p(t) f(s,t) = (rho^2)_ss / p(s)
    t ~ q(t|s) = prod_i q(t_i | t_<i, s),   q(t_i|.) ∝ p(t_i|t_<i) * kappa^[t_i != s_i]   (AR "inclinado" hacia s)
    Z_hat(s) = (1/K) sum_k p(t_k) f(s,t_k) / q(t_k|s)        (insesgado para cualquier kappa > 0)
kappa = 1 -> t ~ p independiente (= pares iid); kappa -> 0 -> t = s (solo d=0).  El optimo pone a t a distancia ~d*.
--kappa auto: piloto sobre una rejilla, elige el kappa de minima varianza relativa de Z_hat por muestra.
Diagnosticos: ESS de los pesos por s (sobre K) y global, CV de P_hat por lote, distancia media de t a s, tiempos.
Compara (mismos parametros) con el estimador nbr de mc_loss (make_mc_stats) a --cmp_pb (p.ej. 8 64): CV por lote y tiempo.
--exact (N<=10): P exacta del modelo (make_loss_fns) para validar el insesgo.
Uso:
  python scripts/s2_tilted.py --params X.msgpack --chi 8 --N 25 --M 256 --K 16 --kappa auto --batches 64 --cmp_pb 8 64 --out f.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src_renyi.mps_ancilla.training import make_sampler, make_loss_fns
from src_renyi.mps_ancilla.mc_loss import make_mc_stats, auto_chunks
from src_renyi.mps_ancilla.models import mps_overlap_aligned
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import load_params
from src_renyi.mps_ancilla.tilted_loss import make_mix_partners

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--N", type=int, required=True); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--h_x", type=float, default=-1.5); ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--M", type=int, default=256); ap.add_argument("--K", type=int, default=16)
ap.add_argument("--kappa", default="auto"); ap.add_argument("--batches", type=int, default=64)
ap.add_argument("--kappas", type=float, nargs="*", default=None, help="mezcla defensiva (anula --kappa)")
ap.add_argument("--cmp_pb", type=int, nargs="*", default=[]); ap.add_argument("--cmp_M", type=int, default=1024)
ap.add_argument("--exact", action="store_true"); ap.add_argument("--seed", type=int, default=4242)
ap.add_argument("--mem_gb", type=float, default=8.0); ap.add_argument("--out", default=None)
a = ap.parse_args()
N, chi = a.N, a.chi
model = build_alt_model(N, chi, a.arch, a.d, 4, a.blocks); params = load_params(a.params, model, N)
sampler = make_sampler(model, N, a.M)


MIX = make_mix_partners(model, N, a.kappas, a.K) if a.kappas else None


@jax.jit
def tilt_weights(params, s, key, logk):
    """s: (M,N) objetivos.  Devuelve log w (M,K), distancia d(s,t) (M,K)."""
    S = jnp.repeat(s, a.K, axis=0)                                       # (M*K, N)
    B = S.shape[0]
    if MIX is not None:
        t, lq = MIX(params, s, key)
        return _weights(params, S, t, lq)

    def body(i, carry):
        t, lq, key = carry
        key, sub = jax.random.split(key)
        lg = model.apply({"params": params}, t, only_logits=True)        # (B, N, 2)
        li = jax.lax.dynamic_index_in_dim(lg, i, axis=1, keepdims=False)  # (B, 2): [p(-1), p(+1)]
        si = jax.lax.dynamic_index_in_dim(S, i, axis=1, keepdims=False)
        other = jnp.stack([si > 0, si < 0], axis=-1).astype(li.dtype)    # 1 en la opcion distinta de s_i
        lt = jax.nn.log_softmax(li + logk * other, axis=-1)
        draw = jax.random.categorical(sub, lt, axis=-1)
        lq = lq + jnp.take_along_axis(lt, draw[:, None], axis=-1)[:, 0]
        return t.at[:, i].set(2.0 * draw - 1.0), lq, key
    t, lq, _ = jax.lax.fori_loop(0, N, body, (jnp.zeros_like(S), jnp.zeros(B), key))
    return _weights(params, S, t, lq)


def _weights(params, S, t, lq):
    lA_t, M_t = model.apply({"params": params}, t)
    lA_s, M_s = model.apply({"params": params}, S)
    n_t = jnp.real(mps_overlap_aligned(M_t, M_t, chi)); n_s = jnp.real(mps_overlap_aligned(M_s, M_s, chi))
    f = jnp.abs(mps_overlap_aligned(M_t, M_s, chi)) ** 2 / (n_t * n_s)
    logw = 2.0 * jnp.real(lA_t) + jnp.log(f) - lq
    d = jnp.sum(t != S, axis=1)
    return logw.reshape(a.M, a.K), d.reshape(a.M, a.K)


# chequeo de consistencia: sum_i log_softmax(logits)[s_i] == 2 Re log A(s)
key = jax.random.PRNGKey(a.seed)
key, k0 = jax.random.split(key); s0 = sampler(params, k0)
lg = jax.nn.log_softmax(model.apply({"params": params}, s0, only_logits=True), axis=-1)
lp_cond = jnp.sum(jnp.take_along_axis(lg, ((s0 > 0).astype(int))[..., None], axis=-1)[..., 0], axis=1)
lp_full = 2 * jnp.real(model.apply({"params": params}, s0)[0])
print(f"N={N} chi={chi} M={a.M} K={a.K}  consistencia logits/logA: max|dif|={float(jnp.max(jnp.abs(lp_cond - lp_full))):.2e}", flush=True)


def run(logk, nb, key):
    Zs, ess_s, ds, lws = [], [], [], []
    t0 = time.time()
    for _ in range(nb):
        key, k1, k2 = jax.random.split(key, 3)
        s = sampler(params, k1)
        lw, d = tilt_weights(params, s, k2, logk)
        lw = np.asarray(lw); d = np.asarray(d)
        lws.append(lw); ds.append(d)
    jax.block_until_ready(lw)
    dt = (time.time() - t0) / nb
    LW = np.concatenate(lws); D = np.concatenate(ds)
    ref = LW.max()
    W = np.exp(LW - ref)                                     # escala comun
    Z = W.mean(1)                                            # Z_hat(s) (escalado)
    ess = (W.sum(1) ** 2 / (W ** 2).sum(1)) / a.K            # ESS por s, fraccion de K
    Pb = Z.reshape(nb, a.M).mean(1)                          # P_hat por lote
    return dict(log_P=float(np.log(Z.mean()) + ref), S2=float(-(np.log(Z.mean()) + ref)),
                cv_Z=float(Z.std() / Z.mean()), cv_batch=float(Pb.std(ddof=1) / Pb.mean()) if nb > 1 else None,
                S2_err=float(Pb.std(ddof=1) / Pb.mean() / np.sqrt(nb)) if nb > 1 else None,
                ess_frac_mean=float(np.mean(ess)), ess_global=float(W.sum() ** 2 / (W ** 2).sum() / W.size),
                d_mean=float(D.mean()), d_hist=np.bincount(D.ravel(), minlength=N + 1)[:12].tolist(),
                P_frac_d=(np.bincount(D.ravel(), weights=W.ravel(), minlength=N + 1)[:12] / W.sum()).round(4).tolist(),
                s_per_batch=dt), key


res = dict(params=a.params, N=N, chi=chi, M=a.M, K=a.K)
if a.kappas:
    kap = "mix:" + ",".join(str(k) for k in a.kappas)
elif a.kappa == "auto":
    grid = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    pil = {}
    for k in grid:
        r, key = run(np.log(k), max(4, a.batches // 8), key)
        pil[k] = r
        print(f"  piloto kappa={k:<5}: S2={r['S2']:.4f}  CV(Z_s)={r['cv_Z']:.3f}  ESS/K={r['ess_frac_mean']:.3f}  <d>={r['d_mean']:.2f}", flush=True)
    kap = min(grid, key=lambda k: pil[k]["cv_Z"])
    res["pilot"] = {str(k): v for k, v in pil.items()}
else:
    kap = float(a.kappa)
r, key = run(0.0 if a.kappas else np.log(kap), a.batches, key)
res.update(kappa=kap, **r)
print(f"TILTED kappa={kap}: S2={r['S2']:.4f} +- {r['S2_err']:.4f}  CV(P_lote)={r['cv_batch']:.3f}  CV(Z_s)={r['cv_Z']:.3f}  "
      f"ESS/K={r['ess_frac_mean']:.3f}  <d>={r['d_mean']:.2f}  hist d={r['d_hist']}  P por d={r['P_frac_d']}  {r['s_per_batch']:.2f} s/lote({a.M}x{a.K})", flush=True)

for pb in a.cmp_pb:
    nch = auto_chunks(a.cmp_M, N, chi, pb, None, a.mem_gb, grad=False)
    st = jax.jit(make_mc_stats(model, N, a.T, a.h_x, a.J_ZZ, None, pb, "nbr", nch))
    smp = make_sampler(model, N, a.cmp_M); Ps = []; t0 = None
    for b in range(a.batches + 1):
        key, k1 = jax.random.split(key)
        _, S2b = st(params, smp(params, k1))
        if b == 0:
            jax.block_until_ready(S2b); t0 = time.time(); continue  # el primero compila
        Ps.append(float(np.exp(-S2b)))
    dt = (time.time() - t0) / a.batches; Ps = np.array(Ps)
    rr = dict(S2=float(-np.log(Ps.mean())), S2_err=float(Ps.std(ddof=1) / Ps.mean() / np.sqrt(len(Ps))),
              cv_batch=float(Ps.std(ddof=1) / Ps.mean()), S2_mean_of_logs=float(np.mean(-np.log(Ps))), s_per_batch=dt)
    res[f"nbr_pb{pb}"] = rr
    print(f"NBR pb={pb} M={a.cmp_M}: S2={rr['S2']:.4f} +- {rr['S2_err']:.4f}  CV(P_lote)={rr['cv_batch']:.3f}  "
          f"<-log P_lote>={rr['S2_mean_of_logs']:.4f}  {dt:.2f} s/lote", flush=True)

if a.exact:
    assert N <= 12
    idx = np.arange(2 ** N); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
    H = np.zeros((2 ** N, 2 ** N)); H[idx, idx] = a.J_ZZ * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
    for i in range(N):
        H[idx, idx ^ (1 << i)] += a.h_x
    loss, _ = make_loss_fns(model, all_s, H, a.T, chunk=64)
    _, (_, S2x) = jax.jit(loss)(params)
    res["S2_exact"] = float(S2x)
    print(f"EXACTO (ED del modelo): S2={float(S2x):.5f}   tilted - exacto = {r['S2'] - float(S2x):+.5f} ({(r['S2'] - float(S2x)) / r['S2_err']:+.1f} sigma)", flush=True)
if a.out:
    assert "/data/0/" not in os.path.abspath(a.out)
    json.dump(res, open(a.out, "w"), indent=1)
