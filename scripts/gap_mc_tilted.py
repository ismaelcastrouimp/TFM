"""
gap_mc_tilted.py
================
Entrenamiento MC (ARViT + ancilla MPS, F = E - T S2) con el termino de pureza estimado por el AR INCLINADO
(src_renyi/mps_ancilla/tilted_loss.py; .claude/notas.md, apartados 10-16).
  energia: igual que mc_loss.py (energia local + score con linea base) sobre M muestras s ~ p.
  pureza : las primeras --Mp muestras s, cada una con --K companeros t ~ q(.|s) (kappa), peso w = p(t) f / q.
           denominador --pden batch (P_hat del lote) | ema (media exponencial de P_hat, --pden_decay).
  kappa  : --kappa inicial; cada --kappa_every pasos se prueba kappa * {0.6,0.8,1,1.25,1.6} en un lote piloto y
           se queda el de minima CV de Z_hat(s) (no sesga: solo cambia la varianza).
Fases main (warmup + coseno) y polish (lr constante, --polish_M / --polish_Mp), EMA de parametros, evaluacion
final con el nbr de siempre (make_evaluator) y con el inclinado; N <= 10: F exacta del estado (ED).
Uso (receta de Q2 desde QD_X5):
  python scripts/gap_mc_tilted.py --N 25 --T 3 --chi 8 --init in/params_QD_X5.msgpack:8 --M 1024 --Mp 256 --K 4 \
      --steps 3000 --lr 1e-3 --polish_steps 500 --polish_M 2048 --polish_Mp 512 --label TL --out_dir out/pden
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

from src_renyi.mps_ancilla.training import make_sampler, make_loss_fns
from src_renyi.mps_ancilla.mc_loss import make_mc_terms, auto_chunks
from src_renyi.mps_ancilla.tilted_loss import make_tilt_sampler, make_tilted_purity, make_mix_partners
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.models import count_params
from src_renyi.mps_ancilla.ladder import init_params, save_params, load_params, make_evaluator, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True); ap.add_argument("--T", type=float, required=True)
ap.add_argument("--h_x", type=float, default=-1.5); ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--chi", type=int, required=True); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--heads", type=int, default=4); ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--init", default="scratch")
ap.add_argument("--M", type=int, default=1024); ap.add_argument("--Mp", type=int, default=256)
ap.add_argument("--K", type=int, default=4); ap.add_argument("--kappa", type=float, default=0.1)
ap.add_argument("--kappa_every", type=int, default=200); ap.add_argument("--kappa_fixed", action="store_true")
ap.add_argument("--kappas", type=float, nargs="*", default=None,
                help="MEZCLA DEFENSIVA: lista de kappas (companero k usa kappa_{k mod J}); desactiva la adaptacion")
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--polish_steps", type=int, default=0)
ap.add_argument("--polish_M", type=int, default=None); ap.add_argument("--polish_Mp", type=int, default=None)
ap.add_argument("--polish_lr", type=float, default=1e-3); ap.add_argument("--clip", type=float, default=10.0)
ap.add_argument("--ema", type=float, default=0.995)
ap.add_argument("--pden", choices=["batch", "ema"], default="batch"); ap.add_argument("--pden_decay", type=float, default=0.99)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--mem_gb", type=float, default=1.5)
ap.add_argument("--eval_M", type=int, default=4096); ap.add_argument("--eval_batches", type=int, default=32)
ap.add_argument("--eval_pb", type=int, default=8)
ap.add_argument("--label", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--freeze", nargs="*", default=None)
a = ap.parse_args()
assert "/data/0" not in os.path.abspath(a.out_dir), "no escribir en data/0"
os.makedirs(a.out_dir, exist_ok=True)
N, T, tag = a.N, a.T, f"[{a.label}] "
model = build_alt_model(N, a.chi, a.arch, a.d, a.heads, a.blocks)
if a.init == "scratch":
    params = init_params(model, N, a.seed)
else:
    path, chi0 = a.init.rsplit(":", 1); assert int(chi0) == a.chi
    params = load_params(path, model, N)
print(f"{tag}N={N} T={T} chi={a.chi} init={a.init} M={a.M} Mp={a.Mp} K={a.K} kappa0={a.kappa} pden={a.pden}  {jax.devices()}", flush=True)

tilt = make_tilt_sampler(model, N)
purity_loss = make_tilted_purity(model, N, a.chi, T)
mix = make_mix_partners(model, N, a.kappas, a.K) if a.kappas else None
if mix is not None:
    a.kappa_fixed = True
    print(f"{tag}mezcla defensiva kappas={a.kappas} (K={a.K})", flush=True)


@jax.jit
def pilot_cv(params, s, key, logk):
    """CV de Z_hat(s) con companeros t ~ q(.|s; kappa) (para elegir kappa)."""
    K = a.K
    t, lq = tilt(params, jnp.repeat(s, K, axis=0), key, logk)
    _, (P, ess, d, _f3) = purity_loss(params, s, t, lq, K)
    from src_renyi.mps_ancilla.tilted_loss import make_pair_logs
    _, lpt, logf = make_pair_logs(model, a.chi)(params, s, t, K)
    Zs = jnp.exp(lpt + logf - lq).reshape(-1, K).mean(1)
    return Zs.std() / Zs.mean(), ess, d


def run_phase(name, params, ema, t_tot, Mv, Mp, n, sched, key, pst, logk, hist):
    nch = auto_chunks(Mv, N, a.chi, 2, None, a.mem_gb, grad=True)
    sampler = make_sampler(model, N, Mv)
    fE = make_mc_terms(model, N, T, a.h_x, a.J_ZZ, None, 2, "iid")
    sch = (nch, Mv // nch, N)
    tx = optax.chain(optax.clip_by_global_norm(a.clip), optax.adam(sched))
    if a.freeze:
        lab = {k: ("frozen" if any(k.startswith(f) for f in a.freeze) else "train") for k in params}
        tx = optax.multi_transform({"train": tx, "frozen": optax.set_to_zero()}, lab)
    opt = tx.init(params)
    dec, K = a.pden_decay, a.K
    dummy = (jnp.zeros(()), jnp.zeros(()), jnp.ones(()))

    @jax.jit
    def step(params, opt, ema, key, pst, logk):
        key, k1, k2 = jax.random.split(key, 3)
        s = sampler(params, k1)
        # energia (como mc_loss, por trozos): pasada 1 Ebar, pasada 2 gradiente
        def fwd(acc, sc):
            _, _, st = fE(params, sc, dummy, Mv)
            return acc + st[0], None
        Ebar, _ = jax.lax.scan(fwd, jnp.zeros(()), s.reshape(sch))
        def bodyE(g, sc):
            gr = jax.grad(lambda p: fE(p, sc, (Ebar, jnp.zeros(()), jnp.ones(())), Mv)[0])(params)
            return jax.tree_util.tree_map(jnp.add, g, gr), None
        gE, _ = jax.lax.scan(bodyE, jax.tree_util.tree_map(jnp.zeros_like, params), s.reshape(sch))
        # pureza inclinada
        sp = s[:Mp]
        if mix is not None:
            t, lq = mix(params, sp, k2)
        else:
            t, lq = tilt(jax.lax.stop_gradient(params), jnp.repeat(sp, K, axis=0), k2, logk)
        if a.pden == "ema":
            wgt = 1.0 - dec ** pst[1]
            den = pst[0] / wgt
            (LP, (P_hat, ess, d, f3)), gP = jax.value_and_grad(purity_loss, has_aux=True)(params, sp, t, lq, K, den, den)
            pst = (dec * pst[0] + (1 - dec) * P_hat, pst[1] + 1)
        else:
            (LP, (P_hat, ess, d, f3)), gP = jax.value_and_grad(purity_loss, has_aux=True)(params, sp, t, lq, K)
        g = jax.tree_util.tree_map(jnp.add, gE, gP)
        upd, opt = tx.update(g, opt, params); params = optax.apply_updates(params, upd)
        ema = jax.tree_util.tree_map(lambda e, p: a.ema * e + (1 - a.ema) * p, ema, params)
        S2 = -jnp.log(P_hat)
        return params, opt, ema, key, pst, (Ebar, S2, Ebar - T * S2, ess, d, f3)

    if a.pden == "ema" and int(pst[1]) == 0:      # sembrar la EMA con un lote
        key, k0, k1 = jax.random.split(key, 3)
        s0 = make_sampler(model, N, Mp)(params, k0)
        t0_, lq0 = (jax.jit(mix)(params, s0, k1) if mix is not None else
                    jax.jit(tilt)(params, jnp.repeat(s0, K, axis=0), k1, logk))
        _, (P0, _, _, _) = jax.jit(purity_loss, static_argnums=4)(params, s0, t0_, lq0, K)
        pst = ((1 - dec) * P0, jnp.ones((), jnp.int32))
    pil_sampler = make_sampler(model, N, min(Mp, 256))
    for it in range(n):
        if not a.kappa_fixed and it % a.kappa_every == 0:
            key, kp, kt = jax.random.split(key, 3)
            sp = pil_sampler(params, kp); best = None
            for fct in (0.6, 0.8, 1.0, 1.25, 1.6):
                lk = float(np.clip(logk + np.log(fct), np.log(0.01), 0.0))
                cv, _, _ = pilot_cv(params, sp, kt, lk)
                if best is None or float(cv) < best[0]:
                    best = (float(cv), lk)
            logk = best[1]
        params, opt, ema, key, pst, (E, S2, F, ess, d, f3) = step(params, opt, ema, key, pst, logk)
        t_tot += 1; hist.append((float(E), float(S2), float(F), float(np.exp(logk)), float(ess), float(d), float(f3)))
        if it % max(1, n // 20) == 0:
            print(f"{tag}[{name}] Step {it:5d} | MC: F={float(F):.5f} E={float(E):.5f} S2={float(S2):.4f} | "
                  f"kappa={'mix' if mix is not None else f'{np.exp(logk):.3f}'} ESS/K={float(ess):.2f} <d>={float(d):.2f} P(d>=3)={float(f3):.2f}", flush=True)
    return params, ema, t_tot, key, pst, logk


t0 = time.time()
key = jax.random.PRNGKey(a.seed)
ema = jax.tree_util.tree_map(jnp.zeros_like, params)
pst = (jnp.zeros(()), jnp.zeros((), jnp.int32)); logk = float(np.log(a.kappa)); hist = []
sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, min(a.warmup, max(a.steps // 5, 1)), a.steps, a.lr * 1e-2)
params, ema, t_tot, key, pst, logk = run_phase("main", params, ema, 0, a.M, a.Mp, a.steps, sched, key, pst, logk, hist)
if a.polish_steps > 0:
    params, ema, t_tot, key, pst, logk = run_phase("polish", params, ema, t_tot, a.polish_M or a.M, a.polish_Mp or a.Mp,
                                                   a.polish_steps, optax.constant_schedule(a.polish_lr), key, pst, logk, hist)
s_per_step = (time.time() - t0) / max(t_tot, 1)
p_final = jax.tree_util.tree_map(lambda e: e / (1.0 - a.ema ** t_tot), ema)
pfile = os.path.join(a.out_dir, f"params_{a.label}.msgpack")
save_params(pfile, p_final)
np.savez(os.path.join(a.out_dir, f"hist_{a.label}.npz"), hist=np.array(hist))   # E, S2, F, kappa, ESS/K, <d>, fraccion de P en d>=3

# evaluacion: nbr de siempre + inclinado (64 lotes de 256 x 16)
nch = auto_chunks(a.eval_M, N, a.chi, a.eval_pb, None, a.mem_gb, grad=False)
one = make_evaluator(model, N, T, a.h_x, a.J_ZZ, a.eval_M, a.eval_pb, purity="nbr", n_chunks=nch)
ev_ = evaluate(one, p_final, T, a.eval_batches, seed=a.seed + 999)
smp = make_sampler(model, N, 256); Ps = []; kk = jax.random.PRNGKey(a.seed + 5)
tl = jax.jit(tilt); pl = jax.jit(purity_loss, static_argnums=4)
mix16 = jax.jit(make_mix_partners(model, N, a.kappas, 16)) if a.kappas else None
for _ in range(64):
    kk, k1, k2 = jax.random.split(kk, 3)
    s = smp(p_final, k1)
    t, lq = mix16(p_final, s, k2) if mix16 is not None else tl(p_final, jnp.repeat(s, 16, axis=0), k2, logk)
    _, (P, _, _, _) = pl(p_final, s, t, lq, 16); Ps.append(float(P))
Ps = np.array(Ps); S2t = float(-np.log(Ps.mean())); S2t_err = float(Ps.std(ddof=1) / Ps.mean() / 8.0)
entry = dict(label=a.label, script="gap_mc_tilted", arch=a.arch, n_params=count_params(p_final), N=N, T=T, chi=a.chi,
             init=a.init, M=a.M, Mp=a.Mp, K=a.K, kappa_final=float(np.exp(logk)), kappas=a.kappas, pden=a.pden, steps=a.steps, lr=a.lr,
             params=pfile, s_per_step=s_per_step, time_s=time.time() - t0, S2_tilted=S2t, S2_tilted_err=S2t_err,
             F_tilted=float(ev_["energy"]) - T * S2t,
             **{k: (float(v) if v is not None else None) for k, v in ev_.items()})
msg = ""
if N <= 10:
    from scipy.optimize import brentq
    idx = np.arange(2 ** N); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
    H = np.zeros((2 ** N, 2 ** N)); H[idx, idx] = a.J_ZZ * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
    for i in range(N):
        H[idx, idx ^ (1 << i)] += a.h_x
    ev = np.linalg.eigvalsh(H)
    c = brentq(lambda c: c - (np.maximum(c - ev, 0) * ev).sum() / np.maximum(c - ev, 0).sum() - 2 * T,
               ev.min() + 1e-9, ev.max() + 20 * T + 10, xtol=1e-12)
    w = np.maximum(c - ev, 0); p = w / w.sum(); Fm = (p * ev).sum() + T * np.log((p ** 2).sum())
    loss, _ = make_loss_fns(model, all_s, H, T, chunk=64)
    Fx, (Ex, Sx) = jax.jit(loss)(p_final)
    entry.update(F_exact_state=float(Fx), S2_exact_state=float(Sx), F_mre=float(Fm), gap_F=float(Fx) - float(Fm))
    msg = f" | EXACTO: F={float(Fx):.5f} S2={float(Sx):.4f} gap con MRE={float(Fx) - Fm:+.4f}"
res_file = os.path.join(a.out_dir, "results.json")
allr = json.load(open(res_file)) if os.path.exists(res_file) else []
allr.append(entry); json.dump(allr, open(res_file, "w"), indent=1)
print(f"{tag}RESULTADO MC: F={ev_['free_energy']:.5f}+-{ev_['free_energy_err']:.5f} E={ev_['energy']:.5f} "
      f"S2(nbr)={ev_['entropy']:.4f}+-{ev_['entropy_err']:.3f} S2(incl)={S2t:.4f}+-{S2t_err:.4f} kappa={np.exp(logk):.3f}"
      f"{msg} | {1e3 * s_per_step:.0f} ms/paso -> {pfile}", flush=True)
