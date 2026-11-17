"""
gap_mc_pema.py
==============
Como gap_mc.py (entrenamiento MC de ARViT + ancilla MPS, F = E - T S2, pureza "nbr"), pero cambiando SOLO el
DENOMINADOR del gradiente de S2.  La perdida de mc_loss.py da  grad S2 = -grad P_lote / P_lote  con numerador y
denominador del MISMO lote (estimador de cociente: sesgado si P_lote tiene colas pesadas).  Aqui:

  --pden batch : P del propio lote (= gap_mc.py; control).
  --pden ema   : P = media exponencial (con correccion de sesgo) de P_near y P_far de los lotes ANTERIORES
                 (--pden_decay, p.ej. 0.99 ~ 100 pasos).  Numerador insesgado / denominador casi determinista.
  --pden indep : P de un SEGUNDO lote independiente (solo forward) en cada paso.  Insesgado en el cociente
                 E[grad P_A] / P_B (salvo el 1/P_B, que es convexo: sesgo O(var P_B / P^2) pero sin correlacion
                 con el numerador).
Las lineas base de los terminos de score (g - Pn, h - Pf) usan los mismos Pn, Pf que el denominador (una constante
como linea base no sesga).  Ebar es siempre la del lote.  Fases: main (warmup + coseno) y polish (lr constante),
EMA de parametros, evaluacion final como gap_mc.py (make_evaluator/evaluate, results.json en --out_dir).
Uso (N=25, misma receta que Q2_free pero con denominador EMA):
  python scripts/gap_mc_pema.py --N 25 --T 3 --chi 8 --arch vit:conv:r=1 --init X.msgpack:8 --M 1024 --pair_block 8 \
      --steps 3000 --lr 1e-3 --warmup 100 --polish_steps 500 --polish_M 2048 --polish_lr 1e-3 \
      --pden ema --pden_decay 0.99 --label P_ema --out_dir data/gap_arch/pden
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
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.models import count_params
from src_renyi.mps_ancilla.ladder import init_params, save_params, load_params, make_evaluator, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True); ap.add_argument("--T", type=float, required=True)
ap.add_argument("--h_x", type=float, default=-1.5); ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--chi", type=int, required=True); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--heads", type=int, default=4); ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--init", default="scratch")
ap.add_argument("--M", type=int, default=1024); ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--polish_steps", type=int, default=0); ap.add_argument("--polish_M", type=int, default=None)
ap.add_argument("--polish_lr", type=float, default=1e-3); ap.add_argument("--pair_block", type=int, default=8)
ap.add_argument("--clip", type=float, default=10.0); ap.add_argument("--ema", type=float, default=0.995)
ap.add_argument("--pden", choices=["batch", "ema", "indep"], default="ema")
ap.add_argument("--pden_decay", type=float, default=0.99)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--mem_gb", type=float, default=1.5)
ap.add_argument("--eval_M", type=int, default=4096); ap.add_argument("--eval_batches", type=int, default=32)
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
print(f"{tag}N={N} T={T} chi={a.chi} init={a.init} M={a.M} pb={a.pair_block} pden={a.pden}"
      f"{'(' + str(a.pden_decay) + ')' if a.pden == 'ema' else ''}  {jax.devices()}", flush=True)


def make_vg(Mv, pb, nch):
    """vg(params, s, den) -> (stats, grad).  den = (Pn, Pf) externos o None (= los del lote, como mc_loss)."""
    f_terms = make_mc_terms(model, N, T, a.h_x, a.J_ZZ, None, pb, "nbr")
    sch_shape = (nch, Mv // nch, N)
    dummy = (jnp.zeros(()), jnp.zeros(()), jnp.ones(()))

    def stats_of(params, s):
        def fwd(acc, sc):
            _, _, st = f_terms(params, sc, dummy, Mv)
            return tuple(x + y for x, y in zip(acc, st)), None
        st, _ = jax.lax.scan(fwd, (jnp.zeros(()),) * 3, s.reshape(sch_shape))
        return st

    def vg(params, s, den):
        st = stats_of(params, s)
        glob = (st[0], st[1], st[2]) if den is None else (st[0], den[0], den[1])

        def body(carry, sc):
            v, gr = jax.value_and_grad(lambda p: sum(f_terms(p, sc, glob, Mv)[:2]))(params)
            return (carry[0] + v, jax.tree_util.tree_map(jnp.add, carry[1], gr)), None
        (_, grad), _ = jax.lax.scan(body, (jnp.zeros(()), jax.tree_util.tree_map(jnp.zeros_like, params)),
                                    s.reshape(sch_shape))
        return st, grad
    return vg, stats_of


def run_phase(name, params, ema, t_tot, Mv, pb, n, sched, key, pstate):
    nch = auto_chunks(Mv, N, a.chi, pb, None, a.mem_gb, grad=True)
    sampler = make_sampler(model, N, Mv)
    vg, stats_of = make_vg(Mv, pb, nch)
    tx = optax.chain(optax.clip_by_global_norm(a.clip), optax.adam(sched))
    if a.freeze:
        lab = {k: ("frozen" if any(k.startswith(f) for f in a.freeze) else "train") for k in params}
        tx = optax.multi_transform({"train": tx, "frozen": optax.set_to_zero()}, lab)
    opt = tx.init(params)
    dec = a.pden_decay

    @jax.jit
    def step(params, opt, ema, key, pst, t):
        key, k1, k2 = jax.random.split(key, 3)
        s = sampler(params, k1)
        if a.pden == "batch":
            den = None
        elif a.pden == "indep":
            st2 = stats_of(params, sampler(params, k2)); den = (st2[1], st2[2])
        else:   # ema con correccion de sesgo (sembrada con un lote independiente antes del primer paso)
            w = 1.0 - dec ** pst[2]
            den = (pst[0] / w, pst[1] / w)
        st, g = vg(params, s, den)
        if a.pden == "ema":
            pst = (dec * pst[0] + (1 - dec) * st[1], dec * pst[1] + (1 - dec) * st[2], pst[2] + 1)
        upd, opt = tx.update(g, opt, params); params = optax.apply_updates(params, upd)
        ema = jax.tree_util.tree_map(lambda e, p: a.ema * e + (1 - a.ema) * p, ema, params)
        S2 = -jnp.log(st[1] + st[2])
        return params, opt, ema, key, pst, (st[0], S2, st[0] - T * S2)

    if a.pden == "ema" and int(pstate[2]) == 0:
        key, k0 = jax.random.split(key)
        st0 = jax.jit(stats_of)(params, sampler(params, k0))
        pstate = ((1 - dec) * st0[1], (1 - dec) * st0[2], jnp.ones((), dtype=jnp.int32))
    hist = []
    for it in range(n):
        params, opt, ema, key, pstate, (E, S2, F) = step(params, opt, ema, key, pstate, t_tot)
        t_tot += 1; hist.append((float(E), float(S2), float(F)))
        if it % max(1, n // 20) == 0:
            extra = ""
            if a.pden == "ema" and float(pstate[2]) > 0:
                w = 1 - dec ** float(pstate[2]); extra = f" | S2(den EMA)={-np.log((float(pstate[0]) + float(pstate[1])) / w):.4f}"
            print(f"{tag}[{name}] Step {it:5d} | MC: F={float(F):.5f} E={float(E):.5f} S2={float(S2):.4f}{extra}", flush=True)
    return params, ema, t_tot, key, pstate, hist


t0 = time.time()
key = jax.random.PRNGKey(a.seed)
ema = jax.tree_util.tree_map(jnp.zeros_like, params)
pstate = (jnp.zeros(()), jnp.zeros(()), jnp.zeros((), dtype=jnp.int32))
sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, min(a.warmup, max(a.steps // 5, 1)), a.steps, a.lr * 1e-2)
params, ema, t_tot, key, pstate, hist = run_phase("main", params, ema, 0, a.M, a.pair_block, a.steps, sched, key, pstate)
if a.polish_steps > 0:
    params, ema, t_tot, key, pstate, h2 = run_phase("polish", params, ema, t_tot, a.polish_M or a.M, a.pair_block,
                                                    a.polish_steps, optax.constant_schedule(a.polish_lr), key, pstate)
    hist += h2
s_per_step = (time.time() - t0) / max(t_tot, 1)
p_final = jax.tree_util.tree_map(lambda e: e / (1.0 - a.ema ** t_tot), ema)
pfile = os.path.join(a.out_dir, f"params_{a.label}.msgpack")
save_params(pfile, p_final)
np.savez(os.path.join(a.out_dir, f"hist_{a.label}.npz"), hist=np.array(hist))

nch = auto_chunks(a.eval_M, N, a.chi, a.pair_block, None, a.mem_gb, grad=False)
one = make_evaluator(model, N, T, a.h_x, a.J_ZZ, a.eval_M, a.pair_block, purity="nbr", n_chunks=nch)
ev_ = evaluate(one, p_final, T, a.eval_batches, seed=a.seed + 999)
entry = dict(label=a.label, script="gap_mc_pema", pden=a.pden, pden_decay=a.pden_decay, arch=a.arch,
             n_params=count_params(p_final), N=N, T=T, chi=a.chi, init=a.init, M=a.M, pair_block=a.pair_block,
             steps=a.steps, lr=a.lr, params=pfile, s_per_step=s_per_step, time_s=time.time() - t0,
             **{k: (float(v) if v is not None else None) for k, v in ev_.items()})
msg = ""
if N <= 10:   # comprobacion exacta (ED) del estado final y del MRE
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
    entry.update(F_exact_state=float(Fx), F_mre=float(Fm), gap_F=float(Fx) - float(Fm))
    msg = f" | EXACTO: F={float(Fx):.5f} gap con MRE={float(Fx) - Fm:+.4f}"
res_file = os.path.join(a.out_dir, "results.json")
allr = json.load(open(res_file)) if os.path.exists(res_file) else []
allr.append(entry); json.dump(allr, open(res_file, "w"), indent=1)
print(f"{tag}RESULTADO MC: F={ev_['free_energy']:.5f}+-{ev_['free_energy_err']:.5f} E={ev_['energy']:.5f} "
      f"S2={ev_['entropy']:.4f}+-{ev_['entropy_err']:.3f}{msg} | {1e3 * s_per_step:.0f} ms/paso -> {pfile}", flush=True)
