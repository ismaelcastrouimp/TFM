"""
distill_pool.py
===============
Destilacion SUPERVISADA de la diagonal p(s) = <s|rho_MRE|s> del TFIM cuantico en el ARViT (sistema del modelo vit:<anc>):
    max_theta E_{s ~ pool}[ log p_theta(s) ]   (= min KL(p_MRE_diag || p_theta) + cte)
con un pool de muestras EXACTAS (scripts/ff_diag_sampler.py, fermiones libres).  La ancilla no interviene (solo se
entrenan los parametros del sistema; los anc_*/mps_copy se congelan y se dejan como en --init).
--init: modelo de partida (p.ej. un X5 chi8 entrenado, o una copia).  --chi/--arch: los de ese modelo.
Al final: NLL en un pool reservado (--heldout) y observables szz_r del modelo (muestreo autorregresivo).
Uso: python scripts/distill_pool.py --pool data/gap_arch/qdistill/mre_diag_N25_T3_s{1,2,3,4}.npz \
        --heldout data/gap_arch/qdistill/mre_diag_N25_T3_s5.npz --init data/gap_arch/params_X5_conv_chi8_from_N16.msgpack \
        --chi 8 --steps 10000 --out data/gap_arch/qdistill/params_QD_X5.msgpack
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import flax.serialization as serialization
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import load_params
from src_renyi.mps_ancilla.training import make_sampler

ap = argparse.ArgumentParser()
ap.add_argument("--pool", nargs="+", required=True); ap.add_argument("--heldout", required=True)
ap.add_argument("--init", required=True); ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--M", type=int, default=1024)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--ema", type=float, default=0.995)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", required=True)
a = ap.parse_args()
assert "/data/0/" not in os.path.abspath(a.out)
pool = np.concatenate([np.load(f)["s"] for f in a.pool]).astype(np.float64)
held = np.load(a.heldout)["s"].astype(np.float64)
N = pool.shape[1]
model = build_alt_model(N, a.chi, a.arch, a.d, 4, a.blocks); params = load_params(a.init, model, N)
lab = {k: ("frozen" if (k.startswith("anc_") or k.startswith("loc_") or k.startswith("res_") or k.startswith("del_")
                        or k == "mps_copy") else "train") for k in params}
print(f"N={N} pool={len(pool)} heldout={len(held)}  entrenables: {sorted(k for k, v in lab.items() if v == 'train')}", flush=True)
tx = optax.multi_transform({"train": optax.chain(optax.clip_by_global_norm(1.0),
                                                  optax.adam(optax.warmup_cosine_decay_schedule(0.0, a.lr, 100, a.steps, 0.05 * a.lr))),
                            "frozen": optax.set_to_zero()}, lab)
opt = tx.init(params)
logp_s = jax.jit(lambda p, s: 2 * jnp.real(model.apply({"params": p}, s)[0]))   # solo el sistema hace falta


def nll_of(p, S, bs=4096):
    return float(-np.mean(np.concatenate([np.asarray(logp_s(p, jnp.asarray(S[i:i + bs]))) for i in range(0, len(S), bs)])))


@jax.jit
def step(p, opt, ema, s):
    nll, gr = jax.value_and_grad(lambda p: -jnp.mean(2 * jnp.real(model.apply({"params": p}, s)[0])))(p)
    upd, opt = tx.update(gr, opt, p); p = optax.apply_updates(p, upd)
    ema = jax.tree_util.tree_map(lambda e, x: a.ema * e + (1 - a.ema) * x, ema, p)
    return p, opt, ema, nll


rng = np.random.default_rng(a.seed)
nll0 = nll_of(params, held)
print(f"NLL heldout inicial (modelo --init): {nll0:.4f}", flush=True)
ema = jax.tree_util.tree_map(jnp.zeros_like, params); t0 = time.time(); hist = []
for it in range(a.steps):
    s = jnp.asarray(pool[rng.integers(0, len(pool), a.M)])
    params, opt, ema, nll = step(params, opt, ema, s); hist.append(float(nll))
    if it % max(1, a.steps // 25) == 0 or it == a.steps - 1:
        print(f"paso {it:5d}  NLL(lote)={float(nll):.4f}  ({time.time() - t0:.0f} s)", flush=True)
ema_hat = jax.tree_util.tree_map(lambda e: e / (1 - a.ema ** a.steps), ema)
# la EMA tambien promedia los congelados (constantes) -> identicos; se restauran por seguridad
ema_hat = {k: (params[k] if lab[k] == "frozen" else ema_hat[k]) for k in params}
nll1 = nll_of(ema_hat, held)
# observables del modelo destilado por muestreo autorregresivo
sampler = make_sampler(model, N, 1024); key = jax.random.PRNGKey(5); S = []
for _ in range(16):
    key, sub = jax.random.split(key); S.append(np.asarray(sampler(ema_hat, sub)))
S = np.concatenate(S)
zz = {f"szz{r}": float(np.mean(S * np.roll(S, -r, 1))) for r in (1, 2, 3)}
zz_ex = {f"szz{r}": float(np.mean(held * np.roll(held, -r, 1))) for r in (1, 2, 3)}
print(f"FINAL: NLL heldout {nll0:.4f} -> {nll1:.4f}   modelo {zz}   exacto(heldout) {zz_ex}", flush=True)
with open(a.out, "wb") as f:
    f.write(serialization.to_bytes(ema_hat))
json.dump(dict(init=a.init, chi=a.chi, arch=a.arch, steps=a.steps, M=a.M, lr=a.lr, nll_init=nll0, nll_final=nll1,
               zz_model=zz, zz_exact=zz_ex, nll_hist=hist[::10]), open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
print("guardado", a.out)
