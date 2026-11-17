"""
distill_classical.py
====================
Destilacion SUPERVISADA del MRE clasico exacto (Ising en anillo, h_x=0) en el ARViT: maxima verosimilitud
    max_theta E_{s ~ p_MRE}[ log p_theta(s) ]       (= min KL(p_MRE || p_theta))
con muestras EXACTAS de p_MRE (capa k con peso g_k (c-E_k)_+, k paredes uniformes entre los N enlaces, signo global al azar).
Separa expresividad de optimizacion: si llega a dF ~ 0 el ARViT PUEDE representar el corte duro y el fallo del
entrenamiento variacional (REINFORCE, gap_arch_queue30.sh) es de optimizacion.
Mismo modelo que la cola 30: vit:conv:r=1, chi=1, ancilla copia exacta congelada (init de make_copy_init.py).
Imprime KL = E_MRE[log p_MRE - log p_theta] (exacta salvo MC) y guarda params (EMA) para scripts/classical_diag.py.
Uso: python scripts/distill_classical.py --N 25 --T 3 --init data/gap_arch/classical/init_copy_N25.msgpack \
        --steps 6000 --M 1024 --lr 1e-3 --out data/gap_arch/classical/params_D0.msgpack
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from math import comb
from scipy.optimize import brentq
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import flax.serialization as serialization
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import load_params

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=25); ap.add_argument("--d", type=int, default=16); ap.add_argument("--blocks", type=int, default=2); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--J", type=float, default=-1.0); ap.add_argument("--init", required=True)
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--M", type=int, default=1024)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--ema", type=float, default=0.995)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", required=True)
a = ap.parse_args()
assert "/data/0/" not in os.path.abspath(a.out)
N, T = a.N, a.T
ks = np.arange(0, N + 1, 2); g = np.array([2.0 * comb(N, int(k)) for k in ks]); Ek = a.J * (N - 2 * ks)
w = lambda c: g * np.maximum(c - Ek, 0)
c = brentq(lambda c: c - (w(c) * Ek).sum() / w(c).sum() - 2 * T, Ek.min() + 1e-12, Ek.max() + 100)
Pk = w(c) / w(c).sum(); Z = w(c).sum()
logp_of_k = np.where(c > Ek, np.log(np.maximum(c - Ek, 1e-300) / Z), -np.inf)
H_ex = -(Pk[Pk > 0] * logp_of_k[Pk > 0]).sum()                 # entropia de Shannon del MRE (para la KL)
rng = np.random.default_rng(a.seed)


def sample_mre(M):
    k = rng.choice(ks, size=M, p=Pk)
    walls = np.zeros((M, N), bool)
    for i in range(M):
        walls[i, rng.choice(N, size=k[i], replace=False)] = True   # pared en el enlace (j, j+1)
    # s_{j+1} = s_j * (-1)^{pared_j}; con k par la cadena cierra en PBC
    sign = np.cumprod(np.where(walls[:, :-1], -1.0, 1.0), axis=1)
    s0 = rng.choice([-1.0, 1.0], size=(M, 1))
    s = np.concatenate([s0, s0 * sign], axis=1)
    assert np.all((s != np.roll(s, -1, 1)).sum(1) == k)
    return s


model = build_alt_model(N, 1, "vit:conv:r=1", a.d, 4, a.blocks); params = load_params(a.init, model, N)
params = params.get("params", params) if isinstance(params, dict) and "params" in params else params
lab = {k: ("frozen" if (k.startswith("anc_") or k == "mps_copy") else "train") for k in params}
tx = optax.multi_transform({"train": optax.chain(optax.clip_by_global_norm(1.0),
                                                  optax.adam(optax.warmup_cosine_decay_schedule(0.0, a.lr, 100, a.steps, 0.05 * a.lr))),
                            "frozen": optax.set_to_zero()}, lab)
opt = tx.init(params)
logp = lambda p, s: 2 * jnp.real(model.apply({"params": p}, s)[0])


@jax.jit
def step(p, opt, ema, s):
    nll, gr = jax.value_and_grad(lambda p: -jnp.mean(logp(p, s)))(p)
    upd, opt = tx.update(gr, opt, p); p = optax.apply_updates(p, upd)
    ema = jax.tree_util.tree_map(lambda e, x: a.ema * e + (1 - a.ema) * x, ema, p)
    return p, opt, ema, nll


ema = jax.tree_util.tree_map(jnp.zeros_like, params)
print(f"N={N} T={T} c={c:.4f}  H_Shannon(MRE)={H_ex:.4f}  -> KL = NLL - H", flush=True)
t0 = time.time(); hist = []
for it in range(a.steps):
    params, opt, ema, nll = step(params, opt, ema, jnp.asarray(sample_mre(a.M)))
    hist.append(float(nll))
    if it % max(1, a.steps // 30) == 0 or it == a.steps - 1:
        print(f"paso {it:5d}  NLL={float(nll):.4f}  KL~{float(nll) - H_ex:+.4f}  ({time.time() - t0:.0f} s)", flush=True)
ema_hat = jax.tree_util.tree_map(lambda e: e / (1 - a.ema ** a.steps), ema)
sv = jnp.asarray(sample_mre(16384))
kl = float(-jnp.mean(jnp.concatenate([logp(ema_hat, sv[i:i + 1024]) for i in range(0, 16384, 1024)]))) - H_ex
print(f"FINAL (EMA, 16384 muestras MRE nuevas): KL(MRE||modelo) = {kl:+.5f}", flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "wb") as f:
    f.write(serialization.to_bytes(ema_hat))
json.dump(dict(N=N, T=T, steps=a.steps, M=a.M, lr=a.lr, init=a.init, KL=kl, H_MRE=H_ex, nll_hist=hist[::10]),
          open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
print("guardado", a.out)
