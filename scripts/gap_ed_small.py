"""
gap_ed_small.py
===============
Diagnostico del hueco de F a T alta: entrena ARViT + MPS ancilla con gradiente EXACTO (ED, sin ruido MC)
a N pequeno y compara con el MRE exacto.  Mismo modelo que la escalera (ladder.build_model, copy_init=True).

Si con gradiente exacto queda un hueco por sitio parecido al de N=25 (~0.07/sitio a T=3), el problema es
expresividad/optimizacion del ansatz y no el estimador MC.

Uso:
  python scripts/gap_ed_small.py --N 10 --T 3.0 --variants 16:4 16:8 32:4 --steps 3000 --lr 5e-3 \
      --out data/gap_ed/ed_N10_T3.json
  variante = d:chi[:blocks]   (heads = 4)
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import brentq

from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import build_model, init_params
from src_renyi.mps_ancilla import training
from src_renyi.mps_ancilla.models import count_params

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=10)
ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--h_x", type=float, default=-1.5)
ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--variants", nargs="+", default=["16:4", "16:8", "32:4"])
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--seeds", type=int, nargs="+", default=[1])
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--arch", default=None,
                help="variante de models_alt (p.ej. vit:mlp, vit:conv:r=2, made:ctx); por defecto el ARViT de siempre")
ap.add_argument("--out", required=True)
args = ap.parse_args()
N, T = args.N, args.T

# ── H en la base de all_s (s_i = +-1, bit i de un entero; +1 <-> bit 1) ───────
idx = np.arange(2 ** N)
bits = (idx[:, None] >> np.arange(N)[None, :]) & 1
all_s = (2 * bits - 1).astype(np.float64)
H = np.zeros((2 ** N, 2 ** N))
zz = sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
H[idx, idx] = args.J_ZZ * zz
for i in range(N):
    H[idx, idx ^ (1 << i)] += args.h_x
ev, V = np.linalg.eigh(H)

def mre(T):
    def sc(c):
        w = np.maximum(c - ev, 0); return c - (w * ev).sum() / w.sum() - 2 * T
    c = brentq(sc, ev.min() + 1e-9, ev.max() + 20 * T + 10, xtol=1e-12)
    w = np.maximum(c - ev, 0); p = w / w.sum()
    return (p * ev).sum(), -np.log((p ** 2).sum())

E_EX, S2_EX = mre(T)
F_EX = E_EX - T * S2_EX
print(f"N={N} T={T}  MRE exacto: E={E_EX:.5f} S2={S2_EX:.5f} F={F_EX:.5f}  dispositivos={jax.devices()}", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
results = json.load(open(args.out)) if os.path.exists(args.out) else []
done = {(r.get("arch"), r["d"], r["chi"], r["blocks"], r["seed"], r["steps"], r["lr"]) for r in results}

for v in args.variants:
    parts = [int(x) for x in v.split(":")]
    d, chi = parts[0], parts[1]
    blocks = parts[2] if len(parts) > 2 else 2
    for seed in args.seeds:
        key = (args.arch, d, chi, blocks, seed, args.steps, args.lr)
        if key in done:
            print(f"[saltado] {key}"); continue
        model = build_alt_model(N, chi, args.arch, d, 4, blocks) if args.arch else build_model(N, chi, d, 4, blocks)
        params = init_params(model, N, seed)
        loss, _ = training.make_loss_fns(model, all_s, H, T, chunk=args.chunk)
        tag = f"[{args.arch or 'vit:ctx(orig)'} d={d} chi={chi} blk={blocks} s={seed}] "
        out = training.train(loss, params, args.steps, args.lr, T, freq=max(1, args.steps // 10), tag=tag)
        r = dict(arch=args.arch, N=N, T=T, d=d, chi=chi, blocks=blocks, seed=seed, steps=args.steps, lr=args.lr,
                 n_params=count_params(params), F=out["F"], E=out["E"], S2=out["S2"],
                 F_ex=F_EX, E_ex=E_EX, S2_ex=S2_EX,
                 gap_F=out["F"] - F_EX, gap_E=out["E"] - E_EX, gap_S2=out["S2"] - S2_EX,
                 gap_F_per_site=(out["F"] - F_EX) / N, s_per_step=out["s_per_step"])
        print(f"{tag}FINAL F={r['F']:.5f}  dF={r['gap_F']:+.4f} ({r['gap_F_per_site']:+.4f}/sitio)  "
              f"dE={r['gap_E']:+.4f}  dS2={r['gap_S2']:+.4f}  #par={r['n_params']}  "
              f"{1e3*r['s_per_step']:.1f} ms/paso", flush=True)
        results.append(r)
        json.dump(results, open(args.out, "w"), indent=1)
