"""
gap_mc.py
=========
Lanzador MC flexible para las pruebas del hueco de F (notas_gap.md).  Usa train_mc / evaluate tal cual.

  * --init scratch               -> arranca de cero (copy_init), como el escalon 0 de la escalera
  * --init fichero.msgpack:chi0  -> carga parametros (EMA) guardados con chi0 y, si --chi > chi0, los amplia (pad_chi)
  * a N <= 10 calcula ademas F, E, S2 EXACTOS (ED) del estado final -> separa error de estimacion de error del ansatz
  * guarda params en --out_dir (nunca escribe fuera de el) y anade una entrada a --out_dir/results.json

Ejemplos:
  # continuar chi=8 de data/0 con mas pasos y M mayor
  python scripts/gap_mc.py --N 25 --T 3.0 --chi 8 --init data/0/mps_N25/params/params_0030_chi8.msgpack:8 \
      --M 2048 --steps 3000 --lr 1e-3 --warmup 30 --polish_steps 0 --label cont_chi8 --out_dir data/gap_mc
  # N=8 con los hiperparametros de N=25, comparando con ED
  python scripts/gap_mc.py --N 8 --T 3.0 --chi 4 --init scratch --M 2048 --steps 4000 --lr 5e-3 --label N8 --out_dir data/gap_mc
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from src_renyi.mps_ancilla.training import train_mc, make_loss_fns
from src_renyi.mps_ancilla.mc_loss import auto_chunks
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.models import count_params
from src_renyi.mps_ancilla.ladder import (build_model, init_params, pad_chi, save_params, load_params,
                                          make_evaluator, evaluate)

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--T", type=float, required=True)
ap.add_argument("--h_x", type=float, default=-1.5)
ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--d", type=int, default=16)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--init", default="scratch")
ap.add_argument("--noise", type=float, default=0.02)
ap.add_argument("--M", type=int, default=1024)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--warmup", type=int, default=30)
ap.add_argument("--polish_steps", type=int, default=0)
ap.add_argument("--polish_M", type=int, default=None)
ap.add_argument("--polish_lr", type=float, default=5e-4)
ap.add_argument("--pair_block", type=int, default=8)
ap.add_argument("--ema", type=float, default=0.995)
ap.add_argument("--purity", default="nbr")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--mem_gb", type=float, default=1.5)
ap.add_argument("--eval_M", type=int, default=4096)
ap.add_argument("--eval_batches", type=int, default=16)
ap.add_argument("--label", required=True)
ap.add_argument("--arch", default=None,
                help="variante de models_alt (p.ej. vit:mlp, vit:conv:r=2, made:ctx); por defecto el ARViT de siempre")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--freeze", nargs="*", default=None, help="prefijos de parametros de primer nivel a congelar (p.ej. anc_ mps_copy res_ loc_ del_)")
args = ap.parse_args()
N, T = args.N, args.T
assert not os.path.abspath(args.out_dir).startswith(os.path.abspath(os.path.join(script_dir, "..", "data", "0"))), \
    "no escribir en data/0"
os.makedirs(args.out_dir, exist_ok=True)
tag = f"[{args.label}] "
print(f"{tag}N={N} T={T} chi={args.chi} d={args.d} blocks={args.blocks} init={args.init} M={args.M} "
      f"steps={args.steps} lr={args.lr} purity={args.purity}  dispositivos={jax.devices()}", flush=True)

def mk(chi):
    return (build_alt_model(N, chi, args.arch, args.d, args.heads, args.blocks) if args.arch
            else build_model(N, chi, args.d, args.heads, args.blocks))
model = mk(args.chi)
if args.init == "scratch":
    params = init_params(model, N, args.seed)
else:
    path, chi0 = args.init.rsplit(":", 1)
    chi0 = int(chi0)
    assert args.arch is None or chi0 == args.chi, "pad_chi solo vale para el ARViT de siempre"
    p0 = load_params(path, mk(chi0), N)
    params = p0 if chi0 == args.chi else pad_chi(p0, chi0, args.chi, N, args.noise / np.sqrt(args.d), args.seed)

# ---- ED opcional (N <= 10) ----
exact_fn, ref = None, None
if N <= 10:
    from scipy.optimize import brentq
    idx = np.arange(2 ** N); bits = (idx[:, None] >> np.arange(N)[None, :]) & 1
    all_s = (2 * bits - 1).astype(np.float64)
    H = np.zeros((2 ** N, 2 ** N))
    H[idx, idx] = args.J_ZZ * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
    for i in range(N):
        H[idx, idx ^ (1 << i)] += args.h_x
    ev = np.linalg.eigvalsh(H)
    def sc(c):
        w = np.maximum(c - ev, 0); return c - (w * ev).sum() / w.sum() - 2 * T
    c = brentq(sc, ev.min() + 1e-9, ev.max() + 20 * T + 10, xtol=1e-12)
    w = np.maximum(c - ev, 0); p = w / w.sum()
    E_ex, S2_ex = (p * ev).sum(), -np.log((p ** 2).sum())
    ref = (E_ex - T * S2_ex, E_ex, S2_ex)
    loss, _ = make_loss_fns(model, all_s, H, T, chunk=64)
    loss = jax.jit(loss)
    def exact_fn(p):
        F, (E, S2) = loss(p); return float(F), float(E), float(S2)

t0 = time.time()
out = train_mc(model, params, N, T, args.h_x, args.J_ZZ, M=args.M, n_steps=args.steps, lr=args.lr,
               warmup=args.warmup, seed=args.seed, pair_block=args.pair_block, freq=max(1, args.steps // 20),
               exact_fn=exact_fn, ref=ref, ema_decay=args.ema, polish_steps=args.polish_steps,
               polish_M=args.polish_M or args.M, polish_lr=args.polish_lr, tag=tag, purity=args.purity,
               mem_gb=args.mem_gb, freeze=args.freeze)
p_final = out["params_ema"]
pfile = os.path.join(args.out_dir, f"params_{args.label}.msgpack")
save_params(pfile, p_final)
np.savez(os.path.join(args.out_dir, f"hist_{args.label}.npz"), hist=out["hist"])

nch = auto_chunks(args.eval_M, N, args.chi, args.pair_block, None, args.mem_gb, grad=False)
one = make_evaluator(model, N, T, args.h_x, args.J_ZZ, args.eval_M, args.pair_block,
                     purity="nbr" if args.purity == "swap" else args.purity, n_chunks=nch)
ev_ = evaluate(one, p_final, T, args.eval_batches, seed=args.seed + 999)
entry = dict(label=args.label, arch=args.arch, n_params=count_params(p_final), N=N, T=T, chi=args.chi, d=args.d, blocks=args.blocks, init=args.init,
             M=args.M, steps=args.steps, lr=args.lr, purity=args.purity, params=pfile,
             s_per_step=out["s_per_step"], time_s=time.time() - t0,
             **{k: (float(v) if v is not None else None) for k, v in ev_.items()})
if exact_fn is not None:
    Fx, Ex, Sx = exact_fn(p_final)
    entry.update(F_exact_state=Fx, E_exact_state=Ex, S2_exact_state=Sx,
                 F_mre=ref[0], E_mre=ref[1], S2_mre=ref[2], gap_F=Fx - ref[0])
res_file = os.path.join(args.out_dir, "results.json")
allr = json.load(open(res_file)) if os.path.exists(res_file) else []
allr.append(entry)
json.dump(allr, open(res_file, "w"), indent=1)
print(f"{tag}RESULTADO MC: F={ev_['free_energy']:.5f}+-{ev_['free_energy_err']:.5f} E={ev_['energy']:.5f} "
      f"S2={ev_['entropy']:.4f}+-{ev_['entropy_err']:.3f}"
      + (f" | EXACTO del estado: F={entry['F_exact_state']:.5f} E={entry['E_exact_state']:.5f} "
         f"S2={entry['S2_exact_state']:.4f}  gap con MRE={entry['gap_F']:+.4f}" if exact_fn else "")
      + f" | {1e3*out['s_per_step']:.0f} ms/paso -> {pfile}", flush=True)
