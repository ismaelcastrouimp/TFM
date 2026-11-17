"""
eval_precise_mps.py
===================
Evaluacion PRECISA (solo forward, sin ED) de varios conjuntos de parametros ya entrenados, con el estimador nbr y
muchas muestras, para poder comparar runs cuya diferencia es menor que la barra de error de la evaluacion normal.

Para cada run:  E, S2, F = E - T*S2  con barra de error, el % de P cubierto exactamente (d<=1) y los CV near/far.
S2 = -log(media de P entre lotes) (insesgado), el error de F es std de F por lote / sqrt(n_lotes).

Cada run se pasa como  etiqueta:fichero.msgpack:chi   (chi = bond dimension con la que se guardo).

Uso:
  python scripts/eval_precise_mps.py --N 25 --T 3.0 --ref_F -62.47691 --M 4096 --batches 32 --mem_gb 1.5 \
     --run viejo_chi2:data/0/mps_N25/params/params_0030_chi2.msgpack:2 \
     --run nuevo_chi2:data/test_swap_N25/params/params_0030_chi2.msgpack:2 \
     --run viejo_chi4:data/0/mps_N25/params/params_0030_chi4.msgpack:4 \
     --run nuevo_chi4:data/test_swap_N25/params/params_0030_chi4.msgpack:4
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from src_renyi.mps_ancilla.ladder import build_model, load_params, make_evaluator, evaluate
from src_renyi.mps_ancilla.mc_loss import auto_chunks

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--T", type=float, required=True)
ap.add_argument("--run", action="append", required=True, help="etiqueta:fichero:chi (repetible)")
ap.add_argument("--h_x", type=float, default=-1.5)
ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--M", type=int, default=4096, help="muestras por lote")
ap.add_argument("--batches", type=int, default=32)
ap.add_argument("--pair_block", type=int, default=32)
ap.add_argument("--mem_gb", type=float, default=None)
ap.add_argument("--sample_chunk", type=int, default=None)
ap.add_argument("--embedding", type=int, default=16)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--blocks_vit", type=int, default=2)
ap.add_argument("--purity", choices=["iid", "nbr"], default="nbr")
ap.add_argument("--ref_F", type=float, default=None, help="F exacta (opcional) para mostrar el hueco")
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--out", type=str, default=None, help="JSON de salida (opcional)")
args = ap.parse_args()

N, T = args.N, args.T
print(f"N={N} T={T} M={args.M} x {args.batches} lotes = {args.M * args.batches:,} muestras/run  purity={args.purity}  "
      f"dispositivos={jax.devices()}\n", flush=True)
results = []
for spec in args.run:
    label, path, chi = spec.split(":")[0], spec.split(":")[1], int(spec.split(":")[2])
    model = build_model(N, chi, args.embedding, args.heads, args.blocks_vit)
    params = load_params(path, model, N)
    nch = auto_chunks(args.M, N, chi, args.pair_block, args.sample_chunk, args.mem_gb, grad=False)
    one = make_evaluator(model, N, T, args.h_x, args.J_ZZ, args.M, args.pair_block, purity=args.purity, n_chunks=nch)
    t0 = time.time()
    ev = evaluate(one, params, T, args.batches, seed=args.seed)
    ev.update(label=label, chi=chi, path=path, time_s=time.time() - t0)
    results.append(ev)
    gap = f"  gap={ev['free_energy'] - args.ref_F:+.3f}" if args.ref_F is not None else ""
    near = f"  near={100 * ev['near_frac']:.0f}% CVfar={ev['far_cv']:.2f}" if ev["near_frac"] is not None else ""
    print(f"{label:>14s} chi={chi:<2d} F={ev['free_energy']:9.4f}±{ev['free_energy_err']:.3f}  E={ev['energy']:8.3f}±{ev['energy_err']:.3f}  "
          f"S2={ev['entropy']:7.3f}±{ev['entropy_err']:.3f}{gap}{near}  [{ev['time_s']:.0f}s]", flush=True)

if args.out:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nJSON: {args.out}")