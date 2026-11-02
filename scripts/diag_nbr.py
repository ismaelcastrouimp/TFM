"""
diag_nbr.py
===========
Diagnostico del estimador de la pureza SIN ED (sirve para N grande) sobre parametros ya entrenados.
Para cada (M, pair_block) y cada estimador (iid / nbr) toma varios lotes independientes y muestra:

  S2        = -log(media de P)           (insesgado entre lotes)
  std S2    = desviacion tipica de S2 por lote   (el ruido que ve un paso de entrenamiento)
  near %    = P_near / P   (nbr: parte de P calculada EXACTA: t=s y vecinos a un espin)
  CV near / far = std/media por lote de P_near y de P_far (cual de las dos domina el ruido)

Lectura: si 'near %' es baja y 'CV far' es grande, la cola iid (d>=2) domina y nbr (d<=1) no basta:
hace falta M mayor, mas pair_block o ampliar nbr a d<=2.

Uso:
  python scripts/diag_nbr.py --N 25 --chi 4 --T 3.0 \
      --params data/mps_N25_test/params/params_0001_chi4.msgpack --mem_gb 1.5
"""
import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from src_renyi.mps_ancilla.training import make_sampler
from src_renyi.mps_ancilla.mc_loss import make_mc_stats, auto_chunks
from src_renyi.mps_ancilla.ladder import build_model, load_params

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--T", type=float, required=True)
ap.add_argument("--params", type=str, required=True)
ap.add_argument("--h_x", type=float, default=-1.5)
ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--Ms", type=int, nargs="+", default=[256, 1024, 4096])
ap.add_argument("--blocks", type=int, nargs="+", default=[8, 32])
ap.add_argument("--batches", type=int, default=16)
ap.add_argument("--mem_gb", type=float, default=None)
ap.add_argument("--sample_chunk", type=int, default=None)
ap.add_argument("--embedding", type=int, default=16)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--blocks_vit", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

N = args.N
model = build_model(N, args.chi, args.embedding, args.heads, args.blocks_vit)
params = load_params(args.params, model, N)
print(f"N={N} chi={args.chi} T={args.T}  params={args.params}  dispositivos={jax.devices()}\n", flush=True)
print(f"{'M':>6s} {'b':>3s} {'estim':>5s} {'trozos':>6s} | {'S2':>8s} {'std S2':>8s} | {'near %':>7s} {'CV near':>8s} {'CV far':>8s}",
      flush=True)

for M in args.Ms:
    for b in args.blocks:
        if M % b:
            continue
        nch = auto_chunks(M, N, args.chi, b, args.sample_chunk, args.mem_gb, grad=False)
        sampler = make_sampler(model, N, M)
        keys = jax.random.split(jax.random.PRNGKey(args.seed), args.batches)
        for pur in ("iid", "nbr"):
            fn = jax.jit(make_mc_stats(model, N, args.T, args.h_x, args.J_ZZ, None, b, pur, nch, diag=True))
            rows = []
            for k in keys:
                s = sampler(params, k)
                rows.append([float(x) for x in fn(params, s)])
            _, Ss, Pn, Pf = np.array(rows).T
            S2 = -np.log(np.mean(np.exp(-Ss)))
            line = f"{M:6d} {b:3d} {pur:>5s} {nch:6d} | {S2:8.3f} {Ss.std(ddof=1):8.3f} | "
            if pur == "nbr":
                line += f"{100 * Pn.mean() / (Pn.mean() + Pf.mean()):6.1f}% {Pn.std(ddof=1) / Pn.mean():8.2f} {Pf.std(ddof=1) / max(Pf.mean(), 1e-300):8.2f}"
            else:
                line += f"{'-':>7s} {'-':>8s} {'-':>8s}"
            print(line, flush=True)