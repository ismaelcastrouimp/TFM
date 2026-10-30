"""
training_mps_ladder.py
======================
Minimiza F = <H> - T*S2 con ARViT + MPS ancilla, todo Monte Carlo, y sube chi en ESCALERA:
cada escalon parte de los parametros (EMA) del anterior ampliados con ceros (+ ruido pequeno).
Guarda los parametros y la F/E/S2 (con error MC) de cada chi para ver la convergencia en chi.

Ejemplo
    python scripts/training_mps_ladder.py --N 10 --T 0.5 1.0 1.5 --idx 3 4 5 --chis 2 4 8


Parada anticipada (opcional): --tol 0.005  => si |F_chi - F_chi_anterior| < tol, no se sigue subiendo
(en modo --stage los jobs posteriores salen solos con codigo 0).

Los argumentos por escalon (--M, --steps, --lr, ...) admiten una lista (uno por escalon);
si la lista es mas corta se repite el ultimo valor.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")      # antes de importar jax

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))                   # raiz del repo -> src_renyi

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from src_renyi.mps_ancilla.training import train_mc
from src_renyi.mps_ancilla.ladder import (
    build_model, init_params, pad_chi, save_params, load_params,
    make_evaluator, evaluate, read_results, get_result, update_results_json_locked,
)

# ── CLI ────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--T", type=float, nargs="+", required=True, help="Una o varias temperaturas.")
ap.add_argument("--idx", type=int, nargs="+", default=None,
                help="Indice de cada T (= SLURM_ARRAY_TASK_ID). Por defecto 0,1,2,...")
ap.add_argument("--N", type=int, default=10)
ap.add_argument("--chis", type=str, nargs="+", required=True, help="Escalera de chi, p. ej. 2 4 8 (o 2,4,8).")
ap.add_argument("--stage", type=int, default=None,
                help="Ejecuta SOLO el escalon k (0-based) cargando el k-1 de disco. Sin esto: toda la escalera.")
ap.add_argument("--tol", type=float, default=None, help="Parada si |dF| entre escalones < tol.")
ap.add_argument("--force", action="store_true", help="Reentrenar aunque ya existan los parametros.")
# entrenamiento (por escalon)
ap.add_argument("--M", type=int, nargs="+", default=[1024], help="Muestras por paso.")
ap.add_argument("--steps", type=int, nargs="+", default=[3000, 1500])
ap.add_argument("--lr", type=float, nargs="+", default=[5e-3, 2e-3])
ap.add_argument("--warmup", type=int, nargs="+", default=[100, 30])
ap.add_argument("--polish_steps", type=int, nargs="+", default=[1000, 500])
ap.add_argument("--polish_M", type=int, nargs="+", default=None,
                help="Por defecto 2*M si chi<=4 y M si chi>4 (a chi alto el buffer de 2*M no cabia en GPU).")
ap.add_argument("--polish_lr", type=float, nargs="+", default=[1e-3, 5e-4])
ap.add_argument("--pair_block", type=int, default=8)
ap.add_argument("--ema", type=float, default=0.995)
ap.add_argument("--noise", type=float, default=0.02, help="Ruido en los bloques nuevos al ampliar chi.")
# modelo
ap.add_argument("--embedding", type=int, default=16)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--mps_init", type=float, default=0.05)
# evaluacion final
ap.add_argument("--eval_M", type=int, default=1024)
ap.add_argument("--eval_batches", type=int, default=32)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--data_dir", type=str, default=None, help="Por defecto <repo>/data/mps_N{N}.")
args = ap.parse_args()

args.chis = [int(x) for tok in args.chis for x in tok.split(",") if x]
Ts = args.T
idxs = args.idx if args.idx is not None else list(range(len(Ts)))
if len(idxs) != len(Ts):
    sys.exit(f"--idx ({len(idxs)}) y --T ({len(Ts)}) deben tener la misma longitud")
if args.stage is not None and not (0 <= args.stage < len(args.chis)):
    sys.exit(f"--stage {args.stage} fuera de rango para --chis {args.chis}")

N = args.N
# ── fisica: TFIM periodico  H = hx sum sigma^x + J sum sigma^z sigma^z  (convencion NetKet) ──
h_x, J_ZZ = -1.5, -1.0

# ── rutas ──────────────────────────────────────────────────────────────────────
data_dir = args.data_dir or os.path.join(script_dir, "..", "data", f"mps_N{N}")
params_dir = os.path.join(data_dir, "params")
hist_dir = os.path.join(data_dir, "hist")
os.makedirs(params_dir, exist_ok=True)
os.makedirs(hist_dir, exist_ok=True)
results_file = os.path.join(data_dir, f"results_mps_N{N}_vs_T.json")


def params_path(idx, chi):
    return os.path.join(params_dir, f"params_{idx:04d}_chi{chi}.msgpack")


def ps(vals, k):
    """Valor del escalon k de un argumento por escalon (se repite el ultimo)."""
    return vals[k] if k < len(vals) else vals[-1]


def stopped_before(idx, k):
    """True si algun escalon anterior marco 'converged' (parada por --tol)."""
    data = read_results(results_file)
    return any(get_result(data, args.chis[j], "converged", idx) for j in range(k))


def run_stage(T, idx, k):
    chi = args.chis[k]
    tag = f"[T={T:.3f} idx={idx} chi={chi}] "
    pfile = params_path(idx, chi)

    if stopped_before(idx, k):
        print(f"{tag}[stop] un escalon anterior ya convergio (|dF|<{args.tol}); no se sube a chi={chi}.")
        return "stopped"
    if os.path.exists(pfile) and not args.force:
        print(f"{tag}[skip] {pfile} ya existe. Usa --force para reentrenar.")
        return "exists"

    model = build_model(N, chi, args.embedding, args.heads, args.blocks, args.mps_init)
    seed = args.seed + 1000 * idx + k

    # ---- parametros iniciales: desde cero (k=0) o del escalon anterior ampliado ----
    if k == 0:
        params = init_params(model, N, seed)
        print(f"{tag}inicio desde cero (copy_init).", flush=True)
    else:
        chi_prev = args.chis[k - 1]
        prev_file = params_path(idx, chi_prev)
        if not os.path.exists(prev_file):
            sys.exit(f"{tag}falta {prev_file}: el escalon chi={chi_prev} no ha terminado (o fallo).")
        model_prev = build_model(N, chi_prev, args.embedding, args.heads, args.blocks, args.mps_init)
        p_prev = load_params(prev_file, model_prev, N)
        params = pad_chi(p_prev, chi_prev, chi, N, args.noise / np.sqrt(args.embedding), seed)
        print(f"{tag}parametros de chi={chi_prev} ampliados a chi={chi} (ruido {args.noise}).", flush=True)

    M = ps(args.M, k)
    assert M % args.pair_block == 0, f"M={M} debe ser multiplo de pair_block={args.pair_block}"
    steps = ps(args.steps, k)
    pst = ps(args.polish_steps, k)
    pM = ps(args.polish_M, k) if args.polish_M else (2 * M if chi <= 4 else M)
    assert pM % args.pair_block == 0, f"polish_M={pM} debe ser multiplo de pair_block"
    warm = ps(args.warmup, k)
    t0 = time.time()
    print(f"{tag}MC: M={M} steps={steps} lr={ps(args.lr, k)} | polish: {pst} pasos, M={pM}, lr={ps(args.polish_lr, k)}",
          flush=True)

    out = train_mc(model, params, N, T, h_x, J_ZZ, M=M, n_steps=steps, lr=ps(args.lr, k), warmup=warm,
                   seed=seed, pair_block=args.pair_block, freq=max(1, steps // 20),
                   ema_decay=args.ema, polish_steps=pst, polish_M=pM, polish_lr=ps(args.polish_lr, k),
                   tag=tag)
    p_final = out["params_ema"]                       # se guarda y se evalua la EMA

    # ---- evaluacion MC final con barra de error ----
    one = make_evaluator(model, N, T, h_x, J_ZZ, args.eval_M, args.pair_block)
    ev = evaluate(one, p_final, T, args.eval_batches, seed=seed + 999)

    # ---- comparacion con el escalon anterior ----
    dF, converged = None, False
    if k > 0:
        data = read_results(results_file)
        Fp = get_result(data, args.chis[k - 1], "free_energy", idx)
        ep = get_result(data, args.chis[k - 1], "free_energy_err", idx)
        if Fp is not None:
            dF = ev["free_energy"] - Fp
            err = float(np.hypot(ev["free_energy_err"], ep or 0.0))
            converged = bool(args.tol is not None and abs(dF) < args.tol)
            print(f"{tag}dF respecto a chi={args.chis[k - 1]}: {dF:+.5f}  (+-{err:.5f})"
                  + ("   -> CONVERGIDO" if converged else ""), flush=True)

    # ---- guardar ----
    save_params(pfile, p_final)
    np.savez(os.path.join(hist_dir, f"hist_{idx:04d}_chi{chi}.npz"), hist=out["hist"])   # columnas: E, S2, F (MC)
    entry = dict(ev, ms_per_step=1e3 * out["s_per_step"], time_s=time.time() - t0, M=M, steps=steps,
                 dF_prev=dF, converged=converged)
    update_results_json_locked(results_file, idx, T, chi, entry, N=N)
    print(f"{tag}RESULTADO: F={ev['free_energy']:.5f}+-{ev['free_energy_err']:.5f}  "
          f"E={ev['energy']:.5f}  S2={ev['entropy']:.4f}  | {out['s_per_step']*1e3:.0f} ms/paso, "
          f"{time.time() - t0:.0f} s  ->  {pfile}", flush=True)
    return "done"


print(f"MPS ladder: N={N}  chis={args.chis}  T={Ts}  idx={idxs}  stage={args.stage}  "
      f"dispositivos={jax.devices()}", flush=True)
for T, idx in zip(Ts, idxs):
    ks = [args.stage] if args.stage is not None else range(len(args.chis))
    for k in ks:
        if run_stage(T, idx, k) == "stopped":
            break
print("[done]")