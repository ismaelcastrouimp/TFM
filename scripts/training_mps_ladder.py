"""
training_mps_ladder.py
======================
Minimiza F = <H> - T*S2 con ARViT + MPS ancilla, todo Monte Carlo, y sube chi en ESCALERA:
cada escalon parte de los parametros (EMA) del anterior ampliados con ceros (+ ruido pequeno).
Guarda los parametros y la F/E/S2 (con error MC) de cada chi para ver la convergencia en chi.

Backward compatible con CLI directo (sin YAML):
    python scripts/training_mps_ladder.py --N 10 --T 0.5 1.0 1.5 --idx 3 4 5 --chis 2 4 8

Uso cluster
    ./slurm/submit_mps_ladder.sh

Uso interactivo con YAML (un solo escalon):
    python scripts/training_mps_ladder.py --config slurm/config_mps.yaml \
        --T 1.5 --idx 0 --chis 2 4 8 --stage 0

Parada anticipada (opcional, via config o --tol):
    --tol 0.005  => si |F_chi - F_chi_anterior| < tol, no se sigue subiendo chi.

Precedencia de parametros:  CLI  >  YAML (--config)  >  defaults hardcoded.
Los argumentos por escalon (--M, --steps, --lr, ...) admiten una lista (uno por escalon);
si la lista es mas corta se repite el ultimo valor; cada elemento puede ser null para
"auto" en polish_M.
"""
import argparse
import os
import sys
import time

import yaml

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
ap.add_argument("--config", type=str, default=None,
                help="YAML con fisica, arquitectura, training, eval, paths. "
                     "(El sweep T lo gestiona el launcher; aqui T e idx llegan por CLI.)")

# Overrides explicitos (default=None para que _get distinga "no pasado" de "pasado").
ap.add_argument("--h_x",  type=float, default=None)
ap.add_argument("--J_ZZ", type=float, default=None)
ap.add_argument("--data_dir", type=str, default=None)
ap.add_argument("--N", type=int, default=None)
ap.add_argument("--embedding", type=int, default=None)
ap.add_argument("--heads", type=int, default=None)
ap.add_argument("--blocks", type=int, default=None)
ap.add_argument("--mps_init", type=float, default=None)
ap.add_argument("--pair_block", type=int, default=None)
ap.add_argument("--ema", type=float, default=None)
ap.add_argument("--noise", type=float, default=None)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--eval_M", type=int, default=None)
ap.add_argument("--eval_batches", type=int, default=None)
ap.add_argument("--tol", type=float, default=None)

# Argumentos obligatorios por job (los pasa el .slurm / launcher).
ap.add_argument("--T", type=float, nargs="+", required=True,
                help="Una o varias temperaturas.")
ap.add_argument("--idx", type=int, nargs="+", default=None,
                help="Indice de cada T (= SLURM_ARRAY_TASK_ID). Por defecto 0,1,2,...")
ap.add_argument("--chis", type=str, nargs="+", default=None,
                help="Escalera de chi, p. ej. '2 4 8' o '2,4,8'. Si se omite, se lee de "
                     "ladder.chis en el YAML o se usa [2, 4, 8].")
ap.add_argument("--stage", type=int, default=None,
                help="Ejecuta SOLO el escalon k (0-based) cargando el k-1 de disco. "
                     "Sin esto: toda la escalera.")
ap.add_argument("--force", action="store_true",
                help="Reentrenar aunque ya existan los parametros o un escalon previo haya convergido.")

# Listas por escalon (default=None; vienen del YAML si no se pasan).
ap.add_argument("--M", type=int, nargs="+", default=None)
ap.add_argument("--steps", type=int, nargs="+", default=None)
ap.add_argument("--lr", type=float, nargs="+", default=None)
ap.add_argument("--warmup", type=int, nargs="+", default=None)
ap.add_argument("--polish_steps", type=int, nargs="+", default=None)
ap.add_argument("--polish_M", type=int, nargs="+", default=None,
                help="Por defecto 2*M si chi<=4 y M si chi>4. Elementos null = auto para ese escalon.")
ap.add_argument("--polish_lr", type=float, nargs="+", default=None)

args = ap.parse_args()


# ── carga del config ──────────────────────────────────────────────────────────
cfg = yaml.safe_load(open(args.config)) if args.config else {}

def _get(cli, section, key, default):
    """CLI > YAML > default. Distingue 'no pasado' (None) de 'pasado' (cualquier otro valor)."""
    if cli is not None:
        return cli
    sec = cfg.get(section, {}) or {}
    v = sec.get(key, None)
    return default if v is None else v

# ── fisica ────────────────────────────────────────────────────────────────────
args.N    = _get(args.N,    "physics", "N",    10)
args.h_x  = _get(args.h_x,  "physics", "h_x",  -1.5)
args.J_ZZ = _get(args.J_ZZ, "physics", "J_ZZ", -1.0)

# ── arquitectura del modelo ───────────────────────────────────────────────────
args.embedding = _get(args.embedding, "model", "embedding", 16)
args.heads     = _get(args.heads,     "model", "heads",      4)
args.blocks    = _get(args.blocks,    "model", "blocks",     2)
args.mps_init  = _get(args.mps_init,  "model", "mps_init",   0.05)

# ── escalares de entrenamiento ────────────────────────────────────────────────
args.pair_block = _get(args.pair_block, "training", "pair_block", 8)
args.ema        = _get(args.ema,        "training", "ema",        0.995)
args.noise      = _get(args.noise,      "training", "noise",      0.02)
args.seed       = _get(args.seed,       "training", "seed",       1)

# ── evaluacion final ──────────────────────────────────────────────────────────
args.eval_M       = _get(args.eval_M,       "evaluation", "eval_M",       1024)
args.eval_batches = _get(args.eval_batches, "evaluation", "eval_batches", 32)

# ── ladder ────────────────────────────────────────────────────────────────────
args.tol = _get(args.tol, "ladder", "tol", None)

ladder_cfg = cfg.get("ladder", {}) or {}

if args.chis is None:
    args.chis = ladder_cfg.get("chis", [2, 4, 8])

if not args.force:
    args.force = bool(ladder_cfg.get("force", False))

# ── listas por escalon (con fallback hardcoded) ───────────────────────────────
_LIST_DEFAULTS = {
    "M":            [1024],
    "steps":        [3000, 1500],
    "lr":           [5e-3, 2e-3],
    "warmup":       [100, 30],
    "polish_steps": [1000, 500],
    "polish_M":     None,          # None = auto por escalon (2*M si chi<=4, M si chi>4)
    "polish_lr":    [1e-3, 5e-4],
}
training_cfg = cfg.get("training", {}) or {}
for key, default in _LIST_DEFAULTS.items():
    v = getattr(args, key)
    if v is None:
        v = training_cfg.get(key, None)
    if v is None:
        v = default
    # Escalar -> [escalar] (permite 'M: 2048' en YAML sin exigir lista)
    if v is not None and not isinstance(v, (list, tuple)):
        v = [v]
    setattr(args, key, v)

# ── rutas ─────────────────────────────────────────────────────────────────────
dd = _get(args.data_dir, "paths", "data_dir", None)
args.data_dir = dd if dd else os.path.join(script_dir, "..", "data", f"mps_N{args.N}")

# ── normalizar chis (acepta '2 4 8', '2,4,8', [2,4,8], [2, '4,8']) ───────────
def _norm_chis(chis):
    out = []
    for tok in chis:
        if isinstance(tok, str):
            for x in tok.split(","):
                x = x.strip()
                if x:
                    out.append(int(x))
        else:
            out.append(int(tok))
    return out

args.chis = _norm_chis(args.chis)

# ── sanity ────────────────────────────────────────────────────────────────────
Ts = args.T
idxs = args.idx if args.idx is not None else list(range(len(Ts)))
if len(idxs) != len(Ts):
    sys.exit(f"--idx ({len(idxs)}) y --T ({len(Ts)}) deben tener la misma longitud")
if args.stage is not None and not (0 <= args.stage < len(args.chis)):
    sys.exit(f"--stage {args.stage} fuera de rango para --chis {args.chis}")

N        = args.N
data_dir = args.data_dir
params_dir = os.path.join(data_dir, "params")
hist_dir   = os.path.join(data_dir, "hist")
os.makedirs(params_dir, exist_ok=True)
os.makedirs(hist_dir, exist_ok=True)
results_file = os.path.join(data_dir, f"results_mps_N{N}_vs_T.json")


# ── helpers ───────────────────────────────────────────────────────────────────
def params_path(idx, chi):
    return os.path.join(params_dir, f"params_{idx:04d}_chi{chi}.msgpack")


def ps(vals, k):
    """Valor del escalon k de un argumento por escalon (se repite el ultimo)."""
    if vals is None:
        return None
    return vals[k] if k < len(vals) else vals[-1]


def stopped_before(idx, k):
    """True si algun escalon anterior marco 'converged'. --force ignora la parada."""
    if args.force:
        return False
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

    # ---- resolucion de hiperparametros por escalon ----
    M = ps(args.M, k)
    if M is None:
        raise ValueError(f"{tag}M no resuelto (training.M en config y --M vacios).")
    assert M % args.pair_block == 0, f"M={M} debe ser multiplo de pair_block={args.pair_block}"

    steps = ps(args.steps, k)
    warm  = ps(args.warmup, k)
    pst   = ps(args.polish_steps, k)
    lr_k  = ps(args.lr, k)
    plr_k = ps(args.polish_lr, k)

    raw_pM = ps(args.polish_M, k) if args.polish_M else None
    pM = raw_pM if raw_pM is not None else (2 * M if chi <= 4 else M)
    assert pM % args.pair_block == 0, f"polish_M={pM} debe ser multiplo de pair_block"

    t0 = time.time()
    print(f"{tag}MC: M={M} steps={steps} lr={lr_k} | polish: {pst} pasos, M={pM}, lr={plr_k}",
          flush=True)

    out = train_mc(model, params, N, T, args.h_x, args.J_ZZ,
                   M=M, n_steps=steps, lr=lr_k, warmup=warm,
                   seed=seed, pair_block=args.pair_block, freq=max(1, steps // 20),
                   ema_decay=args.ema, polish_steps=pst, polish_M=pM, polish_lr=plr_k,
                   tag=tag)
    p_final = out["params_ema"]                       # se guarda y se evalua la EMA

    # ---- evaluacion MC final con barra de error ----
    one = make_evaluator(model, N, T, args.h_x, args.J_ZZ, args.eval_M, args.pair_block)
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
    np.savez(os.path.join(hist_dir, f"hist_{idx:04d}_chi{chi}.npz"),
             hist=out["hist"])   # columnas: E, S2, F (MC)
    entry = dict(ev,
                 ms_per_step=1e3 * out["s_per_step"],
                 time_s=time.time() - t0,
                 M=M, steps=steps,
                 h_x=float(args.h_x), J_ZZ=float(args.J_ZZ),
                 dF_prev=dF, converged=converged)
    update_results_json_locked(results_file, idx, T, chi, entry, N=N)
    print(f"{tag}RESULTADO: F={ev['free_energy']:.5f}+-{ev['free_energy_err']:.5f}  "
          f"E={ev['energy']:.5f}  S2={ev['entropy']:.4f}  | {out['s_per_step']*1e3:.0f} ms/paso, "
          f"{time.time() - t0:.0f} s  ->  {pfile}", flush=True)
    return "done"


# ── main ──────────────────────────────────────────────────────────────────────
print(f"MPS ladder: N={N}  chis={args.chis}  T={Ts}  idx={idxs}  stage={args.stage}  "
      f"h_x={args.h_x}  J_ZZ={args.J_ZZ}  config={args.config}  "
      f"dispositivos={jax.devices()}", flush=True)

for T, idx in zip(Ts, idxs):
    ks = [args.stage] if args.stage is not None else range(len(args.chis))
    for k in ks:
        if run_stage(T, idx, k) == "stopped":
            break
print("[done]")