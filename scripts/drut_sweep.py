"""
drut_analysis.py
================
Análisis comparativo de estimadores de S₂ y su gradiente en NQS,
con foco en el método Drut–Porter (TI-2).

Estructura:
  1. Entrena vstates a múltiples temperaturas (o los carga de disco).
  2. Corre un barrido de configuraciones para Drut y compara con los
     otros métodos (swap, λ-i, increment).
  3. Repite cada config n_rep veces para separar ruido de sesgo.
  4. Guarda todo a JSON ANTES de cualquier plotting.
  5. Genera figuras (PGF si hay LaTeX, PDF si no).

Uso:
    # Entrenar + analizar
    python drut_analysis.py
    # Solo analizar (usa vstates guardados)
    python drut_analysis.py --skip-train
    # Solo re-generar plots desde JSON existente
    python drut_analysis.py --only-plot
"""

import os
import sys
import time
import json
import copy
import argparse
import shutil
from itertools import combinations

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax

# ── matplotlib: detección de LaTeX ────────────────────────────────────────────
import matplotlib
if shutil.which("pdflatex") is not None:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })
    PLOT_EXT = "pgf"
    HAS_LATEX = True
else:
    matplotlib.use("pdf")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
        "mathtext.fontset": "cm",
    })
    PLOT_EXT = "pdf"
    HAS_LATEX = False

import matplotlib.pyplot as plt

from src_renyi.entropy import (
    renyi2_entropy_and_grad_sampled,
    renyi2_entropy_and_grad_lambda_integral,
    renyi2_drut_sampling,
    renyi2_increment_sampling,
)
from src_renyi.training import free_energy_minimize

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

N              = 12
N_A            = 12
GAMMA          = -1.5
V              = -1.0
TEMPS          = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
N_STEPS_TRAIN  = 300
N_SAMPLES_TRAIN = 2**11

# Barrido de configs para Drut (de barata a cara)
DRUT_CONFIGS = [
    dict(name="A", n_lambda=10, n_sweeps_per_lam=15, n_chains=64),
    dict(name="B", n_lambda=15, n_sweeps_per_lam=25, n_chains=128),
    dict(name="C", n_lambda=20, n_sweeps_per_lam=50, n_chains=256),
    dict(name="D", n_lambda=30, n_sweeps_per_lam=100, n_chains=512),
]

# Parámetros para los otros métodos (comparación)
SWAP_N_SAMPLES       = 2**11
LAMBDA_I_N_LAMBDA    = 30
INCREMENT_CHAINS     = 256
INCREMENT_SWEEPS     = 50

# Repeticiones
N_REP = 5

# Directorios
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "..", "data", f"drut_analysis_N{N}")
PLOTS_DIR = os.path.join(ROOT, "..", "plots", f"drut_analysis_N{N}")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(DATA_DIR, f"results_N{N}.json")

partition = list(range(N))


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path + f".{PLOT_EXT}", bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    print(f"  guardado: {path}.{PLOT_EXT}")
    plt.close(fig)


def grad_norm(g):
    flat, _ = jax.flatten_util.ravel_pytree(g)
    return float(jnp.linalg.norm(jnp.array(flat, float)))


def cosine_similarity(g1, g2):
    flat1, _ = jax.flatten_util.ravel_pytree(g1)
    flat2, _ = jax.flatten_util.ravel_pytree(g2)
    flat1 = jnp.array(flat1, float)
    flat2 = jnp.array(flat2, float)
    return float(
        jnp.dot(flat1, flat2) /
        (jnp.linalg.norm(flat1) * jnp.linalg.norm(flat2) + 1e-30)
    )


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {
        "N": N, "N_A": N_A, "GAMMA": GAMMA, "V": V,
        "TEMPS": list(TEMPS),
        "DRUT_CONFIGS": DRUT_CONFIGS,
        "SWAP_N_SAMPLES": SWAP_N_SAMPLES,
        "LAMBDA_I_N_LAMBDA": LAMBDA_I_N_LAMBDA,
        "INCREMENT_CHAINS": INCREMENT_CHAINS,
        "INCREMENT_SWEEPS": INCREMENT_SWEEPS,
        "N_REP": N_REP,
        "results": {},   # results[str(T)][method][cfg_name or "-"] = list of dicts
    }


def save_results(data):
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MODELO
# ══════════════════════════════════════════════════════════════════════════════

def build_model_and_hamiltonian():
    hi = nk.hilbert.Spin(s=1/2, N=N + N_A)
    H_ext = 0
    for i in range(N):
        H_ext += GAMMA * sigmax(hi, i)
        H_ext += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)
    model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                                 activation=jax.nn.gelu)
    sampler = nk.sampler.ARDirectSampler(hi)
    return hi, H_ext, model, sampler


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def train_all(TEMPS, verbose=True):
    """Entrena un vstate por cada T y los devuelve en un dict."""
    _, H_ext, model, sampler = build_model_and_hamiltonian()
    vstates = {}
    for T in TEMPS:
        if verbose:
            print(f"\n[entrenando T={T}]")
        vstate = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES_TRAIN)
        lr = optax.linear_schedule(0.05, 0.001, N_STEPS_TRAIN)
        t0 = time.time()
        free_energy_minimize(
            vstate, T, partition, H_ext, n_steps=N_STEPS_TRAIN,
            optimizer=optax.sgd(lr),
            plot=False, verbose=False,
            chunk_size=vstate.n_samples // 2,
        )
        if verbose:
            print(f"  entrenado en {time.time() - t0:.1f} s")
        vstates[T] = vstate
    return vstates


# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_method(vstate, T, method, cfg=None, verbose=False):
    """Corre un método y devuelve (S2, grad_norm, tiempo)."""
    t0 = time.time()
    try:
        if method == "swap":
            S2, g = renyi2_entropy_and_grad_sampled(
                vstate, partition, SWAP_N_SAMPLES
            )
        elif method == "lambda_i":
            S2, g = renyi2_entropy_and_grad_lambda_integral(
                vstate, partition, SWAP_N_SAMPLES,
                n_lambda=LAMBDA_I_N_LAMBDA, debug=False,
            )
        elif method == "drut":
            S2, g = renyi2_drut_sampling(
                vstate, partition,
                n_chains=cfg["n_chains"],
                n_sweeps_per_lam=cfg["n_sweeps_per_lam"],
                n_props_per_sweep=2 * (N + N_A),
                n_lambda=cfg["n_lambda"],
                debug=False,
            )
        elif method == "increment":
            S2, g, _, _ = renyi2_increment_sampling(
                vstate, partition,
                n_chains=INCREMENT_CHAINS,
                n_sweeps_per_site=INCREMENT_SWEEPS,
                n_props_per_sweep=2 * (N + N_A),
                debug=False,
            )
        else:
            raise ValueError(f"Método desconocido: {method}")
    except Exception as e:
        if verbose:
            print(f"  [ERROR] {method} cfg={cfg}: {e}")
        return None

    dt = time.time() - t0
    return {
        "S2": float(S2),
        "grad_norm": grad_norm(g),
        "time": dt,
        "grad_flat": [float(x) for x in
                       np.asarray(jax.flatten_util.ravel_pytree(g)[0])],
    }


def run_analysis(vstates, data):
    """Corre todos los métodos, todas las configs, todas las reps."""

    for T, vstate in vstates.items():
        T_str = str(float(T))
        if T_str not in data["results"]:
            data["results"][T_str] = {
                "swap": [],
                "lambda_i": [],
                "drut": {cfg["name"]: [] for cfg in DRUT_CONFIGS},
                "increment": [],
            }

        print(f"\n{'=' * 70}\nT = {T}\n{'=' * 70}")

        # ── Métodos de referencia ────────────────────────────────────────
        for method in ["swap", "lambda_i", "increment"]:
            if len(data["results"][T_str][method]) >= N_REP:
                print(f"  [{method}] ya tiene {N_REP} reps, saltando")
                continue
            print(f"  [{method}]")
            for rep in range(len(data["results"][T_str][method]), N_REP):
                res = evaluate_method(vstate, T, method)
                if res is not None:
                    data["results"][T_str][method].append(res)
                    print(f"    rep {rep+1}/{N_REP}: "
                          f"S2={res['S2']:.4f} t={res['time']:.2f}s")

        # ── Drut con todas las configs ───────────────────────────────────
        for cfg in DRUT_CONFIGS:
            name = cfg["name"]
            if len(data["results"][T_str]["drut"][name]) >= N_REP:
                print(f"  [drut-{name}] ya tiene {N_REP} reps, saltando")
                continue
            print(f"  [drut-{name}]  λ={cfg['n_lambda']} "
                  f"sweeps={cfg['n_sweeps_per_lam']} chains={cfg['n_chains']}")
            for rep in range(len(data["results"][T_str]["drut"][name]), N_REP):
                res = evaluate_method(vstate, T, "drut", cfg=cfg)
                if res is not None:
                    data["results"][T_str]["drut"][name].append(res)
                    print(f"    rep {rep+1}/{N_REP}: "
                          f"S2={res['S2']:.4f} t={res['time']:.2f}s")

        # Guardar tras cada T (a prueba de crashes)
        save_results(data)

    return data


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS Y PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def summarize(data):
    """Extrae medias, std y errores por T/método/config."""
    summary = {}
    for T_str, methods in data["results"].items():
        T = float(T_str)
        summary[T] = {}

        # Swap, λ-i, increment: métodos únicos
        for method in ["swap", "lambda_i", "increment"]:
            reps = methods.get(method, [])
            if not reps:
                continue
            S2_arr = np.array([r["S2"] for r in reps])
            t_arr = np.array([r["time"] for r in reps])
            summary[T][method] = {
                "S2_mean": float(S2_arr.mean()),
                "S2_std": float(S2_arr.std()),
                "time_mean": float(t_arr.mean()),
                "n_rep": len(reps),
            }

        # Drut: múltiples configs
        drut = methods.get("drut", {})
        for name, reps in drut.items():
            if not reps:
                continue
            S2_arr = np.array([r["S2"] for r in reps])
            t_arr = np.array([r["time"] for r in reps])
            summary[T][f"drut_{name}"] = {
                "S2_mean": float(S2_arr.mean()),
                "S2_std": float(S2_arr.std()),
                "time_mean": float(t_arr.mean()),
                "n_rep": len(reps),
            }
    return summary


def compute_errors(summary):
    """
    Para cada T, toma la config D (la más cara) como referencia y calcula
    el error absoluto y relativo de las demás.
    """
    errors = {}
    for T, methods in summary.items():
        ref_key = "drut_D"
        if ref_key not in methods:
            continue
        S2_ref = methods[ref_key]["S2_mean"]
        errors[T] = {"S2_ref": S2_ref}
        for name, m in methods.items():
            errors[T][name] = {
                "S2_err": abs(m["S2_mean"] - S2_ref),
                "S2_err_rel": abs(m["S2_mean"] - S2_ref) / max(S2_ref, 1e-10),
                "time": m["time_mean"],
            }
    return errors


def plot_all(summary, errors):
    """Genera todas las figuras."""
    Ts = sorted(summary.keys())

    # ── 1. S2 vs T ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in ["swap", "lambda_i", "increment"]:
        S2_mean = [summary[T][method]["S2_mean"] for T in Ts if method in summary[T]]
        S2_std  = [summary[T][method]["S2_std"]  for T in Ts if method in summary[T]]
        Ts_ok   = [T for T in Ts if method in summary[T]]
        if not Ts_ok:
            continue
        ax.errorbar(Ts_ok, S2_mean, yerr=S2_std, fmt='o-', label=method, capsize=3)
    for cfg in DRUT_CONFIGS:
        name = f"drut_{cfg['name']}"
        Ts_ok = [T for T in Ts if name in summary[T]]
        S2_mean = [summary[T][name]["S2_mean"] for T in Ts_ok]
        S2_std  = [summary[T][name]["S2_std"]  for T in Ts_ok]
        ax.errorbar(Ts_ok, S2_mean, yerr=S2_std, fmt='s--', label=name, capsize=3)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$S_2$")
    ax.set_title(fr"$S_2$ vs $T$  ($N={N}$)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "S2_vs_T")

    # ── 2. Error vs tiempo ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for T in Ts:
        if T not in errors:
            continue
        for name, e in errors[T].items():
            if name == "S2_ref":
                continue
            ax.scatter(e["time"], e["S2_err"], label=f"T={T}" if name == "swap" else None)
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$|\Delta S_2|$ vs config D")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Error vs coste computacional")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    save_fig(fig, "error_vs_time")

    # ── 3. Error relativo de las configs Drut ────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for cfg in DRUT_CONFIGS:
        name = f"drut_{cfg['name']}"
        Ts_ok = [T for T in Ts if T in errors and name in errors[T]]
        if not Ts_ok:
            continue
        err = [errors[T][name]["S2_err_rel"] for T in Ts_ok]
        tm  = [errors[T][name]["time"] for T in Ts_ok]
        ax.plot(Ts_ok, err, 'o-', label=f"Drut-{cfg['name']}")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$|S_2 - S_2^{ref}| / S_2^{ref}$")
    ax.set_title("Error relativo por config Drut")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    save_fig(fig, "drut_error_vs_T")

    # ── 4. Tiempo vs T ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in ["swap", "lambda_i", "increment"]:
        Ts_ok = [T for T in Ts if method in summary[T]]
        tm = [summary[T][method]["time_mean"] for T in Ts_ok]
        if Ts_ok:
            ax.plot(Ts_ok, tm, 'o-', label=method)
    for cfg in DRUT_CONFIGS:
        name = f"drut_{cfg['name']}"
        Ts_ok = [T for T in Ts if name in summary[T]]
        tm = [summary[T][name]["time_mean"] for T in Ts_ok]
        if Ts_ok:
            ax.plot(Ts_ok, tm, 's--', label=name)
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$t$ (s)")
    ax.set_title("Coste por evaluación")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    save_fig(fig, "time_vs_T")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="No reentrenar, pero correr análisis")
    parser.add_argument("--only-plot", action="store_true",
                        help="Solo generar plots desde JSON existente")
    args = parser.parse_args()

    data = load_results()

    if args.only_plot:
        summary = summarize(data)
        errors = compute_errors(summary)
        plot_all(summary, errors)
        return

    # Entrenar (o cargar vstates existentes)
    vstates = train_all(TEMPS)  # nota: no guarda vstates a disco aún

    # Análisis
    data = run_analysis(vstates, data)

    # Resumen y plots
    summary = summarize(data)
    errors = compute_errors(summary)

    # Guardar resumen
    with open(os.path.join(DATA_DIR, "summary.json"), "w") as f:
        json.dump({
            "summary": {str(k): v for k, v in summary.items()},
            "errors": {str(k): v for k, v in errors.items()},
        }, f, indent=2, default=float)

    plot_all(summary, errors)

    # Tabla final en consola
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    for T in sorted(summary.keys()):
        print(f"\nT = {T}")
        print(f"  {'método':<15}  {'S2 (mean±std)':<20}  {'t(s)':>8}")
        for name, m in sorted(summary[T].items()):
            print(f"  {name:<15}  {m['S2_mean']:.4f} ± {m['S2_std']:.4f}     "
                  f"{m['time_mean']:>8.2f}")


if __name__ == "__main__":
    main()