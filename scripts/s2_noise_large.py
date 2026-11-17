"""
s2_noise_large.py
====================
Comparación de estimadores de S₂ y su gradiente para tamaños donde ED no es posible, a COSTE IGUALADO
(evaluaciones del ansatz).  Versión 16-11-2026: AR inclinado, Wang CS, muestreo directo (DS, n=2) e Increment
(sin DRUT ni swap).

Estructura:
  1. Entrena vstates a distintas temperaturas con Wang CS (fase única); la red reversa se entrena dentro de la fase
     Wang (initial_train + warm_start) y se reutiliza para el estimador de Wang.
  2. Barre el presupuesto B por temperatura.
  3. Análisis a presupuesto fijo: S₂, σ(S₂) entre réplicas, autoconsistencia del gradiente, tiempo.
  4. Error (σ entre réplicas) vs coste, y consistencia mutua de los gradientes medios.

Coste en evaluaciones del ansatz (muestreo AR = 1 evaluación por sitio muestreado, como en Wang):
  - Inclinado: M·N_total + M·K·(N_total + 5)
  - Wang:      (2·N_total + 7) por muestra
  - DS (n=2):  2·N_total + 4 por pareja
  - Increment: N_A · n_sweeps_per_site · n_props · 2 por cadena

Gráficas en ../plots/S2_noise_large/ (.pdf y .pgf).
Uso: python scripts/s2_noise_large.py [--quick]    (--quick: N=6, pocos pasos; gráficas en S2_noise_large_quick/)
"""

import argparse
import os
import sys
import time
import copy
import itertools
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax
import subprocess
import tempfile
import matplotlib


def _latex_ok():
    """True si pdflatex compila un documento mínimo (en un sandbox sin acceso a texmf falla)."""
    try:
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "t.tex"), "w").write(r"\documentclass{article}\begin{document}x\end{document}")
            return subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"], cwd=d, capture_output=True,
                                  timeout=60).returncode == 0
    except Exception:
        return False


LATEX = _latex_ok()
if LATEX:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })
else:                                   # sin LaTeX: mathtext y solo .pdf
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    print("AVISO: pdflatex no disponible -> figuras con mathtext, solo .pdf (sin .pgf)")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src_renyi.physical_ancilla.bridge import renyi2_bridge   # 17-11: estimador puente
from src_renyi.physical_ancilla.entropy import (
    renyi2_tilted,
    renyi2_wang_cs,
    renyin_entropy_and_grad_sampled,
    renyi2_increment_sampling,
)
from src_renyi.physical_ancilla.training import free_energy_minimize_phases

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="prueba rápida (N=6, pocos pasos y presupuestos)")
ap.add_argument("--train_steps", type=int, default=None, help="pasos de la fase Wang (por defecto 400; quick 20)")
_args = ap.parse_args(); QUICK = _args.quick

# ══════════════════════════════════════════════════════════════════════════════
# Configuración
# ══════════════════════════════════════════════════════════════════════════════
N              = 6 if QUICK else 20
N_A            = N
GAMMA          = -1.5
V              = -1.0
TEMPS          = [3.0, 7.0] if QUICK else [3.0, 4.0, 5.0, 7.0]
n_rep          = 3 if QUICK else 10

# AR inclinado
TILT_K      = 5
TILT_KAPPAS = (0.05, 0.1, 0.2, 0.4, 1.0)
TILT_CHUNK  = 256          # nº de x1 por trozo (memoria); se ajusta a un divisor de M

# Increment (Hastings 2010)
INCREMENT_N_SWEEPS_PER_SITE = 20
INCREMENT_N_PROPS_PER_SWEEP = 2 * (N + N_A)

# ── Hiperparámetros de entrenamiento (Wang único) ──────────────────────────────
N_STEPS_WANG  = _args.train_steps or (20 if QUICK else 400)
LR_WANG       = 1e-3
INIT_N_STEPS  = 100 if QUICK else 3000
INIT_BATCH    = 1024 if QUICK else 4096
LR_INIT       = 1e-3
WARM_EVERY    = 5
WARM_N_STEPS  = 20
WARM_BATCH    = 2**12 if QUICK else 2**14
LR_WARM       = 5e-4

N_TOTAL = N + N_A

COST_TILT_PER_X1 = N_TOTAL + TILT_K * (N_TOTAL + 5)
BRIDGE_K, BRIDGE_NA = 5, 2      # puente: compañeros s2 por s1 y muestras de ancilla por par
COST_BRIDGE_PER_S1 = N + BRIDGE_K * (N + 1 + BRIDGE_NA * (2 * N_A + 2))
COST_WANG_PER_SAMPLE = 2 * N_TOTAL + 7
COST_DS_PER_PAIR = 2 * N_TOTAL + 4
COST_INC_PER_CHAIN = N_A * INCREMENT_N_SWEEPS_PER_SITE * INCREMENT_N_PROPS_PER_SWEEP * 2

BUDGET_LIST = [2**14, 2**16] if QUICK else [2**14, 2**16, 2**18, 2**20]      # 16-11: sin 2**22 (Ismael)
BUDGET_FIXED = BUDGET_LIST[-1]
print(f"Presupuestos a barrer: {BUDGET_LIST}")


def sizes(B):
    return dict(tilted=max(8, B // COST_TILT_PER_X1), bridge=max(8, B // COST_BRIDGE_PER_S1), wang=max(16, B // COST_WANG_PER_SAMPLE),
                ds=max(16, B // COST_DS_PER_PAIR), inc=max(8, B // COST_INC_PER_CHAIN))


def costs(nsz):
    return dict(tilted=nsz["tilted"] * COST_TILT_PER_X1, bridge=nsz["bridge"] * COST_BRIDGE_PER_S1, wang=nsz["wang"] * COST_WANG_PER_SAMPLE,
                ds=nsz["ds"] * COST_DS_PER_PAIR, inc=nsz["inc"] * COST_INC_PER_CHAIN)


def tilt_chunk(M):
    c = min(TILT_CHUNK, M)
    while M % c:
        c -= 1
    return c


for B in BUDGET_LIST:
    print(f"  B={B:>10,} → {sizes(B)}")

partition = list(range(N))

# Cada método: (etiqueta, marcador, color, fn(vstate, vstate_R, n, rep) -> (S2, grad))
METHODS = {
    "tilted": ("AR inclinado", "o", "C0",
               lambda vs, vR, n, rep: renyi2_tilted(vs, partition, n, K=TILT_K, kappas=TILT_KAPPAS, key=rep,
                                                    chunk_size=tilt_chunk(n))),
    "bridge": ("Puente", "P", "C2",
               lambda vs, vR, n, rep: renyi2_bridge(vs, partition, n, K=BRIDGE_K, kappas=TILT_KAPPAS,
                                                    n_a=BRIDGE_NA, key=rep, chunk_size=tilt_chunk(n))),
    "wang":   ("Wang CS", "d", "C3",
               lambda vs, vR, n, rep: renyi2_wang_cs(vs, vR, partition, n_samples=n, key=rep)),
    "ds":     ("Direct sampling", "s", "C1",
               lambda vs, vR, n, rep: renyin_entropy_and_grad_sampled(vs, partition, 2, n, key=rep)),
    "inc":    ("Increment", "v", "C4",
               lambda vs, vR, n, rep: renyi2_increment_sampling(
                   vs, partition, n_chains=n, n_sweeps_per_site=INCREMENT_N_SWEEPS_PER_SITE,
                   n_props_per_sweep=INCREMENT_N_PROPS_PER_SWEEP, key=rep)[:2]),
}

# ── directorios ────────────────────────────────────────────────────────────────
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots/S2_noise_large" + ("_quick" if QUICK else ""))
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    if LATEX:
        fig.savefig(path + ".pgf", bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    print(f"  guardado: {path}.pdf")
    plt.close(fig)


def _flat(g):
    return jnp.array(jax.flatten_util.ravel_pytree(g)[0], float)


def cos_(a, b):
    return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))


# ── hilbert y hamiltoniano ─────────────────────────────────────────────────────
hi = nk.hilbert.Spin(s=1/2, N=N + N_A)
H_extended = 0
for i in range(N):
    H_extended += GAMMA * sigmax(hi, i)
    H_extended += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)

model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16, activation=jax.nn.gelu)
sampler    = nk.sampler.ARDirectSampler(hi)
vstate_ref = nk.vqs.MCState(sampler, model, n_samples=2**12 if QUICK else 2**16)


# ══════════════════════════════════════════════════════════════════════════════
# Entrenamiento (Wang CS como única fase)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"Entrenando vstates  (N = {N}, N_A = {N_A}) con Wang CS")
print("=" * 70)

trained_vstates   = {}
trained_vstates_R = {}

for T in TEMPS:
    vstate = copy.deepcopy(vstate_ref)
    print(f"\n  T = {T:.2f}")
    model_R = nk.models.ARNNDense(hilbert=hi, layers=1, features=16, activation=jax.nn.gelu)
    vstate_R = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model_R, n_samples=vstate.n_samples)
    phases = [{
        "method": "wang",
        "n_steps": N_STEPS_WANG,
        "optimizer": optax.adam(LR_WANG),
        "vstate_R": vstate_R,
        "initial_train": {"n_steps": INIT_N_STEPS, "batch": INIT_BATCH, "lr": LR_INIT},
        "warm_start": {"every": WARM_EVERY, "n_steps": WARM_N_STEPS, "lr": LR_WARM, "batch": WARM_BATCH},
        "wang_kwargs": {"n_samples": vstate.n_samples},
    }]
    t0 = time.time()
    free_energy_minimize_phases(
        vstate, T, partition, H_extended, phases,
        chunk_size=vstate.n_samples // 2,
        verbose=False, freq=50, plot=False, timing=False,
    )
    print(f"         entrenamiento (Wang): {time.time() - t0:.1f} s")
    trained_vstates[T]   = vstate
    trained_vstates_R[T] = vstate_R


# ══════════════════════════════════════════════════════════════════════════════
# Sección 1: barrido de presupuesto por temperatura
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. Error vs presupuesto (por temperatura)")
print("=" * 70)

# results[T][método][B] = {"S2": [...], "grads": [...], "time": [...], "cost": ..., "n_size": ..., "cos_self": ...}
results = {T: {k: {} for k in METHODS} for T in TEMPS}

for T, vstate in trained_vstates.items():
    vstate_R = trained_vstates_R[T]
    print(f"\n  T = {T:.2f}")
    for B in BUDGET_LIST:
        nsz, cst = sizes(B), costs(sizes(B))
        for k in METHODS:
            s2, grads, ts = [], [], []
            for rep in range(n_rep):
                t0 = time.time()
                S2, g = METHODS[k][3](vstate, vstate_R, nsz[k], rep)
                ts.append(time.time() - t0); s2.append(float(S2)); grads.append(g)
            fl = [_flat(g) for g in grads]
            cself = [cos_(fl[i], fl[j]) for i, j in itertools.combinations(range(n_rep), 2)]
            results[T][k][B] = dict(S2=s2, grads=grads, time=ts, cost=cst[k], n_size=nsz[k],
                                    cos_self=float(np.mean(cself)))
            print(f"    B={B:>10,}  {k:6s}(n={nsz[k]:>8,}): ⟨S₂⟩={np.nanmean(s2):.4f}  σ={np.nanstd(s2):.4f}  "
                  f"cos_self={np.mean(cself):.3f}  t={np.mean(ts):.3f}s", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sección 2: análisis a presupuesto fijo
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"2. Análisis a presupuesto fijo B = {BUDGET_FIXED:,}")
print("=" * 70)

stat = lambda k, f: np.array([f(results[T][k][BUDGET_FIXED]) for T in TEMPS])
S2_consensus = np.nanmean([stat(k, lambda r: np.nanmean(r["S2"])) for k in METHODS], axis=0)

fig, ax = plt.subplots(figsize=(6.5, 4))
for k, (lab, mk, c, _) in METHODS.items():
    ax.errorbar(TEMPS, stat(k, lambda r: np.nanmean(r["S2"])), yerr=stat(k, lambda r: np.nanstd(r["S2"])),
                fmt=mk + "-", color=c, label=lab, capsize=3)
ax.plot(TEMPS, S2_consensus, "k--", alpha=0.4, label="consenso")
ax.set_xlabel(r"$T$"); ax.set_ylabel(r"$S_2$")
ax.set_title(fr"$S_2$ vs $T$ ($N={N}$, $B={BUDGET_FIXED:,}$)")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); save_fig(fig, "S2_vs_T")

for name, f, ylab, title, logy in (
        ("S2_std_vs_T", lambda r: np.nanstd(r["S2"]), r"$\mathrm{std}(S_2)$ entre réplicas",
         "Varianza del estimador de $S_2$", True),
        ("cos_self_vs_T", lambda r: r["cos_self"], r"$\langle \cos(\nabla S_2^{(i)}, \nabla S_2^{(j)}) \rangle$",
         "Autoconsistencia del gradiente", False),
        ("time_vs_T", lambda r: np.mean(r["time"]), r"$t$ (s)", "Tiempo por llamada (presupuesto fijo)", True)):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, (lab, mk, c, _) in METHODS.items():
        ax.plot(TEMPS, stat(k, f), mk + "-", color=c, label=lab)
    if name == "cos_self_vs_T":
        ax.axhline(0.7, color="r", ls=":", alpha=0.5, label="umbral 0.7"); ax.set_ylim(0, 1.05)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(r"$T$"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout(); save_fig(fig, name)


# ══════════════════════════════════════════════════════════════════════════════
# Sección 3: LA FIGURA CLAVE — error vs coste
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. Error vs coste (figura clave)")
print("=" * 70)

fig, axes = plt.subplots(1, len(TEMPS), figsize=(4 * len(TEMPS), 3.5), sharey=False)
for ax, T in zip(np.atleast_1d(axes), TEMPS):
    for k, (lab, mk, c, _) in METHODS.items():
        ax.loglog([results[T][k][B]["cost"] for B in BUDGET_LIST],
                  [np.nanstd(results[T][k][B]["S2"]) for B in BUDGET_LIST], mk + "-", color=c, label=lab)
    ax.set_xlabel(r"$B$ (evaluaciones del ansatz)"); ax.set_ylabel(r"$\sigma(S_2)$ entre réplicas")
    ax.set_title(fr"$T = {T}$"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")
fig.tight_layout(); save_fig(fig, "error_vs_cost_per_T")


# ══════════════════════════════════════════════════════════════════════════════
# Sección 4: consistencia mutua entre métodos
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. Consistencia mutua (gradientes medios)")
print("=" * 70)

pairs = list(itertools.combinations(METHODS, 2))
cos_mut = {p: [] for p in pairs}
for T in TEMPS:
    gm = {k: jnp.mean(jnp.stack([_flat(g) for g in results[T][k][BUDGET_FIXED]["grads"]]), axis=0) for k in METHODS}
    for a, b in pairs:
        cos_mut[(a, b)].append(cos_(gm[a], gm[b]))
    print(f"  T = {T:.2f}   " + "   ".join(f"cos({a},{b}) = {cos_mut[(a, b)][-1]:.4f}" for a, b in pairs))

fig, ax = plt.subplots(figsize=(6.5, 4))
for i, (a, b) in enumerate(pairs):
    ax.plot(TEMPS, cos_mut[(a, b)], "o-", color=f"C{i}", label=rf"$\cos(\nabla S_2^{{{a}}}, \nabla S_2^{{{b}}})$")
ax.axhline(0.7, color="r", ls=":", alpha=0.5, label="umbral 0.7")
ax.set_xlabel(r"$T$"); ax.set_ylabel(r"$\cos$"); ax.set_title("Consistencia mutua del gradiente")
ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
fig.tight_layout(); save_fig(fig, "cos_mutual_vs_T")


# ══════════════════════════════════════════════════════════════════════════════
# Tabla resumen
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Resumen final")
print("=" * 70)
print(f"\n{'T':>5}  {'método':>7}  {'S₂':>8}  {'σ(S₂)':>8}  {'cos_self':>9}  {'t(s)':>8}  {'coste':>15}  {'n_size':>10}")
for T in TEMPS:
    for k in METHODS:
        r = results[T][k][BUDGET_FIXED]
        print(f"{T:>5.1f}  {k:>7}  {np.nanmean(r['S2']):>8.4f}  {np.nanstd(r['S2']):>8.4f}  {r['cos_self']:>9.3f}  "
              f"{np.mean(r['time']):>8.3f}  {r['cost']:>15,}  {r['n_size']:>10,}")
    print()

print("\nDone. Gráficas guardadas en", os.path.abspath(PLOTS_DIR))
