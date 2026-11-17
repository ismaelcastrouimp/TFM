"""
s2_noise.py
===========
Comparación justa de estimadores de S₂ y su gradiente a presupuesto igualado, frente al EXACTO (ED).
Estimadores (versión 16-11-2026; sin DRUT, swap ni λ-i):
  - AR inclinado (renyi2_tilted, mezcla de κ's)
  - Wang CS (renyi2_wang_cs, con red reversa entrenada)
  - Muestreo directo, DS (renyin_entropy_and_grad_sampled con n=2)
  - Increment (Hastings 2010; renyi2_increment_sampling)

Presupuesto común: evaluaciones del ansatz, contando el muestreo AR como 1 evaluación por sitio muestreado
(mismo convenio que Wang):
  - Inclinado: M·N_total (x1) + M·K·(N_total + 5)   (x2 AR, 1 pasada de condicionales para q, 4 de ln ψ)
  - Wang CS:   (2·N_total + 7) por muestra
  - DS (n=2):  2·N_total + 4 por pareja  (2 configuraciones AR + 4 ln ψ)
  - Increment: N_A · n_sweeps_per_site · n_props · 2 por cadena

Métricas del gradiente: cos(ĝ, g_ex) (dirección); β = <ĝ, g_ex>/<g_ex, g_ex> (ganancia: con F = E − T·S₂ un
estimador con β ≠ 1 entrena a T_eff = β·T); ‖ĝ‖/‖g_ex‖.

Gráficas en ../plots/S2_noise/ (.pgf y .pdf).
Uso: python scripts/s2_noise.py [--quick]     (--quick: prueba rápida, N=4, pocos pasos; gráficas en S2_noise_quick/)
"""

import argparse
import os
import sys
import copy
import time
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax
from scipy.optimize import curve_fit
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
    renyi2_entropy_and_grad_exact,
    renyi2_tilted,
    renyi2_wang_cs,
    renyin_entropy_and_grad_sampled,
    renyi2_increment_sampling,
    train_reverse_network,
)
from src_renyi.physical_ancilla.training import free_energy_minimize

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="prueba rápida (N=4, pocos pasos y presupuestos)")
QUICK = ap.parse_args().quick

# ── configuración ──────────────────────────────────────────────────────────────
N              = 4 if QUICK else 10
GAMMA          = -1.5
V              = -1.0
TEMPS          = [2.0, 5.0] if QUICK else [2.0, 3.0, 4.0, 5.0]
n_rep          = 3 if QUICK else 10
TRAIN_STEPS    = 30 if QUICK else 300
REV_STEPS      = 200 if QUICK else 4000

BUDGET_FIXED   = 2**16 if QUICK else 2**20
BUDGET_LIST    = [2**14, 2**16] if QUICK else [2**14, 2**16, 2**18, 2**20, 2**22]

# AR inclinado
TILT_K      = 5
TILT_KAPPAS = (0.05, 0.1, 0.2, 0.4, 1.0)

# Increment (Hastings 2010)
INCREMENT_N_SWEEPS_PER_SITE = 20
INCREMENT_N_PROPS_PER_SWEEP = 2 * N

# ── Hilbert ────────────────────────────────────────────────────────────────────
subsystem = list(range(N))
partition = subsystem
hi = nk.hilbert.Spin(s=1/2, N=2 * N)   # N_S = N, N_A = N
N_TOTAL = 2 * N

# Coste por unidad de cada estimador (evaluaciones del ansatz)
COST_TILT_PER_X1 = N_TOTAL + TILT_K * (N_TOTAL + 5)
BRIDGE_K, BRIDGE_NA = 5, 2      # puente: compañeros s2 por s1 y muestras de ancilla por par
COST_BRIDGE_PER_S1 = N + BRIDGE_K * (N + 1 + BRIDGE_NA * (2 * N + 2))
COST_WANG_PER_SAMPLE = 2 * N_TOTAL + 7
COST_DS_PER_PAIR = 2 * N_TOTAL + 4
COST_INC_PER_CHAIN = N * INCREMENT_N_SWEEPS_PER_SITE * INCREMENT_N_PROPS_PER_SWEEP * 2


def sizes(B):
    """Tamaños de cada estimador para un presupuesto B."""
    return dict(tilted=max(8, B // COST_TILT_PER_X1), bridge=max(8, B // COST_BRIDGE_PER_S1), wang=max(16, B // COST_WANG_PER_SAMPLE),
                ds=max(16, B // COST_DS_PER_PAIR), inc=max(8, B // COST_INC_PER_CHAIN))


# Cada método: (etiqueta, marcador, color, fn(vstate, vstate_R, n, rep) -> (S2, grad))
METHODS = {
    "tilted": ("AR inclinado", "o", "C0",
               lambda vs, vR, n, rep: renyi2_tilted(vs, subsystem, n, K=TILT_K, kappas=TILT_KAPPAS, key=rep)),
    "bridge": ("Puente", "P", "C2",
               lambda vs, vR, n, rep: renyi2_bridge(vs, subsystem, n, K=BRIDGE_K, kappas=TILT_KAPPAS,
                                                    n_a=BRIDGE_NA, key=rep)),
    "wang":   ("Wang CS", "d", "C3",
               lambda vs, vR, n, rep: renyi2_wang_cs(vs, vR, subsystem, n_samples=n, key=rep)),
    "ds":     ("Direct sampling", "s", "C1",
               lambda vs, vR, n, rep: renyin_entropy_and_grad_sampled(vs, subsystem, 2, n, key=rep)),
    "inc":    ("Increment", "v", "C4",
               lambda vs, vR, n, rep: renyi2_increment_sampling(
                   vs, subsystem, n_chains=n, n_sweeps_per_site=INCREMENT_N_SWEEPS_PER_SITE,
                   n_props_per_sweep=INCREMENT_N_PROPS_PER_SWEEP, key=rep)[:2]),
}

print(f"Presupuesto fijo: {BUDGET_FIXED} evaluaciones  →  {sizes(BUDGET_FIXED)}")
print(f"  S₂_max = {N * np.log(2):.2f}  (N_A · ln 2)")

# ── directorios ────────────────────────────────────────────────────────────────
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots/S2_noise" + ("_quick" if QUICK else ""))
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    if LATEX:
        fig.savefig(path + ".pgf", bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    print(f"  guardado: {path}.pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Métricas de calidad del gradiente
# ══════════════════════════════════════════════════════════════════════════════

def _flat(g):
    return jnp.array(jax.flatten_util.ravel_pytree(g)[0], float)


def cosine_similarity(g1, g2):
    """cos(ĝ, g_ex): mide solo dirección."""
    f1, f2 = _flat(g1), _flat(g2)
    return float(jnp.dot(f1, f2) / (jnp.linalg.norm(f1) * jnp.linalg.norm(f2) + 1e-30))


def gradient_slope(g_est, g_ref):
    """β = <ĝ, g_ref>/<g_ref, g_ref> = (‖ĝ‖/‖g_ref‖)·cos: ganancia efectiva (T_eff = β·T)."""
    f1, f2 = _flat(g_est), _flat(g_ref)
    return float(jnp.dot(f1, f2) / (jnp.dot(f2, f2) + 1e-30))


def gradient_norm_ratio(g_est, g_ref):
    """‖ĝ‖/‖g_ref‖."""
    return float(jnp.linalg.norm(_flat(g_est)) / (jnp.linalg.norm(_flat(g_ref)) + 1e-30))


def run_method(key, vstate, vstate_R, n, grad_ex):
    """n_rep llamadas -> dict de arrays: S2, cos, beta, nrm, t."""
    out = {k: [] for k in ("S2", "cos", "beta", "nrm", "t")}
    for rep in range(n_rep):
        t0 = time.time()
        S2, g = METHODS[key][3](vstate, vstate_R, n, rep)
        out["t"].append(time.time() - t0)
        out["S2"].append(float(S2)); out["cos"].append(cosine_similarity(g, grad_ex))
        out["beta"].append(gradient_slope(g, grad_ex)); out["nrm"].append(gradient_norm_ratio(g, grad_ex))
    return {k: np.array(v) for k, v in out.items()}


# ── Hamiltonianos ──────────────────────────────────────────────────────────────
H_extended = 0
for i in range(N):
    H_extended += GAMMA * sigmax(hi, i)
    H_extended += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)

# ── Modelos ────────────────────────────────────────────────────────────────────
model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16, activation=jax.nn.gelu)
sampler = nk.sampler.ARDirectSampler(hi)
vstate_ref = nk.vqs.MCState(sampler, model, n_samples=2**12 if QUICK else 2**16)


# ══════════════════════════════════════════════════════════════════════════════
# Entrenamiento de vstates y de las redes reversas (para Wang)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Entrenando vstates (forward) y vstate_R (reverse)")
print("=" * 60)

trained_vstates   = {}
trained_vstates_R = {}

for T in TEMPS:
    vstate = copy.deepcopy(vstate_ref)
    lr = optax.linear_schedule(0.05, 0.001, TRAIN_STEPS)
    print(f"\n  T={T:.2f}")
    free_energy_minimize(
        vstate, T, partition, H_extended, n_steps=TRAIN_STEPS,
        optimizer=optax.sgd(lr),
        plot=False, verbose=False,
        chunk_size=vstate.n_samples // 2,
    )
    trained_vstates[T] = vstate
    S2_ex, _ = renyi2_entropy_and_grad_exact(vstate, subsystem, hi)
    print(f"         S₂ exacto = {float(S2_ex):.4f}")

    model_R = nk.models.ARNNDense(hilbert=hi, layers=1, features=16, activation=jax.nn.gelu)
    vstate_R = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model_R, n_samples=2**12 if QUICK else 2**16)
    train_reverse_network(vstate, vstate_R, n_steps=REV_STEPS, batch=4096, lr=1e-3, verbose=True,
                          freq=max(1, REV_STEPS // 8))
    trained_vstates_R[T] = vstate_R


# ══════════════════════════════════════════════════════════════════════════════
# Sección 1: error vs entropía a PRESUPUESTO FIJO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"1. Error vs entropía (presupuesto fijo = {BUDGET_FIXED} evals)")
print("=" * 60)
nsz = sizes(BUDGET_FIXED)
print("tamaños:", nsz)

S2_exact_list = []
res1 = {k: [] for k in METHODS}          # res1[m][iT] = dict de arrays
for T, vstate in trained_vstates.items():
    S2_ex, grad_ex = renyi2_entropy_and_grad_exact(vstate, subsystem, hi)
    S2_ex = float(S2_ex); S2_exact_list.append(S2_ex)
    print(f"\n  T={T:.2f}  S₂={S2_ex:.4f}")
    for k in METHODS:
        r = run_method(k, vstate, trained_vstates_R[T], nsz[k], grad_ex)
        r["err"] = np.abs(r["S2"] - S2_ex); res1[k].append(r)
        print(f"    {k:7s} |ΔS₂|={r['err'].mean():.4f}  cos={r['cos'].mean():.4f}  "
              f"β={r['beta'].mean():.4f}±{r['beta'].std():.4f}  t={r['t'].mean():.2f}s")

S2_arr = np.array(S2_exact_list)
agg = lambda k, f: np.array([f(r) for r in res1[k]])

# ── tabla resumen ──
print("\n" + "=" * 90)
print("TABLA RESUMEN: β = (ĝ · g_ex)/(g_ex · g_ex)  [temperatura efectiva = β·T]")
print("=" * 90)
print(f"{'T':>5}  {'S₂':>7}  " + "  ".join(f"{METHODS[k][0] + ' β':>16}" for k in METHODS))
print("-" * 90)
for iT, T in enumerate(TEMPS):
    print(f"{T:>5.2f}  {S2_exact_list[iT]:>7.3f}  " + "  ".join(
        f"{res1[k][iT]['beta'].mean():>7.3f}±{res1[k][iT]['beta'].std():>6.3f}  " for k in METHODS))
print("=" * 90)
for k in METHODS:
    dev = np.abs(agg(k, lambda r: r["beta"].mean()) - 1.0)
    if np.any(dev > 0.10):
        print(f"  ⚠ {k}: |E[β]-1| > 0.10 en algún T (máx {dev.max():.3f} @ T={TEMPS[int(dev.argmax())]})")
        print(f"    → este método entrena a T_eff ≈ β·T, no a T")

# ── figura: error absoluto y relativo ──
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for k, (lab, mk, c, _) in METHODS.items():
    em = agg(k, lambda r: r["err"].mean())
    axes[0].plot(S2_arr, em, mk + "-", color=c, label=lab)
    axes[0].fill_between(S2_arr, agg(k, lambda r: np.percentile(r["err"], 25)),
                         agg(k, lambda r: np.percentile(r["err"], 75)), alpha=0.25, color=c)
    axes[1].plot(S2_arr, em / S2_arr, mk + "-", color=c, label=lab)
S2_ref = np.linspace(S2_arr.min(), S2_arr.max(), 100)
axes[0].plot(S2_ref, np.exp(S2_ref) / np.sqrt(nsz["ds"]), "k--", label=r"$e^{S_2}/\sqrt{N_\mathrm{pairs}}$")
axes[0].set_title(fr"$S_2$ absolute error (B = {BUDGET_FIXED})"); axes[0].set_ylabel(r"$|\Delta S_2|$")
axes[1].set_title("Relative error"); axes[1].set_ylabel(r"$|\Delta S_2| / S_2$")
for ax in axes:
    ax.set_xlabel(r"$S_2$ (exact)"); ax.set_yscale("log"); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
save_fig(fig, "s2_error_vs_entropy")

# ── figura: calidad del gradiente (cos + β) ──
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for k, (lab, mk, c, _) in METHODS.items():
    axes[0].errorbar(S2_arr, agg(k, lambda r: r["cos"].mean()), yerr=agg(k, lambda r: r["cos"].std()),
                     fmt=mk + "-", color=c, label=lab, capsize=3)
    axes[1].errorbar(S2_arr, agg(k, lambda r: r["beta"].mean()), yerr=agg(k, lambda r: r["beta"].std()),
                     fmt=mk + "-", color=c, label=lab, capsize=3)
axes[0].axhline(0.7, color="k", ls=":", alpha=0.4)
axes[0].set_ylabel(r"$\cos(\nabla S_2^\mathrm{est}, \nabla S_2^\mathrm{ex})$")
axes[0].set_title(r"Direction quality: cos($\hat g$, g$_\mathrm{ex}$)")
axes[1].axhline(1.0, color="k", ls="--", lw=1.5, alpha=0.6, label=r"$\beta=1$ (insesgado)")
axes[1].set_ylabel(r"$\beta = \langle \hat g, g_\mathrm{ex}\rangle / \langle g_\mathrm{ex}, g_\mathrm{ex}\rangle$")
axes[1].set_title(r"Magnitude+gain: $\beta$  ($T_\mathrm{eff}=\beta\, T$)")
for ax in axes:
    ax.set_xlabel(r"$S_2$ (exact)"); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "grad_quality_vs_entropy")


# ══════════════════════════════════════════════════════════════════════════════
# Sección 2: error vs presupuesto (S₂ baja y alta)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Error vs presupuesto (evaluaciones)")
print("=" * 60)


def inv_sqrt(n, alpha):
    return alpha / np.sqrt(n)


for label, T in [("low", TEMPS[0]), ("high", TEMPS[-1])]:
    vstate, vstate_R = trained_vstates[T], trained_vstates_R[T]
    S2_ex, grad_ex = renyi2_entropy_and_grad_exact(vstate, subsystem, hi)
    S2_ex = float(S2_ex)
    print(f"\n  S₂ {label} (T={T})  S₂={S2_ex:.4f}")
    res2 = {k: [] for k in METHODS}
    for B in BUDGET_LIST:
        nsz = sizes(B)
        print(f"\n  B={B}  → {nsz}")
        for k in METHODS:
            res2[k].append(run_method(k, vstate, vstate_R, nsz[k], grad_ex))
        print("    |ΔS₂|: " + "  ".join(f"{k}={np.abs(res2[k][-1]['S2'] - S2_ex).mean():.4f}" for k in METHODS))
        print("    β:     " + "  ".join(f"{k}={res2[k][-1]['beta'].mean():.3f}" for k in METHODS))

    B_arr = np.array(BUDGET_LIST, dtype=float)
    B_fit = np.geomspace(B_arr.min(), B_arr.max(), 200)

    # ── figura: error vs budget ──
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, (lab, mk, c, _) in METHODS.items():
        em = np.array([np.abs(r["S2"] - S2_ex).mean() for r in res2[k]])
        ax.loglog(B_arr, em, mk + "-", color=c, label=lab)
        if k in ("tilted", "ds") and np.all(np.isfinite(em)):
            alpha, _ = curve_fit(inv_sqrt, B_arr, em, p0=[1.0])
            ax.loglog(B_fit, inv_sqrt(B_fit, alpha), "--", color=c, label=rf"$\alpha={alpha[0]:.2f}/\sqrt{{B}}$")
    ax.set_xlabel(r"Ansatz evaluations $B$"); ax.set_ylabel(r"$|\Delta S_2|$")
    ax.set_title(fr"Error vs budget — {label} $S_2$ ($S_2={S2_ex:.2f}$)")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    save_fig(fig, f"s2_error_vs_budget_{label}")

    # ── figura: calidad del gradiente (cos + β) vs budget ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for k, (lab, mk, c, _) in METHODS.items():
        axes[0].semilogx(B_arr, [r["cos"].mean() for r in res2[k]], mk + "-", color=c, label=lab)
        axes[1].errorbar(B_arr, [r["beta"].mean() for r in res2[k]], yerr=[r["beta"].std() for r in res2[k]],
                         fmt=mk + "-", color=c, label=lab, capsize=3)
    axes[0].axhline(0.7, color="k", ls=":", alpha=0.4)
    axes[0].set_ylabel(r"$\cos(\nabla S_2^\mathrm{est}, \nabla S_2^\mathrm{ex})$")
    axes[0].set_title(f"Direction quality — {label} $S_2$")
    axes[1].axhline(1.0, color="k", ls="--", lw=1.5, alpha=0.6, label=r"$\beta=1$ (insesgado)")
    axes[1].set_xscale("log"); axes[1].set_ylabel(r"$\beta$")
    axes[1].set_title(fr"Gain $\beta$ — {label} $S_2$  ($T_\mathrm{{eff}}=\beta\, T$)")
    for ax in axes:
        ax.set_xlabel(r"Ansatz evaluations $B$"); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    save_fig(fig, f"grad_quality_vs_budget_{label}")

print("\nDone. Gráficas guardadas en", os.path.abspath(PLOTS_DIR))
