"""
s2_noise_large.py
====================
Comparación de swap trick, DRUT y Wang CS para tamaños donde ED no es posible,
a COSTE IGUALADO (evaluaciones del ansatz).

Estructura:
  1. Entrena vstates a distintas temperaturas con Wang CS (fase única).
     La red reversa se entrena dentro de la fase Wang (initial_train + warm_start).
  2. Compara a un presupuesto fijo (para ver tendencia con T).
  3. Barre el presupuesto B y mide error vs B.
  4. Analiza: error, ESS, cos, y coste computacional.

Coste en evaluaciones del ansatz (no en segundos, para reproducibilidad):
  - Swap: 4 × n_samples
  - DRUT: n_chains × n_lambda × (4 × n_sweeps × n_props + 4)
  - Wang: (2 × N_total + 7) × n_samples
          (2·N_total de sampling AR en las 4 réplicas correlacionadas,
           3 de logP, 4 de Ω)

Gráficas en ../plots/S2_noise_large/ en formato .pdf y .pgf
"""

import os
import time
import copy
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
})
import matplotlib.pyplot as plt

from src_renyi.entropy import (
    renyi2_entropy_and_grad_sampled,
    renyi2_drut_sampling2,
    renyi2_wang_cs,
)
from src_renyi.training import free_energy_minimize_phases

# ══════════════════════════════════════════════════════════════════════════════
# Coste en evaluaciones del ansatz
# ══════════════════════════════════════════════════════════════════════════════
def cost_swap(n_samples):
    """Swap trick: ψ(s1), ψ(s2), ψ(s1'), ψ(s2') por muestra."""
    return 4 * n_samples

def cost_drut(n_chains, n_lambda, n_sweeps, n_props):
    """
    DRUT: por cadena y por λ,
      - n_sweeps × n_props pasos MH × 4 ψ evals por paso
      - 4 ψ evals para lnR final
    """
    per_chain_per_lam = 4 * n_sweeps * n_props + 4
    return n_chains * n_lambda * per_chain_per_lam

def cost_wang(n_samples, N_total):
    """
    Wang CS: por muestra,
      - 2·N_total evals (sampling AR de las 4 réplicas)
      - 3 evals (3 llamadas a conditionals_log_psi para logP)
      - 4 evals (Ω en c11, c21, c22, c12)
    """
    return (2 * N_total + 7) * n_samples

# ══════════════════════════════════════════════════════════════════════════════
# Configuración
# ══════════════════════════════════════════════════════════════════════════════
N              = 20
N_A            = N
GAMMA          = -1.5
V              = -1.0
TEMPS          = [3.0, 4.0, 5.0, 7.0]

n_rep          = 10

# DRUT hyperparameters
DRUT_N_CHAINS  = 512
DRUT_N_LAMBDA  = 15
DRUT_N_SWEEPS  = 100
DRUT_N_PROPS   = 2 * (N + N_A)

# Coste por cadena de DRUT
DRUT_COST_PER_CHAIN = DRUT_N_LAMBDA * (4 * DRUT_N_SWEEPS * DRUT_N_PROPS + 4)

# ── Hiperparámetros de entrenamiento (Wang único) ──────────────────────────────
N_STEPS_WANG  = 400
LR_WANG       = 1e-3

INIT_N_STEPS  = 3000
INIT_BATCH    = 4096
LR_INIT       = 1e-3

WARM_EVERY    = 5
WARM_N_STEPS  = 20
WARM_BATCH    = 2**14
LR_WARM       = 5e-4

# ── Número de sitios totales del sistema purificado ──
N_TOTAL = N + N_A

print(f"DRUT: {DRUT_N_CHAINS} chains × {DRUT_N_LAMBDA} λ × {DRUT_N_SWEEPS} sweeps × {DRUT_N_PROPS} props")
print(f"  → coste por cadena: {DRUT_COST_PER_CHAIN:,} evals")
print(f"  → coste total (n_chains={DRUT_N_CHAINS}): "
      f"{DRUT_COST_PER_CHAIN * DRUT_N_CHAINS:,} evals")

WANG_COST_PER_SAMPLE = 2 * N_TOTAL + 7
print(f"\nWang CS: {WANG_COST_PER_SAMPLE} evals por muestra (N_total = {N_TOTAL})")

# Presupuestos para el barrido
BUDGET_LIST = [2**14, 2**16, 2**18, 2**20, 2**22]
print(f"\nPresupuestos a barrer: {BUDGET_LIST}")

# Para el análisis a presupuesto fijo (Sección 2), usamos el máximo
BUDGET_FIXED = 2**22

# ── directorios ────────────────────────────────────────────────────────────────
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots/S2_noise_large")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path + ".pgf", bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    print(f"  guardado: {path}.pgf")
    plt.close(fig)


def cosine_similarity(g1, g2):
    flat1, _ = jax.flatten_util.ravel_pytree(g1)
    flat2, _ = jax.flatten_util.ravel_pytree(g2)
    flat1 = jnp.array(flat1, float)
    flat2 = jnp.array(flat2, float)
    return float(
        jnp.dot(flat1, flat2) /
        (jnp.linalg.norm(flat1) * jnp.linalg.norm(flat2) + 1e-30)
    )


def grad_norm(g):
    flat, _ = jax.flatten_util.ravel_pytree(g)
    return float(jnp.linalg.norm(jnp.array(flat, float)))


partition = list(range(N))

# ── hilbert y hamiltoniano ─────────────────────────────────────────────────────
hi = nk.hilbert.Spin(s=1/2, N=N + N_A)

H_extended = 0
for i in range(N):
    H_extended += GAMMA * sigmax(hi, i)
    H_extended += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)

model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16, activation=jax.nn.gelu)
sampler    = nk.sampler.ARDirectSampler(hi)
vstate_ref = nk.vqs.MCState(sampler, model, n_samples=2**16)


# ══════════════════════════════════════════════════════════════════════════════
# Entrenamiento (Wang CS como única fase)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"Entrenando vstates  (N = {N}, N_A = {N_A}) con Wang CS")
print("=" * 70)

trained_vstates   = {}
trained_vstates_R = {}

for i, T in enumerate(TEMPS):
    vstate = copy.deepcopy(vstate_ref)
    print(f"\n  T = {T:.2f}")

    # ── Red reversa (necesaria para Wang) ──
    model_R = nk.models.ARNNDense(
        hilbert=hi, layers=1, features=16, activation=jax.nn.gelu
    )
    vstate_R = nk.vqs.MCState(
        nk.sampler.ARDirectSampler(hi), model_R, n_samples=2**16
    )

    # ── Fase única: Wang CS ──
    phases = [
        {
            "method": "wang",
            "n_steps": N_STEPS_WANG,
            "optimizer": optax.adam(LR_WANG),
            "vstate_R": vstate_R,
            "initial_train": {
                "n_steps": INIT_N_STEPS,
                "batch":   INIT_BATCH,
                "lr":      LR_INIT,
            },
            "warm_start": {
                "every":   WARM_EVERY,
                "n_steps": WARM_N_STEPS,
                "lr":      LR_WARM,
                "batch":   WARM_BATCH,
            },
            "wang_kwargs": {"n_samples": vstate.n_samples},
        },
    ]

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

# Results[T][method][B] = {"S2": [...], "grads": [...], "time": [...], "cost": ...}
results = {T: {"swap": {}, "drut": {}, "wang": {}} for T in TEMPS}

for T, vstate in trained_vstates.items():
    vstate_R = trained_vstates_R[T]
    print(f"\n  T = {T:.2f}")

    for B in BUDGET_LIST:
        # Derivar tamaños
        ns_sw       = max(1,  B // 4)
        n_chains_dr = max(8,  B // DRUT_COST_PER_CHAIN)
        ns_wang     = max(16, B // WANG_COST_PER_SAMPLE)

        s2_sw, s2_dr, s2_wa       = [], [], []
        cos_sw_self, cos_dr_self, cos_wa_self = [], [], []
        t_sw, t_dr, t_wa          = [], [], []
        grads_sw, grads_dr, grads_wa = [], [], []

        for rep in range(n_rep):
            # ── Swap ─────────────────────────────────────────────
            t0 = time.time()
            S2, g = renyi2_entropy_and_grad_sampled(
                vstate, partition, ns_sw, chunk_size=ns_sw // 64 * 8
            )
            t_sw.append(time.time() - t0)
            s2_sw.append(float(S2))
            grads_sw.append(g)

            # ── DRUT ─────────────────────────────────────────────
            t0 = time.time()
            S2, g = renyi2_drut_sampling2(
                vstate, partition,
                n_chains=n_chains_dr,
                n_lambda=DRUT_N_LAMBDA,
                n_sweeps_per_lam=DRUT_N_SWEEPS,
                n_props_per_sweep=DRUT_N_PROPS,
                K=2,
                debug=False,
            )
            t_dr.append(time.time() - t0)
            s2_dr.append(float(S2))
            grads_dr.append(g)

            # ── Wang CS ──────────────────────────────────────────
            t0 = time.time()
            S2, g = renyi2_wang_cs(
                vstate, vstate_R, partition,
                n_samples=ns_wang, key=rep,
            )
            t_wa.append(time.time() - t0)
            s2_wa.append(float(S2))
            grads_wa.append(g)

        # Autoconsistencia
        for i_ in range(n_rep):
            for j_ in range(i_ + 1, n_rep):
                cos_sw_self.append(cosine_similarity(grads_sw[i_], grads_sw[j_]))
                cos_dr_self.append(cosine_similarity(grads_dr[i_], grads_dr[j_]))
                cos_wa_self.append(cosine_similarity(grads_wa[i_], grads_wa[j_]))

        # Guardar
        results[T]["swap"][B] = {
            "S2": s2_sw, "grads": grads_sw, "time": t_sw,
            "cost": cost_swap(ns_sw), "n_size": ns_sw,
            "cos_self": np.mean(cos_sw_self),
        }
        results[T]["drut"][B] = {
            "S2": s2_dr, "grads": grads_dr, "time": t_dr,
            "cost": cost_drut(n_chains_dr, DRUT_N_LAMBDA,
                              DRUT_N_SWEEPS, DRUT_N_PROPS),
            "n_size": n_chains_dr,
            "cos_self": np.mean(cos_dr_self),
        }
        results[T]["wang"][B] = {
            "S2": s2_wa, "grads": grads_wa, "time": t_wa,
            "cost": cost_wang(ns_wang, N_TOTAL), "n_size": ns_wang,
            "cos_self": np.mean(cos_wa_self),
        }

        print(f"    B={B:>10,}  swap(n={ns_sw:>8,}): "
              f"⟨S₂⟩={np.mean(s2_sw):.4f}  cos_self={np.mean(cos_sw_self):.3f}  "
              f"t={np.mean(t_sw):.3f}s")
        print(f"    {'':>14}  drut(n_ch={n_chains_dr:>5}): "
              f"⟨S₂⟩={np.mean(s2_dr):.4f}  cos_self={np.mean(cos_dr_self):.3f}  "
              f"t={np.mean(t_dr):.3f}s")
        print(f"    {'':>14}  wang(n={ns_wang:>8,}): "
              f"⟨S₂⟩={np.mean(s2_wa):.4f}  cos_self={np.mean(cos_wa_self):.3f}  "
              f"t={np.mean(t_wa):.3f}s")


# ══════════════════════════════════════════════════════════════════════════════
# Sección 2: análisis a presupuesto fijo
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"2. Análisis a presupuesto fijo B = {BUDGET_FIXED:,}")
print("=" * 70)

B_selected = min(BUDGET_LIST, key=lambda b: abs(b - BUDGET_FIXED))
print(f"Usando B = {B_selected:,}")

S2_swap_mean, S2_swap_std = [], []
S2_drut_mean, S2_drut_std = [], []
S2_wang_mean, S2_wang_std = [], []
cos_self_swap, cos_self_drut, cos_self_wang = [], [], []
t_swap, t_drut, t_wang = [], [], []

for T in TEMPS:
    r_sw = results[T]["swap"][B_selected]
    r_dr = results[T]["drut"][B_selected]
    r_wa = results[T]["wang"][B_selected]

    S2_swap_mean.append(np.mean(r_sw["S2"])); S2_swap_std.append(np.std(r_sw["S2"]))
    S2_drut_mean.append(np.mean(r_dr["S2"])); S2_drut_std.append(np.std(r_dr["S2"]))
    S2_wang_mean.append(np.mean(r_wa["S2"])); S2_wang_std.append(np.std(r_wa["S2"]))
    cos_self_swap.append(r_sw["cos_self"])
    cos_self_drut.append(r_dr["cos_self"])
    cos_self_wang.append(r_wa["cos_self"])
    t_swap.append(np.mean(r_sw["time"]))
    t_drut.append(np.mean(r_dr["time"]))
    t_wang.append(np.mean(r_wa["time"]))

S2_consensus = (np.array(S2_swap_mean) + np.array(S2_drut_mean) +
                np.array(S2_wang_mean)) / 3.0

# ── Figura: S₂ vs T ──
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.errorbar(TEMPS, S2_swap_mean, yerr=S2_swap_std,
            fmt='o-', color='C0', label='Swap trick', capsize=3)
ax.errorbar(TEMPS, S2_drut_mean, yerr=S2_drut_std,
            fmt='s-', color='C2', label='DRUT', capsize=3)
ax.errorbar(TEMPS, S2_wang_mean, yerr=S2_wang_std,
            fmt='d-', color='C3', label='Wang CS', capsize=3)
ax.plot(TEMPS, S2_consensus, 'k--', alpha=0.4, label='consenso')
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$S_2$')
ax.set_title(fr'$S_2$ vs $T$ ($N={N}$, $B={B_selected:,}$)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "S2_vs_T")

# ── Figura: desviación estándar de S₂ (varianza del estimador) ──
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(TEMPS, S2_swap_std, 'o-', color='C0', label='Swap trick')
ax.plot(TEMPS, S2_drut_std, 's-', color='C2', label='DRUT')
ax.plot(TEMPS, S2_wang_std, 'd-', color='C3', label='Wang CS')
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$\mathrm{std}(S_2)$ entre réplicas')
ax.set_title('Varianza del estimador de $S_2$')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
fig.tight_layout()
save_fig(fig, "S2_std_vs_T")

# ── Figura: cos de autoconsistencia ──
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(TEMPS, cos_self_swap, 'o-', color='C0', label='Swap trick')
ax.plot(TEMPS, cos_self_drut, 's-', color='C2', label='DRUT')
ax.plot(TEMPS, cos_self_wang, 'd-', color='C3', label='Wang CS')
ax.axhline(0.7, color='r', ls=':', alpha=0.5, label='umbral 0.7')
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$\langle \cos(\nabla S_2^{(i)}, \nabla S_2^{(j)}) \rangle$')
ax.set_title('Autoconsistencia del gradiente')
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "cos_self_vs_T")

# ── Figura: tiempo por llamada ──
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(TEMPS, t_swap, 'o-', color='C0', label='Swap trick')
ax.plot(TEMPS, t_drut, 's-', color='C2', label='DRUT')
ax.plot(TEMPS, t_wang, 'd-', color='C3', label='Wang CS')
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$t$ (s)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
ax.set_title('Tiempo por llamada (presupuesto fijo)')
fig.tight_layout()
save_fig(fig, "time_vs_T")


# ══════════════════════════════════════════════════════════════════════════════
# Sección 3: LA FIGURA CLAVE — error vs coste
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. Error vs coste (figura clave)")
print("=" * 70)

fig, axes = plt.subplots(1, len(TEMPS), figsize=(4 * len(TEMPS), 3.5),
                          sharey=False)

for ax, T in zip(axes, TEMPS):
    costs_sw, errs_sw = [], []
    costs_dr, errs_dr = [], []
    costs_wa, errs_wa = [], []

    for B in BUDGET_LIST:
        r_sw = results[T]["swap"][B]
        r_dr = results[T]["drut"][B]
        r_wa = results[T]["wang"][B]

        errs_sw.append(np.std(r_sw["S2"]))
        errs_dr.append(np.std(r_dr["S2"]))
        errs_wa.append(np.std(r_wa["S2"]))

        costs_sw.append(r_sw["cost"])
        costs_dr.append(r_dr["cost"])
        costs_wa.append(r_wa["cost"])

    ax.loglog(costs_sw, errs_sw, 'o-', color='C0', label='Swap')
    ax.loglog(costs_dr, errs_dr, 's-', color='C2', label='DRUT')
    ax.loglog(costs_wa, errs_wa, 'd-', color='C3', label='Wang CS')
    ax.set_xlabel(r'$B$ (evaluaciones del ansatz)')
    ax.set_ylabel(r'$\sigma(S_2)$ entre réplicas')
    ax.set_title(fr'$T = {T}$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

fig.tight_layout()
save_fig(fig, "error_vs_cost_per_T")


# ══════════════════════════════════════════════════════════════════════════════
# Sección 4: consistencia mutua entre métodos
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. Consistencia mutua (gradientes medios)")
print("=" * 70)

cos_sw_dr, cos_sw_wa, cos_dr_wa = [], [], []
for T in TEMPS:
    r_sw = results[T]["swap"][B_selected]
    r_dr = results[T]["drut"][B_selected]
    r_wa = results[T]["wang"][B_selected]

    def mean_grad(grads):
        return jnp.mean(jnp.stack([
            jnp.array(jax.flatten_util.ravel_pytree(g)[0], float)
            for g in grads
        ]), axis=0)

    g_sw = mean_grad(r_sw["grads"])
    g_dr = mean_grad(r_dr["grads"])
    g_wa = mean_grad(r_wa["grads"])

    def cos_(a, b):
        return float(jnp.dot(a, b) /
                     (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))

    c1, c2, c3 = cos_(g_sw, g_dr), cos_(g_sw, g_wa), cos_(g_dr, g_wa)
    cos_sw_dr.append(c1); cos_sw_wa.append(c2); cos_dr_wa.append(c3)
    print(f"  T = {T:.2f}   cos(swap,drut) = {c1:.4f}   "
          f"cos(swap,wang) = {c2:.4f}   cos(drut,wang) = {c3:.4f}")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(TEMPS, cos_sw_dr, 'o-', color='C0', label=r'$\cos(\nabla S_2^{sw}, \nabla S_2^{dr})$')
ax.plot(TEMPS, cos_sw_wa, 's-', color='C1', label=r'$\cos(\nabla S_2^{sw}, \nabla S_2^{wa})$')
ax.plot(TEMPS, cos_dr_wa, 'd-', color='C2', label=r'$\cos(\nabla S_2^{dr}, \nabla S_2^{wa})$')
ax.axhline(0.7, color='r', ls=':', alpha=0.5, label='umbral 0.7')
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$\cos$')
ax.set_title('Consistencia mutua del gradiente')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "cos_mutual_vs_T")


# ══════════════════════════════════════════════════════════════════════════════
# Tabla resumen
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Resumen final")
print("=" * 70)
print(f"\n{'T':>5}  {'método':>7}  {'S₂':>8}  {'σ(S₂)':>8}  "
      f"{'cos_self':>9}  {'t(s)':>8}  {'coste':>15}  {'n_size':>10}")
for T in TEMPS:
    r_sw = results[T]["swap"][B_selected]
    r_dr = results[T]["drut"][B_selected]
    r_wa = results[T]["wang"][B_selected]
    for name, r in [("swap", r_sw), ("drut", r_dr), ("wang", r_wa)]:
        print(f"{T:>5.1f}  {name:>7}  {np.mean(r['S2']):>8.4f}  "
              f"{np.std(r['S2']):>8.4f}  {r['cos_self']:>9.3f}  "
              f"{np.mean(r['time']):>8.3f}  {r['cost']:>15,}  "
              f"{r['n_size']:>10,}")
    print()

print("\nDone. Gráficas guardadas en", os.path.abspath(PLOTS_DIR))