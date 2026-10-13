"""
sn_noise.py
===========
Comparación de estimadores de S_n y su gradiente para varios órdenes n
a presupuesto igualado.

Métodos incluidos (solo los que soportan n arbitrario):
  - Swap trick (direct sampling): renyin_entropy_and_grad_sampled
  - Wang CS (conditional sampling): renyin_wang_cs

Referencia exacta: renyin_entropy_exact / renyin_entropy_and_grad_exact

Coste por unidad (evaluaciones del ansatz):

  Swap: 3n por batch de n configuraciones
        1 (logP, sin grad) + 2n (una diag + una off por réplica)
        Coste por configuración = 3, independiente de n.

  Wang: n·N_total + 2n + 3 por secuencia
        n·N_total para el sampling AR (n pasos en A + n pasos en B)
        + 2n para las evaluaciones de log ψ en el loss
        + 3 para logP (amortizado sobre el batch).

Gráficas en ../plots/Sn_noise/ en formato .pgf y .pdf.
"""

import os
import copy
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax
from scipy.optimize import curve_fit
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
    renyin_entropy_exact,
    renyin_entropy_and_grad_exact,
    renyin_entropy_and_grad_sampled,
    renyin_wang_cs,
    train_reverse_network,
)
from src_renyi.training import free_energy_minimize

# ── directorios ────────────────────────────────────────────────────────────────
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots/Sn_noise")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path + ".pgf", bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    print(f"  guardado: {path}.pdf")
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


# ── configuración ──────────────────────────────────────────────────────────────
N              = 10
GAMMA          = -1.5
V              = -1.0
TEMPS          = [2.0, 2.5, 3.0, 3.5, 4.0]
N_ORDERS       = [2, 3, 4, 5, 6]
n_rep          = 10

BUDGET_FIXED   = 2**20
BUDGET_LIST    = [2**14, 2**16, 2**18, 2**20, 2**22]


# Coste por unidad (evaluaciones del ansatz)
def cost_swap(n):
    """Coste por batch de n configuraciones."""
    return 3 * n

def cost_wang(n, N_total):
    """Coste por secuencia."""
    return n * N_total + 2 * n + 3


# ── Hilbert ────────────────────────────────────────────────────────────────────
subsystem = list(range(N))
partition = subsystem

hi = nk.hilbert.Spin(s=1/2, N=2 * N)   # 2N sitios: A (N) + ancilla (N)
N_TOTAL = 2 * N

print(f"Presupuesto fijo: {BUDGET_FIXED} evaluaciones")
for n in N_ORDERS:
    print(f"  n={n}:  swap_n_samples = {BUDGET_FIXED // cost_swap(n):>8}   "
          f"wang_n_samples = {BUDGET_FIXED // cost_wang(n, N_TOTAL):>8}")


# ── Hamiltoniano del sistema físico (extendido trivialmente) ──────────────────
H_extended = 0
for i in range(N):
    H_extended += GAMMA * sigmax(hi, i)
    H_extended += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)


# ── Modelos ────────────────────────────────────────────────────────────────────
model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                            activation=jax.nn.gelu)
sampler = nk.sampler.ARDirectSampler(hi)
vstate_ref = nk.vqs.MCState(sampler, model, n_samples=2**16)


# ══════════════════════════════════════════════════════════════════════════════
# Entrenamiento de vstates y redes reversas (una vstate_R por T)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Entrenando vstates y vstate_R")
print("=" * 60)

trained_vstates   = {}
trained_vstates_R = {}

for T in TEMPS:
    vstate = copy.deepcopy(vstate_ref)
    lr = optax.linear_schedule(0.05, 0.001, 300)
    print(f"\n  T={T:.2f}")
    free_energy_minimize(
        vstate, T, partition, H_extended, n_steps=300,
        optimizer=optax.sgd(lr),
        plot=False, verbose=False,
        chunk_size=vstate.n_samples // 2,
    )
    trained_vstates[T] = vstate

    S2_ex, _ = renyin_entropy_exact(vstate, subsystem, 2)
    print(f"         S₂ exacto = {float(S2_ex):.4f}")

    # Red reversa (compartida entre n's: modela p, no depende de n)
    model_R = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                                   activation=jax.nn.gelu)
    sampler_R = nk.sampler.ARDirectSampler(hi)
    vstate_R = nk.vqs.MCState(sampler_R, model_R, n_samples=2**16)

    train_reverse_network(
        vstate, vstate_R,
        n_steps=4000, batch=4096, lr=1e-3,
        verbose=True, freq=500,
    )
    trained_vstates_R[T] = vstate_R


# ══════════════════════════════════════════════════════════════════════════════
# Referencias exactas de S_n y ∇S_n
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Referencias exactas (ED)")
print("=" * 60)

exact_Sn  = {n: {} for n in N_ORDERS}    # exact_Sn[n][T]  = S_n
exact_gSn = {n: {} for n in N_ORDERS}    # exact_gSn[n][T] = grad

for T, vstate in trained_vstates.items():
    for n in N_ORDERS:
        S_n, g = renyin_entropy_and_grad_exact(
            vstate, subsystem, n, hi_extended=hi, isFullSum=False
        )
        exact_Sn[n][T]  = float(S_n)
        exact_gSn[n][T] = g
        print(f"  T={T:.2f}  n={n}  S_n = {float(S_n):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Sección 1: comparación a PRESUPUESTO FIJO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"1. Comparación a presupuesto fijo = {BUDGET_FIXED} evals")
print("=" * 60)

# results[n][method][T] = dict con stats
results = {n: {"swap": {}, "wang": {}} for n in N_ORDERS}

for n in N_ORDERS:
    ns_swap = BUDGET_FIXED // cost_swap(n)
    ns_wang = BUDGET_FIXED // cost_wang(n, N_TOTAL)
    print(f"\n  ── n = {n} ──")
    print(f"     swap: n_samples = {ns_swap}   "
          f"wang: n_samples = {ns_wang}")

    for T, vstate in trained_vstates.items():
        vstate_R = trained_vstates_R[T]
        S_n_ex   = exact_Sn[n][T]
        g_n_ex   = exact_gSn[n][T]

        s_sw, cos_sw     = [], []
        s_wang, cos_wang = [], []

        for rep in range(n_rep):
            # Swap
            S_est, g = renyin_entropy_and_grad_sampled(
                vstate, subsystem, n, n_samples=ns_swap, key=rep
            )
            s_sw.append(float(S_est))
            cos_sw.append(cosine_similarity(g, g_n_ex))

            # Wang CS
            S_est, g = renyin_wang_cs(
                vstate, vstate_R, subsystem, n,
                n_samples=ns_wang, key=rep
            )
            s_wang.append(float(S_est))
            cos_wang.append(cosine_similarity(g, g_n_ex))

        s_sw   = np.array(s_sw)
        s_wang = np.array(s_wang)
        err_sw   = np.abs(s_sw   - S_n_ex)
        err_wang = np.abs(s_wang - S_n_ex)

        results[n]["swap"][T] = {
            "S_n": S_n_ex,
            "err_mean": err_sw.mean(),
            "err_p25":  np.percentile(err_sw, 25),
            "err_p75":  np.percentile(err_sw, 75),
            "cos_mean": float(np.mean(cos_sw)),
            "cos_std":  float(np.std(cos_sw)),
        }
        results[n]["wang"][T] = {
            "S_n": S_n_ex,
            "err_mean": err_wang.mean(),
            "err_p25":  np.percentile(err_wang, 25),
            "err_p75":  np.percentile(err_wang, 75),
            "cos_mean": float(np.mean(cos_wang)),
            "cos_std":  float(np.std(cos_wang)),
        }

        print(f"     T={T:.2f}  S_n={S_n_ex:.4f}  "
              f"swap |Δ|/S={err_sw.mean()/S_n_ex:.4f} "
              f"cos={np.mean(cos_sw):.3f}  |  "
              f"wang |Δ|/S={err_wang.mean()/S_n_ex:.4f} "
              f"cos={np.mean(cos_wang):.3f}")


# ── Figura 1: S_n exacto vs T (una curva por n) ──
fig, ax = plt.subplots(figsize=(6, 3.8))
cmap = plt.cm.viridis
for k, n in enumerate(N_ORDERS):
    Ts = np.array(TEMPS)
    Ss = np.array([exact_Sn[n][T] for T in TEMPS])
    ax.plot(Ts, Ss, "o-", color=cmap(k / max(1, len(N_ORDERS) - 1)),
            label=f"$S_{n}$")
ax.axhline(N * np.log(2), color="k", ls=":", alpha=0.4,
           label=r"$N_A \ln 2$")
ax.set_xlabel(r"$T$")
ax.set_ylabel(r"$S_n$")
ax.set_title(fr"Exact Rényi entropies (N={N})")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
save_fig(fig, "Sn_exact_vs_T")


# ── Figura 2: error relativo vs T, un panel por n ──
fig, axes = plt.subplots(1, len(N_ORDERS),
                          figsize=(4.5 * len(N_ORDERS), 3.5),
                          sharey=True, squeeze=False)
axes = axes[0]
for ax, n in zip(axes, N_ORDERS):
    Ts = np.array(TEMPS)
    err_sw   = np.array([results[n]["swap"][T]["err_mean"] for T in TEMPS])
    err_wang = np.array([results[n]["wang"][T]["err_mean"] for T in TEMPS])
    S_ref    = np.array([results[n]["swap"][T]["S_n"]    for T in TEMPS])

    sw_p25 = np.array([results[n]["swap"][T]["err_p25"] for T in TEMPS]) / S_ref
    sw_p75 = np.array([results[n]["swap"][T]["err_p75"] for T in TEMPS]) / S_ref
    wa_p25 = np.array([results[n]["wang"][T]["err_p25"] for T in TEMPS]) / S_ref
    wa_p75 = np.array([results[n]["wang"][T]["err_p75"] for T in TEMPS]) / S_ref

    ax.plot(Ts, err_sw / S_ref,   "o-", color="C0", label="Swap")
    ax.fill_between(Ts, sw_p25, sw_p75, alpha=0.2, color="C0")
    ax.plot(Ts, err_wang / S_ref, "d-", color="C3", label="Wang CS")
    ax.fill_between(Ts, wa_p25, wa_p75, alpha=0.2, color="C3")

    ax.set_xlabel(r"$T$")
    if ax is axes[0]:
        ax.set_ylabel(r"$|\Delta S_n| / S_n$")
    ax.set_yscale("log")
    ax.set_title(fr"$n = {n}$")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)

fig.suptitle(fr"Relative error at $B = {BUDGET_FIXED}$", y=1.02)
fig.tight_layout()
save_fig(fig, "Sn_rel_error_vs_T")


# ── Figura 3: cos(∇S_n) vs T, un panel por n ──
fig, axes = plt.subplots(1, len(N_ORDERS),
                          figsize=(4.5 * len(N_ORDERS), 3.5),
                          sharey=True, squeeze=False)
axes = axes[0]
for ax, n in zip(axes, N_ORDERS):
    Ts = np.array(TEMPS)
    cos_sw   = np.array([results[n]["swap"][T]["cos_mean"] for T in TEMPS])
    cos_wang = np.array([results[n]["wang"][T]["cos_mean"] for T in TEMPS])
    std_sw   = np.array([results[n]["swap"][T]["cos_std"]  for T in TEMPS])
    std_wang = np.array([results[n]["wang"][T]["cos_std"]  for T in TEMPS])

    ax.errorbar(Ts, cos_sw,   yerr=std_sw,   fmt="o-", color="C0",
                label="Swap", capsize=3)
    ax.errorbar(Ts, cos_wang, yerr=std_wang, fmt="d-", color="C3",
                label="Wang CS", capsize=3)
    ax.axhline(0.7, color="k", ls=":", alpha=0.4)
    ax.set_xlabel(r"$T$")
    if ax is axes[0]:
        ax.set_ylabel(r"$\cos(\nabla S_n^\mathrm{est}, "
                       r"\nabla S_n^\mathrm{ex})$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(fr"$n = {n}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle(fr"Gradient quality at $B = {BUDGET_FIXED}$", y=1.02)
fig.tight_layout()
save_fig(fig, "Sn_grad_cos_vs_T")


# ══════════════════════════════════════════════════════════════════════════════
# Sección 2: error vs presupuesto, para cada n
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Error vs presupuesto para cada n")
print("=" * 60)

T_scan = TEMPS[len(TEMPS) // 2]        # T central
vstate_scan   = trained_vstates[T_scan]
vstate_R_scan = trained_vstates_R[T_scan]

for n in N_ORDERS:
    print(f"\n  ── n = {n},  T = {T_scan} ──")
    S_n_ex = exact_Sn[n][T_scan]
    g_n_ex = exact_gSn[n][T_scan]

    err_sw_list, err_wang_list = [], []
    cos_sw_list, cos_wang_list = [], []

    for B in BUDGET_LIST:
        ns_swap = max(4, B // cost_swap(n))
        ns_wang = max(4, B // cost_wang(n, N_TOTAL))

        s_sw, cos_sw     = [], []
        s_wang, cos_wang = [], []

        for rep in range(n_rep):
            S_est, g = renyin_entropy_and_grad_sampled(
                vstate_scan, subsystem, n, n_samples=ns_swap, key=rep
            )
            s_sw.append(float(S_est))
            cos_sw.append(cosine_similarity(g, g_n_ex))

            S_est, g = renyin_wang_cs(
                vstate_scan, vstate_R_scan, subsystem, n,
                n_samples=ns_wang, key=rep
            )
            s_wang.append(float(S_est))
            cos_wang.append(cosine_similarity(g, g_n_ex))

        err_sw   = np.abs(np.array(s_sw)   - S_n_ex).mean()
        err_wang = np.abs(np.array(s_wang) - S_n_ex).mean()
        err_sw_list.append(err_sw)
        err_wang_list.append(err_wang)
        cos_sw_list.append(float(np.mean(cos_sw)))
        cos_wang_list.append(float(np.mean(cos_wang)))

        print(f"     B={B:9d}  "
              f"swap: |Δ|={err_sw:.4f} cos={np.mean(cos_sw):.3f}  |  "
              f"wang: |Δ|={err_wang:.4f} cos={np.mean(cos_wang):.3f}")

    B_arr = np.array(BUDGET_LIST, dtype=float)

    def inv_sqrt(B, alpha):
        return alpha / np.sqrt(B)

    try:
        alpha_sw, _ = curve_fit(inv_sqrt, B_arr, err_sw_list,   p0=[1.0])
        alpha_wa, _ = curve_fit(inv_sqrt, B_arr, err_wang_list, p0=[1.0])
    except Exception:
        alpha_sw, alpha_wa = [0.0], [0.0]

    # ── |Δ S_n| vs B ──
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.loglog(B_arr, err_sw_list,   "o-", color="C0", label="Swap")
    ax.loglog(B_arr, err_wang_list, "d-", color="C3", label="Wang CS")

    B_fit = np.geomspace(B_arr.min(), B_arr.max(), 200)
    if alpha_sw[0] > 0:
        ax.loglog(B_fit, inv_sqrt(B_fit, alpha_sw[0]), "--", color="C0",
                  label=rf"$\alpha={alpha_sw[0]:.2f}/\sqrt{{B}}$")
    if alpha_wa[0] > 0:
        ax.loglog(B_fit, inv_sqrt(B_fit, alpha_wa[0]), "--", color="C3",
                  label=rf"$\alpha={alpha_wa[0]:.2f}/\sqrt{{B}}$")

    ax.set_xlabel(r"Ansatz evaluations $B$")
    ax.set_ylabel(r"$|\Delta S_n|$")
    ax.set_title(fr"$n={n}$, $T={T_scan}$, $S_n^{{ex}}={S_n_ex:.2f}$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    save_fig(fig, f"Sn_error_vs_budget_n{n}")

    # ── cos vs B ──
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.semilogx(B_arr, cos_sw_list,   "o-", color="C0", label="Swap")
    ax.semilogx(B_arr, cos_wang_list, "d-", color="C3", label="Wang CS")
    ax.axhline(0.7, color="k", ls=":", alpha=0.4)
    ax.set_xlabel(r"Ansatz evaluations $B$")
    ax.set_ylabel(r"$\cos(\nabla S_n^\mathrm{est}, \nabla S_n^\mathrm{ex})$")
    ax.set_title(fr"Gradient quality, $n={n}$, $T={T_scan}$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"Sn_grad_cos_vs_budget_n{n}")


print("\nDone. Gráficas guardadas en", os.path.abspath(PLOTS_DIR))