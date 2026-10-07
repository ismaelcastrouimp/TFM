"""
s2_noise.py
===========
Comparación justa de estimadores de S₂ y su gradiente a presupuesto igualado.

Presupuesto común: número de evaluaciones del ansatz (forward passes).

  - Swap trick: 4 · N_samples evaluaciones.
  - λ-i (reweighting): 4 · N_lambda · N_samples.
  - DRUT: n_chains × n_lambda × n_sweeps × n_props × 2.
  - Wang CS: (2·N_total + 7) · N_samples  (2·N_total de sampling AR +
             3 de logP + 4 de Ω).

Gráficas en ../plots/S2_noise/ en formato .pgf
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
    renyi2_entropy_and_grad_sampled,
    renyi2_entropy_and_grad_lambda_integral,
    renyi2_entropy_and_grad_exact,
    renyi2_drut_sampling2,
    renyi2_wang_cs,
    train_reverse_network,
    _per_site_log_psi,
)
from src_renyi.training import free_energy_minimize

# ── directorios ────────────────────────────────────────────────────────────────
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots/S2_noise")
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
TEMPS          = [2.0]
n_rep          = 10

BUDGET_FIXED   = 2**20
BUDGET_LIST    = [2**14, 2**16, 2**18, 2**20, 2**22]

# DRUT
DRUT_N_LAMBDA = 15
DRUT_N_SWEEPS = 100
DRUT_N_PROPS  = 2 * N
DRUT_K        = 2
DRUT_COST_PER_CHAIN = DRUT_N_LAMBDA * DRUT_N_SWEEPS * DRUT_N_PROPS * 2

# λ-i
LAMBDA_I_N_LAMBDA = 60

# ── Hilbert ────────────────────────────────────────────────────────────────────
subsystem = list(range(N))
partition = subsystem

hi = nk.hilbert.Spin(s=1/2, N=2 * N)   # N_S = N, N_A = N
N_TOTAL = 2 * N

# Coste por muestra de Wang CS en evaluaciones de ansatz
WANG_COST_PER_SAMPLE = 2 * N_TOTAL + 7

print(f"Presupuesto fijo: {BUDGET_FIXED} evaluaciones")
print(f"  → swap:  n_samples = {BUDGET_FIXED // 4}")
print(f"  → drut:  n_chains  = {BUDGET_FIXED // DRUT_COST_PER_CHAIN}")
print(f"  → wang:  n_samples = {BUDGET_FIXED // WANG_COST_PER_SAMPLE}")


# ── Hamiltonianos ──────────────────────────────────────────────────────────────
hi_sys = nk.hilbert.Spin(s=1/2, N=N)
H_sys = 0
H_extended = 0
for i in range(N):
    H_sys += GAMMA * sigmax(hi_sys, i)
    H_sys += V * sigmaz(hi_sys, i) @ sigmaz(hi_sys, (i + 1) % N)
    H_extended += GAMMA * sigmax(hi, i)
    H_extended += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)


# ── Modelos ────────────────────────────────────────────────────────────────────
model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                            activation=jax.nn.gelu)
sampler = nk.sampler.ARDirectSampler(hi)
vstate_ref = nk.vqs.MCState(sampler, model, n_samples=2**16)


# ══════════════════════════════════════════════════════════════════════════════
# Entrenamiento de vstates y de las redes reversas
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Entrenando vstates (forward) y vstate_R (reverse)")
print("=" * 60)

trained_vstates    = {}
trained_vstates_R  = {}

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

    S2_ex, _ = renyi2_entropy_and_grad_exact(vstate, subsystem, hi)
    print(f"         S₂ exacto = {float(S2_ex):.4f}")

    # ── Red reversa ──
    model_R = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                                activation=jax.nn.gelu)   # ← sin override
    sampler_R = nk.sampler.ARDirectSampler(hi)
    vstate_R = nk.vqs.MCState(sampler_R, model_R, n_samples=2**16)

    train_reverse_network(
        vstate, vstate_R,
        n_steps=4000, batch=4096, lr=1e-3,
        verbose=True, freq=500,
    )
    trained_vstates_R[T] = vstate_R


# ══════════════════════════════════════════════════════════════════════════════
# Sección 1: error vs entropía a PRESUPUESTO FIJO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"1. Error vs entropía (presupuesto fijo = {BUDGET_FIXED} evals)")
print("=" * 60)

n_samples_sw       = BUDGET_FIXED // 4
LAMBDA_I_COST_FACTOR = 4 * LAMBDA_I_N_LAMBDA
n_samples_lam_i    = BUDGET_FIXED // LAMBDA_I_COST_FACTOR
n_chains_dr        = max(16, BUDGET_FIXED // DRUT_COST_PER_CHAIN)
n_samples_wang     = BUDGET_FIXED // WANG_COST_PER_SAMPLE
print(f"n_samples (swap) = {n_samples_sw}")
print(f"n_chains  (drut) = {n_chains_dr}")
print(f"n_samples (wang) = {n_samples_wang}")

S2_exact_list = []
err_swap_list,      cos_swap_list      = [], []
err_lam_i_r_list,   cos_lam_i_r_list   = [], []
err_drut_list,      cos_drut_list      = [], []
err_wang_list,      cos_wang_list      = [], []

for T, vstate in trained_vstates.items():
    vstate_R = trained_vstates_R[T]
    S2_ex, grad_ex = renyi2_entropy_and_grad_exact(vstate, subsystem, hi)
    S2_ex = float(S2_ex)
    S2_exact_list.append(S2_ex)
    print(f"\n  T={T:.2f}  S₂={S2_ex:.4f}")

    s2_sw,      cos_sw      = [], []
    s2_lam_i_r, cos_lam_i_r = [], []
    s2_drut,    cos_drut    = [], []
    s2_wang,    cos_wang    = [], []

    for rep in range(n_rep):
        # Swap trick
        S2_est, grad_est = renyi2_entropy_and_grad_sampled(
            vstate, subsystem, n_samples_sw
        )
        s2_sw.append(float(S2_est))
        cos_sw.append(cosine_similarity(grad_est, grad_ex))

        # λ-i (reweighting)
        S2_est, grad_est = renyi2_entropy_and_grad_lambda_integral(
            vstate, subsystem, n_samples_lam_i, n_lambda=LAMBDA_I_N_LAMBDA
        )
        s2_lam_i_r.append(float(S2_est))
        cos_lam_i_r.append(cosine_similarity(grad_est, grad_ex))

        # DRUT
        S2_est, grad_est = renyi2_drut_sampling2(
            vstate, subsystem,
            n_chains=n_chains_dr,
            n_lambda=DRUT_N_LAMBDA,
            n_sweeps_per_lam=DRUT_N_SWEEPS,
            n_props_per_sweep=DRUT_N_PROPS,
            K=DRUT_K,
            debug=False,
        )
        s2_drut.append(float(S2_est))
        cos_drut.append(cosine_similarity(grad_est, grad_ex))

        # Wang CS
        S2_est, grad_est = renyi2_wang_cs(
            vstate, vstate_R, subsystem,
            n_samples=n_samples_wang,
            key=rep,
            debug=True
        )
        s2_wang.append(float(S2_est))
        cos_wang.append(cosine_similarity(grad_est, grad_ex))
        

    err_swap_list.append(np.abs(np.array(s2_sw)      - S2_ex)); cos_swap_list.append(np.array(cos_sw))
    err_lam_i_r_list.append(np.abs(np.array(s2_lam_i_r) - S2_ex)); cos_lam_i_r_list.append(np.array(cos_lam_i_r))
    err_drut_list.append(np.abs(np.array(s2_drut)    - S2_ex)); cos_drut_list.append(np.array(cos_drut))
    err_wang_list.append(np.abs(np.array(s2_wang)    - S2_ex)); cos_wang_list.append(np.array(cos_wang))

    print(f"    swap    |ΔS₂|={err_swap_list[-1].mean():.4f}  cos={np.mean(cos_sw):.4f}")
    print(f"    λ-i (r) |ΔS₂|={err_lam_i_r_list[-1].mean():.4f}  cos={np.mean(cos_lam_i_r):.4f}")
    print(f"    drut    |ΔS₂|={err_drut_list[-1].mean():.4f}  cos={np.mean(cos_drut):.4f}")
    print(f"    wang    |ΔS₂|={err_wang_list[-1].mean():.4f}  cos={np.mean(cos_wang):.4f}")


S2_arr = np.array(S2_exact_list)
err_sw_mean      = np.array([e.mean() for e in err_swap_list])
err_lam_i_r_mean = np.array([e.mean() for e in err_lam_i_r_list])
err_drut_mean    = np.array([e.mean() for e in err_drut_list])
err_wang_mean    = np.array([e.mean() for e in err_wang_list])
cos_sw_mean      = np.array([c.mean() for c in cos_swap_list])
cos_lam_i_r_mean = np.array([c.mean() for c in cos_lam_i_r_list])
cos_drut_mean    = np.array([c.mean() for c in cos_drut_list])
cos_wang_mean    = np.array([c.mean() for c in cos_wang_list])


def percentile_bands(err_list):
    p25 = np.array([np.percentile(e, 25) for e in err_list])
    p75 = np.array([np.percentile(e, 75) for e in err_list])
    return p25, p75


sw_p25, sw_p75          = percentile_bands(err_swap_list)
lam_i_r_p25, lam_i_r_p75 = percentile_bands(err_lam_i_r_list)
drut_p25, drut_p75      = percentile_bands(err_drut_list)
wang_p25, wang_p75      = percentile_bands(err_wang_list)

rel_sw_mean      = err_sw_mean      / S2_arr
rel_lam_i_r_mean = err_lam_i_r_mean / S2_arr
rel_drut_mean    = err_drut_mean    / S2_arr
rel_wang_mean    = err_wang_mean    / S2_arr

# ── figura: error absoluto y relativo ──
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

ax = axes[0]
ax.plot(S2_arr, err_sw_mean, 'o-', color='C0', label='Swap trick')
ax.fill_between(S2_arr, sw_p25, sw_p75, alpha=0.25, color='C0')
ax.plot(S2_arr, err_lam_i_r_mean, 's-', color='C1', label=r'$\lambda_i (r)$')
ax.fill_between(S2_arr, lam_i_r_p25, lam_i_r_p75, alpha=0.25, color='C1')
ax.plot(S2_arr, err_drut_mean, '^-', color='C2', label='DRUT')
ax.fill_between(S2_arr, drut_p25, drut_p75, alpha=0.25, color='C2')
ax.plot(S2_arr, err_wang_mean, 'd-', color='C3', label='Wang CS')
ax.fill_between(S2_arr, wang_p25, wang_p75, alpha=0.25, color='C3')

S2_ref = np.linspace(S2_arr.min(), S2_arr.max(), 100)
ax.plot(S2_ref, np.exp(S2_ref) / np.sqrt(n_samples_sw), 'k--',
        label=r'$e^{S_2}/\sqrt{N_\mathrm{eval}}$')

ax.set_xlabel(r'$S_2$ (exact)')
ax.set_ylabel(r'$|\Delta S_2|$')
ax.set_yscale('log')
ax.legend(fontsize=8)
ax.set_title(fr'$S_2$ absolute error (B = {BUDGET_FIXED})')
ax.grid(True, alpha=0.3, which='both')

ax = axes[1]
ax.plot(S2_arr, rel_sw_mean,      'o-', color='C0', label='Swap trick')
ax.plot(S2_arr, rel_lam_i_r_mean, 's-', color='C1', label=r'$\lambda_i (r)$')
ax.plot(S2_arr, rel_drut_mean,    '^-', color='C2', label='DRUT')
ax.plot(S2_arr, rel_wang_mean,    'd-', color='C3', label='Wang CS')
ax.set_xlabel(r'$S_2$ (exact)')
ax.set_ylabel(r'$|\Delta S_2| / S_2$')
ax.set_yscale('log')
ax.set_title('Relative error')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, which='both')

fig.tight_layout()
save_fig(fig, "s2_error_vs_entropy")

# ── figura: calidad del gradiente ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))
cos_sw_std      = np.array([c.std() for c in cos_swap_list])
cos_lam_i_r_std = np.array([c.std() for c in cos_lam_i_r_list])
cos_drut_std    = np.array([c.std() for c in cos_drut_list])
cos_wang_std    = np.array([c.std() for c in cos_wang_list])

ax.errorbar(S2_arr, cos_sw_mean,      yerr=cos_sw_std,      fmt='o-', color='C0', label='Swap trick',    capsize=3)
ax.errorbar(S2_arr, cos_lam_i_r_mean, yerr=cos_lam_i_r_std, fmt='s-', color='C1', label=r'$\lambda_i (r)$', capsize=3)
ax.errorbar(S2_arr, cos_drut_mean,    yerr=cos_drut_std,    fmt='^-', color='C2', label='DRUT',          capsize=3)
ax.errorbar(S2_arr, cos_wang_mean,    yerr=cos_wang_std,    fmt='d-', color='C3', label='Wang CS',       capsize=3)
ax.axhline(0.7, color='k', ls=':', alpha=0.4)
ax.set_xlabel(r'$S_2$ (exact)')
ax.set_ylabel(r'$\cos(\nabla S_2^\mathrm{est}, \nabla S_2^\mathrm{ex})$')
ax.set_title(r"Gradient quality vs entropy (fixed budget)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "grad_cosine_vs_entropy")


# ══════════════════════════════════════════════════════════════════════════════
# Sección 2: error vs presupuesto (S₂ baja y alta)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. Error vs presupuesto (evaluaciones)")
print("=" * 60)

for label, T in [("low", TEMPS[0]), ("high", TEMPS[-1])]:
    vstate   = trained_vstates[T]
    vstate_R = trained_vstates_R[T]
    S2_ex, grad_ex = renyi2_entropy_and_grad_exact(vstate, subsystem, hi)
    S2_ex = float(S2_ex)
    print(f"\n  S₂ {label} (T={T})  S₂={S2_ex:.4f}")

    s2_sw_all, s2_lam_i_r_all, s2_drut_all, s2_wang_all = [], [], [], []
    cos_sw_ns, cos_lam_i_r_ns, cos_drut_ns, cos_wang_ns  = [], [], [], []
    budget_used = []

    for B in BUDGET_LIST:
        ns_sw       = B // 4
        ns_lam_i    = B // LAMBDA_I_COST_FACTOR
        n_chains_dr = max(16, B // DRUT_COST_PER_CHAIN)
        ns_wang     = max(16, B // WANG_COST_PER_SAMPLE)

        print(f"\n  B={B}  → sw: {ns_sw}  λ-i: {ns_lam_i}  "
              f"drut: {n_chains_dr}  wang: {ns_wang}")
        print(f"    coste real: sw={4*ns_sw}  λ-i={LAMBDA_I_COST_FACTOR*ns_lam_i}  "
              f"drut={DRUT_COST_PER_CHAIN*n_chains_dr}  "
              f"wang={WANG_COST_PER_SAMPLE*ns_wang}")

        s2_sw,       cos_sw       = [], []
        s2_lam_i_r,  cos_lam_i_r  = [], []
        s2_drut,     cos_drut     = [], []
        s2_wang,     cos_wang     = [], []

        for rep in range(n_rep):
            S2_est, g = renyi2_entropy_and_grad_sampled(vstate, subsystem, ns_sw)
            s2_sw.append(float(S2_est)); cos_sw.append(cosine_similarity(g, grad_ex))

            S2_est, g = renyi2_entropy_and_grad_lambda_integral(
                vstate, subsystem, ns_lam_i, n_lambda=LAMBDA_I_N_LAMBDA
            )
            s2_lam_i_r.append(float(S2_est)); cos_lam_i_r.append(cosine_similarity(g, grad_ex))

            S2_est, g = renyi2_drut_sampling2(
                vstate, subsystem,
                n_chains=n_chains_dr,
                n_lambda=DRUT_N_LAMBDA,
                n_sweeps_per_lam=DRUT_N_SWEEPS,
                n_props_per_sweep=DRUT_N_PROPS,
                K=DRUT_K, debug=False,
            )
            s2_drut.append(float(S2_est)); cos_drut.append(cosine_similarity(g, grad_ex))

            S2_est, g = renyi2_wang_cs(
                vstate, vstate_R, subsystem,
                n_samples=ns_wang, key=rep,
            )
            s2_wang.append(float(S2_est)); cos_wang.append(cosine_similarity(g, grad_ex))

        s2_sw_all.append(s2_sw);             s2_lam_i_r_all.append(s2_lam_i_r)
        s2_drut_all.append(s2_drut);         s2_wang_all.append(s2_wang)
        cos_sw_ns.append(np.mean(cos_sw));   cos_lam_i_r_ns.append(np.mean(cos_lam_i_r))
        cos_drut_ns.append(np.mean(cos_drut)); cos_wang_ns.append(np.mean(cos_wang))
        budget_used.append(B)

        sw_mean = np.abs(np.array(s2_sw)      - S2_ex).mean()
        li_mean = np.abs(np.array(s2_lam_i_r) - S2_ex).mean()
        dr_mean = np.abs(np.array(s2_drut)    - S2_ex).mean()
        wa_mean = np.abs(np.array(s2_wang)    - S2_ex).mean()
        print(f"    B={B:8d}  swap={sw_mean:.4f}  λ-i(r)={li_mean:.4f}  "
              f"drut={dr_mean:.4f}  wang={wa_mean:.4f}")

    B_arr = np.array(budget_used, dtype=float)
    err_sw_m      = np.array([np.abs(np.array(s) - S2_ex).mean() for s in s2_sw_all])
    err_lam_i_r_m = np.array([np.abs(np.array(s) - S2_ex).mean() for s in s2_lam_i_r_all])
    err_drut_m    = np.array([np.abs(np.array(s) - S2_ex).mean() for s in s2_drut_all])
    err_wang_m    = np.array([np.abs(np.array(s) - S2_ex).mean() for s in s2_wang_all])

    def inv_sqrt(n, alpha):
        return alpha / np.sqrt(n)

    alpha_sw,   _ = curve_fit(inv_sqrt, B_arr, err_sw_m,      p0=[1.0])
    alpha_lam,  _ = curve_fit(inv_sqrt, B_arr, err_lam_i_r_m, p0=[1.0])
    alpha_dr,   _ = curve_fit(inv_sqrt, B_arr, err_drut_m,    p0=[1.0])
    alpha_wang, _ = curve_fit(inv_sqrt, B_arr, err_wang_m,    p0=[1.0])
    B_fit = np.geomspace(B_arr.min(), B_arr.max(), 200)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.loglog(B_arr, err_sw_m,      'o-', color='C0', label='Swap trick')
    ax.loglog(B_arr, err_lam_i_r_m, 's-', color='C1', label=r'$\lambda_i (r)$')
    ax.loglog(B_arr, err_drut_m,    '^-', color='C2', label='DRUT')
    ax.loglog(B_arr, err_wang_m,    'd-', color='C3', label='Wang CS')
    ax.loglog(B_fit, inv_sqrt(B_fit, alpha_sw),   '--', color='C0',
              label=rf'$\alpha={alpha_sw[0]:.2f}/\sqrt{{B}}$')
    ax.loglog(B_fit, inv_sqrt(B_fit, alpha_wang), '--', color='C3',
              label=rf'$\alpha={alpha_wang[0]:.2f}/\sqrt{{B}}$')
    ax.set_xlabel(r'Ansatz evaluations $B$')
    ax.set_ylabel(r'$|\Delta S_2|$')
    ax.set_title(fr'Error vs budget — {label} $S_2$ ($S_2={S2_ex:.2f}$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    save_fig(fig, f"s2_error_vs_budget_{label}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogx(B_arr, cos_sw_ns,      'o-', color='C0', label='Swap trick')
    ax.semilogx(B_arr, cos_lam_i_r_ns, 's-', color='C1', label=r'$\lambda_i (r)$')
    ax.semilogx(B_arr, cos_drut_ns,    '^-', color='C2', label='DRUT')
    ax.semilogx(B_arr, cos_wang_ns,    'd-', color='C3', label='Wang CS')
    ax.axhline(0.7, color='k', ls=':', alpha=0.4)
    ax.set_xlabel(r'Ansatz evaluations $B$')
    ax.set_ylabel(r'$\cos(\nabla S_2^\mathrm{est}, \nabla S_2^\mathrm{ex})$')
    ax.set_title(f'Gradient quality — {label} $S_2$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"grad_cosine_vs_budget_{label}")

print("\nDone. Gráficas guardadas en", os.path.abspath(PLOTS_DIR))