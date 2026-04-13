"""
s2_noise.py
====================
Demuestra la principal limitación del método: el ruido en el estimador
del gradiente de S₂ a alta entropía.

Para N<12 (donde ED es posible) entrena el ansatz a distintas temperaturas
T ∈ TEMPS (barriendo el rango de S₂ de baja a alta) y compara:
  - renyi2_entropy_and_grad_sampled         (swap trick estándar)
  - renyi2_entropy_and_grad_lambda_integral  (integración termodinámica)
  - renyi2_entropy_and_grad_exact            (referencia exacta)

Métricas:
  - Error absoluto en S₂
  - Coseno entre gradiente estimado y exacto
  - Evolución con n_samples (para S₂ baja y alta)
  - Crecimiento exponencial del error con S₂  (N_eff ≈ N · e^{-S₂})

Gráficas guardadas en ../plots/ en formato .pgf
"""

import os
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
    renyi2_entropy_and_grad_lambda_integral,
    renyi2_entropy_and_grad_exact,
)
from src_renyi.training import free_energy_minimize

# ── directorios ────────────────────────────────────────────────────────────────
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
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


# ── configuración ──────────────────────────────────────────────────────────────
N              = 5
GAMMA          = -1.5
V              = -1.0
TEMPS          = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
n_rep          = 20
chunk_size     = 128
n_samples_diag = 4096
N_SAMPLES_LIST = [256, 512, 1024, 2048, 4096, 8192]

subsystem = list(range(N))
partition = subsystem

# ── hilbert y hamiltoniano ────────────────

hi_sys = nk.hilbert.Spin(s=1/2, N=N)
hi_anc = nk.hilbert.Spin(s=1/2, N=N)
hi = nk.hilbert.Spin(s=1/2, N=N+N)
Gamma = -1.5  
V = -1
H_sys=0
H_extended = 0
for i in range(N):
    H_sys+= Gamma * sigmax(hi_sys, i)
    H_sys += V * sigmaz(hi_sys, i) @ sigmaz(hi_sys, (i + 1) % N)
for i in range(N):
    H_extended += Gamma * sigmax(hi, i)
    H_extended += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)

model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16, activation=jax.nn.gelu)
sampler    = nk.sampler.ARDirectSampler(hi)
vstate_ref = nk.vqs.MCState(sampler, model, n_samples=n_samples_diag)


# ── entrenamiento ───────────────────────────────────────────────────
print("=" * 60)
print("Entrenando vstates")
print("=" * 60)

trained_vstates = {}
params_prev     = None

for i, T in enumerate(TEMPS):
    vstate = copy.deepcopy(vstate_ref)
    lr = optax.linear_schedule(0.05, 0.001, 400)
    print(f"\n  T={T:.2f}")
    free_energy_minimize(
        vstate, T, partition, H_extended, n_steps=400,
        optimizer=optax.sgd(lr),
        chunk_size=chunk_size,
        plot=False, verbose=False,
    )
    trained_vstates[T] = vstate
    params_prev        = copy.deepcopy(vstate.parameters)

    S2_ex, _ = renyi2_entropy_and_grad_exact(vstate, subsystem, hi)
    print(f"         S₂ exacto = {float(S2_ex):.4f}")


# ── 1. error vs entropía  (n_samples fijo) ────────────────────────────────────
print("\n" + "=" * 60)
print("1. Error vs entropía  (n_samples fijo)")
print("=" * 60)

S2_exact_list                = []
err_swap_list, cos_swap_list = [], []
err_ti_list,   cos_ti_list   = [], []

for T, vstate in trained_vstates.items():
    S2_ex, grad_ex = renyi2_entropy_and_grad_exact(vstate, subsystem, hi)
    S2_ex = float(S2_ex)
    S2_exact_list.append(S2_ex)
    print(f"\n  T={T:.2f}  S₂={S2_ex:.4f}")

    s2_sw, cos_sw = [], []
    s2_ti, cos_ti = [], []

    for rep in range(n_rep):
        S2_est, grad_est = renyi2_entropy_and_grad_sampled(
            vstate, subsystem, n_samples_diag
        )
        s2_sw.append(float(S2_est))
        cos_sw.append(cosine_similarity(grad_est, grad_ex))

        S2_est, grad_est = renyi2_entropy_and_grad_lambda_integral(
            vstate, subsystem, n_samples_diag, n_lambda=30
        )
        s2_ti.append(float(S2_est))
        cos_ti.append(cosine_similarity(grad_est, grad_ex))

    err_swap_list.append(np.abs(np.array(s2_sw) - S2_ex))
    cos_swap_list.append(np.array(cos_sw))
    err_ti_list.append(np.abs(np.array(s2_ti) - S2_ex))
    cos_ti_list.append(np.array(cos_ti))

    print(f"    swap  |ΔS₂|={err_swap_list[-1].mean():.4f}  cos={np.mean(cos_sw):.4f}")
    print(f"    TI    |ΔS₂|={err_ti_list[-1].mean():.4f}  cos={np.mean(cos_ti):.4f}")

S2_arr      = np.array(S2_exact_list)
err_sw_mean = np.array([e.mean() for e in err_swap_list])
err_sw_std  = np.array([e.std()  for e in err_swap_list])
err_ti_mean = np.array([e.mean() for e in err_ti_list])
err_ti_std  = np.array([e.std()  for e in err_ti_list])
cos_sw_mean = np.array([c.mean() for c in cos_swap_list])
cos_ti_mean = np.array([c.mean() for c in cos_ti_list])

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.errorbar(S2_arr, err_sw_mean, yerr=err_sw_std,
            marker='o', capsize=3, label='Swap trick')
ax.errorbar(S2_arr, err_ti_mean, yerr=err_ti_std,
            marker='s', capsize=3, label='TI')
S2_ref = np.linspace(S2_arr.min(), S2_arr.max(), 100)
ax.plot(S2_ref, np.exp(S2_ref) / n_samples_diag, 'k--',
        label=r'$e^{S_2}/N_\mathrm{samp}$')
ax.set_xlabel(r'$S_2$ (exacto)')
ax.set_ylabel(r'$|\Delta S_2|$')
ax.set_yscale('log')
ax.legend()
ax.set_title(r"Error en $S_2$ vs entropia")
fig.tight_layout()
save_fig(fig, "s2_error_vs_entropy")

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(S2_arr, cos_sw_mean, 'o-', label='Swap trick')
ax.plot(S2_arr, cos_ti_mean, 's-', label='TI')
ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel(r'$S_2$ (exacto)')
ax.set_ylabel(r'$\cos(\nabla S_2^\mathrm{est},\, \nabla S_2^\mathrm{ex})$')
ax.set_ylim(-0.1, 1.05)
ax.legend()
ax.set_title(r"Calidad del gradiente vs entropia")
fig.tight_layout()
save_fig(fig, "grad_cosine_vs_entropy")


# ── 2. error vs n_samples  (S₂ baja y alta) ───────────────────────────────────
print("\n" + "=" * 60)
print("2. Error vs n_samples")
print("=" * 60)

for label, T in [("baja", TEMPS[0]), ("alta", TEMPS[-1])]:
    vstate = trained_vstates[T]
    S2_ex, grad_ex = renyi2_entropy_and_grad_exact(vstate, subsystem, hi)
    S2_ex = float(S2_ex)
    print(f"\n  S2 {label}  (T={T})  S2={S2_ex:.4f}")

    err_sw_ns, err_ti_ns = [], []
    cos_sw_ns, cos_ti_ns = [], []

    for ns in N_SAMPLES_LIST:
        cs = min(chunk_size, ns // 2)
        s2_sw, cos_sw = [], []
        s2_ti, cos_ti = [], []

        for rep in range(n_rep):
            S2_est, grad_est = renyi2_entropy_and_grad_sampled(
                vstate, subsystem, ns
            )
            s2_sw.append(float(S2_est))
            cos_sw.append(cosine_similarity(grad_est, grad_ex))

            S2_est, grad_est = renyi2_entropy_and_grad_lambda_integral(
                vstate, subsystem, ns, n_lambda=30
            )
            s2_ti.append(float(S2_est))
            cos_ti.append(cosine_similarity(grad_est, grad_ex))

        err_sw_ns.append((np.abs(np.array(s2_sw) - S2_ex).mean(),
                          np.abs(np.array(s2_sw) - S2_ex).std()))
        err_ti_ns.append((np.abs(np.array(s2_ti) - S2_ex).mean(),
                          np.abs(np.array(s2_ti) - S2_ex).std()))
        cos_sw_ns.append(np.mean(cos_sw))
        cos_ti_ns.append(np.mean(cos_ti))
        print(f"    n={ns:5d}  swap |DS2|={err_sw_ns[-1][0]:.4f}"
              f"  TI |DS2|={err_ti_ns[-1][0]:.4f}")

    err_sw_m = np.array([e[0] for e in err_sw_ns])
    err_sw_s = np.array([e[1] for e in err_sw_ns])
    err_ti_m = np.array([e[0] for e in err_ti_ns])
    err_ti_s = np.array([e[1] for e in err_ti_ns])
    ns_arr   = np.array(N_SAMPLES_LIST)
    Neff_ref = ns_arr * np.exp(-S2_ex)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, (err_m, err_s, lbl) in zip(axes, [
        (err_sw_m, err_sw_s, 'Swap trick'),
        (err_ti_m, err_ti_s, 'TI'),
    ]):
        ax.errorbar(ns_arr, err_m, yerr=err_s, marker='o', capsize=3, label=lbl)
        ax.plot(ns_arr, 1.0 / np.sqrt(Neff_ref), 'k--',
                label=r'$1/\sqrt{N_\mathrm{eff}}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$N_\mathrm{samples}$')
        ax.set_ylabel(r'$|\Delta S_2|$')
        ax.set_title(f'{lbl}  --  $S_2$ {label}')
        ax.legend()
    fig.tight_layout()
    save_fig(fig, f"s2_error_vs_nsamples_{label}")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogx(ns_arr, cos_sw_ns, 'o-', label='Swap trick')
    ax.semilogx(ns_arr, cos_ti_ns, 's-', label='TI')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(r'$N_\mathrm{samples}$')
    ax.set_ylabel(r'$\cos(\nabla S_2^\mathrm{est},\, \nabla S_2^\mathrm{ex})$')
    ax.set_title(f'Calidad del gradiente  --  $S_2$ {label}')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, f"grad_cosine_vs_nsamples_{label}")


# ── 3. colapso exponencial de N_eff con S₂ ────────────────────────────────────
print("\n" + "=" * 60)
print("3. N_eff vs S2")
print("=" * 60)

Neff_data = []

for T, vstate in trained_vstates.items():
    S2_ex, _ = renyi2_entropy_and_grad_exact(vstate, subsystem, hi)
    S2_ex    = float(S2_ex)

    s2_vals = []
    for rep in range(n_rep * 2):
        S2_est, _ = renyi2_entropy_and_grad_sampled(
            vstate, subsystem, n_samples_diag
        )
        s2_vals.append(float(S2_est))

    var      = np.var(s2_vals)
    Neff_emp = 1.0 / (var + 1e-30)
    Neff_th  = n_samples_diag * np.exp(-S2_ex)
    Neff_data.append((S2_ex, Neff_emp, Neff_th))
    print(f"  T={T:.2f}  S2={S2_ex:.3f}"
          f"  N_eff empirico={Neff_emp:.1f}  teorico={Neff_th:.1f}")

S2_neff  = np.array([x[0] for x in Neff_data])
Neff_emp = np.array([x[1] for x in Neff_data])
Neff_th  = np.array([x[2] for x in Neff_data])

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.semilogy(S2_neff, Neff_emp, 'o-', label=r'$N_\mathrm{eff}$ empirico')
ax.semilogy(S2_neff, Neff_th,  'k--',
            label=r'$N_\mathrm{samp} \cdot e^{-S_2}$')
ax.set_xlabel(r'$S_2$')
ax.set_ylabel(r'$N_\mathrm{eff}$')
ax.set_title(r'Colapso exponencial de $N_\mathrm{eff}$')
ax.legend()
fig.tight_layout()
save_fig(fig, "neff_vs_s2")

print("\nDone. Graficas guardadas en", os.path.abspath(PLOTS_DIR))