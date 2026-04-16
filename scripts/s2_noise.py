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
)
from src_renyi.training import free_energy_minimize

# ── directorios ────────────────────────────────────────────────────────────────
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots/S2_noise")
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
N              = 6
GAMMA          = -1.5
V              = -1.0
TEMPS          = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]
n_rep          = 50
n_samples_diag = 16384
N_SAMPLES_LIST = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

subsystem = list(range(N))
partition = subsystem

# ── hilbert y hamiltoniano ────────────────

hi_sys = nk.hilbert.Spin(s=1/2, N=N)
hi_anc = nk.hilbert.Spin(s=1/2, N=N)
hi = nk.hilbert.Spin(s=1/2, N=N+N)
H_sys=0
H_extended = 0
for i in range(N):
    H_sys+= GAMMA * sigmax(hi_sys, i)
    H_sys += V * sigmaz(hi_sys, i) @ sigmaz(hi_sys, (i + 1) % N)
for i in range(N):
    H_extended += GAMMA * sigmax(hi, i)
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
    lr = optax.linear_schedule(0.05, 0.001, 300)
    print(f"\n  T={T:.2f}")
    free_energy_minimize(
        vstate, T, partition, H_extended, n_steps=300,
        optimizer=optax.sgd(lr),
        plot=False, verbose=False,
        chunk_size=vstate.n_samples//2
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
            vstate, subsystem, n_samples_diag, n_lambda=60
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
err_ti_mean = np.array([e.mean() for e in err_ti_list])
cos_sw_mean = np.array([c.mean() for c in cos_swap_list])
cos_ti_mean = np.array([c.mean() for c in cos_ti_list])

# Percentiles para barras asimétricas robustas
def percentile_bands(err_list):
    p25 = np.array([np.percentile(e, 25) for e in err_list])
    p75 = np.array([np.percentile(e, 75) for e in err_list])
    return p25, p75

sw_p25, sw_p75 = percentile_bands(err_swap_list)
ti_p25, ti_p75 = percentile_bands(err_ti_list)

# Error relativo
rel_sw_mean = err_sw_mean / S2_arr
rel_ti_mean = err_ti_mean / S2_arr
rel_sw_p25  = sw_p25 / S2_arr
rel_sw_p75  = sw_p75 / S2_arr
rel_ti_p25  = ti_p25 / S2_arr
rel_ti_p75  = ti_p75 / S2_arr

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# Panel izquierdo: error absoluto con bandas de percentiles
ax = axes[0]
ax.plot(S2_arr, err_sw_mean, 'o-', color='C0', label='Swap trick')
ax.fill_between(S2_arr, sw_p25, sw_p75, alpha=0.25, color='C0')
ax.plot(S2_arr, err_ti_mean, 's-', color='C1', label='TI')
ax.fill_between(S2_arr, ti_p25, ti_p75, alpha=0.25, color='C1')
# Referencia teórica corregida: std ~ e^{S2/2} / sqrt(N_samp)
S2_ref = np.linspace(S2_arr.min(), S2_arr.max(), 100)
ax.plot(S2_ref, np.exp(S2_ref / 2) / np.sqrt(n_samples_diag), 'k--',
        label=r'$e^{S_2/2}/\sqrt{N_\mathrm{samp}}$')
ax.set_xlabel(r'$S_2$ (exacto)')
ax.set_ylabel(r'$|\Delta S_2|$')
ax.set_yscale('log')
ax.legend()
ax.set_title(r'Error absoluto en $S_2$')

# Panel derecho: error relativo (sin escala log, más legible)
ax = axes[1]
ax.plot(S2_arr, rel_sw_mean, 'o-', color='C0', label='Swap trick')
ax.fill_between(S2_arr, rel_sw_p25, rel_sw_p75, alpha=0.25, color='C0')
ax.plot(S2_arr, rel_ti_mean, 's-', color='C1', label='TI')
ax.fill_between(S2_arr, rel_ti_p25, rel_ti_p75, alpha=0.25, color='C1')
ax.set_xlabel(r'$S_2$ (exacto)')
ax.set_ylabel(r'$|\Delta S_2| / S_2$')
ax.set_title(r'Error relativo en $S_2$')
ax.legend()

fig.tight_layout()
save_fig(fig, "s2_error_vs_entropy")

fig, ax = plt.subplots(figsize=(5, 3.5))

# Calcular std
cos_sw_std = np.array([c.std() for c in cos_swap_list])
cos_ti_std = np.array([c.std() for c in cos_ti_list])

# Graficar con bandas de ±1σ
ax.plot(S2_arr, cos_sw_mean, 'o-', color='C0', label='Swap trick')
ax.fill_between(S2_arr, cos_sw_mean - cos_sw_std, cos_sw_mean + cos_sw_std, 
                alpha=0.2, color='C0')

ax.plot(S2_arr, cos_ti_mean, 's-', color='C1', label='TI')
ax.fill_between(S2_arr, cos_ti_mean - cos_ti_std, cos_ti_mean + cos_ti_std, 
                alpha=0.2, color='C1')

ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)

ax.set_xlabel(r'$S_2$ (exacto)')
ax.set_ylabel(r'$\langle \cos(\nabla S_2^\mathrm{est}, \nabla S_2^\mathrm{ex}) \rangle$')
ax.legend()
ax.set_title(r"Calidad del gradiente vs entropía")
ax.grid(True, alpha=0.3)

fig.tight_layout()
save_fig(fig, "grad_cosine_vs_entropy")


# ── 2. error vs n_samples  (S₂ baja y alta) ───────────────────────────────────
from scipy.optimize import curve_fit

print("\n" + "=" * 60)
print("2. Error vs n_samples")
print("=" * 60)

for label, T in [("baja", TEMPS[0]), ("alta", TEMPS[-1])]:
    vstate = trained_vstates[T]
    S2_ex, grad_ex = renyi2_entropy_and_grad_exact(vstate, subsystem, hi)
    S2_ex = float(S2_ex)
    print(f"\n  S2 {label}  (T={T})  S2={S2_ex:.4f}")

    s2_sw_all, s2_ti_all = [], []
    cos_sw_ns, cos_ti_ns = [], []

    for ns in N_SAMPLES_LIST:
        s2_sw, cos_sw = [], []
        s2_ti, cos_ti = [], []

        for rep in range(n_rep):
            S2_est, grad_est = renyi2_entropy_and_grad_sampled(
                vstate, subsystem, ns
            )
            s2_sw.append(float(S2_est))
            cos_sw.append(cosine_similarity(grad_est, grad_ex))

            S2_est, grad_est = renyi2_entropy_and_grad_lambda_integral(
                vstate, subsystem, ns, n_lambda=60
            )
            s2_ti.append(float(S2_est))
            cos_ti.append(cosine_similarity(grad_est, grad_ex))

        s2_sw_all.append(s2_sw)
        s2_ti_all.append(s2_ti)
        cos_sw_ns.append(np.mean(cos_sw))
        cos_ti_ns.append(np.mean(cos_ti))

        sw_mean = np.abs(np.array(s2_sw) - S2_ex).mean()
        ti_mean = np.abs(np.array(s2_ti) - S2_ex).mean()
        print(f"    n={ns:5d}  swap |DS2|={sw_mean:.4f}  TI |DS2|={ti_mean:.4f}")

    ns_arr   = np.array(N_SAMPLES_LIST, dtype=float)
    err_sw_m = np.array([np.abs(np.array(s) - S2_ex).mean() for s in s2_sw_all])
    err_ti_m = np.array([np.abs(np.array(s) - S2_ex).mean() for s in s2_ti_all])
    sw_p25   = np.array([np.percentile(np.abs(np.array(s) - S2_ex), 25) for s in s2_sw_all])
    sw_p75   = np.array([np.percentile(np.abs(np.array(s) - S2_ex), 75) for s in s2_sw_all])
    ti_p25   = np.array([np.percentile(np.abs(np.array(s) - S2_ex), 25) for s in s2_ti_all])
    ti_p75   = np.array([np.percentile(np.abs(np.array(s) - S2_ex), 75) for s in s2_ti_all])

    # fit α / √N para cada método
    def inv_sqrt(n, alpha):
        return alpha / np.sqrt(n)

    alpha_sw, _ = curve_fit(inv_sqrt, ns_arr, err_sw_m, p0=[1.0])
    alpha_ti, _ = curve_fit(inv_sqrt, ns_arr, err_ti_m, p0=[1.0])
    ns_fit = np.geomspace(ns_arr.min(), ns_arr.max(), 200)

    # ── plot error absoluto ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.loglog(ns_arr, err_sw_m, 'o-', color='C0', label='Swap trick')
    ax.fill_between(ns_arr, sw_p25, sw_p75, alpha=0.25, color='C0')
    ax.loglog(ns_arr, err_ti_m, 's-', color='C1', label='TI')
    ax.fill_between(ns_arr, ti_p25, ti_p75, alpha=0.25, color='C1')
    ax.loglog(ns_fit, inv_sqrt(ns_fit, alpha_sw), '--', color='C0',
              label=rf'$\alpha_\mathrm{{sw}}/\sqrt{{N}}$, $\alpha={alpha_sw[0]:.3f}$')
    ax.loglog(ns_fit, inv_sqrt(ns_fit, alpha_ti), '--', color='C1',
              label=rf'$\alpha_\mathrm{{TI}}/\sqrt{{N}}$, $\alpha={alpha_ti[0]:.3f}$')
    ax.set_xlabel(r'$N_\mathrm{samples}$')
    ax.set_ylabel(r'$|\Delta S_2|$')
    ax.set_title(fr'Error absoluto — $S_2$ {label} ($S_2={S2_ex:.2f}$)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, f"s2_error_vs_nsamples_{label}")

    # ── plot coseno (sin cambios) ──────────────────────────────────────────────
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

print("\nDone. Graficas guardadas en", os.path.abspath(PLOTS_DIR))