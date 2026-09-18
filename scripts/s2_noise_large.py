"""
s2_noise_large.py
====================
Comparación de estimadores de S₂ y su gradiente para tamaños donde ED
NO es posible.

Para cada temperatura:
  1. Entrena un vstate minimizando F₂ = <H> - T·S₂ (swap trick estándar).
  2. Estima S₂ y ∇S₂ con los 4 métodos:
       - swap trick estándar
       - λ-integral (reweighting)
       - Drut & Porter (Metropolis + annealing en λ)
       - Increment trick de Hastings (Metropolis + annealing en regiones)
  3. Analiza:
       - Media y desviación estándar de S₂ por método
       - Norma y varianza del gradiente
       - Coseno entre réplicas independientes (autoconsistencia)
       - Coseno entre las medias de los 4 métodos (consistencia mutua)
       - Tiempo de cómputo por llamada

Sin ED, la "referencia" es el consenso entre métodos: la mediana de las
medias de los 4 métodos, para calcular errores relativos.

Gráficas guardadas en ../plots/S2_noise_large/ en formato .dpf y .pgf
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
    renyi2_entropy_and_grad_lambda_integral,
    renyi2_drut_sampling,
    renyi2_increment_sampling,
)
from src_renyi.training import free_energy_minimize

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


# ── configuración ──────────────────────────────────────────────────────────────
N              = 12                       # tamaño del sistema (sin ancilla)
N_A            = N                        # tamaño de la ancilla 
GAMMA          = -1.5                     # coeficiente de σˣ
V              = -1.0                     # coeficiente de σᶻσᶻ
TEMPS          = [3.0, 4.0, 5.0, 7.0]

n_rep          = 10                       # nº de réplicas por método y T
n_samples_diag = 16384                    # para swap y λ-i (nº de muestras)

# Parámetros de Metropolis (Drut e Increment)
n_chains       = 512
n_sweeps_met   = 100                      # sweeps por λ (Drut) / por sitio (Increment)
n_props_sweep  = 2 * (N + N_A)            # propuestas por sweep
n_lambda       = 20                       # puntos en la rejilla λ (Drut)

partition = list(range(N))

# ── hilbert y hamiltoniano ─────────────────────────────────────────────────────
hi = nk.hilbert.Spin(s=1/2, N=N + N_A)

H_extended = 0
for i in range(N):
    H_extended += GAMMA * sigmax(hi, i)
    H_extended += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)

model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16, activation=jax.nn.gelu)
sampler    = nk.sampler.ARDirectSampler(hi)
vstate_ref = nk.vqs.MCState(sampler, model, n_samples=n_samples_diag)


# ── entrenamiento ──────────────────────────────────────────────────────────────
print("=" * 70)
print(f"Entrenando vstates  (N = {N}, N_A = {N_A})")
print("=" * 70)

trained_vstates = {}

for i, T in enumerate(TEMPS):
    vstate = copy.deepcopy(vstate_ref)
    lr = optax.linear_schedule(0.05, 0.001, 300)
    print(f"\n  T = {T:.2f}")
    t0 = time.time()
    free_energy_minimize(
        vstate, T, partition, H_extended, n_steps=300,
        optimizer=optax.sgd(lr),
        plot=False, verbose=False,
        chunk_size=vstate.n_samples // 2,
    )
    print(f"         entrenamiento: {time.time() - t0:.1f} s")
    trained_vstates[T] = vstate


# ── evaluación de los 4 métodos ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Comparación de estimadores  (sin ED)")
print("=" * 70)

METHODS = ["swap", "lambda_i", "drut", "increment"]
COLORS  = {"swap": "C0", "lambda_i": "C1", "drut": "C2", "increment": "C3"}
LABELS  = {"swap": "Swap trick",
           "lambda_i": r"$\lambda$-i",
           "drut": r"$\lambda$-i (TI-2)",
           "increment": "Increment"}

# Resultados: por T, por método → listas de S₂, listas de gradientes, tiempos
results = {T: {m: {"S2": [], "grads": [], "time": []} for m in METHODS}
           for T in TEMPS}

for T, vstate in trained_vstates.items():
    print(f"\n  T = {T:.2f}")

    for rep in range(n_rep):
        # ── swap trick ────────────────────────────────────────────────
        t0 = time.time()
        S2, g = renyi2_entropy_and_grad_sampled(
            vstate, partition, n_samples_diag
        )
        results[T]["swap"]["S2"].append(float(S2))
        results[T]["swap"]["grads"].append(g)
        results[T]["swap"]["time"].append(time.time() - t0)

        # ── λ-integral (reweighting) ──────────────────────────────────
        t0 = time.time()
        S2, g = renyi2_entropy_and_grad_lambda_integral(
            vstate, partition, n_samples_diag,
            n_lambda=n_lambda, debug=False,
        )
        results[T]["lambda_i"]["S2"].append(float(S2))
        results[T]["lambda_i"]["grads"].append(g)
        results[T]["lambda_i"]["time"].append(time.time() - t0)

        # ── Drut (Metropolis + annealing en λ) ───────────────────────
        t0 = time.time()
        S2, g = renyi2_drut_sampling(
            vstate, partition,
            n_chains=n_chains,
            n_sweeps_per_lam=n_sweeps_met,
            n_props_per_sweep=n_props_sweep,
            n_lambda=n_lambda,
            debug=False,
        )
        results[T]["drut"]["S2"].append(float(S2))
        results[T]["drut"]["grads"].append(g)
        results[T]["drut"]["time"].append(time.time() - t0)

        # ── Increment (Metropolis + annealing en regiones) ───────────
        t0 = time.time()
        S2, g, _, _ = renyi2_increment_sampling(
            vstate, partition,
            n_chains=n_chains,
            n_sweeps_per_site=n_sweeps_met,
            n_props_per_sweep=n_props_sweep,
            debug=False,
        )
        results[T]["increment"]["S2"].append(float(S2))
        results[T]["increment"]["grads"].append(g)
        results[T]["increment"]["time"].append(time.time() - t0)

        if (rep + 1) % 5 == 0:
            print(f"    rep {rep+1}/{n_rep}")

    # ── resumen por T ─────────────────────────────────────────────────
    print(f"    {'método':<12}  {'S₂':>10}  {'std(S₂)':>10}  "
          f"{'|∇S₂|':>10}  {'std(|∇|)':>10}  {'t (s)':>8}")
    for m in METHODS:
        s2_arr   = np.array(results[T][m]["S2"])
        norms    = np.array([grad_norm(g) for g in results[T][m]["grads"]])
        times    = np.array(results[T][m]["time"])
        print(f"    {m:<12}  {s2_arr.mean():>10.4f}  {s2_arr.std():>10.4f}  "
              f"{norms.mean():>10.4f}  {norms.std():>10.4f}  "
              f"{times.mean():>8.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Análisis
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. S₂ vs T, con error bars y "consenso" entre métodos ─────────────────────
print("\n" + "=" * 70)
print("Análisis: S₂ vs T")
print("=" * 70)

S2_mean = {m: [] for m in METHODS}
S2_std  = {m: [] for m in METHODS}
S2_consensus = []   # mediana de las medias de los 4 métodos

for T in TEMPS:
    means = [np.mean(results[T][m]["S2"]) for m in METHODS]
    S2_consensus.append(np.median(means))
    for m in METHODS:
        arr = np.array(results[T][m]["S2"])
        S2_mean[m].append(arr.mean())
        S2_std[m].append(arr.std())

S2_consensus = np.array(S2_consensus)

fig, ax = plt.subplots(figsize=(6, 4))
for m in METHODS:
    ax.errorbar(TEMPS, S2_mean[m], yerr=S2_std[m],
                fmt='o-', color=COLORS[m], label=LABELS[m],
                capsize=3, markersize=4)
ax.plot(TEMPS, S2_consensus, 'k--', alpha=0.4, label='consenso (mediana)')
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$S_2$')
ax.set_title(fr'$S_2$ vs $T$  ($N={N}$, $N_A={N_A}$)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "S2_vs_T")

# ── 2. Error relativo vs consenso ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
for m in METHODS:
    rel_err = np.abs(np.array(S2_mean[m]) - S2_consensus) / S2_consensus
    ax.plot(TEMPS, rel_err, 'o-', color=COLORS[m], label=LABELS[m])
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$|S_2^\mathrm{met} - S_2^\mathrm{cons}| / S_2^\mathrm{cons}$')
ax.set_title('Error relativo vs consenso')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "S2_relative_error_vs_T")

# ── 3. Desviación estándar de S₂ (varianza del estimador) ─────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
for m in METHODS:
    ax.plot(TEMPS, S2_std[m], 'o-', color=COLORS[m], label=LABELS[m])
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$\mathrm{std}(S_2)$ entre réplicas')
ax.set_title('Varianza del estimador de $S_2$')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "S2_std_vs_T")

# ── 4. Autoconsistencia: coseno entre réplicas del mismo método ───────────────
print("\n" + "=" * 70)
print("Autoconsistencia del gradiente  (cos entre réplicas del mismo método)")
print("=" * 70)

cos_self = {m: [] for m in METHODS}
cos_self_std = {m: [] for m in METHODS}

for T in TEMPS:
    for m in METHODS:
        grads = results[T][m]["grads"]
        cos_vals = []
        for i in range(len(grads)):
            for j in range(i + 1, len(grads)):
                cos_vals.append(cosine_similarity(grads[i], grads[j]))
        cos_self[m].append(np.mean(cos_vals))
        cos_self_std[m].append(np.std(cos_vals))

for m in METHODS:
    print(f"  {m:<12}  cos_self  =  " +
          "  ".join(f"{c:.3f}" for c in cos_self[m]))

fig, ax = plt.subplots(figsize=(6, 4))
for m in METHODS:
    ax.errorbar(TEMPS, cos_self[m], yerr=cos_self_std[m],
                fmt='o-', color=COLORS[m], label=LABELS[m], capsize=3)
ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$\langle \cos(\nabla S_2^{(i)}, \nabla S_2^{(j)}) \rangle$  (mismo método)')
ax.set_title('Autoconsistencia del gradiente')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "cos_self_vs_T")

# ── 5. Consistencia mutua: coseno entre medias de distintos métodos ───────────
print("\n" + "=" * 70)
print("Consistencia mutua: cos entre medias de distintos métodos")
print("=" * 70)

# Media de gradientes por método y T (gradiente "consenso" de cada método)
mean_grads = {T: {} for T in TEMPS}
for T in TEMPS:
    for m in METHODS:
        flat_list = [jnp.array(jax.flatten_util.ravel_pytree(g)[0], float)
                     for g in results[T][m]["grads"]]
        mean_flat = jnp.mean(jnp.stack(flat_list), axis=0)
        mean_grads[T][m] = mean_flat

# Matriz de cosenos entre métodos, por T
cos_matrix = np.zeros((len(TEMPS), len(METHODS), len(METHODS)))
for it, T in enumerate(TEMPS):
    for im, m1 in enumerate(METHODS):
        for jm, m2 in enumerate(METHODS):
            v1, v2 = mean_grads[T][m1], mean_grads[T][m2]
            cos_matrix[it, im, jm] = float(
                jnp.dot(v1, v2) /
                (jnp.linalg.norm(v1) * jnp.linalg.norm(v2) + 1e-30)
            )

# Imprimir la matriz para cada T
for it, T in enumerate(TEMPS):
    print(f"\n  T = {T:.2f}")
    print("             " + "".join(f"{m:>12}" for m in METHODS))
    for im, m1 in enumerate(METHODS):
        row = "".join(f"{cos_matrix[it, im, jm]:>12.3f}"
                      for jm in range(len(METHODS)))
        print(f"    {m1:<10}{row}")

# Plot: coseno medio vs los otros 3 métodos, en función de T
fig, ax = plt.subplots(figsize=(6, 4))
for im, m in enumerate(METHODS):
    # media de cos con los otros métodos
    others = [cos_matrix[:, im, jm] for jm in range(len(METHODS)) if jm != im]
    mean_others = np.mean(others, axis=0)
    ax.plot(TEMPS, mean_others, 'o-', color=COLORS[m], label=LABELS[m])
ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$\langle \cos(\nabla S_2^m, \nabla S_2^{m^\prime}) \rangle$')
ax.set_title('Consistencia mutua del gradiente')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "cos_mutual_vs_T")

# ── 6. Tiempo de cómputo ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Tiempo de cómputo")
print("=" * 70)

time_mean = {m: [] for m in METHODS}
time_std  = {m: [] for m in METHODS}
for T in TEMPS:
    for m in METHODS:
        arr = np.array(results[T][m]["time"])
        time_mean[m].append(arr.mean())
        time_std[m].append(arr.std())
    print(f"  T = {T:.2f}  " +
          "  ".join(f"{m}={time_mean[m][-1]:.1f}s" for m in METHODS))

fig, ax = plt.subplots(figsize=(6, 4))
for m in METHODS:
    ax.errorbar(TEMPS, time_mean[m], yerr=time_std[m],
                fmt='o-', color=COLORS[m], label=LABELS[m], capsize=3)
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$t$ (s) por llamada')
ax.set_title('Coste de cómputo')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "timing_vs_T")

# ── 7. Varianza del gradiente: norma² de la desviación típica por componente ──
print("\n" + "=" * 70)
print("Varianza del gradiente  (norma de la std por componente)")
print("=" * 70)

grad_var_norm = {m: [] for m in METHODS}
for T in TEMPS:
    for m in METHODS:
        flats = np.stack([np.array(jax.flatten_util.ravel_pytree(g)[0], float)
                          for g in results[T][m]["grads"]])
        std_per_component = flats.std(axis=0)
        grad_var_norm[m].append(float(np.linalg.norm(std_per_component)))

for m in METHODS:
    print(f"  {m:<12}  " +
          "  ".join(f"{v:.3e}" for v in grad_var_norm[m]))

fig, ax = plt.subplots(figsize=(6, 4))
for m in METHODS:
    ax.plot(TEMPS, grad_var_norm[m], 'o-', color=COLORS[m], label=LABELS[m])
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$\|\sigma_{\nabla S_2}\|_2$  (std por componente)')
ax.set_title('Varianza del gradiente')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "grad_variance_vs_T")

print("\nDone. Gráficas guardadas en", os.path.abspath(PLOTS_DIR))