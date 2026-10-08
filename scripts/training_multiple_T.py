"""
training_multiple_T.py
===========
Entrenamiento NQS en array de temperaturas para minimizar <H>-TS₂.

Fases por temperatura: swap (coarse) + Wang (refinado con red reversa).
Ver training_single_T.py para la versión monoT.

Uso:
    Editar la sección "CONFIGURACIÓN" y ejecutar:
        python scripts/training_multiple_T.py
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import flax.serialization as serialization
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from src_renyi import (
    free_energy_minimize_phases,
    renyi2_entropy_and_grad_sampled,
    renyi2_wang_cs,
)

# ── CONFIGURACIÓN  ─────────────────────────────────────────────────────────────

N, N_A = 2, 2
J_ZZ, J_XX, h_x, h_z = -1.0, 0.0, -1.5, 0.0

T_min, T_max, N_Temps = 0.0, 4.0, 41
linear_T = True                    # Si False, malla no lineal (solo N<10)

N_SAMPLES       = 2**14
N_STEPS_SWAP    = 300
N_STEPS_WANG    = 50
INIT_N_STEPS    = 500
INIT_BATCH      = N_SAMPLES
WARM_EVERY      = 5
WARM_N_STEPS    = 20
WARM_BATCH      = N_SAMPLES

chunk_size = N_SAMPLES // 2

N_REP_COSINE = 10

# ── funciones auxiliares ───────────────────────────────────────────────────────

def cosine_similarity(g1, g2):
    """Calcula el coseno entre dos gradientes (pytrees)."""
    flat1, _ = jax.flatten_util.ravel_pytree(g1)
    flat2, _ = jax.flatten_util.ravel_pytree(g2)
    flat1 = jnp.array(flat1, float)
    flat2 = jnp.array(flat2, float)
    return float(jnp.dot(flat1, flat2) /
                 (jnp.linalg.norm(flat1) * jnp.linalg.norm(flat2) + 1e-30))


def compute_jump_temperatures(evals, max_jumps=None, tol=1e-10):
    # Colapsar niveles degenerados
    unique_evals = []
    for e in evals:
        if len(unique_evals) == 0 or abs(e - unique_evals[-1]) > tol:
            unique_evals.append(e)
    unique_evals = np.array(unique_evals)

    n_levels = len(unique_evals) if max_jumps is None else min(max_jumps + 1, len(unique_evals))
    T_jumps = []
    for n in range(1, n_levels):
        weights = unique_evals[n] - unique_evals[:n]
        E_bar = np.dot(unique_evals[:n], weights) / weights.sum()
        T_jumps.append((unique_evals[n] - E_bar) / 2)
    return np.array(T_jumps)


def make_optimal_T_array(T_jumps, T_min=0.05, T_max=4.0, n_total=200, width_factor=0.1):
    """
    Crea un array de temperaturas con alta densidad cerca de cada T_n.
    """
    n_base = n_total // 3
    T_base = np.linspace(T_min, T_max, n_base)

    T_extra = []
    jumps_in_range = T_jumps[(T_jumps > T_min) & (T_jumps < T_max)]
    for i, Tn in enumerate(jumps_in_range):
        gaps = np.abs(jumps_in_range - Tn)
        gaps = gaps[gaps > 1e-10]
        local_gap = gaps.min() if len(gaps) > 0 else (T_max - T_min) / len(jumps_in_range)
        width = width_factor * local_gap

        n_local = n_total // (2 * len(jumps_in_range))
        T_extra.append(np.linspace(Tn - 2 * width, Tn + 2 * width, n_local))

    T_all = np.concatenate([T_base] + T_extra)
    T_all = np.clip(T_all, T_min, T_max)
    T_all = np.unique(np.round(T_all, 8))
    return T_all

# ───────────────────────────────────────────────────────────────────────────────


# ── construir hilbert, hamiltoniano y modelos ──────────────────────────────────
hi_sys = nk.hilbert.Spin(s=1/2, N=N)
hi     = nk.hilbert.Spin(s=1/2, N=N+N_A)

H_sys = 0
H_extended = 0
for i in range(N):
    H_sys += h_x * sigmax(hi_sys, i)
    H_sys += h_z * sigmaz(hi_sys, i)
    H_sys += J_ZZ * sigmaz(hi_sys, i) @ sigmaz(hi_sys, (i + 1) % N)
    H_sys += J_XX * sigmax(hi_sys, i) @ sigmax(hi_sys, (i + 1) % N)

    H_extended += h_x * sigmax(hi, i)
    H_extended += h_z * sigmaz(hi, i)
    H_extended += J_ZZ * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)
    H_extended += J_XX * sigmax(hi, i) @ sigmax(hi, (i + 1) % N)

model     = nk.models.ARNNDense(hilbert=hi, layers=1, features=16, activation=jax.nn.tanh)
sampler   = nk.sampler.ARDirectSampler(hi)

model_R   = nk.models.ARNNDense(hilbert=hi, layers=1, features=16, activation=jax.nn.tanh)
sampler_R = nk.sampler.ARDirectSampler(hi)

partition = list(range(N))
# ───────────────────────────────────────────────────────────────────────────────


# ── crear array de temperaturas ────────────────────────────────────────────────
if N > 10 and (not linear_T):
    linear_T = True

if not linear_T:
    eigvals, _ = np.linalg.eigh(H_sys.to_dense())
    T_jumps = compute_jump_temperatures(eigvals)
    T_array = make_optimal_T_array(T_jumps, T_min=T_min, T_max=T_max, n_total=N_Temps)
else:
    T_array = np.linspace(T_min, T_max, N_Temps)
# ───────────────────────────────────────────────────────────────────────────────


# ── rutas ──────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(script_dir, "..", "data")
if N == N_A:
    data_dir = os.path.join(base_data_dir, f"N{N}")
else:
    data_dir = os.path.join(base_data_dir, f"N{N}_NA_{N_A}")
params_dir = os.path.join(data_dir, "params")
os.makedirs(params_dir, exist_ok=True)

results_file = os.path.join(data_dir, f"results_N{N}_vs_T.json")

if os.path.exists(results_file):
    with open(results_file, "r") as f:
        all_data = json.load(f)
else:
    all_data = {"T": [], "energy": [], "entropy": [], "free_energy": [],
                "reliability": [], "param_index": []}
# ───────────────────────────────────────────────────────────────────────────────


# ── ENTRENAMIENTO ──────────────────────────────────────────────────────────────
energy_results      = []
entropy_results     = []
free_energy_results = []

print(f"TRAINING N={N}, N_A={N_A}  | {len(T_array)} temperatures")

for T_loop_idx, T in enumerate(tqdm(T_array, desc="Temperaturas")):

    # ── determinar idx (reusar si T ya existe) ──
    if T in all_data["T"]:
        idx = all_data["T"].index(T)
    else:
        idx = len(all_data["T"])

    print(f"\n=== T = {T:.4f}  (idx={idx}) ===")

    # ── estados variacionales frescos por temperatura ──
    vstate   = nk.vqs.MCState(sampler,   model,   n_samples=N_SAMPLES)
    vstate_R = nk.vqs.MCState(sampler_R, model_R, n_samples=N_SAMPLES)

    # ── fases: swap coarse + Wang refinado ──
    phases = [
        {
            "method": "swap",
            "n_steps": N_STEPS_SWAP,
            "optimizer": optax.sgd(optax.linear_schedule(0.05, 0.005, 2000)),
        },
        {
            "method": "wang",
            "n_steps": N_STEPS_WANG,
            "optimizer": optax.adam(1e-3),
            "vstate_R": vstate_R,
            "initial_train": {"n_steps": INIT_N_STEPS, "batch": INIT_BATCH, "lr": 1e-3},
            "warm_start":    {"every": WARM_EVERY, "n_steps": WARM_N_STEPS,
                              "lr": 5e-4, "batch": WARM_BATCH},
            "wang_kwargs":   {"n_samples": N_SAMPLES},
        },
    ]

    # ── entrenamiento ──
    _, f_best, E_best, S_best = free_energy_minimize_phases(
        vstate, T, partition, H_extended, phases,
        chunk_size=chunk_size,
        verbose=True, freq=10,
        plot=False, timing=True,
        history_index=idx,
    )
    print(f"  Best: S₂={S_best:.6f}, E={E_best:.6f}, F={f_best:.6f}")

    # ── consistencia del gradiente: SWAP ──
    grads_swap = []
    for rep in range(N_REP_COSINE):
        _, grad_est = renyi2_entropy_and_grad_sampled(
            vstate, partition, N_SAMPLES, chunk_size=chunk_size
        )
        grads_swap.append(grad_est)

    cos_vals_swap = [
        cosine_similarity(grads_swap[i], grads_swap[j])
        for i in range(N_REP_COSINE) for j in range(i + 1, N_REP_COSINE)
    ]
    cos_swap_mean = float(np.mean(cos_vals_swap))
    cos_swap_std  = float(np.std(cos_vals_swap))
    print(f"  [swap] cos = {cos_swap_mean:.4f} ± {cos_swap_std:.4f}")

    # ── consistencia del gradiente: WANG ──
    grads_wang = []
    for rep in range(N_REP_COSINE):
        _, grad_est = renyi2_wang_cs(
            vstate, vstate_R, partition, N_SAMPLES, key=rep
        )
        grads_wang.append(grad_est)

    cos_vals_wang = [
        cosine_similarity(grads_wang[i], grads_wang[j])
        for i in range(N_REP_COSINE) for j in range(i + 1, N_REP_COSINE)
    ]
    cos_wang_mean = float(np.mean(cos_vals_wang))
    cos_wang_std  = float(np.std(cos_vals_wang))
    print(f"  [wang] cos = {cos_wang_mean:.4f} ± {cos_wang_std:.4f}")

    # ── guardar parámetros con el índice ──
    best_params = vstate.parameters
    filename = os.path.join(params_dir, f"params_{idx:04d}.msgpack")
    with open(filename, "wb") as f:
        f.write(serialization.to_bytes(best_params))

    # ── actualizar JSON (upsert) ──
    if T not in all_data["T"]:
        all_data["T"].append(float(T))
        all_data["energy"].append(None)
        all_data["entropy"].append(None)
        all_data["free_energy"].append(None)
        all_data["reliability"].append(None)
        all_data["param_index"].append(idx)

    all_data["energy"][idx]      = float(E_best)
    all_data["entropy"][idx]     = float(S_best)
    all_data["free_energy"][idx] = float(f_best)
    all_data["reliability"][idx] = {"cos_mean": cos_wang_mean,
                                    "cos_std":  cos_wang_std}

    with open(results_file, "w") as f:
        json.dump(all_data, f, indent=2)

    # ── acumular para plotting ──
    energy_results.append(float(E_best))
    entropy_results.append(float(S_best))
    free_energy_results.append(float(f_best))
# ───────────────────────────────────────────────────────────────────────────────


# ── VISUALIZACIÓN ──────────────────────────────────────────────────────────────
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.plot(T_array, energy_results, 'o-')
plt.xlabel('T'); plt.ylabel('Energía'); plt.title('Energía vs T'); plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(T_array, entropy_results, 's-')
plt.xlabel('T'); plt.ylabel('S₂'); plt.title('S₂ vs T'); plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(T_array, free_energy_results, 'd-')
plt.xlabel('T'); plt.ylabel('F'); plt.title('Energía libre vs T'); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()