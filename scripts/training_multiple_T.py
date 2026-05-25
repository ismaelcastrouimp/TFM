"""
training_multiple_T.py
===========
Entrenamiento NQS en array de temperaturas para minimizar <H>-TS₂.

Uso:
    Editar la sección "CONFIGURACIÓN" y ejecutar:
        python scripts/training_single_T.py
"""

import os
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax
import numpy as np
import matplotlib.pyplot as plt
import json
import flax.serialization as serialization
from tqdm import tqdm

from src_renyi import free_energy_minimize, renyi2_entropy_and_grad_sampled, free_energy_minimize_exact, ARNN_Z2

# ── CONFIGURACIÓN  ─────────────────────────────────────────────────────────────
N          = 15
N_A        = 10
N_SAMPLES  = 2**21

J_ZZ       = -1.0
J_XX       = 0.0
h_x        = -1.1
h_z        = 0.0

T_min      = 0.2
T_max      = 0.9
N_Temps    = 21
linear_T   = True  #If False, creates non linear T distribution
                    #following cutoff temperatures (only for N<10)

N_STEPS    = 300
chunk_size = N_SAMPLES//16
clip_norm  = None
lr         = optax.linear_schedule(0.05, 0.001, N_STEPS)
optimizer  = optax.sign_sgd(lr)
sr         = None

N_REP_COSINE = 10
# ───────────────────────────────────────────────────────────────────────────────


# ── construir hilbert, hamiltoniano y vstate ───────────────────────────────────
hi_sys = nk.hilbert.Spin(s=1/2, N=N)
hi = nk.hilbert.Spin(s=1/2, N=N+N_A)
H_sys=0
H_extended = 0
for i in range(N):
    H_sys+= h_x * sigmax(hi_sys, i)
    H_sys+= h_z * sigmaz(hi_sys, i)
    H_sys += J_ZZ * sigmaz(hi_sys, i) @ sigmaz(hi_sys, (i + 1) % N)
    H_sys += J_XX * sigmax(hi_sys, i) @ sigmax(hi_sys, (i + 1) % N)
    H_extended += h_x * sigmax(hi, i)
    H_extended += h_z * sigmaz(hi, i)
    H_extended += J_ZZ * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)
    H_extended += J_XX * sigmax(hi, i) @ sigmax(hi, (i + 1) % N)

model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16, activation=jax.nn.gelu)
sampler = nk.sampler.ARDirectSampler(hi)
vstate  = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES)

partition = list(range(N))
# ───────────────────────────────────────────────────────────────────────────────

# ── funciones auxiliares ───────────────────────────────────────────────────────
def save_results(results_file, T_idx, T, best_energy, best_entropy, best_F, cos_mean, cos_std):
    """Upsert: carga el JSON existente y actualiza/añade la entrada para T_idx."""
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            data = json.load(f)
    else:
        data = {"T": [], "energy": [], "entropy": [], "free_energy": [],
                "reliability": [], "param_index": []}

    # Extender listas si es un índice nuevo
    while len(data["T"]) <= T_idx:
        data["T"].append(None)
        data["energy"].append(None)
        data["entropy"].append(None)
        data["free_energy"].append(None)
        data["reliability"].append(None)
        data["param_index"].append(None)

    data["T"][T_idx]           = float(T)
    data["energy"][T_idx]      = float(best_energy)
    data["entropy"][T_idx]     = float(best_entropy)
    data["free_energy"][T_idx] = float(best_F)
    data["reliability"][T_idx] = {"cos_mean": float(cos_mean), "cos_std": float(cos_std)}
    data["param_index"][T_idx] = T_idx

    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)

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
    
    - Empieza con una malla base uniforme gruesa
    - Añade puntos extra concentrados alrededor de cada T_n con una
      ventana gaussiana de anchura width_factor * gap_local
    """
    # Malla base uniforme (1/3 de los puntos)
    n_base = n_total // 3
    T_base = np.linspace(T_min, T_max, n_base)
    # Puntos extra alrededor de cada salto
    T_extra = []
    jumps_in_range = T_jumps[(T_jumps > T_min) & (T_jumps < T_max)]
    # Anchura local = fracción del gap al salto más cercano
    for i, Tn in enumerate(jumps_in_range):
        gaps = np.abs(jumps_in_range - Tn)
        gaps = gaps[gaps > 1e-10]  # excluir el propio Tn
        local_gap = gaps.min() if len(gaps) > 0 else (T_max - T_min) / len(jumps_in_range)
        width = width_factor * local_gap
        
        n_local = n_total // (2 * len(jumps_in_range))  # repartir el resto entre saltos
        T_extra.append(np.linspace(Tn - 2*width, Tn + 2*width, n_local))
    # Unir, ordenar y eliminar duplicados
    T_all = np.concatenate([T_base] + T_extra)
    T_all = np.clip(T_all, T_min, T_max)
    T_all = np.unique(np.round(T_all, 8))  # elimina duplicados numéricos
    return T_all
# ───────────────────────────────────────────────────────────────────────────────

# ── crear array de temperaturas ────────────────────────────────────────────────
if N>10 and (not(linear_T)):
    linear_T=True
if not linear_T:
    eigvals, _ = np.linalg.eigh(H_sys.to_dense())
    T_jumps = compute_jump_temperatures(eigvals)
    T_array = make_optimal_T_array(T_jumps, T_min=T_min, T_max=T_max, n_total=N_Temps)
else:
    T_array = np.linspace(T_min, T_max, N_Temps)
# ───────────────────────────────────────────────────────────────────────────────

# ── ENTRENAMIENTO  ─────────────────────────────────────────────────────────────
energy_results = []
entropy_results = []
free_energy_results = []
reliability_results = []
all_histories = []
best_params = None

script_dir = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(script_dir, "..", "data")
if N==N_A:
    data_dir = os.path.join(base_data_dir, f"N{N}")
else:
    data_dir = os.path.join(base_data_dir, f"N{N}_NA_{N_A}")
params_dir = os.path.join(data_dir, "params")
results_file = os.path.join(data_dir, f"results_N{N}_vs_T.json")
os.makedirs(params_dir, exist_ok=True)

print(f"TRAINING N={N}, N_A={N_A}")
for T_idx, T in enumerate(tqdm(T_array, desc="Temperaturas")):
    print(f"\n=== Temperatura T = {T:.3f} ===")
    vstate.init_parameters()

    free_energy_history, best_F, best_energy, best_entropy = free_energy_minimize(
        vstate=vstate, T=T, partition=partition, Hamiltonian=H_extended, n_steps=N_STEPS,
        verbose=True, plot=False, optimizer=optimizer, chunk_size=chunk_size, clip_norm=clip_norm,
        sr=sr, n_samples_sr=2**12
    )
    best_params = vstate.parameters

    filename = os.path.join(params_dir, f"params_{T_idx:04d}.msgpack")
    with open(filename, "wb") as f:
        f.write(serialization.to_bytes(best_params))

    print(f"  Mejor resultado: E={best_energy:.4f}, S₂={best_entropy:.4f}, F={best_F:.4f}")

    grads = []
    for rep in range(N_REP_COSINE):
        _, grad_est = renyi2_entropy_and_grad_sampled(
            vstate, partition, N_SAMPLES, chunk_size=chunk_size
        )
        grads.append(grad_est)

    cos_vals = [cosine_similarity(grads[i], grads[j])
                for i in range(N_REP_COSINE) for j in range(i+1, N_REP_COSINE)]
    cos_mean, cos_std = np.mean(cos_vals), np.std(cos_vals)
    print(f"  Consistencia del gradiente: cos = {cos_mean:.4f} ± {cos_std:.4f}")

    save_results(results_file, T_idx, T, best_energy, best_entropy, best_F, cos_mean, cos_std)
    energy_results.append(best_energy)
    entropy_results.append(best_entropy)
    free_energy_results.append(best_F)
# ───────────────────────────────────────────────────────────────────────────────


# ── VISUALIZACIÓN  ─────────────────────────────────────────────────────────────
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.plot(T_array, energy_results, 'o-')
plt.xlabel('T'); plt.ylabel('Energía'); plt.title('Energía vs Temperatura'); plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(T_array, entropy_results, 's-')
plt.xlabel('T'); plt.ylabel('S₂'); plt.title('Entropía de Rényi-2 vs Temperatura'); plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(T_array, free_energy_results, 'd-')
plt.xlabel('T'); plt.ylabel('F'); plt.title('Energía Libre vs Temperatura'); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
