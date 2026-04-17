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

from src_renyi import free_energy_minimize, renyi2_entropy_and_grad_sampled

# ── CONFIGURACIÓN  ─────────────────────────────────────────────────────────────
N          = 50
N_SAMPLES  = 2**16
GAMMA      = -1.5
V          = -1.0
T_array    = np.linspace(0, 4, 41)
N_STEPS    = 300
chunk_size = N_SAMPLES//4
clip_norm  = None
lr         = optax.linear_schedule(0.05, 0.001, N_STEPS)
optimizer  = optax.sign_sgd(lr)

N_REP_COSINE = 10
# ───────────────────────────────────────────────────────────────────────────────


# ── construir hilbert, hamiltoniano y vstate ───────────────────────────────────
hi_sys = nk.hilbert.Spin(s=1/2, N=N)
hi_anc = nk.hilbert.Spin(s=1/2, N=N)
hi = nk.hilbert.Spin(s=1/2, N=N+N)
H_sys=0
H_extended = 0
for i in range(N):
    H_sys+= GAMMA * sigmax(hi_sys, i)
    H_sys += V * sigmaz(hi_sys, i) @ sigmaz(hi_sys, (i + 1) % N)
    H_extended += GAMMA * sigmax(hi, i)
    H_extended += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)

model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16, activation=jax.nn.gelu)
sampler = nk.sampler.ARDirectSampler(hi)
vstate  = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES)

partition = list(range(N))
# ───────────────────────────────────────────────────────────────────────────────

# ── funciones auxiliares ───────────────────────────────────────────────────────
def cosine_similarity(g1, g2):
    """Calcula el coseno entre dos gradientes (pytrees)."""
    flat1, _ = jax.flatten_util.ravel_pytree(g1)
    flat2, _ = jax.flatten_util.ravel_pytree(g2)
    flat1 = jnp.array(flat1, float)
    flat2 = jnp.array(flat2, float)
    return float(jnp.dot(flat1, flat2) /
                 (jnp.linalg.norm(flat1) * jnp.linalg.norm(flat2) + 1e-30))
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
data_dir = os.path.join(base_data_dir, f"N{N}")
params_dir = os.path.join(data_dir, "params")
os.makedirs(params_dir, exist_ok=True)

for T_idx, T in enumerate(tqdm(T_array, desc="Temperaturas")):
    print(f"\n=== Temperatura T = {T:.3f} ===")
    vstate.init_parameters()

    free_energy_history, best_F, best_energy, best_entropy = free_energy_minimize(
        vstate=vstate, T=T, partition=partition, Hamiltonian=H_extended, n_steps=N_STEPS,
        verbose=True, plot=False, optimizer=optimizer, chunk_size=chunk_size, clip_norm=clip_norm
    )
    best_params = vstate.parameters

    filename = os.path.join(params_dir, f"params_T_{T:.3f}.msgpack")
    with open(filename, "wb") as f:
        f.write(serialization.to_bytes(best_params))

    energy_results.append(best_energy)
    entropy_results.append(best_entropy)
    free_energy_results.append(best_F)

    all_histories.append({
        "T": T,
        "free_energy": free_energy_history,
    })

    print(f"  Mejor resultado: E={best_energy:.4f}, S₂={best_entropy:.4f}, F={best_F:.4f}")

    # ── EVALUACIÓN DE FIABILIDAD (coseno entre réplicas del gradiente) ──
    grads = []
    for rep in range(N_REP_COSINE):
        _, grad_est = renyi2_entropy_and_grad_sampled(
            vstate, partition, N_SAMPLES, chunk_size=chunk_size
        )
        grads.append(grad_est)
    
    # Calcular coseno medio entre todos los pares de réplicas
    cos_vals = []
    for i in range(N_REP_COSINE):
        for j in range(i+1, N_REP_COSINE):
            cos_vals.append(cosine_similarity(grads[i], grads[j]))
    
    cos_mean = np.mean(cos_vals)
    cos_std = np.std(cos_vals)
    
    reliability_results.append({
        "cos_mean": float(cos_mean),
        "cos_std": float(cos_std),
    })
    
    print(f"  Consistencia del gradiente: cos = {cos_mean:.4f} ± {cos_std:.4f}")

results_file = os.path.join(data_dir, f"results_N{N}_vs_T.txt")
data = {
    "T": [float(T) for T in T_array],
    "energy": [float(x) for x in energy_results],
    "entropy": [float(x) for x in entropy_results],
    "free_energy": [float(x) for x in free_energy_results],
    "reliability": reliability_results,
}
with open(results_file, "w") as f:
    json.dump(data, f, indent=2)
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