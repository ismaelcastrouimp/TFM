"""
training_single_T.py
===========
Entrenamiento NQS a temperatura T para minimizar <H>-TS₂.

Uso:
    Editar la sección "CONFIGURACIÓN" y ejecutar:
        python scripts/training_single_T.py
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
from src_renyi import free_energy_minimize, renyi2_entropy_and_grad_sampled, renyi2_entropy_and_grad_exact, renyi2_drut_sampling

# ── CONFIGURACIÓN  ─────────────────────────────────────────────────────────────
N          = 10
N_A        = 10
N_SAMPLES  = 2**10

J_ZZ       = -1.0
J_XX       = 0.0
h_x        = -1.5
h_z        = 0.0

T          = 4
N_STEPS    = 350
N_STEPS_FINE   = 20

chunk_size = N_SAMPLES//2
clip_norm  = None
lr         = optax.linear_schedule(0.05, 0.001, N_STEPS)
lr_fine    = None
optimizer  = optax.sign_sgd(lr)

drut_kwargs = dict(n_chains=512, n_lambda=20, n_sweeps_per_lam=100, n_props_per_sweep=4*N, K=1)

N_REP_COSINE = 10
N_REP_DRUT = 3
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


# ── ENTRENAMIENTO  ─────────────────────────────────────────────────────────────
print(f"TRAINING N={N} at T={T}, N_A={N_A}")
_,f_best,E_best,S_best = free_energy_minimize(vstate, T, partition, H_extended, N_STEPS, freq=20,
                                               optimizer=optimizer, clip_norm=clip_norm, timing=True,
                                               chunk_size=chunk_size, plot=True, 
                                               fine_steps=N_STEPS_FINE, fine_drut_kwargs=drut_kwargs, fine_lr=lr_fine)

print(f"Best solution: S₂={S_best:.6f}, E={E_best:.6f}, F={f_best:.6f}")

# ── consistencia del gradiente: SWAP ─────────────────────────────────────────
grads_swap = []
for rep in range(N_REP_COSINE):
    _, grad_est = renyi2_entropy_and_grad_sampled(
        vstate, partition, N_SAMPLES, chunk_size=chunk_size
    )
    grads_swap.append(grad_est)

cos_vals_swap = []
for i in range(N_REP_COSINE):
    for j in range(i + 1, N_REP_COSINE):
        cos_vals_swap.append(cosine_similarity(grads_swap[i], grads_swap[j]))
cos_swap_mean = np.mean(cos_vals_swap)
cos_swap_std  = np.std(cos_vals_swap)
print(f"[swap] cos = {cos_swap_mean:.4f} ± {cos_swap_std:.4f}")

# ── consistencia del gradiente: DRUT ─────────────────────────────────────────
grads_drut = []
for rep in range(N_REP_DRUT):
    _, grad_est = renyi2_drut_sampling(
        vstate, partition,
        n_chains=512, n_lambda=20, n_sweeps_per_lam=100,
        n_props_per_sweep=4 * N, K=1, key=rep,
    )
    grads_drut.append(grad_est)

cos_vals_drut = []
for i in range(N_REP_DRUT):
    for j in range(i + 1, N_REP_DRUT):
        cos_vals_drut.append(cosine_similarity(grads_drut[i], grads_drut[j]))
cos_drut_mean = np.mean(cos_vals_drut)
cos_drut_std  = np.std(cos_vals_drut)
print(f"[drut] cos = {cos_drut_mean:.4f} ± {cos_drut_std:.4f}")

# ── guardar parámetros ─────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(script_dir, "..", "data")
if N==N_A:
    data_dir = os.path.join(base_data_dir, f"N{N}")
else:
    data_dir = os.path.join(base_data_dir, f"N{N}_NA_{N_A}")
params_dir = os.path.join(data_dir, "params")
os.makedirs(params_dir, exist_ok=True)
best_params = vstate.parameters

results_file = os.path.join(data_dir, f"results_N{N}_vs_T.json")

# Cargar JSON existente o crear uno vacío
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        all_data = json.load(f)
else:
    all_data = {"T": [], "energy": [], "entropy": [], "free_energy": [],
                "reliability": [], "param_index": []}

# Buscar si esta T ya existe (para sobreescribir) o añadir
if T in all_data["T"]:
    idx = all_data["T"].index(T)
else:
    idx = len(all_data["T"])
    all_data["T"].append(float(T))
    all_data["energy"].append(None)
    all_data["entropy"].append(None)
    all_data["free_energy"].append(None)
    all_data["reliability"].append(None)
    all_data["param_index"].append(idx)

# Actualizar valores
all_data["energy"][idx]      = float(E_best)
all_data["entropy"][idx]     = float(S_best)
all_data["free_energy"][idx] = float(f_best)
all_data["reliability"][idx] = {"cos_mean": float(cos_swap_mean), "cos_std": float(cos_swap_std)}

# Guardar parámetros con índice
filename = os.path.join(params_dir, f"params_{idx:04d}.msgpack")
with open(filename, "wb") as f:
    f.write(serialization.to_bytes(best_params))

# Guardar JSON actualizado
with open(results_file, "w") as f:
    json.dump(all_data, f, indent=2)
