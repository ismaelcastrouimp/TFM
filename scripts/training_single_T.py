"""
training_single_T.py
===========
Entrenamiento NQS a temperatura T para minimizar <H>-TS₂.

Uso interactivo:
    python scripts/training_single_T.py --T 2.5 --idx 25

Uso en SLURM job array (ver scripts/submit_sweep.sh):
    python scripts/training_single_T.py --T $T --idx $SLURM_ARRAY_TASK_ID
"""
import argparse
import fcntl
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import flax.serialization as serialization
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax

from src_renyi import (
    free_energy_minimize_phases,
    renyi2_entropy_and_grad_sampled,
    renyi2_wang_cs,
)


# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--T", type=float, required=True,
                    help="Temperatura de esta entrada del barrido.")
parser.add_argument("--idx", type=int, required=True,
                    help="Índice de esta T en el barrido (= SLURM_ARRAY_TASK_ID).")
parser.add_argument("--force", action="store_true",
                    help="Reentrenar aunque ya exista params_{idx}.")
args = parser.parse_args()

T   = args.T
idx = args.idx


# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
N, N_A = 10, 10
J_ZZ, J_XX, h_x, h_z = -1.0, 0.0, -1.5, 0.0

N_SAMPLES       = 2**14
N_STEPS_SWAP    = 400
N_STEPS_WANG    = 400
INIT_N_STEPS    = 3000
INIT_BATCH      = N_SAMPLES
WARM_EVERY      = 5
WARM_N_STEPS    = 20
WARM_BATCH      = N_SAMPLES

chunk_size = N_SAMPLES // 2

phases = [
    {
        "method": "swap",
        "n_steps": N_STEPS_SWAP,
        "optimizer": optax.sgd(optax.linear_schedule(0.05, 0.005, 2000)),
    },
    {
        "method": "wang", "n_steps": N_STEPS_WANG,
        "optimizer": optax.adam(1e-3),
        "vstate_R": None,
        "initial_train": {"n_steps": INIT_N_STEPS, "batch": INIT_BATCH, "lr": 1e-3},
        "warm_start":    {"every": WARM_EVERY, "n_steps": WARM_N_STEPS,
                          "lr": 5e-4, "batch": WARM_BATCH},
        "wang_kwargs":   {"n_samples": N_SAMPLES},
    },
]

N_REP_COSINE = 10
# ───────────────────────────────────────────────────────────────────────────────


# ── funciones auxiliares ───────────────────────────────────────────────────────
def cosine_similarity(g1, g2):
    flat1, _ = jax.flatten_util.ravel_pytree(g1)
    flat2, _ = jax.flatten_util.ravel_pytree(g2)
    flat1 = jnp.array(flat1, float)
    flat2 = jnp.array(flat2, float)
    return float(jnp.dot(flat1, flat2) /
                 (jnp.linalg.norm(flat1) * jnp.linalg.norm(flat2) + 1e-30))


def update_results_json_locked(results_file, idx, T,
                               E, S2, F, cos_mean, cos_std):
    """
    Read-modify-write del JSON de resultados con lock entre procesos.
    Sin el lock, dos jobs SLURM concurrentes pueden pisarse.
    """
    lock_path = results_file + ".lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    all_data = json.load(f)
            else:
                all_data = {
                    "T": [], "energy": [], "entropy": [], "free_energy": [],
                    "reliability": [], "param_index": [],
                }

            # Extender listas hasta cubrir idx
            while len(all_data["T"]) <= idx:
                all_data["T"].append(None)
                all_data["energy"].append(None)
                all_data["entropy"].append(None)
                all_data["free_energy"].append(None)
                all_data["reliability"].append(None)
                all_data["param_index"].append(None)

            all_data["T"][idx]           = float(T)
            all_data["energy"][idx]      = float(E)
            all_data["entropy"][idx]     = float(S2)
            all_data["free_energy"][idx] = float(F)
            all_data["reliability"][idx] = {
                "cos_mean": float(cos_mean),
                "cos_std":  float(cos_std),
            }
            all_data["param_index"][idx] = int(idx)

            with open(results_file, "w") as f:
                json.dump(all_data, f, indent=2)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
# ───────────────────────────────────────────────────────────────────────────────


# ── rutas ──────────────────────────────────────────────────────────────────────
script_dir    = os.path.dirname(os.path.abspath(__file__))
base_data_dir = os.path.join(script_dir, "..", "data")
if N == N_A:
    data_dir = os.path.join(base_data_dir, f"N{N}")
else:
    data_dir = os.path.join(base_data_dir, f"N{N}_NA_{N_A}")
params_dir = os.path.join(data_dir, "params")
os.makedirs(params_dir, exist_ok=True)

results_file = os.path.join(data_dir, f"results_N{N}_vs_T.json")
params_file  = os.path.join(params_dir, f"params_{idx:04d}.msgpack")

# Skip si ya está hecho
if os.path.exists(params_file) and not args.force:
    print(f"[skip] {params_file} ya existe. Usa --force para reentrenar.")
    raise SystemExit(0)
# ───────────────────────────────────────────────────────────────────────────────


# ── construir hilbert, hamiltoniano y vstate ───────────────────────────────────
hi_sys = nk.hilbert.Spin(s=1/2, N=N)
hi     = nk.hilbert.Spin(s=1/2, N=N + N_A)

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

model   = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                              activation=jax.nn.tanh)
sampler = nk.sampler.ARDirectSampler(hi)
vstate  = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES)

model_R   = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                                activation=jax.nn.tanh)
sampler_R = nk.sampler.ARDirectSampler(hi)
vstate_R  = nk.vqs.MCState(sampler_R, model_R, n_samples=N_SAMPLES)

partition = list(range(N))
phases[1]["vstate_R"] = vstate_R
# ───────────────────────────────────────────────────────────────────────────────


# ── ENTRENAMIENTO ──────────────────────────────────────────────────────────────
print(f"TRAINING N={N} at T={T}, N_A={N_A}, idx={idx}")

_, f_best, E_best, S_best = free_energy_minimize_phases(
    vstate, T, partition, H_extended, phases,
    chunk_size=chunk_size,
    verbose=True, freq=10,
    plot=False, timing=True,
    history_index=idx,
)
print(f"Best solution: S₂={S_best:.6f}, E={E_best:.6f}, F={f_best:.6f}")


# ── consistencia del gradiente: SWAP ───────────────────────────────────────────
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
print(f"[swap] cos = {cos_swap_mean:.4f} ± {cos_swap_std:.4f}")

# ── consistencia del gradiente: WANG ───────────────────────────────────────────
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
print(f"[wang] cos = {cos_wang_mean:.4f} ± {cos_wang_std:.4f}")


# ── guardar parámetros + JSON ──────────────────────────────────────────────────
best_params = vstate.parameters

with open(params_file, "wb") as f:
    f.write(serialization.to_bytes(best_params))

update_results_json_locked(
    results_file, idx, T,
    E_best, S_best, f_best,
    cos_wang_mean, cos_wang_std,
)

print(f"[done] T={T}  idx={idx}  →  {params_file}")