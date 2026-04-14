"""
training_single_T.py
===========
Entrenamiento NQS a temperatura T para minimizar <H>-TS₂.

Uso:
    Editae la sección "CONFIGURACIÓN" y ejecutar:
        python scripts/training_single_T.py
"""

import os
import sys
os.environ["JAX_PLATFORM_NAME"] = "gpu"

import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax
import numpy as np
import matplotlib.pyplot as plt
import json
import flax.serialization as serialization
from tqdm import tqdm

import importlib
try:
    importlib.reload(sys.modules["src_renyi"])
except KeyError: 
    pass
from src_renyi import free_energy_minimize

# ── CONFIGURACIÓN  ─────────────────────────────────────────────────────────────
N          = 30
N_SAMPLES  = 2**16
GAMMA      = -1.5
V          = -1.0
T          = 1.9
N_STEPS    = 300
chunk_size = N_SAMPLES//2
lr         = optax.linear_schedule(0.05, 0.001, N_STEPS)
optimizer  = optax.sign_sgd(lr)
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


# ── ENTRENAMIENTO  ─────────────────────────────────────────────────────────────
_,f_best,E_best,S_best = free_energy_minimize(vstate, T, partition, H_extended, N_STEPS, freq=20,
                                               optimizer=optimizer, clip_norm=None, timing=True,
                                               chunk_size=vstate.n_samples//16)

print(f"Best solution: S₂={S_best:.6f}, E={E_best:.6f}, F={f_best:.6f}")