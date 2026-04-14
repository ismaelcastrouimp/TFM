"""
diagnose.py
===========
Diagnóstico previo al entrenamiento para una configuración (N, n_samples).

Calcula:
  - chunk_size óptimo (el mayor que no da OOM)
  - tiempo real por step (excluyendo el primer step de compilación JIT)
  - clip_norm recomendado a partir de las normas de gradiente

Uso:
    Edita la sección "CONFIGURACIÓN" y ejecuta:
        python scripts/diagnose.py
"""

import time
import copy
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax

from src_renyi.observables import FreeRenyiEnergyObservable
from src_renyi.training import free_energy_minimize

# ── CONFIGURACIÓN  ─────────────────────────────────────────────────
N         = 30
N_SAMPLES = 2**19
GAMMA     = -1.5
V         = -1.0
T         = 1.0         # temperatura de diagnóstico
N_STEPS   = 5           # steps totales (1 de JIT warmup + N_STEPS-1 medidos)
N_GRAD    = 20           # steps para medir normas de gradiente y clip_norm
LR        = 0.05         # lr para el diagnóstico (no afecta a los resultados)
# ──────────────────────────────────────────────────────────────────────────────

# ── construir hilbert, hamiltoniano y vstate ───────────────────────────────────
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
sampler = nk.sampler.ARDirectSampler(hi)
vstate  = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES)

partition = list(range(N))

print("=" * 60)
print(f"Diagnóstico  —  N={N}  n_samples={N_SAMPLES}  T={T}")
print("=" * 60)


# ── 1. chunk_size ──────────────────────────────────────────────────────────────
print("\n── 1. Buscando chunk_size ────────────────────────────────")

n = N_SAMPLES // 2
candidate = n 
chunk_size = None

while candidate >= 1:
    if n % candidate != 0:
        candidate //= 2
        continue
    try:
        op_test = FreeRenyiEnergyObservable(
            hi, H_extended, partition, T, chunk_size=candidate
        )
        _ = vstate.expect_and_grad(op_test)
        jax.effects_barrier()
        print(f"  ✓ chunk_size = {candidate}  (n // {N_SAMPLES // candidate})")
        chunk_size = candidate
        break
    except Exception as e:
        print(f"  ✗ chunk_size = {candidate}  → OOM")
        candidate //= 2

if chunk_size is None:
    raise RuntimeError("No se encontró chunk_size válido. Reduce N o n_samples.")



# ── 2. tiempo por step ────────────────────────────────────────────────────────
print("\n── 2. Tiempo por step ────────────────────────────────────")

op_time   = FreeRenyiEnergyObservable(hi, H_extended, partition, T, chunk_size=chunk_size)
optimizer = optax.sgd(LR)
opt_state = optimizer.init(vstate.parameters)
params_bak = copy.deepcopy(vstate.parameters)

step_times = []

for step in range(N_STEPS):
    t0 = time.perf_counter()
    F_stats, F_grad = vstate.expect_and_grad(op_time)
    updates, opt_state = optimizer.update(F_grad, opt_state, vstate.parameters)
    vstate.parameters = optax.apply_updates(vstate.parameters, updates)
    jax.effects_barrier()
    elapsed = time.perf_counter() - t0

    if step != 0:
        step_times.append(elapsed)
        

step_times = np.array(step_times)
t_mean = step_times.mean()
t_std  = step_times.std()

print(f"  Tiempo por step : {t_mean:.3f}s ± {t_std:.3f}s")
print(f"  Estimación 500 steps : {500 * t_mean / 60:.1f} min")

# restaurar parámetros originales
vstate.parameters = params_bak


# ── 3. clip_norm ──────────────────────────────────────────────────────────────
print("\n── 3. Normas de gradiente y clip_norm ───────────────────")

op_clip   = FreeRenyiEnergyObservable(hi, H_extended, partition, T, chunk_size=chunk_size)
optimizer2 = optax.sgd(LR)
opt_state2 = optimizer2.init(vstate.parameters)

grad_norms = []

for step in range(N_GRAD):
    F_stats, F_grad = vstate.expect_and_grad(op_clip)
    flat, _ = jax.flatten_util.ravel_pytree(F_grad)
    grad_norms.append(float(jnp.linalg.norm(flat)))
    updates, opt_state2 = optimizer2.update(F_grad, opt_state2, vstate.parameters)
    vstate.parameters = optax.apply_updates(vstate.parameters, updates)

grad_norms = jnp.array(grad_norms)
med  = float(jnp.median(grad_norms))
p75  = float(jnp.percentile(grad_norms, 75))
p90  = float(jnp.percentile(grad_norms, 90))
p95  = float(jnp.percentile(grad_norms, 95))
mx   = float(jnp.max(grad_norms))
spread = mx / (med + 1e-10)

clip_recommended = p90 if spread > 5 else p95

print(f"  median : {med:.4f}")
print(f"  p75    : {p75:.4f}")
print(f"  p90    : {p90:.4f}")
print(f"  p95    : {p95:.4f}")
print(f"  max    : {mx:.4f}  (ratio max/median = {spread:.1f})")
print(f"  {'cola larga → usando p90' if spread > 5 else 'distribución compacta → usando p95'}")


# ── resumen final ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print(f"  N                : {N}")
print(f"  n_samples        : {N_SAMPLES}")
print(f"  chunk_size       : {chunk_size}")
print(f"  tiempo por step  : {t_mean:.3f}s ± {t_std:.3f}s")
print(f"  500 steps        : ~{500 * t_mean / 60:.1f} min")
print(f"  clip_norm        : {clip_recommended:.4f}")
