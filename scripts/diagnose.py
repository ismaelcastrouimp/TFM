"""
diagnose.py
===========
Diagnóstico previo al entrenamiento.

Calcula, para un único `chunk_size`:
  - el mayor `chunk_size` que no da OOM
  - tiempo por step real (swap / wang)
  - clip_norm recomendado

Uso:
    Editar CONFIGURACIÓN y ejecutar:
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
from src_renyi.entropy import train_reverse_network

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
N         = 50
N_A       = 50
N_SAMPLES = 2**18
GAMMA     = -1.5
V         = -1.0
T         = 1.0
N_STEPS   = 5           # steps para medir tiempo (1 JIT warmup + resto)
N_GRAD    = 20          # steps para medir normas de gradiente
LR        = 0.05        # lr de diagnóstico

# Rango de búsqueda de chunk_size (potencias de 2)
CHUNK_MIN = 2**4
CHUNK_MAX = N_SAMPLES // 2      # swap usa N_SAMPLES//2 configuraciones
WANG_KWARGS      = dict(n_samples=4096)
WANG_TRAIN_STEPS = 500
# ──────────────────────────────────────────────────────────────────────────────


# ── utilidades ────────────────────────────────────────────────────────────────

OOM_MARKERS = ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory")

def is_oom(exc: BaseException) -> bool:
    msg = str(exc)
    return any(m in msg for m in OOM_MARKERS)


def find_max_chunk_size(probe_fn, chunk_min=CHUNK_MIN, chunk_max=CHUNK_MAX,
                        label=""):
    """Dobla desde abajo; devuelve el mayor cs que no da OOM (o None)."""
    cs, best = chunk_min, None
    while cs <= chunk_max:
        try:
            probe_fn(cs)
            jax.effects_barrier()
            print(f"  ✓ {label}chunk_size = {cs}")
            best = cs
            cs *= 2
        except Exception as e:
            if is_oom(e):
                print(f"  ✗ {label}chunk_size = {cs}  → OOM, usar {best}")
                break
            else:
                print(f"  ✗ {label}chunk_size = {cs}  → error no-OOM: "
                      f"{type(e).__name__}: {e}")
                raise
    return best


# ── construir hilbert, hamiltoniano y vstates ─────────────────────────────────

hi_sys = nk.hilbert.Spin(s=1/2, N=N)
hi = nk.hilbert.Spin(s=1/2, N=N + N_A)

H_sys, H_extended = 0, 0
for i in range(N):
    H_sys += GAMMA * sigmax(hi_sys, i)
    H_sys += V * sigmaz(hi_sys, i) @ sigmaz(hi_sys, (i + 1) % N)
    H_extended += GAMMA * sigmax(hi, i)
    H_extended += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)

model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                            activation=jax.nn.gelu)
vstate = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model,
                        n_samples=N_SAMPLES)

model_R = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                              activation=jax.nn.gelu)
vstate_R = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model_R,
                          n_samples=N_SAMPLES)

partition = list(range(N))

print("=" * 60)
print(f"Diagnóstico  —  N={N}  N_A={N_A}  n_samples={N_SAMPLES}  T={T}")
print("=" * 60)


# ── 1. chunk_size único ───────────────────────────────────────────────────────
print("\n── 1. chunk_size óptimo (fuente única: op.chunk_size) ───")

def probe(cs):
    op = FreeRenyiEnergyObservable(
        hi, H_extended, partition, T, chunk_size=cs, method="swap"
    )
    _ = vstate.expect_and_grad(op)

chunk_size_opt = find_max_chunk_size(probe, label="")
if chunk_size_opt is None:
    raise RuntimeError("Ningún chunk_size funciona. Reduce N o N_SAMPLES.")


# ── 2a. Tiempo por step (swap) ────────────────────────────────────────────────
print("\n── 2a. Tiempo por step (swap) ───────────────────────────")

op_swap = FreeRenyiEnergyObservable(
    hi, H_extended, partition, T, chunk_size=chunk_size_opt, method="swap"
)
opt = optax.adam(LR)
opt_state = opt.init(vstate.parameters)
params_bak = copy.deepcopy(vstate.parameters)

times = []
for step in range(N_STEPS):
    t0 = time.perf_counter()
    F_stats, F_grad = vstate.expect_and_grad(op_swap)
    updates, opt_state = opt.update(F_grad, opt_state, vstate.parameters)
    vstate.parameters = optax.apply_updates(vstate.parameters, updates)
    jax.effects_barrier()
    if step != 0:
        times.append(time.perf_counter() - t0)

t_swap_mean, t_swap_std = float(np.mean(times)), float(np.std(times))
print(f"  t/step = {t_swap_mean:.3f}s ± {t_swap_std:.3f}s")
print(f"  500 steps ~ {500 * t_swap_mean / 60:.1f} min")
vstate.parameters = params_bak


# ── 2b. Tiempo por step (wang) ────────────────────────────────────────────────
print("\n── 2b. Tiempo por step (wang) ───────────────────────────")
print(f"    Pre-entrenando N_R ({WANG_TRAIN_STEPS} steps) ...")
train_reverse_network(
    vstate, vstate_R, n_steps=WANG_TRAIN_STEPS, batch=1024, lr=1e-3,
    verbose=False,
)

op_wang = FreeRenyiEnergyObservable(
    hi, H_extended, partition, T, chunk_size=chunk_size_opt, method="wang",
    vstate_R=vstate_R, wang_kwargs=WANG_KWARGS,
)
opt2 = optax.adam(LR)
opt_state2 = opt2.init(vstate.parameters)
vstate.parameters = params_bak

times_w = []
for step in range(N_STEPS):
    t0 = time.perf_counter()
    F_stats, F_grad = vstate.expect_and_grad(op_wang)
    updates, opt_state2 = opt2.update(F_grad, opt_state2, vstate.parameters)
    vstate.parameters = optax.apply_updates(vstate.parameters, updates)
    jax.effects_barrier()
    if step != 0:
        times_w.append(time.perf_counter() - t0)

t_wang_mean, t_wang_std = float(np.mean(times_w)), float(np.std(times_w))
print(f"  t/step = {t_wang_mean:.3f}s ± {t_wang_std:.3f}s")
print(f"  500 steps ~ {500 * t_wang_mean / 60:.1f} min")
vstate.parameters = params_bak


# ── 3. clip_norm ──────────────────────────────────────────────────────────────
print("\n── 3. Normas de gradiente y clip_norm (swap) ────────────")

op_clip = FreeRenyiEnergyObservable(
    hi, H_extended, partition, T, chunk_size=chunk_size_opt, method="swap"
)
opt3 = optax.adam(LR)
opt_state3 = opt3.init(vstate.parameters)

norms = []
for _ in range(N_GRAD):
    F_stats, F_grad = vstate.expect_and_grad(op_clip)
    flat, _ = jax.flatten_util.ravel_pytree(F_grad)
    norms.append(float(jnp.linalg.norm(flat)))
    updates, opt_state3 = opt3.update(F_grad, opt_state3, vstate.parameters)
    vstate.parameters = optax.apply_updates(vstate.parameters, updates)

norms = jnp.array(norms)
med = float(jnp.median(norms))
p90 = float(jnp.percentile(norms, 90))
p95 = float(jnp.percentile(norms, 95))
mx  = float(jnp.max(norms))
spread = mx / (med + 1e-10)
clip_rec = p90 if spread > 5 else p95

print(f"  median : {med:.4f}")
print(f"  p90    : {p90:.4f}")
print(f"  p95    : {p95:.4f}")
print(f"  max    : {mx:.4f}  (max/median = {spread:.1f})")
print(f"  → clip_norm = {clip_rec:.4f} "
      f"({'p90, cola larga' if spread > 5 else 'p95, distribución compacta'})")
vstate.parameters = params_bak


# ── resumen final ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print(f"  N, N_A           : {N}, {N_A}")
print(f"  n_samples        : {N_SAMPLES}")
print(f"  chunk_size       : {chunk_size_opt}  "
      f"({chunk_size_opt / N_SAMPLES:.4f} × N_SAMPLES, "
      f"i.e. N_SAMPLES/{N_SAMPLES // chunk_size_opt})")
print(f"  t/step (swap)    : {t_swap_mean:.3f}s ± {t_swap_std:.3f}s")
print(f"  t/step (wang)    : {t_wang_mean:.3f}s ± {t_wang_std:.3f}s")
print(f"  clip_norm        : {clip_rec:.4f}")
print("=" * 60)