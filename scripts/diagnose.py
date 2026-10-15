"""
diagnose.py
===========
Diagnóstico previo al entrenamiento.

Devuelve, para una única configuración:
  1a. Mayor `chunk_size` sin OOM (swap).
  2a. Tiempo por step real (swap).
  2c. Warmup rápido de vstate (pocos pasos de swap).
  3a. clip_norm recomendado (swap).
  1b. Mayor `chunk_size` sin OOM (wang).
  2b. Tiempo por step real (wang).
  3b. clip_norm recomendado (wang).
  4.  Curva DKL(step) de la red reversa para varios lr → elige lr óptimo
      y entrena vstate_R final.
  5.  Escalado de std(∇S2) con n_samples (Wang) → n_samples_wang óptimo.
  6a. Deriva DKL al entrenar vstate con **Wang** (no swap) sin warm-start
      → WARM_EVERY sugerido.
  6b. Recuperación DKL con warm-start → WARM_N_STEPS sugerido.
  7.  Contabilidad completa del coste Wang (init + main + warm-starts).
  8.  Memoria del proceso (VmHWM) → --mem sugerido para SLURM.

DISEÑO ROBUSTO: todas las secciones a partir de la 5 están envueltas en
try/except. Si una falla (p. ej. OOM en un budget grande), se reporta y
el script continúa hasta el resumen final.

Uso:
    Editar CONFIGURACIÓN y ejecutar:
        python scripts/diagnose.py
"""

import time
import copy
import traceback
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax

from src_renyi.observables import FreeRenyiEnergyObservable
from src_renyi.entropy import (
    train_reverse_network,
    renyi2_wang_cs,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
N         = 50
N_A       = 50
N_SAMPLES = 2**15
GAMMA     = -1.5
V         = -1.0
T         = 2

# LRs de diagnóstico (deben reflejar los LRs reales de producción)
LR_SWAP   = 0.05      # swap + SGD
LR_WANG   = 1e-3      # wang + adam
LR_DRIFT  = LR_WANG   # el drift lo medimos con el LR real de Wang

# Warmup de vstate (pocos pasos de swap)
WARMUP_SWAP_STEPS = 100
WARMUP_LR_INIT    = 0.05
WARMUP_LR_FINAL   = 0.005

# chunk_size
CHUNK_MIN = 2**4
CHUNK_MAX = N_SAMPLES // 2

# Cap duro para chunk_wang: el probe 1b.2 solo ve 1 chunk, pero en
# producción con n_samples grandes hay múltiples y JAX materializa
# tensores intermedios más grandes. 8192 está verificado.
CHUNK_WANG_HARD_CAP = 8192

# tiempos por step
N_STEPS   = 5

# normas de gradiente
N_GRAD    = 20

# Wang para diagnóstico preliminar (2b, 3b, 5) y para el drift (6)
WANG_KWARGS_DIAG = dict(n_samples=4096)

# ── Sección 4: barrido de lr para la red reversa ──
DKL_LRS          = [3e-4, 1e-3, 3e-3]
DKL_BATCH        = 1024
DKL_N_STEPS_MAX  = 5000
DKL_SMOOTH_WIN   = 200
DKL_PLATEAU_TOL  = 0.02
DKL_PRINT_FREQ   = 500

# ── Sección 5: escalado con n_samples ──
NS_BUDGET_LIST   = [2**8, 2**10, 2**12, 2**14, 2**16]
NS_N_REP         = 20
NS_EPS_GRAD      = 0.10

# ── Sección 6: deriva y recuperación (drift con Wang) ──
DRIFT_N_STEPS       = 100
DRIFT_DKL_SAMPLES   = 4096
DRIFT_DKL_THRESH    = 2.0

RECOVER_N_STEPS_MAX = 300
RECOVER_DKL_SAMPLES = 4096
RECOVER_TOL         = 1.10
RECOVER_PRINT_CURVE = True

# ── Sección 7: contabilidad ──
INIT_BATCH_HINT       = 4096
WARM_BATCH_HINT       = 2**14
WANG_N_STEPS_HINT     = 400

# ──────────────────────────────────────────────────────────────────────────────


# ── utilidades ────────────────────────────────────────────────────────────────

OOM_MARKERS = ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory")


def is_oom(exc: BaseException) -> bool:
    msg = str(exc)
    return any(m in msg for m in OOM_MARKERS)


def find_max_chunk_size(probe_fn, chunk_min=CHUNK_MIN, chunk_max=CHUNK_MAX,
                        label=""):
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
            else:
                print(f"  ✗ {label}chunk_size = {cs}  → error no-OOM: "
                      f"{type(e).__name__}: {e}")
            break
    return best


def cosine_similarity(g1, g2):
    flat1, _ = jax.flatten_util.ravel_pytree(g1)
    flat2, _ = jax.flatten_util.ravel_pytree(g2)
    flat1 = jnp.array(flat1, float)
    flat2 = jnp.array(flat2, float)
    return float(jnp.dot(flat1, flat2) /
                 (jnp.linalg.norm(flat1) * jnp.linalg.norm(flat2) + 1e-30))


def grad_norm(g):
    flat, _ = jax.flatten_util.ravel_pytree(g)
    return float(jnp.linalg.norm(jnp.array(flat, float)))


def moving_average(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x.copy()
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def detect_plateau(history, window=DKL_SMOOTH_WIN, tol_rel=DKL_PLATEAU_TOL):
    h = np.asarray(history, dtype=float)
    if len(h) < 2 * window:
        return len(h) - 1, float(h[-1])
    h_smooth = moving_average(h, window)
    offset = window // 2
    for i in range(len(h_smooth) - window):
        improvement = (h_smooth[i] - h_smooth[i + window]) / (
            abs(h_smooth[i]) + 1e-10
        )
        if improvement < tol_rel:
            return i + offset, float(h_smooth[i])
    return len(h) - 1, float(h[-1])


def compute_kl_local(vstate, vstate_R, n_samples=4096):
    s = vstate.sample(n_samples=n_samples).reshape(-1, vstate.hilbert.size)
    lp = 2.0 * jnp.real(vstate._apply_fun(
        {"params": vstate.parameters, **vstate.model_state}, s))
    lpR = 2.0 * jnp.real(vstate_R._apply_fun(
        {"params": vstate_R.parameters, **vstate_R.model_state},
        jnp.flip(s, axis=-1)))
    return float(jnp.mean(lp - lpR))


def measure_rev_step_time(vstate, vstate_R, batch, n_measure=2):
    params_R_bak = copy.deepcopy(vstate_R.parameters)
    opt = optax.adam(1e-3)
    opt_state = opt.init(vstate_R.parameters)

    def loss_fn(params_R, s):
        log_p = 2.0 * jnp.real(vstate._apply_fun(
            {"params": vstate.parameters, **vstate.model_state}, s))
        log_pR = 2.0 * jnp.real(vstate_R._apply_fun(
            {"params": params_R, **vstate_R.model_state},
            jnp.flip(s, axis=-1)))
        return jnp.mean(log_p - log_pR)

    times = []
    for step in range(n_measure + 1):
        s = vstate.sample(n_samples=batch).reshape(-1, vstate.hilbert.size)
        t0 = time.perf_counter()
        _, grads = jax.value_and_grad(loss_fn)(vstate_R.parameters, s)
        updates, opt_state = opt.update(grads, opt_state, vstate_R.parameters)
        vstate_R.parameters = optax.apply_updates(vstate_R.parameters, updates)
        jax.effects_barrier()
        if step != 0:
            times.append(time.perf_counter() - t0)

    vstate_R.parameters = params_R_bak
    return float(np.mean(times))


def print_curve(history, window=100, label="", n_points=10):
    h = np.asarray(history, dtype=float)
    h_smooth = moving_average(h, window)
    if len(h_smooth) == 0:
        return
    idx = np.linspace(0, len(h_smooth) - 1, n_points).astype(int)
    print(f"    curva suavizada ({label}):")
    for i in idx:
        print(f"      step≈{i + window // 2:>5}  DKL≈{h_smooth[i]:.4f}")


def section_guard(name):
    """Decorador simple para envolver bloques en try/except."""
    class _Guard:
        def __enter__(self):
            print(f"\n>>> inicio: {name}")
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                print(f"<<< fin: {name}")
                return False
            if is_oom(exc_val):
                print(f"!!! {name} abortada por OOM: {exc_val}")
            else:
                print(f"!!! {name} abortada por "
                      f"{exc_type.__name__}: {exc_val}")
            # Imprimir traza corta para diagnóstico
            tb = "".join(traceback.format_exception(exc_type, exc_val,
                                                    exc_tb, limit=3))
            print("    " + tb.replace("\n", "\n    ").rstrip())
            return True   # swallow exception
    return _Guard()


# ── memoria del proceso y GPU ────────────────────────────────────────────────

def get_process_memory_mb():
    keys = ("VmRSS", "VmHWM", "VmSize", "VmPeak")
    info = {k: 0.0 for k in keys}
    with open("/proc/self/status") as f:
        for line in f:
            for k in keys:
                if line.startswith(k + ":"):
                    info[k] = int(line.split()[1]) / 1024.0
    return info


def get_gpu_memory_mb():
    out = {}
    for d in jax.local_devices():
        try:
            stats = d.memory_stats()
        except Exception:
            stats = None
        if stats and "bytes_in_use" in stats and "bytes_limit" in stats:
            out[d.id] = (
                stats["bytes_in_use"] / 1024.0 ** 2,
                stats["bytes_limit"]   / 1024.0 ** 2,
            )
    return out


def memory_report(label=""):
    mem = get_process_memory_mb()
    gpu = get_gpu_memory_mb()

    print(f"  memoria del proceso ({label}):")
    print(f"    VmRSS  (ahora)   = {mem['VmRSS']/1024:.2f} GB")
    print(f"    VmHWM  (pico)    = {mem['VmHWM']/1024:.2f} GB   ← contra --mem")
    print(f"    VmSize (ahora)   = {mem['VmSize']/1024:.2f} GB   (virtual)")
    print(f"    VmPeak (pico)    = {mem['VmPeak']/1024:.2f} GB   (virtual)")

    if gpu:
        for dev_id, (used, limit) in gpu.items():
            print(f"    GPU {dev_id}: {used/1024:.2f} GB usados / "
                  f"{limit/1024:.2f} GB límite")
    else:
        print(f"    GPU: no disponible o stats no accesibles")

    suggested_mem_gb = max(4, int(np.ceil(mem["VmHWM"] / 1024 * 3)))
    print(f"    → --mem sugerido (3 × VmHWM): {suggested_mem_gb} GB")
    return suggested_mem_gb


# ── construir hilbert, hamiltoniano y vstates ────────────────────────────────

hi = nk.hilbert.Spin(s=1/2, N=N + N_A)

H_extended = 0
for i in range(N):
    H_extended += GAMMA * sigmax(hi, i)
    H_extended += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)

model   = nk.models.ARNNDense(hilbert=hi, layers=2, features=24,
                              activation=jax.nn.gelu)
vstate  = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model,
                         n_samples=N_SAMPLES)

model_R   = nk.models.ARNNDense(hilbert=hi, layers=2, features=24,
                                activation=jax.nn.gelu)
vstate_R  = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model_R,
                           n_samples=N_SAMPLES)

partition = list(range(N))

print("=" * 66)
print(f"Diagnóstico  —  N={N}  N_A={N_A}  n_samples={N_SAMPLES}  T={T}")
print("=" * 66)


# ── estado global para el resumen (defaults por si algo falla) ───────────────
results_summary = {
    "chunk_swap":         None,
    "chunk_wang":         None,
    "chunk_wang_energy":  None,
    "t_swap_mean":        None,
    "t_wang_mean":        None,
    "clip_swap":          None,
    "clip_wang":          None,
    "best_lr":            None,
    "best_dkl":           None,
    "suggested_init_n_steps": None,
    "C_g":                None,
    "slope":              None,
    "mean_grad_norm":     None,
    "n_samples_wang_opt_log2": None,
    "n_samples_wang_opt": None,
    "warm_every_rec":     None,
    "warm_n_steps_rec":   None,
    "t_init_phase":       None,
    "t_main_phase":       None,
    "t_warm_phase":       None,
    "t_total":            None,
    "frac_net":           None,
    "F_warmup_last":      None,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1a. chunk_size óptimo (swap)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 1a. chunk_size óptimo (swap) ─────────────────────────────")

def probe_swap(cs):
    op = FreeRenyiEnergyObservable(
        hi, H_extended, partition, T, chunk_size=cs, method="swap"
    )
    _ = vstate.expect_and_grad(op)

chunk_swap = find_max_chunk_size(probe_swap, label="[swap] ")
if chunk_swap is None:
    chunk_swap = 1024
    print(f"  ⚠ ningún chunk_size válido para swap; usando {chunk_swap}")
results_summary["chunk_swap"] = chunk_swap


# ══════════════════════════════════════════════════════════════════════════════
# 2a. Tiempo por step (swap)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 2a. Tiempo por step (swap) ───────────────────────────────")

op_swap = FreeRenyiEnergyObservable(
    hi, H_extended, partition, T, chunk_size=chunk_swap, method="swap"
)
opt = optax.sgd(LR_SWAP)
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

t_swap_mean = float(np.mean(times)) if times else float("nan")
t_swap_std  = float(np.std(times))  if times else float("nan")
print(f"  t/step = {t_swap_mean:.3f}s ± {t_swap_std:.3f}s")
print(f"  500 steps ~ {500 * t_swap_mean / 60:.1f} min")
vstate.parameters = params_bak
results_summary["t_swap_mean"] = t_swap_mean


# ══════════════════════════════════════════════════════════════════════════════
# 2c. Warmup de vstate (swap, pocos pasos)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n── 2c. Warmup de vstate ({WARMUP_SWAP_STEPS} steps de swap) ──")

warmup_lr = optax.linear_schedule(
    WARMUP_LR_INIT, WARMUP_LR_FINAL, WARMUP_SWAP_STEPS
)
opt_warmup = optax.sgd(warmup_lr)
opt_state_warmup = opt_warmup.init(vstate.parameters)

F_warmup_last = None
for step in range(WARMUP_SWAP_STEPS):
    F_stats, F_grad = vstate.expect_and_grad(op_swap)
    updates, opt_state_warmup = opt_warmup.update(
        F_grad, opt_state_warmup, vstate.parameters
    )
    vstate.parameters = optax.apply_updates(vstate.parameters, updates)
    F_warmup_last = float(F_stats.mean.real)
jax.effects_barrier()

print(f"  F tras warmup = {F_warmup_last:.4f}")
results_summary["F_warmup_last"] = F_warmup_last

# Re-capturamos params_bak en el estado warmed-up
params_bak = copy.deepcopy(vstate.parameters)
print(f"  params_bak renovado al estado warmed-up")


# ══════════════════════════════════════════════════════════════════════════════
# 3a. Normas de gradiente y clip_norm (swap)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3a. Normas de gradiente y clip_norm (swap) ──────────────")

opt3 = optax.sgd(LR_SWAP)
opt_state3 = opt3.init(vstate.parameters)

norms_swap = []
for _ in range(N_GRAD):
    _, F_grad = vstate.expect_and_grad(op_swap)
    norms_swap.append(grad_norm(F_grad))
    updates, opt_state3 = opt3.update(F_grad, opt_state3, vstate.parameters)
    vstate.parameters = optax.apply_updates(vstate.parameters, updates)

norms_swap = jnp.array(norms_swap)
med_s = float(jnp.median(norms_swap))
p90_s = float(jnp.percentile(norms_swap, 90))
p95_s = float(jnp.percentile(norms_swap, 95))
mx_s  = float(jnp.max(norms_swap))
spread_s = mx_s / (med_s + 1e-10)
clip_swap = p90_s if spread_s > 5 else p95_s

print(f"  median : {med_s:.4f}")
print(f"  p90    : {p90_s:.4f}")
print(f"  p95    : {p95_s:.4f}")
print(f"  max    : {mx_s:.4f}  (max/median = {spread_s:.1f})")
print(f"  → clip_norm = {clip_swap:.4f} "
      f"({'p90, cola larga' if spread_s > 5 else 'p95, distribución compacta'})")
vstate.parameters = params_bak
results_summary["clip_swap"] = clip_swap


# ══════════════════════════════════════════════════════════════════════════════
# 1b. chunk_size óptimo (wang) — ENERGÍA y ENTROPÍA por separado
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 1b.1. chunk_size óptimo (wang · ENERGÍA) ─────────────────")
print("    (con vstate_R virgen — solo medimos coste computacional)")

CHUNK_WANG_S_FIXED = min(N_SAMPLES // 2, CHUNK_WANG_HARD_CAP)

def probe_wang_energy(cs):
    op = FreeRenyiEnergyObservable(
        hi, H_extended, partition, T,
        chunk_size=CHUNK_WANG_S_FIXED,
        chunk_size_energy=cs,
        method="wang",
        vstate_R=vstate_R, wang_kwargs=WANG_KWARGS_DIAG,
    )
    _ = vstate.expect_and_grad(op)

chunk_wang_energy = find_max_chunk_size(probe_wang_energy, label="[wang·E] ")
if chunk_wang_energy is None:
    chunk_wang_energy = 1024
    print(f"  ⚠ ningún chunk_size válido para la energía de Wang; "
          f"usando {chunk_wang_energy}")

# Cap de seguridad para energía
if chunk_wang_energy > CHUNK_WANG_HARD_CAP:
    print(f"  ⚠ cap energía: {chunk_wang_energy} → {CHUNK_WANG_HARD_CAP}")
    chunk_wang_energy = CHUNK_WANG_HARD_CAP


print("\n── 1b.2. chunk_size óptimo (wang · ENTROPÍA) ────────────────")

def probe_wang_S(cs):
    op = FreeRenyiEnergyObservable(
        hi, H_extended, partition, T,
        chunk_size=cs,
        chunk_size_energy=chunk_wang_energy,
        method="wang",
        vstate_R=vstate_R, wang_kwargs=WANG_KWARGS_DIAG,
    )
    _ = vstate.expect_and_grad(op)

chunk_wang = find_max_chunk_size(probe_wang_S, label="[wang·S] ")
if chunk_wang is None:
    chunk_wang = chunk_swap
    print(f"  ⚠ ningún chunk_size válido para Wang; usando {chunk_wang}")

# ── Cap de seguridad: 8192 (verificado con múltiples chunks) ─────────────
if chunk_wang > CHUNK_WANG_HARD_CAP:
    print(f"  ⚠ cap de seguridad Wang·S: {chunk_wang} → "
          f"{CHUNK_WANG_HARD_CAP} (evita OOM con múltiples chunks)")
    chunk_wang = CHUNK_WANG_HARD_CAP

results_summary["chunk_wang"]        = chunk_wang
results_summary["chunk_wang_energy"] = chunk_wang_energy


# ══════════════════════════════════════════════════════════════════════════════
# 2b. Tiempo por step (wang)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 2b. Tiempo por step (wang) ───────────────────────────────")

op_wang = FreeRenyiEnergyObservable(
    hi, H_extended, partition, T,
    chunk_size=chunk_wang,
    chunk_size_energy=chunk_wang_energy,
    method="wang",
    vstate_R=vstate_R, wang_kwargs=WANG_KWARGS_DIAG,
)
opt2 = optax.adam(LR_WANG)
opt_state2 = opt2.init(vstate.parameters)

times_w = []
for step in range(N_STEPS):
    t0 = time.perf_counter()
    F_stats, F_grad = vstate.expect_and_grad(op_wang)
    updates, opt_state2 = opt2.update(F_grad, opt_state2, vstate.parameters)
    vstate.parameters = optax.apply_updates(vstate.parameters, updates)
    jax.effects_barrier()
    if step != 0:
        times_w.append(time.perf_counter() - t0)

t_wang_mean = float(np.mean(times_w)) if times_w else float("nan")
t_wang_std  = float(np.std(times_w))  if times_w else float("nan")
print(f"  t/step = {t_wang_mean:.3f}s ± {t_wang_std:.3f}s")
print(f"  500 steps ~ {500 * t_wang_mean / 60:.1f} min")
vstate.parameters = params_bak
results_summary["t_wang_mean"] = t_wang_mean


# ══════════════════════════════════════════════════════════════════════════════
# 3b. Normas de gradiente y clip_norm (wang)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3b. Normas de gradiente y clip_norm (wang) ──────────────")

opt3b = optax.adam(LR_WANG)
opt_state3b = opt3b.init(vstate.parameters)

norms_wang = []
for _ in range(N_GRAD):
    _, F_grad = vstate.expect_and_grad(op_wang)
    norms_wang.append(grad_norm(F_grad))
    updates, opt_state3b = opt3b.update(F_grad, opt_state3b, vstate.parameters)
    vstate.parameters = optax.apply_updates(vstate.parameters, updates)

norms_wang = jnp.array(norms_wang)
med_w = float(jnp.median(norms_wang))
p90_w = float(jnp.percentile(norms_wang, 90))
p95_w = float(jnp.percentile(norms_wang, 95))
mx_w  = float(jnp.max(norms_wang))
spread_w = mx_w / (med_w + 1e-10)
clip_wang = p90_w if spread_w > 5 else p95_w

print(f"  median : {med_w:.4f}")
print(f"  p90    : {p90_w:.4f}")
print(f"  p95    : {p95_w:.4f}")
print(f"  max    : {mx_w:.4f}  (max/median = {spread_w:.1f})")
print(f"  → clip_norm = {clip_wang:.4f} "
      f"({'p90, cola larga' if spread_w > 5 else 'p95, distribución compacta'})")
vstate.parameters = params_bak
results_summary["clip_wang"] = clip_wang


# ══════════════════════════════════════════════════════════════════════════════
# 4. Curva DKL(step) de la red reversa para varios lr
# ══════════════════════════════════════════════════════════════════════════════
with section_guard("4. Barrido de lr red reversa"):
    print(f"    lrs: {DKL_LRS}   batch: {DKL_BATCH}   "
          f"n_steps_max: {DKL_N_STEPS_MAX}")

    dkl_curves = {}
    best_lr = None
    best_dkl = float("inf")

    for lr in DKL_LRS:
        print(f"\n    lr = {lr:.1e}")
        model_R_dkl = nk.models.ARNNDense(hilbert=hi, layers=2, features=24,
                                          activation=jax.nn.gelu)
        vstate_R_dkl = nk.vqs.MCState(
            nk.sampler.ARDirectSampler(hi), model_R_dkl,
            n_samples=N_SAMPLES,
        )

        t0 = time.time()
        _, dkl_hist = train_reverse_network(
            vstate, vstate_R_dkl,
            n_steps=DKL_N_STEPS_MAX, batch=DKL_BATCH, lr=lr,
            verbose=False, freq=DKL_PRINT_FREQ,
            return_history=True,
        )
        t_train = time.time() - t0

        plateau_step, plateau_dkl = detect_plateau(dkl_hist)
        dkl_curves[lr] = {
            "history":      dkl_hist,
            "plateau_step": int(plateau_step),
            "plateau_dkl":  float(plateau_dkl),
            "t_train":      t_train,
        }
        print(f"      DKL(0) = {dkl_hist[0]:.4f}   "
              f"DKL(final) = {dkl_hist[-1]:.4f}")
        print(f"      plateau @ step {plateau_step}  "
              f"(DKL ≈ {plateau_dkl:.4f}, t = {t_train:.1f}s)")

        if plateau_dkl < best_dkl:
            best_dkl = plateau_dkl
            best_lr  = lr

    print(f"\n  → lr óptimo = {best_lr:.1e}  (DKL_plateau = {best_dkl:.4f})")
    suggested_init_n_steps = int(dkl_curves[best_lr]["plateau_step"] * 1.2)

    print(f"\n  Reentrenando vstate_R final con lr={best_lr:.1e} "
          f"durante {suggested_init_n_steps} steps ...")
    model_R_final = nk.models.ARNNDense(hilbert=hi, layers=2, features=24,
                                        activation=jax.nn.gelu)
    vstate_R_final = nk.vqs.MCState(
        nk.sampler.ARDirectSampler(hi), model_R_final,
        n_samples=N_SAMPLES,
    )
    vstate_R_final = train_reverse_network(
        vstate, vstate_R_final,
        n_steps=suggested_init_n_steps, batch=DKL_BATCH, lr=best_lr,
        verbose=False,
    )
    dkl_final_trained = compute_kl_local(vstate, vstate_R_final,
                                         n_samples=DRIFT_DKL_SAMPLES)
    print(f"  DKL(vstate || vstate_R_final) = {dkl_final_trained:.4f}")

    vstate_R = vstate_R_final

    results_summary["best_lr"]   = best_lr
    results_summary["best_dkl"]  = best_dkl
    results_summary["suggested_init_n_steps"] = suggested_init_n_steps


# ══════════════════════════════════════════════════════════════════════════════
# 5. Escalado de S2 / ∇S2 con n_samples (Wang) — robusto
# ══════════════════════════════════════════════════════════════════════════════
with section_guard("5. Escalado de S2/∇S2 con n_samples"):
    print(f"    budgets: {NS_BUDGET_LIST}   n_rep: {NS_N_REP}")
    print(f"    chunk_wang base: {chunk_wang}")

    ns_results = {}
    for ns in NS_BUDGET_LIST:
        # chunk adaptativo: al menos 4 chunks, cap al hard cap
        cs_eff = min(chunk_wang, max(256, ns // 4))
        cs_eff = 2 ** int(np.floor(np.log2(cs_eff)))
        cs_eff = max(256, cs_eff)

        print(f"\n    budget n={ns:,}  →  chunk_size efectivo = {cs_eff}")

        s2_list, grad_norm_list, cos_self_list, t_list = [], [], [], []
        grads = []

        try:
            for rep in range(NS_N_REP):
                t0 = time.perf_counter()
                S2, g = renyi2_wang_cs(
                    vstate, vstate_R, partition,
                    n_samples=ns, key=rep, chunk_size=cs_eff,
                )
                t_list.append(time.perf_counter() - t0)
                s2_list.append(float(S2))
                grad_norm_list.append(grad_norm(g))
                grads.append(g)
        except Exception as e:
            print(f"    ✗ {type(e).__name__} con n={ns}, chunk={cs_eff}. "
                  f"Saltando este budget.")
            continue

        for i_ in range(len(grads)):
            for j_ in range(i_ + 1, len(grads)):
                cos_self_list.append(cosine_similarity(grads[i_], grads[j_]))

        ns_results[ns] = {
            "S2":        s2_list,
            "grad_norm": grad_norm_list,
            "cos_self":  float(np.mean(cos_self_list)) if cos_self_list else float("nan"),
            "time":      float(np.mean(t_list)),
            "chunk":     cs_eff,
        }
        print(f"    n={ns:>8,}  ⟨S₂⟩={np.mean(s2_list):.4f}  "
              f"std(S₂)={np.std(s2_list):.4f}  "
              f"⟨||∇S₂||⟩={np.mean(grad_norm_list):.4f}  "
              f"std(||∇S₂||)={np.std(grad_norm_list):.4f}  "
              f"cos_self={np.mean(cos_self_list):.3f}  "
              f"t={np.mean(t_list):.3f}s")

    if len(ns_results) >= 2:
        ns_arr  = np.array(sorted(ns_results.keys()), dtype=float)
        std_arr = np.array([np.std(ns_results[int(n)]["grad_norm"])
                            for n in ns_arr])
        log_n   = np.log(ns_arr)
        log_std = np.log(std_arr)
        slope, intercept = np.polyfit(log_n, log_std, 1)
        C_g = float(np.exp(intercept))
        print(f"\n  Ajuste log-log: slope = {slope:.3f}  (teórico -0.5)")
        print(f"  → C_g ≈ {C_g:.4f}  (std(||∇S₂||) = C_g / √n)")

        mean_grad_norm = float(np.mean(
            ns_results[max(ns_results)]["grad_norm"]
        ))
        n_samples_wang_opt = int(np.ceil(
            (C_g / (NS_EPS_GRAD * mean_grad_norm)) ** 2
        ))
        n_samples_wang_opt = 2 ** int(np.ceil(
            np.log2(max(n_samples_wang_opt, 1))
        ))
        n_samples_wang_opt_log2 = int(np.round(np.log2(n_samples_wang_opt)))

        print(f"  ||∇S₂|| característica ≈ {mean_grad_norm:.4f}")
        print(f"  → n_samples_wang_opt = 2**{n_samples_wang_opt_log2} "
              f"= {n_samples_wang_opt:,}  "
              f"(para std(||∇S₂||)/||∇S₂|| = {NS_EPS_GRAD:.0%})")

        results_summary["C_g"]        = C_g
        results_summary["slope"]      = slope
        results_summary["mean_grad_norm"] = mean_grad_norm
        results_summary["n_samples_wang_opt_log2"] = n_samples_wang_opt_log2
        results_summary["n_samples_wang_opt"]      = n_samples_wang_opt
    else:
        print(f"\n  ⚠ solo {len(ns_results)} presupuestos válidos; "
              f"no se puede ajustar el escalado.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Deriva y recuperación de DKL  (con Wang)
# ══════════════════════════════════════════════════════════════════════════════
with section_guard("6a. Deriva DKL"):
    op_wang_final = FreeRenyiEnergyObservable(
        hi, H_extended, partition, T,
        chunk_size=chunk_wang,
        chunk_size_energy=chunk_wang_energy,
        method="wang",
        vstate_R=vstate_R,
        wang_kwargs=WANG_KWARGS_DIAG,
    )

    dkl0 = compute_kl_local(vstate, vstate_R, n_samples=DRIFT_DKL_SAMPLES)
    print(f"  DKL inicial = {dkl0:.4f}   umbral deriva = "
          f"{DRIFT_DKL_THRESH}× DKL₀ = {DRIFT_DKL_THRESH * dkl0:.4f}")

    drift_dkl = [dkl0]
    drift_step_thresh = DRIFT_N_STEPS

    opt_drift = optax.adam(LR_DRIFT)
    opt_state_drift = opt_drift.init(vstate.parameters)
    for step in range(DRIFT_N_STEPS):
        _, F_grad = vstate.expect_and_grad(op_wang_final)
        updates, opt_state_drift = opt_drift.update(
            F_grad, opt_state_drift, vstate.parameters
        )
        vstate.parameters = optax.apply_updates(vstate.parameters, updates)
        jax.effects_barrier()

        dkl_now = compute_kl_local(vstate, vstate_R,
                                   n_samples=DRIFT_DKL_SAMPLES)
        drift_dkl.append(dkl_now)

        if (dkl_now > DRIFT_DKL_THRESH * dkl0
                and drift_step_thresh == DRIFT_N_STEPS):
            drift_step_thresh = step + 1
            print(f"  → DKL cruzó {DRIFT_DKL_THRESH}× DKL₀ en step "
                  f"{step+1}  (DKL={dkl_now:.4f})")

    print(f"  DKL final tras {DRIFT_N_STEPS} steps: {drift_dkl[-1]:.4f}  "
          f"({drift_dkl[-1] / (dkl0 + 1e-10):.2f}× DKL₀)")

    print_curve(drift_dkl, window=10, label="drift DKL(step)", n_points=10)

    if drift_step_thresh == DRIFT_N_STEPS:
        print(f"  ⚠ no se cruzó el umbral en {DRIFT_N_STEPS} steps. "
              f"Recomendación provisional WARM_EVERY = {DRIFT_N_STEPS}")
        warm_every_rec = DRIFT_N_STEPS
    else:
        warm_every_rec = max(1, int(drift_step_thresh * 0.5))
        print(f"  → WARM_EVERY sugerido = {warm_every_rec}  "
              f"(mitad del tiempo a deriva)")

    vstate.parameters = params_bak
    results_summary["warm_every_rec"] = warm_every_rec


with section_guard("6b. Recuperación DKL"):
    print(f"  Simulando deriva ({warm_every_rec} steps de Wang) ...")
    opt_drift2 = optax.adam(LR_DRIFT)
    opt_state_drift2 = opt_drift2.init(vstate.parameters)
    for _ in range(warm_every_rec):
        _, F_grad = vstate.expect_and_grad(op_wang_final)
        updates, opt_state_drift2 = opt_drift2.update(
            F_grad, opt_state_drift2, vstate.parameters
        )
        vstate.parameters = optax.apply_updates(vstate.parameters, updates)
        jax.effects_barrier()

    dkl_drifted = compute_kl_local(vstate, vstate_R,
                                   n_samples=RECOVER_DKL_SAMPLES)
    print(f"  DKL tras deriva = {dkl_drifted:.4f}")

    params_R_bak = copy.deepcopy(vstate_R.parameters)
    vstate_R_recover = copy.deepcopy(vstate_R)
    vstate_R_recover, dkl_recover_hist = train_reverse_network(
        vstate, vstate_R_recover,
        n_steps=RECOVER_N_STEPS_MAX, batch=DRIFT_DKL_SAMPLES, lr=best_lr,
        verbose=False, return_history=True,
    )

    if RECOVER_PRINT_CURVE:
        print_curve(dkl_recover_hist, window=10, label="recover DKL(step)",
                    n_points=10)

    dkl_target = RECOVER_TOL * dkl0
    recover_step = RECOVER_N_STEPS_MAX
    for k, dkl_val in enumerate(dkl_recover_hist):
        if dkl_val <= dkl_target:
            recover_step = k + 1
            break

    print(f"  Objetivo (≤ {RECOVER_TOL}× DKL₀) = {dkl_target:.4f}")
    if recover_step == RECOVER_N_STEPS_MAX:
        print(f"  ⚠ no se recuperó en {RECOVER_N_STEPS_MAX} steps. "
              f"WARM_N_STEPS = {RECOVER_N_STEPS_MAX} (provisional)")
        warm_n_steps_rec = RECOVER_N_STEPS_MAX
    else:
        warm_n_steps_rec = int(recover_step * 1.2)
        print(f"  → recuperado en step {recover_step}. "
              f"WARM_N_STEPS sugerido = {warm_n_steps_rec}")

    vstate_R.parameters = params_R_bak
    vstate.parameters = params_bak
    results_summary["warm_n_steps_rec"] = warm_n_steps_rec


# ══════════════════════════════════════════════════════════════════════════════
# 7. Contabilidad del coste Wang
# ══════════════════════════════════════════════════════════════════════════════
with section_guard("7. Contabilidad coste Wang"):
    print(f"  Midiendo t_rev_step con batch={INIT_BATCH_HINT} ...")
    t_rev_init = measure_rev_step_time(vstate, vstate_R, INIT_BATCH_HINT,
                                       n_measure=2)
    print(f"    t_rev_step({INIT_BATCH_HINT}) = {t_rev_init*1000:.1f} ms")

    print(f"  Midiendo t_rev_step con batch={WARM_BATCH_HINT} ...")
    t_rev_warm = measure_rev_step_time(vstate, vstate_R, WARM_BATCH_HINT,
                                       n_measure=2)
    print(f"    t_rev_step({WARM_BATCH_HINT}) = {t_rev_warm*1000:.1f} ms")

    N_WANG = WANG_N_STEPS_HINT
    t_init_phase  = (results_summary["suggested_init_n_steps"] or 0) * t_rev_init
    t_main_phase  = N_WANG * t_wang_mean
    t_warm_per_it = (results_summary["warm_n_steps_rec"] or 0) * t_rev_warm
    we = results_summary["warm_every_rec"] or N_WANG
    n_warm_events = N_WANG // max(1, we)
    t_warm_phase  = n_warm_events * t_warm_per_it

    t_total = t_init_phase + t_main_phase + t_warm_phase
    frac_net = t_main_phase / t_total if t_total > 0 else 0.0

    print(f"\n  Estimación para {N_WANG} pasos Wang:")
    print(f"    t_init_train  = {t_init_phase:>8.1f}s")
    print(f"    t_main_wang   = {t_main_phase:>8.1f}s")
    print(f"    t_warm_starts = {t_warm_phase:>8.1f}s")
    print(f"    ────────────────────────────────────────")
    print(f"    t_total       = {t_total:>8.1f}s")
    print(f"    fracción 'neta' (main / total) = {frac_net:.2%}")

    results_summary["t_init_phase"] = t_init_phase
    results_summary["t_main_phase"] = t_main_phase
    results_summary["t_warm_phase"] = t_warm_phase
    results_summary["t_total"]      = t_total
    results_summary["frac_net"]     = frac_net


# ══════════════════════════════════════════════════════════════════════════════
# 8. Memoria del proceso (VmHWM → --mem sugerido para SLURM)
# ══════════════════════════════════════════════════════════════════════════════
suggested_mem_gb = None
with section_guard("8. Memoria del proceso"):
    _ = vstate.expect(H_extended)
    _ = vstate.expect_and_grad(op_wang_final)
    _ = measure_rev_step_time(vstate, vstate_R, INIT_BATCH_HINT, n_measure=1)
    suggested_mem_gb = memory_report(label="tras ops pesadas")


# ══════════════════════════════════════════════════════════════════════════════
# 9. RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
def _fmt(x, fmt=".4f", default="—"):
    if x is None:
        return default
    try:
        return format(float(x), fmt)
    except Exception:
        return str(x)


print("\n" + "=" * 66)
print("RESUMEN")
print("=" * 66)
print(f"  N, N_A                  : {N}, {N_A}")
print(f"  n_samples (vstate)      : {N_SAMPLES}")
print(f"  warmup swap             : {WARMUP_SWAP_STEPS} steps "
      f"(F_final = {_fmt(results_summary['F_warmup_last'])})")
print()
print(f"  chunk_size (swap)       : {results_summary['chunk_swap']}")
print(f"  chunk_size (wang·S)     : {results_summary['chunk_wang']}")
print(f"  chunk_size (wang·E)     : {results_summary['chunk_wang_energy']}")
print()
print(f"  t/step (swap)           : {_fmt(results_summary['t_swap_mean'], '.3f')}s")
print(f"  t/step (wang)           : {_fmt(results_summary['t_wang_mean'], '.3f')}s")
print()
print(f"  clip_norm (swap)        : {_fmt(results_summary['clip_swap'])}")
print(f"  clip_norm (wang)        : {_fmt(results_summary['clip_wang'])}")
print()
print(f"  Red reversa:")
print(f"    lr óptimo             : {results_summary['best_lr']}")
print(f"    INIT_N_STEPS          : {results_summary['suggested_init_n_steps']}")
print(f"    DKL(plateau)          : {_fmt(results_summary['best_dkl'])}")
print()
print(f"  Wang – escalado:")
print(f"    C_g                   : {_fmt(results_summary['C_g'])}  "
      f"(slope={_fmt(results_summary['slope'], '.3f')})")
print(f"    ||∇S₂|| típico       : {_fmt(results_summary['mean_grad_norm'])}")
print(f"    n_samples_wang_opt    : "
      f"2**{results_summary['n_samples_wang_opt_log2']} "
      f"= {results_summary['n_samples_wang_opt']}")
print()
print(f"  Warm-start (drift medido con Wang):")
print(f"    WARM_EVERY sugerido   : {results_summary['warm_every_rec']}")
print(f"    WARM_N_STEPS sugerido : {results_summary['warm_n_steps_rec']}")
print()
print(f"  Coste Wang:")
print(f"    init_train            : {_fmt(results_summary['t_init_phase'], '.1f')}s")
print(f"    main                  : {_fmt(results_summary['t_main_phase'], '.1f')}s")
print(f"    warm-starts           : {_fmt(results_summary['t_warm_phase'], '.1f')}s")
print(f"    total                 : {_fmt(results_summary['t_total'], '.1f')}s  "
      f"(neto: {_fmt(results_summary['frac_net'], '.1%')})")
print()
print(f"  --mem sugerido          : {suggested_mem_gb} GB (3 × VmHWM)")
print("=" * 66)