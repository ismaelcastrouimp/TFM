"""
compare_FvN_vs_F2.py
====================

Test N=10 con ED para comparar:
  - Minimización de F_2  = E - T·S_2   (Wang)
  - Minimización de F_vN = E - T·S_vN  (chain rule cuadrática en n)
  - Referencias: ρ_G (Gibbs) y ρ_R2 (Rényi-2 exactos por ED)

Para cada T:
  - Entrena vstate con F_2  (swap + wang)
  - Entrena vstate con F_vN (FreeEnergyObservable con Wang + cuadrática n)
  - Extrae ρ_phys vía SVD y compara con ρ_G y ρ_R2

Gráficas en ../plots/FvN_vs_F2/
"""

import os
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

from src_renyi.physical_ancilla.entropy import (
    renyi2_wang_cs,
    train_reverse_network,
)
from src_renyi.physical_ancilla.training import free_energy_minimize_phases
from src_renyi.physical_ancilla.observables import FreeEnergyObservable

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots/FvN_vs_F2")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    print(f"  guardado: {path}.pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Referencias exactas
# ══════════════════════════════════════════════════════════════════════════════

def exact_gibbs_state(H_mat, T):
    """ρ_G = e^{-H/T}/Z."""
    E, V = np.linalg.eigh(H_mat)
    E_shift = E - E.min()
    w = np.exp(-E_shift / T)
    w /= w.sum()
    return V @ np.diag(w) @ V.T.conj(), w


def exact_renyi2_state(H_mat, T, n_steps=500, lr=1e-2):
    """ρ_{R2} = argmin_ρ [E - T·S_2(ρ)] parametrizando en la base propia de H."""
    E, V = np.linalg.eigh(H_mat)
    E_jax = jnp.array(E)

    def obj(logits):
        p = jax.nn.softmax(logits)
        return jnp.sum(p * E_jax) + T * jnp.log(jnp.sum(p**2))

    logits = jnp.zeros(len(E_jax))
    opt = optax.adam(lr)
    opt_state = opt.init(logits)

    @jax.jit
    def step(logits, opt_state):
        loss, grad = jax.value_and_grad(obj)(logits)
        updates, opt_state = opt.update(grad, opt_state)
        logits = optax.apply_updates(logits, updates)
        return logits, opt_state, loss

    for _ in range(n_steps):
        logits, opt_state, _ = step(logits, opt_state)

    p_opt = np.asarray(jax.nn.softmax(logits))
    return V @ np.diag(p_opt) @ V.T.conj(), p_opt


def entropy_from_rho(rho):
    p = np.linalg.eigvalsh(rho)
    p = np.clip(p, 1e-30, None)
    return -float(np.sum(p * np.log(p))), -float(np.log(np.sum(p**2)))


def trace_norm(rho1, rho2):
    return float(np.sum(np.abs(np.linalg.eigvalsh(rho1 - rho2))))


def local_observables_from_rho(rho, ops):
    return {name: float(np.real(np.trace(rho @ O))) for name, O in ops.items()}


def build_physical_observables(hi_phys, N):
    ops = {}
    for i in range(N):
        ops[f"X_{i}"] = np.asarray(sigmax(hi_phys, i).to_dense())
        ops[f"Z_{i}"] = np.asarray(sigmaz(hi_phys, i).to_dense())
    for i in range(N):
        j = (i + 1) % N
        ops[f"ZZ_{i}"] = np.asarray(
            (sigmaz(hi_phys, i) @ sigmaz(hi_phys, j)).to_dense()
        )
    return ops


# ══════════════════════════════════════════════════════════════════════════════
# Extracción de ρ_phys de un vstate
# ══════════════════════════════════════════════════════════════════════════════

def extract_rho_phys(vstate, subsystem, chunk=4096):
    hi = vstate.hilbert
    N_total = hi.size
    nA = len(subsystem); nB = N_total - nA
    complement = [i for i in range(N_total) if i not in subsystem]

    all_states = hi.all_states()
    N_states = all_states.shape[0]
    psi = np.zeros(N_states, dtype=np.complex128)
    for i in range(0, N_states, chunk):
        s = all_states[i:i+chunk]
        psi[i:i+chunk] = np.asarray(vstate.log_value(s)).reshape(-1)
    psi = np.exp(psi)
    psi = psi / np.sqrt(np.sum(np.abs(psi)**2))

    bits = ((np.asarray(all_states) + 1) // 2).astype(np.int32)
    a_idx = np.zeros(N_states, dtype=np.int64)
    for q in subsystem:
        a_idx = (a_idx << 1) | bits[:, q]
    b_idx = np.zeros(N_states, dtype=np.int64)
    for q in complement:
        b_idx = (b_idx << 1) | bits[:, q]

    Psi = np.zeros((2**nA, 2**nB), dtype=np.complex128)
    Psi[a_idx, b_idx] = psi
    return Psi @ Psi.conj().T


# ══════════════════════════════════════════════════════════════════════════════
# Minimización de F_vN
# ══════════════════════════════════════════════════════════════════════════════

def minimize_FvN(vstate, vstate_R, H_extended, partition, T,
                 n_steps=200, lr_peak=1e-2, clip_norm=5.0,
                 chunk_size=512, chunk_size_energy=256,
                 n_orders=(2, 3, 4, 5), n_samples_wang=2**14,
                 warm_start=None,                    # ← NUEVO
                 verbose=True):
    op = FreeEnergyObservable(
        vstate.hilbert, H_extended, partition, T,
        chunk_size=chunk_size,
        chunk_size_energy=chunk_size_energy,
        n_orders=list(n_orders),
        n_samples_wang=n_samples_wang,
        vstate_R=vstate_R,
        extrapolation_method="quadratic_n",
    )

    lr = optax.warmup_cosine_decay_schedule(
        init_value=1e-4, peak_value=lr_peak,
        warmup_steps=50, decay_steps=n_steps, end_value=1e-4,
    )
    opt = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(lr))
    opt_state = opt.init(vstate.parameters)

    history = []
    for step in range(n_steps):
        F_stats, F_grad = vstate.expect_and_grad(op)
        updates, opt_state = opt.update(F_grad, opt_state, vstate.parameters)
        vstate.parameters = optax.apply_updates(vstate.parameters, updates)
        F_val = float(F_stats.mean.real)
        history.append(F_val)

        # ── warm-start: re-entrenar vstate_R sobre el vstate_vN actual ──
        if (warm_start is not None and step > 0
                and step % warm_start["every"] == 0):
            vstate_R, dkl_traj = train_reverse_network(
                vstate, vstate_R,
                n_steps=warm_start.get("n_steps", 10),
                batch=warm_start.get("batch", 2048),
                lr=warm_start.get("lr", 1e-4),
                verbose=False,
                return_history=True,
                refresh_every=warm_start.get("refresh_every", 10),
            )
            if verbose:
                print(f"    [FvN warm-start @ step {step}] "
                      f"D_KL: {dkl_traj[0]:.2e} → {dkl_traj[-1]:.2e}")

        if verbose and step % 20 == 0:
            print(f"  [FvN] step {step:3d}  F={F_val:.4f}")

    return history


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
N = 5
N_A = 5
N_TOTAL = N + N_A
J_ZZ, h_x = -1.0, -1.5

TEMPS = [0.5, 1.0, 1.5, 2.0, 3.0]

N_SAMPLES_TRAIN   = 2**14
N_STEPS_F2        = 300
N_STEPS_FvN       = 200
CHUNK_SIZE        = N_SAMPLES_TRAIN
CHUNK_SIZE_ENERGY = N_SAMPLES_TRAIN
N_ORDERS_SVN      = (2, 3, 4, 5)
N_SAMPLES_WANG    = 2**14


# ══════════════════════════════════════════════════════════════════════════════
# Hilbert y operadores
# ══════════════════════════════════════════════════════════════════════════════
hi_phys = nk.hilbert.Spin(s=1/2, N=N)
hi      = nk.hilbert.Spin(s=1/2, N=N_TOTAL)

H_phys = 0
for i in range(N):
    H_phys += h_x * sigmax(hi_phys, i)
    H_phys += J_ZZ * sigmaz(hi_phys, i) @ sigmaz(hi_phys, (i + 1) % N)
H_phys_mat = np.asarray(H_phys.to_dense())

H_extended = 0
for i in range(N):
    H_extended += h_x * sigmax(hi, i)
    H_extended += J_ZZ * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)

print("Construyendo observables locales...")
LOCAL_OPS = build_physical_observables(hi_phys, N)
print(f"  {len(LOCAL_OPS)} operadores")


# ══════════════════════════════════════════════════════════════════════════════
# Referencias exactas
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Referencias exactas (ED)")
print("=" * 60)

exact = {}
for T in TEMPS:
    rho_G, _ = exact_gibbs_state(H_phys_mat, T)
    rho_R2, _ = exact_renyi2_state(H_phys_mat, T)

    E_G  = float(np.trace(rho_G  @ H_phys_mat))
    E_R2 = float(np.trace(rho_R2 @ H_phys_mat))
    S_vN_G,  S_2_G  = entropy_from_rho(rho_G)
    S_vN_R2, S_2_R2 = entropy_from_rho(rho_R2)

    exact[T] = {
        "rho_G": rho_G, "rho_R2": rho_R2,
        "E_G": E_G, "E_R2": E_R2,
        "S_vN_G": S_vN_G, "S_2_G": S_2_G,
        "S_vN_R2": S_vN_R2, "S_2_R2": S_2_R2,
        "F_G": E_G - T * S_vN_G,
        "F_2_R2": E_R2 - T * S_2_R2,
        "F_vN_R2": E_R2 - T * S_vN_R2,
        "F_2_G": E_G - T * S_2_G,
        "obs_G":  local_observables_from_rho(rho_G,  LOCAL_OPS),
        "obs_R2": local_observables_from_rho(rho_R2, LOCAL_OPS),
        "trace_G_R2": trace_norm(rho_G, rho_R2),
    }
    print(f"  T={T:.2f}  ||ρ_G-ρ_R2||_1={exact[T]['trace_G_R2']:.4f}  "
          f"(F_2(R2)={exact[T]['F_2_R2']:.3f} < F_2(G)={exact[T]['F_2_G']:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# Entrenamiento a cada T
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Entrenamiento (F_2 y F_vN) para cada T")
print("=" * 60)

partition = list(range(N))
results = {}

for T in TEMPS:
    print(f"\n── T = {T:.2f} ──")

    # Modelos frescos
    model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                                activation=jax.nn.gelu)
    sampler = nk.sampler.ARDirectSampler(hi)
    vstate_F2 = nk.vqs.MCState(sampler, model, n_samples=N_SAMPLES_TRAIN)

    model_R = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                                  activation=jax.nn.gelu)
    sampler_R = nk.sampler.ARDirectSampler(hi)
    vstate_R = nk.vqs.MCState(sampler_R, model_R, n_samples=N_SAMPLES_TRAIN)

    print("  Entrenando vstate_R...")
    train_reverse_network(vstate_F2, vstate_R,
                          n_steps=1000, batch=2048, lr=1e-3,
                          verbose=False)

    vstate_vN = copy.deepcopy(vstate_F2)

    # ── F_2 ──
    print(f"  [F_2] {N_STEPS_F2} steps...")
    phases_F2 = [
        {
            "method": "swap",
            "n_steps": N_STEPS_F2 // 3,
            "optimizer": optax.adam(optax.warmup_cosine_decay_schedule(
                init_value=1e-4, peak_value=1e-2,
                warmup_steps=50, decay_steps=N_STEPS_F2 // 3, end_value=5e-4,
            )),
            "chunk_size_energy": CHUNK_SIZE_ENERGY,
        },
        {
            "method": "wang",
            "n_steps": 2 * N_STEPS_F2 // 3,
            "optimizer": optax.adam(optax.warmup_cosine_decay_schedule(
                init_value=1e-4, peak_value=3e-3,
                warmup_steps=100, decay_steps=2 * N_STEPS_F2 // 3, end_value=1e-4,
            )),
            "vstate_R": vstate_R,
            "initial_train": {"n_steps": 200, "batch": 2048, "lr": 1e-3},
            "warm_start": {"every": 20, "n_steps": 10, "lr": 1e-4, "batch": 2048},
            "wang_kwargs": {"n_samples": N_SAMPLES_TRAIN},
            "chunk_size_energy": CHUNK_SIZE_ENERGY,
        },
    ]
    hist_F2, _, _, _ = free_energy_minimize_phases(
        vstate_F2, T, partition, H_extended, phases_F2,
        chunk_size=CHUNK_SIZE, clip_norm=5.0,
        verbose=True, plot=False,
    )

    # ── F_vN ──
    print(f"  [F_vN] {N_STEPS_FvN} steps...")
    hist_FvN = minimize_FvN(
        vstate_vN, vstate_R, H_extended, partition, T,
        n_steps=N_STEPS_FvN,
        lr_peak=1e-2,
        clip_norm=5.0,
        chunk_size=CHUNK_SIZE,
        chunk_size_energy=CHUNK_SIZE_ENERGY,
        n_orders=N_ORDERS_SVN,
        n_samples_wang=N_SAMPLES_WANG,
        warm_start={"every": 20, "n_steps": 10,
                    "lr": 1e-4, "batch": 2048},
        verbose=True,
    )

    # ── Extracción y métricas ──
    rho_F2 = extract_rho_phys(vstate_F2, partition)
    rho_vN = extract_rho_phys(vstate_vN, partition)

    S_vN_F2, S_2_F2 = entropy_from_rho(rho_F2)
    S_vN_vN, S_2_vN = entropy_from_rho(rho_vN)

    E_F2 = float(vstate_F2.expect(H_extended).mean.real)
    E_vN = float(vstate_vN.expect(H_extended).mean.real)

    results[T] = {
        "hist_F2":  hist_F2,
        "hist_FvN": hist_FvN,
        "E_F2": E_F2, "E_vN": E_vN,
        "S_2_F2": S_2_F2, "S_2_vN": S_2_vN,
        "S_vN_F2": S_vN_F2, "S_vN_vN": S_vN_vN,
        "F_2_F2":  E_F2 - T * S_2_F2,
        "F_vN_vN": E_vN - T * S_vN_vN,
        "obs_F2":  local_observables_from_rho(rho_F2, LOCAL_OPS),
        "obs_vN":  local_observables_from_rho(rho_vN, LOCAL_OPS),
        "trace_F2_G":  trace_norm(rho_F2, exact[T]["rho_G"]),
        "trace_vN_G":  trace_norm(rho_vN, exact[T]["rho_G"]),
        "trace_F2_R2": trace_norm(rho_F2, exact[T]["rho_R2"]),
        "trace_vN_R2": trace_norm(rho_vN, exact[T]["rho_R2"]),
    }

    print(f"  E:   G={exact[T]['E_G']:.3f}   F2={E_F2:.3f}   FvN={E_vN:.3f}")
    print(f"  SvN: G={exact[T]['S_vN_G']:.3f}  F2={S_vN_F2:.3f}  FvN={S_vN_vN:.3f}")
    print(f"  S2:  G={exact[T]['S_2_G']:.3f}   R2={exact[T]['S_2_R2']:.3f}  "
          f"F2={S_2_F2:.3f}   FvN={S_2_vN:.3f}")
    print(f"  ||ρ-ρ_G||_1:   F2={results[T]['trace_F2_G']:.4f}   "
          f"FvN={results[T]['trace_vN_G']:.4f}")
    print(f"  ||ρ-ρ_R2||_1:  F2={results[T]['trace_F2_R2']:.4f}   "
          f"FvN={results[T]['trace_vN_R2']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1: convergencia F vs step
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(TEMPS), figsize=(3.0 * len(TEMPS), 3.5),
                         squeeze=False)
axes = axes[0]

for ax, T in zip(axes, TEMPS):
    ax.plot(results[T]["hist_F2"],  "o-", ms=3, color="C0", label=r"$F_2$")
    ax.plot(results[T]["hist_FvN"], "s-", ms=3, color="C3", label=r"$F_{\rm vN}$")
    ax.axhline(exact[T]["F_G"], color="k", ls=":", alpha=0.5, label=r"$F_G$ exacta")
    ax.set_xlabel("step")
    ax.set_ylabel("F")
    ax.set_title(f"T={T:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle(r"Convergencia de $F_2$ y $F_{\rm vN}$", y=1.02)
fig.tight_layout()
save_fig(fig, "convergence_F2_vs_FvN")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2: S_2, S_vN y distancias vs T
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
Ts = np.array(TEMPS)

ax = axes[0]
ax.plot(Ts, [exact[T]["S_2_G"]  for T in TEMPS], "k-",  lw=2, label=r"$S_2(\rho_G)$")
ax.plot(Ts, [exact[T]["S_2_R2"] for T in TEMPS], "k--", lw=2, label=r"$S_2(\rho_{R2})$")
ax.plot(Ts, [results[T]["S_2_F2"] for T in TEMPS], "o", color="C0", label=r"$F_2$ entrenado")
ax.plot(Ts, [results[T]["S_2_vN"] for T in TEMPS], "s", color="C3", label=r"$F_{\rm vN}$ entrenado")
ax.set_xlabel("T"); ax.set_ylabel(r"$S_2$"); ax.set_title(r"$S_2$")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(Ts, [exact[T]["S_vN_G"]  for T in TEMPS], "k-",  lw=2, label=r"$S_{\rm vN}(\rho_G)$")
ax.plot(Ts, [results[T]["S_vN_F2"] for T in TEMPS], "o", color="C0", label=r"$F_2$ entrenado")
ax.plot(Ts, [results[T]["S_vN_vN"] for T in TEMPS], "s", color="C3", label=r"$F_{\rm vN}$ entrenado")
ax.set_xlabel("T"); ax.set_ylabel(r"$S_{\rm vN}$"); ax.set_title(r"$S_{\rm vN}$")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(Ts, [results[T]["trace_F2_R2"] for T in TEMPS], "o-",
        color="C0", label=r"$\|\rho_{F_2} - \rho_{R2}\|_1$")
ax.plot(Ts, [results[T]["trace_vN_G"]  for T in TEMPS], "s-",
        color="C3", label=r"$\|\rho_{F_{\rm vN}} - \rho_G\|_1$")
ax.plot(Ts, [results[T]["trace_vN_R2"] for T in TEMPS], "s--",
        color="C3", alpha=0.5)
ax.plot(Ts, [results[T]["trace_F2_G"]  for T in TEMPS], "o--",
        color="C0", alpha=0.5)
ax.plot(Ts, [exact[T]["trace_G_R2"]    for T in TEMPS], "k:", lw=1.5,
        label=r"$\|\rho_G - \rho_{R2}\|_1$ (ED)")
ax.set_xlabel("T"); ax.set_ylabel(r"$\|\cdot\|_1$")
ax.set_title(r"Distancia a referencia")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig.tight_layout()
save_fig(fig, "entropy_and_distance_vs_T")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 3: observables locales (un T representativo)
# ══════════════════════════════════════════════════════════════════════════════
T_rep = TEMPS[len(TEMPS) // 2]
sites = np.arange(N)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, key, label in [
    (axes[0], "Z",  r"$\langle\sigma_i^z\rangle$"),
    (axes[1], "X",  r"$\langle\sigma_i^x\rangle$"),
    (axes[2], "ZZ", r"$\langle\sigma_i^z\sigma_{i+1}^z\rangle$"),
]:
    ax.plot(sites, [exact[T_rep]["obs_G"][f"{key}_{i}"]  for i in sites],
            "k-", lw=2, label="Gibbs")
    ax.plot(sites, [results[T_rep]["obs_F2"][f"{key}_{i}"] for i in sites],
            "o-", color="C0", label=r"$F_2$")
    ax.plot(sites, [results[T_rep]["obs_vN"][f"{key}_{i}"] for i in sites],
            "s-", color="C3", label=r"$F_{\rm vN}$")
    ax.set_xlabel("site"); ax.set_ylabel(label)
    ax.set_title(f"T={T_rep:.2f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.tight_layout()
save_fig(fig, f"observables_T{T_rep}")


# ══════════════════════════════════════════════════════════════════════════════
# Tabla resumen
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 95)
print("TABLA RESUMEN")
print("=" * 95)
print(f"{'T':>5}  {'Method':>8}  {'E':>10}  {'S_2':>8}  {'S_vN':>8}  "
      f"{'||ρ-ρ_G||_1':>12}  {'||ρ-ρ_R2||_1':>13}")
print("-" * 95)
for T in TEMPS:
    e = exact[T]
    r = results[T]
    print(f"{T:>5.2f}  {'G (ED)':>8}  {e['E_G']:>10.4f}  {e['S_2_G']:>8.4f}  "
          f"{e['S_vN_G']:>8.4f}  {'—':>12}  {'—':>13}")
    print(f"{'':>5}  {'R2 (ED)':>8}  {e['E_R2']:>10.4f}  {e['S_2_R2']:>8.4f}  "
          f"{e['S_vN_R2']:>8.4f}  {e['trace_G_R2']:>12.4f}  {'—':>13}")
    print(f"{'':>5}  {'F2':>8}  {r['E_F2']:>10.4f}  {r['S_2_F2']:>8.4f}  "
          f"{r['S_vN_F2']:>8.4f}  {r['trace_F2_G']:>12.4f}  {r['trace_F2_R2']:>13.4f}")
    print(f"{'':>5}  {'FvN':>8}  {r['E_vN']:>10.4f}  {r['S_2_vN']:>8.4f}  "
          f"{r['S_vN_vN']:>8.4f}  {r['trace_vN_G']:>12.4f}  {r['trace_vN_R2']:>13.4f}")
print("=" * 95)

print("\nDone. Gráficas en", os.path.abspath(PLOTS_DIR))