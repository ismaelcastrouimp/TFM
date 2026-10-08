import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import optax
import os
import json
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from .observables import FreeRenyiEnergyObservable
from .entropy import (
    renyi2_entropy_and_grad_sampled,
    renyi2_entropy_sampled,
    renyi2_entropy_exact,
    renyi2_entropy_and_grad_exact,     
    train_reverse_network,    
)

def free_energy_minimize(vstate, T, partition, Hamiltonian, n_steps=1000,
                         fine_steps=0, fine_drut_kwargs=None, fine_lr=None,
                         verbose=True, freq=50, plot=True, optimizer=None,
                         learning_rate=None, clip_norm=None, timing=False,
                         chunk_size=256, sr=None, n_samples_sr=None):
    """
    Minimiza F = E - T·S₂ con dos fases: coarse (swap) + fine (Drut).


    Parámetros
    ----------
    vstate       : MCState de NetKet con el modelo variacional.
    T            : Temperatura.
    partition    : Lista de sitios del subsistema A para S₂.
    Hamiltonian  : Operador H compatible con NetKet.
    n_steps      : Número de pasos de optimización.
    verbose      : Si True, imprime progreso cada `freq` pasos.
    freq         : Frecuencia de impresión.
    plot         : Si True, muestra gráfica de F al final.
    optimizer    : optax.GradientTransformation
    learning_rate: Schedule o escalar de optax. Por defecto warmup_cosine_decay.
    clip_norm    : float o None. Si no es None, aplica clip_by_global_norm.
    timing       : Si True, mide y muestra el tiempo real de cada step en los prints.
    chunk_size   : tamaño de los chunks para procesar FreeRenyiEnergyObservable.
    sr           : nk.optimizer.SR
    n_samples_sr : Muestras usadas en el paso de SR (< n_samples completo).
    fine_steps      : nº de pasos con Drut. Si 0, no se ejecuta la fase 2.
    fine_drut_kwargs: dict con argumentos para renyi2_drut_sampling
                      (n_chains, n_lambda, n_sweeps_per_lam, n_props_per_sweep, K, ...).
    fine_lr : float o None. Si None, hereda el LR final del coarse.

    Devuelve
    -------
    (free_energy_history, best_F, E_best, S2_best)
    """
    # ── Defaults ────────────────────────────────────────────────────────
    if optimizer is None:
        if learning_rate is None:
            if n_steps > 0:
                learning_rate = optax.linear_schedule(0.01, 0.001, n_steps)
            else:
                learning_rate = 0.001
        optimizer = optax.sgd(learning_rate)

    # ── Extraer LR final del coarse ─────────────────────────────────────
    coarse_final_lr = None
    if learning_rate is not None:
        try:
            if callable(learning_rate):
                coarse_final_lr = float(learning_rate(n_steps - 1))
            else:
                coarse_final_lr = float(learning_rate)
        except Exception:
            coarse_final_lr = None

    # ── Resolver fine_lr ────────────────────────────────────────────────
    if fine_steps > 0 and fine_lr is None:
        if coarse_final_lr is not None:
            fine_lr = coarse_final_lr
            if verbose:
                print(f"[fine] fine_lr no especificado → "
                      f"usando LR final del coarse: {fine_lr:.6e}")
        else:
            fine_lr = 0.001
            if verbose:
                print("[fine] no se pudo extraer LR final del coarse; "
                      "usando fine_lr=1e-3 por defecto")

    # ── Construcción del optimizer ──────────────────────────────────────
    if clip_norm is not None:
        gradient_transform = optax.chain(
            optax.clip_by_global_norm(clip_norm), optimizer
        )
    else:
        gradient_transform = optimizer
    opt_state = gradient_transform.init(vstate.parameters)

    free_renyi_op = FreeRenyiEnergyObservable(
        vstate.hilbert, Hamiltonian, partition, T, chunk_size
    )
    n_samples_full = vstate.n_samples

    free_energy_history = []
    coarse_best_F = float("inf")
    coarse_best_params = None
    fine_best_F = float("inf")
    fine_best_params = None

    # ═══════════════════════════════════════════════════════════════════
    # FASE 1: coarse (swap)
    # ═══════════════════════════════════════════════════════════════════
    if verbose:
        print(f"[coarse] {n_steps} pasos con swap trick")

    for step in range(n_steps):
        if timing:
            t0 = time.time()

        F_stats, F_grad = vstate.expect_and_grad(free_renyi_op)

        if sr is not None:
            if n_samples_sr is not None:
                vstate.n_samples = n_samples_sr
            F_grad = sr(vstate, F_grad, step)
            if n_samples_sr is not None:
                vstate.n_samples = n_samples_full

        updates, opt_state = gradient_transform.update(
            F_grad, opt_state, vstate.parameters
        )
        vstate.parameters = optax.apply_updates(vstate.parameters, updates)

        if timing:
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready(), vstate.parameters
            )

        F_val = float(F_stats.mean.real)
        free_energy_history.append(F_val)

        # Guardamos el mejor coarse SIEMPRE (para diagnóstico),
        # pero solo lo usaremos como "best" final si fine_steps == 0.
        if F_val < coarse_best_F:
            coarse_best_F = F_val
            coarse_best_params = vstate.parameters

        if step % freq == 0 and verbose:
            msg = f"  Step {step:4d} | F={F_val:.6f}"
            if timing:
                msg += f" | t={time.time()-t0:.3f}s"
            print(msg)

    # ═══════════════════════════════════════════════════════════════════
    # FASE 2: fine (Drut)
    # ═══════════════════════════════════════════════════════════════════
    if fine_steps > 0:
        if fine_drut_kwargs is None:
            fine_drut_kwargs = dict(
                n_chains=256,
                n_lambda=12,
                n_sweeps_per_lam=50,
                n_props_per_sweep=2 * vstate.hilbert.size,
                K=2,
            )
        fine_renyi_op = FreeRenyiEnergyObservable(
            vstate.hilbert,
            Hamiltonian,
            partition,
            T,
            chunk_size=chunk_size,
            method="drut",
            drut_kwargs=fine_drut_kwargs,
            drut_seed=12345,
        )
        if verbose:
            print(f"[fine] {fine_steps} pasos con Drut")
            print(f"       kwargs: {fine_drut_kwargs}")

        # Optimizer del fine con LR constante heredado
        fine_opt = optax.sgd(fine_lr)
        if clip_norm is not None:
            fine_opt = optax.chain(
                optax.clip_by_global_norm(clip_norm), fine_opt
            )
        fine_opt_state = fine_opt.init(vstate.parameters)

        for step in range(fine_steps):
            if timing:
                t0 = time.time()

            F_stats, grad_F = vstate.expect_and_grad(fine_renyi_op)

            updates, fine_opt_state = fine_opt.update(
                grad_F, fine_opt_state, vstate.parameters
            )
            vstate.parameters = optax.apply_updates(vstate.parameters, updates)

            if timing:
                jax.tree_util.tree_map(
                    lambda x: x.block_until_ready(), vstate.parameters
                )

            F_val = float(F_stats.mean.real)
            free_energy_history.append(F_val)

            # Best del fine (el único que cuenta si fine_steps > 0)
            if F_val < fine_best_F:
                fine_best_F = F_val
                fine_best_params = vstate.parameters

            if step % max(1, freq // 5) == 0 and verbose:
                msg = f"  [fine] Step {step:3d} | F={F_val:.6f}"
                if timing:
                    msg += f" | t={time.time()-t0:.2f}s"
                print(msg)

    # ═══════════════════════════════════════════════════════════════════
    # Selección del best final
    # ═══════════════════════════════════════════════════════════════════
    if fine_steps > 0 and fine_best_params is not None:
        best_params = fine_best_params
        best_F = fine_best_F
        if verbose:
            print(f"[best] fine: F={fine_best_F:.6f} | "
                  f"coarse (referencia): F={coarse_best_F:.6f}")
    elif coarse_best_params is not None:
        best_params = coarse_best_params
        best_F = coarse_best_F
        if verbose:
            print(f"[best] coarse: F={best_F:.6f}")
    else:
        best_params = vstate.parameters
        best_F = float("inf")

    # ── Restaurar y reevaluar ───────────────────────────────────────────
    vstate.parameters = best_params
    jax.clear_caches()

    vstate.chunk_size = chunk_size
    E_best = float(vstate.expect(Hamiltonian).mean.real)
    vstate.chunk_size = None
    S2_best = renyi2_entropy_sampled(
        vstate, partition, n_samples_full, chunk_size=chunk_size
    )
    best_F = E_best - T * S2_best

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(free_energy_history, label=r"$F$")
        if fine_steps > 0:
            ax.axvline(n_steps, color='k', ls='--', alpha=0.4,
                       label=f'inicio fine')
            ax.axhline(coarse_best_F, color='r', ls=':', alpha=0.4,
                       label=f'coarse best = {coarse_best_F:.2f}')
        ax.set_xlabel("Step")
        ax.set_ylabel(r"$F$")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return free_energy_history, best_F, E_best, S2_best

def compute_kl(vstate, vstate_R, n_samples=2048):
    """D_KL(p_θ || p_R) estimada con muestras de vstate."""
    s = vstate.sample(n_samples=n_samples).reshape(-1, vstate.hilbert.size)
    lp = 2.0 * jnp.real(vstate._apply_fun(
        {"params": vstate.parameters, **vstate.model_state}, s))
    lpR = 2.0 * jnp.real(vstate_R._apply_fun(
        {"params": vstate_R.parameters, **vstate_R.model_state},
        jnp.flip(s, axis=-1)))
    return float(jnp.mean(lp - lpR))

def _auto_training_history_path(vstate, partition, tag):
    """Ruta del historial completo de entrenamiento.

    `tag` se usa tal cual en el nombre del archivo (típicamente el índice
    del run, formateado como `0001`, `0002`, ...). Se guarda un archivo
    por run dentro de `training_histories/`.
    """
    N   = len(partition)
    N_A = vstate.hilbert.size - N
    here = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(here, "..", "data")
    folder = f"N{N}" if N == N_A else f"N{N}_NA_{N_A}"
    data_dir = os.path.join(base_data_dir, folder, "training_histories")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, f"training_history_{tag}.json")

def free_energy_minimize_phases(
    vstate, T, partition, Hamiltonian, phases,
    verbose=True, freq=50, plot=True,
    chunk_size=256, timing=False,
    learning_rate=None, clip_norm=None, sr=None, n_samples_sr=None,
    monitor_every=20,
    dkl_n_samples=2048,
    history_index=None,
):
    """
    Minimiza F = E - T·S₂ pasando por una secuencia de fases.

    Guarda un archivo por temperatura:
        data/N{N}[_NA_{N_A}]/training_history_T{T}.json

    con dos bloques:
        "steps"       : [{global_step, phase, phase_step, method, F}, ...]
        "dkl_records" : [{tag, phase, phase_step, global_step, inner_step?, dkl}, ...]

    Tags de dkl_records:
        "initial_before"     : antes de initial_train
        "initial_train_step" : pasos internos de initial_train (inner_step = 0..n-1)
        "initial_after"      : después de initial_train
        "warm_start_before"  : antes de cada warm-start
        "warm_start_step"    : pasos internos de cada warm-start
        "warm_start_after"   : después de cada warm-start
        "monitor"            : monitor periódico
    """
    n_samples_full = vstate.n_samples
    free_energy_history = []
    best_F = float("inf")
    best_params = None
    phase_boundaries = []

    has_wang = any(p["method"] == "wang" for p in phases)
    global_step = 0

    # ── contenedor del historial completo ──
    training_history = {
        "N":          len(partition),
        "N_A":        vstate.hilbert.size - len(partition),
        "T":          float(T),
        "chunk_size": int(chunk_size),
        "phases":     [{"method": p["method"], "n_steps": p["n_steps"]}
                       for p in phases],
        "steps":       [],
        "dkl_records": [],
    }

    def _record_dkl(tag, phase_idx, phase_step, vstate_R, dkl_value=None,
                    inner_step=None):
        if dkl_value is None:
            dkl_value = compute_kl(vstate, vstate_R, n_samples=dkl_n_samples)
        rec = {
            "tag":         tag,
            "phase":       int(phase_idx),
            "phase_step":  int(phase_step),
            "global_step": int(global_step),
            "dkl":         float(dkl_value),
        }
        if inner_step is not None:
            rec["inner_step"] = int(inner_step)
        training_history["dkl_records"].append(rec)
        return dkl_value

    for i_phase, phase in enumerate(phases):
        method        = phase["method"]
        n_steps       = phase["n_steps"]
        vstate_R      = phase.get("vstate_R", None)
        warm_start    = phase.get("warm_start", None)
        drut_kwargs   = phase.get("drut_kwargs", None)
        wang_kwargs   = phase.get("wang_kwargs", None)
        optimizer     = phase.get("optimizer", None)
        lr_phase      = phase.get("learning_rate", learning_rate)
        initial_train = phase.get("initial_train", None)

        if method == "wang" and vstate_R is None:
            raise ValueError("Phase with method='wang' requires a 'vstate_R' entry.")

        op = FreeRenyiEnergyObservable(
            vstate.hilbert, Hamiltonian, partition, T,
            chunk_size=chunk_size,
            method=method,
            drut_kwargs=drut_kwargs,
            wang_kwargs=wang_kwargs,
            vstate_R=vstate_R,
        )
        if optimizer is None:
            lr = lr_phase if lr_phase is not None else 1e-3
            optimizer = optax.sgd(lr)
        grad_transform = optimizer
        if clip_norm is not None:
            grad_transform = optax.chain(
                optax.clip_by_global_norm(clip_norm), optimizer
            )
        opt_state = grad_transform.init(vstate.parameters)

        if verbose:
            print(f"\n[phase {i_phase}] method={method}, n_steps={n_steps}")
            if method == "wang" and warm_start is not None:
                print(f"             warm_start: {warm_start}")

        phase_start = len(free_energy_history)
        phase_boundaries.append(phase_start)

        # ── initial_train ──
        if initial_train is not None:
            dkl_before = _record_dkl("initial_before", i_phase, -1, vstate_R)
            vstate_R, dkl_traj = train_reverse_network(
                vstate, vstate_R,
                n_steps=initial_train["n_steps"],
                batch=initial_train.get("batch", 4096),
                lr=initial_train.get("lr", 1e-3),
                verbose=verbose, freq=500,
                return_history=True,
            )
            for k, dkl_val in enumerate(dkl_traj):
                _record_dkl("initial_train_step", i_phase, -1, vstate_R,
                            dkl_value=dkl_val, inner_step=k)
            dkl_after = _record_dkl("initial_after", i_phase, -1, vstate_R)
            if verbose:
                print(f"    D_KL: {dkl_before:.2e} → {dkl_after:.2e}")

        # ── loop de la fase ──
        for step in range(n_steps):
            if timing:
                t0 = time.time()

            F_stats, F_grad = vstate.expect_and_grad(op)

            if sr is not None:
                if n_samples_sr is not None:
                    vstate.n_samples = n_samples_sr
                F_grad = sr(vstate, F_grad, step)
                if n_samples_sr is not None:
                    vstate.n_samples = n_samples_full

            updates, opt_state = grad_transform.update(
                F_grad, opt_state, vstate.parameters
            )
            vstate.parameters = optax.apply_updates(vstate.parameters, updates)

            if timing:
                jax.tree_util.tree_map(
                    lambda x: x.block_until_ready(), vstate.parameters
                )

            F_val = float(F_stats.mean.real)
            free_energy_history.append(F_val)

            training_history["steps"].append({
                "global_step": int(global_step),
                "phase":       int(i_phase),
                "phase_step":  int(step),
                "method":      method,
                "F":           float(F_val),
            })

            if F_val < best_F:
                best_F = F_val
                best_params = vstate.parameters

            # ── warm-start ──
            if (method == "wang" and warm_start is not None
                    and step > 0 and step % warm_start["every"] == 0):
                dkl_pre = _record_dkl("warm_start_before", i_phase, step, vstate_R)
                vstate_R, dkl_traj = train_reverse_network(
                    vstate, vstate_R,
                    n_steps=warm_start["n_steps"],
                    batch=warm_start.get("batch", 1024),
                    lr=warm_start["lr"],
                    verbose=False,
                    return_history=True,
                )
                for k, dkl_val in enumerate(dkl_traj):
                    _record_dkl("warm_start_step", i_phase, step, vstate_R,
                                dkl_value=dkl_val, inner_step=k)
                dkl_post = _record_dkl("warm_start_after", i_phase, step, vstate_R)
                if verbose:
                    print(f"    [warm-start @ step {step}] "
                          f"D_KL: {dkl_pre:.2e} → {dkl_post:.2e}")

            # ── monitor periódico ──
            if (method == "wang" and monitor_every
                    and step > 0 and step % monitor_every == 0):
                dkl_now = _record_dkl("monitor", i_phase, step, vstate_R)
                if verbose:
                    print(f"    [monitor @ step {step}] "
                          f"D_KL(vstate || vstate_R) = {dkl_now:.2e}")

            if step % freq == 0 and verbose:
                msg = f"  [{method}] Step {step:4d} | F={F_val:.6f}"
                if timing:
                    msg += f" | t={time.time()-t0:.3f}s"
                print(msg)

            global_step += 1

    # ── guardar historial completo (una vez por entrenamiento) ──
    if has_wang:
        if history_index is not None:
            tag = f"{history_index:04d}"
        else:
            tag = str(T)                      # fallback por compatibilidad
        dkl_save_path = _auto_training_history_path(vstate, partition, tag)
        with open(dkl_save_path, "w") as f:
            json.dump(training_history, f, indent=2)
        if verbose:
            print(f"\n[training_history] {len(training_history['steps'])} steps + "
                  f"{len(training_history['dkl_records'])} dkl records → "
                  f"{dkl_save_path}")

    # ── restaurar y reevaluar ──
    vstate.parameters = best_params
    jax.clear_caches()
    vstate.chunk_size = chunk_size
    E_best = float(vstate.expect(Hamiltonian).mean.real)
    vstate.chunk_size = None
    S2_best = renyi2_entropy_sampled(
        vstate, partition, n_samples_full, chunk_size=chunk_size
    )
    best_F = E_best - T * S2_best

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(free_energy_history, label=r"$F$")
        for i, b in enumerate(phase_boundaries[1:], start=1):
            ax.axvline(b, color='k', ls='--', alpha=0.3,
                       label=f"phase {i}: {phases[i]['method']}")
        ax.set_xlabel("Step")
        ax.set_ylabel(r"$F$")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    return free_energy_history, best_F, E_best, S2_best

def free_energy_minimize_exact(
    vstate,         
    T,
    partition,
    Hamiltonian,
    hilbert,
    isFullSum = True,
    n_steps=500,
    verbose=True,
    freq=50,
    plot=True,
    optimizer=None,
    learning_rate=None,
    clip_norm=None,
    timing=False,
    sr=None,
):
    """
    Minimiza F = E - T·S₂ usando gradientes exactos.

    El gradiente de E se calcula via vstate.expect_and_grad(H).
    El gradiente de S2 se calcula via renyi2_entropy_and_grad_exact.
    No usa FreeRenyiEnergyObservable ni el swap trick.

    Parámetros
    ----------
    vstate      : MCState de NetKet con el modelo variacional.
    T           : Temperatura.
    partition   : Lista de sitios del subsistema A para S₂.
    Hamiltonian : Operador H compatible con NetKet.
    n_steps     : Número de pasos de optimización.
    verbose     : Si True, imprime progreso cada `freq` pasos.
    freq        : Frecuencia de impresión.
    plot        : Si True, muestra gráfica de F al final.
    optimizer   : optax.GradientTransformation. Por defecto SGD.
    learning_rate: Schedule o escalar de optax.
    clip_norm   : float o None. Clipping del gradiente global.
    timing      : Si True, mide tiempo por step.
    sr          : nk.optimizer.SR o None.

    Devuelve
    -------
    (free_energy_history, best_F, E_best, S2_best)
    """

    # --- learning rate por defecto ---
    if learning_rate is None:
        learning_rate = optax.warmup_cosine_decay_schedule(
            0.1, 0.1, 100, n_steps, 0.001
        )

    # --- optimizador por defecto ---
    if optimizer is None:
        optimizer = optax.sgd(learning_rate)

    # --- gradient transform ---
    if clip_norm is not None:
        gradient_transform = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optimizer
        )
    else:
        gradient_transform = optimizer

    opt_state = gradient_transform.init(vstate.parameters)

    free_energy_history = []
    E_history = []
    S2_history = []
    best_F = float("inf")
    best_params = None

    for step in range(n_steps):

        if timing:
            t0 = time.time()

        # ── Gradiente de E (exacto via FullSumState) ──────────────
        E_stats, grad_E = vstate.expect_and_grad(Hamiltonian)
        E_val = float(E_stats.mean.real)

        # ── S2 y su gradiente (exacto) ────────────────────────────
        S2_val, grad_S2 = renyi2_entropy_and_grad_exact(vstate, partition, hilbert, isFullSum=isFullSum)
        S2_val = float(S2_val)

        # ── Gradiente de F = E - T*S2 ─────────────────────────────
        grad_F = jax.tree_util.tree_map(
            lambda ge, gs: ge - T * gs,
            grad_E, grad_S2
        )

        F_val = E_val - T * S2_val

        # ── SR (opcional) ─────────────────────────────────────────
        if sr is not None:
            grad_F = sr(vstate, grad_F, step)

        # ── Update ────────────────────────────────────────────────
        updates, opt_state = gradient_transform.update(
            grad_F, opt_state, vstate.parameters
        )
        vstate.parameters = optax.apply_updates(vstate.parameters, updates)

        if timing:
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready(), vstate.parameters
            )

        free_energy_history.append(F_val)
        E_history.append(E_val)
        S2_history.append(S2_val)

        if F_val < best_F:
            best_F = F_val
            best_params = vstate.parameters

        if step % freq == 0 and verbose:
            msg = f"Step {step:4d} | F={F_val:.6f} | E={E_val:.6f} | S2={S2_val:.6f}"
            if timing:
                msg += f" | t={time.time()-t0:.3f}s"
            print(msg)

    # ── Restaurar mejores parámetros ──────────────────────────────
    vstate.parameters = best_params

    # Evaluación final exacta
    E_best = float(vstate.expect(Hamiltonian).mean.real)
    S2_best, _ = renyi2_entropy_and_grad_exact(vstate, partition, hilbert, isFullSum=isFullSum)
    S2_best = float(S2_best)
    best_F = E_best - T * S2_best

    if verbose:
        print(f"\nFinal | F={best_F:.6f} | E={E_best:.6f} | S2={S2_best:.6f}")

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(free_energy_history, label=r"$F$")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel(r"$F = E - T \cdot S_2$")
        axes[0].set_title("Energía libre")
        axes[0].legend()

        axes[1].plot(E_history, label=r"$\langle H \rangle$", color="tab:orange")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel(r"$\langle H \rangle$")
        axes[1].set_title("Energía")
        axes[1].legend()

        axes[2].plot(S2_history, label=r"$S_2$", color="tab:green")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel(r"$S_2$")
        axes[2].set_title("Entropía Rényi-2")
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    return (free_energy_history, best_F, E_best, S2_best)

def free_energy_minimize_scipy(
    vstate, T, partition, Hamiltonian,
    method="L-BFGS-B", options=None, verbose=True,
):
    """
    Minimiza F = E - T·S₂ usando scipy (método determinista, exacto).

    Adecuado para sistemas pequeños donde S₂ se puede calcular exactamente.
    Usa L-BFGS-B por defecto — no requiere gradiente explícito (lo estima
    numéricamente), pero es más lento que SR+SGD para sistemas grandes.

    Parámetros
    ----------
    vstate      : MCState de NetKet con el modelo variacional.
    T           : Temperatura.
    partition   : Lista de sitios del subsistema A para S₂.
    Hamiltonian : Operador H compatible con NetKet.
    method      : Método de scipy.optimize.minimize. Por defecto "L-BFGS-B".
    options     : Diccionario de opciones para scipy. Por defecto {"maxiter": 200}.
    verbose     : Si True, imprime F, E y S₂ en cada evaluación.

    Devuelve
    -------
    Diccionario con:
        result              : Objeto OptimizeResult de scipy.
        free_energy_history : Lista de F en cada evaluación.
        energy_history      : Lista de E en cada evaluación.
        entropy_history     : Lista de S₂ en cada evaluación.
        best_F              : Mejor F alcanzado.
        best_energy         : E en el mejor F.
        best_entropy        : S₂ en el mejor F.
    """
    flat_params0, unravel_fn = ravel_pytree(vstate.parameters)

    free_energy_history = []
    energy_history = []
    entropy_history = []

    def objective(flat_params):
        vstate.parameters = unravel_fn(flat_params)

        E = vstate.expect(Hamiltonian)
        S2, _ = renyi2_entropy_exact(vstate, partition)
        F = float(E.mean.real) - T * float(S2)

        free_energy_history.append(F)
        energy_history.append(float(E.mean.real))
        entropy_history.append(float(S2))

        if verbose:
            print(f"F={F:.6f} | E={float(E.mean.real):.6f} | S₂={float(S2):.6f}")

        return F

    res = minimize(
        fun=objective,
        x0=np.array(flat_params0, dtype=np.float64),
        method=method,
        options=options or {"maxiter": 200},
    )

    vstate.parameters = unravel_fn(res.x)
    best_idx = int(np.argmin(free_energy_history))

    return {
        "result":              res,
        "free_energy_history": free_energy_history,
        "energy_history":      energy_history,
        "entropy_history":     entropy_history,
        "best_F":              free_energy_history[best_idx],
        "best_energy":         energy_history[best_idx],
        "best_entropy":        entropy_history[best_idx],
    }