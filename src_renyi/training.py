import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from .observables import FreeRenyiEnergyObservable
from .entropy import renyi2_entropy_and_grad_sampled, renyi2_entropy_sampled, renyi2_entropy_exact, renyi2_entropy_and_grad_lambda_integral


def free_energy_minimize_SR_SGD(
    vstate, T, partition, Hamiltonian, n_steps=1000,
    verbose=True, freq=50, plot=True,
    learning_rate=None, diag_shift=None,
    n_samples_sr=4096, timing=False,
):
    """
    Minimiza F = E - T·S₂ con SR + SGD.

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
    learning_rate: Schedule o escalar de optax. Por defecto warmup_cosine_decay.
    diag_shift   : Schedule o escalar para SR. Por defecto linear 1e-1 → 1e-4.
    n_samples_sr : Muestras usadas en el paso de SR (< n_samples completo).
    timing       : Si True, mide y muestra el tiempo real de cada step en los prints.

    Devuelve
    -------
    (free_energy_history, best_F, E_best, S2_best)
    """
    if learning_rate is None:
        learning_rate = optax.warmup_cosine_decay_schedule(0.1, 0.1, 100, n_steps, 0.001)
    if diag_shift is None:
        diag_shift = optax.linear_schedule(1e-1, 1e-4, n_steps)

    sr = nk.optimizer.SR(diag_shift=diag_shift)
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(vstate.parameters)

    free_renyi_op = FreeRenyiEnergyObservable(vstate.hilbert, Hamiltonian, partition, T)
    n_samples_full = vstate.n_samples

    free_energy_history = []
    best_F = float("inf")
    best_params = None

    for step in range(n_steps):
        if timing:
            t0 = time.time()

        F_stats, F_grad = vstate.expect_and_grad(free_renyi_op)

        vstate.n_samples = n_samples_sr
        delta = sr(vstate, F_grad, step)
        vstate.n_samples = n_samples_full

        updates, opt_state = optimizer.update(delta, opt_state)
        vstate.parameters = optax.apply_updates(vstate.parameters, updates)

        if timing:
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), vstate.parameters)

        F_val = float(F_stats.mean.real)
        free_energy_history.append(F_val)

        if F_val < best_F:
            best_F = F_val
            best_params = vstate.parameters

        if step % freq == 0 and verbose:
            if timing:
                print(f"Step {step:4d} | F={F_val:.6f} | t={time.time()-t0:.3f}s")
            else:
                print(f"Step {step:4d} | F={F_val:.6f}")

    # Restaurar mejores parámetros y evaluar
    vstate.parameters = best_params
    jax.clear_caches()

    vstate.chunk_size = 64
    E_best = float(vstate.expect(Hamiltonian).mean.real)
    S2_best = renyi2_entropy_sampled(vstate, partition, n_samples_full)
    best_F = E_best - T * S2_best
    vstate.chunk_size = None

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(free_energy_history, color="tab:blue", label=r"$F$")
        ax.set_xlabel("Step")
        ax.set_ylabel(r"$F$", color="tab:blue")
        plt.tight_layout()
        plt.show()

    return free_energy_history, best_F, E_best, S2_best

def free_energy_minimize(vstate, T, partition, Hamiltonian, n_steps=1000, verbose=True, freq=50,
    plot=True, optimizer=None, learning_rate=None, clip_norm=None, timing=False):
    """
    Minimiza F = E - T·S₂ con optimizador general.

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
    optimizer : optax.GradientTransformation
    learning_rate: Schedule o escalar de optax. Por defecto warmup_cosine_decay.
    clip_norm : float o None. Si no es None, aplica clip_by_global_norm.
    timing       : Si True, mide y muestra el tiempo real de cada step en los prints.

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

    # --- construir gradient_transform ---
    if clip_norm is not None:
        gradient_transform = optax.chain(
            optax.clip_by_global_norm(clip_norm),
            optimizer
        )
    else:
        gradient_transform = optimizer

    opt_state = gradient_transform.init(vstate.parameters)

    free_renyi_op = FreeRenyiEnergyObservable(
        vstate.hilbert,
        Hamiltonian,
        partition,
        T
    )

    n_samples_full = vstate.n_samples

    free_energy_history = []
    best_F = float("inf")
    best_params = None

    for step in range(n_steps):

        if timing:
            t0 = time.time()

        F_stats, F_grad = vstate.expect_and_grad(free_renyi_op)

        updates, opt_state = gradient_transform.update(F_grad, opt_state, vstate.parameters)

        vstate.parameters = optax.apply_updates(vstate.parameters, updates)

        if timing:
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), vstate.parameters)

        F_val = float(F_stats.mean.real)
        free_energy_history.append(F_val)

        if F_val < best_F:
            best_F = F_val
            best_params = vstate.parameters

        if step % freq == 0 and verbose:
            if timing:
                print(
                    f"Step {step:4d} | "
                    f"F={F_val:.6f} | "
                    f"t={time.time()-t0:.3f}s"
                )
            else:
                print(
                    f"Step {step:4d} | "
                    f"F={F_val:.6f}"
                )

    # --- restaurar mejores parámetros ---

    vstate.parameters = best_params
    jax.clear_caches()

    vstate.chunk_size = 64
    E_best = float(vstate.expect(Hamiltonian).mean.real)
    S2_best = renyi2_entropy_sampled(vstate, partition, n_samples_full)
    best_F = E_best - T * S2_best
    vstate.chunk_size = None

    if plot:

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.plot(
            free_energy_history,
            label=r"$F$",
        )

        ax.set_xlabel("Step")
        ax.set_ylabel(r"$F$")

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

def renyi_entropy_maximize_SR_SGD(
    vstate, partition, n_steps=1000,
    verbose=True, freq=50, plot=True,
    learning_rate=None, diag_shift=None,
    n_samples_sr=4096, timing=False,
):
    """
    Maximiza S₂ del subsistema A usando SR + SGD.

    Parámetros
    ----------
    vstate       : MCState de NetKet con el modelo variacional.
    partition    : Lista de sitios del subsistema A para S₂.
    n_steps      : Número de pasos de optimización.
    verbose      : Si True, imprime progreso cada `freq` pasos.
    freq         : Frecuencia de impresión.
    plot         : Si True, muestra gráfica de S₂ al final.
    learning_rate: Schedule o escalar de optax. Por defecto warmup_cosine_decay.
    diag_shift   : Schedule o escalar para SR. Por defecto linear 1e-1 → 1e-4.
    n_samples_sr : Muestras usadas en el paso de SR (< n_samples completo).
    timing       : Si True, mide y muestra el tiempo real de cada step en los prints.

    Devuelve
    -------
    (entropy_history, best_S2, best_params)
    """
    if learning_rate is None:
        learning_rate = optax.warmup_cosine_decay_schedule(0.1, 0.1, 100, n_steps, 0.001)
    if diag_shift is None:
        diag_shift = optax.linear_schedule(1e-1, 1e-4, n_steps)

    sr = nk.optimizer.SR(diag_shift=diag_shift)
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(vstate.parameters)
    n_samples_full = vstate.n_samples

    entropy_history = []
    best_S2 = float("-inf")
    best_params = None

    for step in range(n_steps):
        if timing:
            t0 = time.time()

        # S2, grad_S2 = renyi2_entropy_and_grad_sampled(vstate, partition, n_samples_full)
        S2, grad_S2 = renyi2_entropy_and_grad_lambda_integral(vstate, partition, n_samples_full, 30)

        S2_max = len(partition) * jnp.log(2.0)
        mask = float(S2 < S2_max)

        # Gradiente con mask aplicado (se anula si S2 >= S2_max)
        masked_grad = jax.tree_util.tree_map(lambda g: mask * g, grad_S2)

        vstate.n_samples = n_samples_sr
        delta = sr(vstate, masked_grad, step)
        vstate.n_samples = n_samples_full

        # Negamos delta para maximizar
        neg_delta = jax.tree_util.tree_map(lambda d: -d, delta)
        updates, opt_state = optimizer.update(neg_delta, opt_state)
        vstate.parameters = optax.apply_updates(vstate.parameters, updates)
        vstate.parameters = optax.apply_updates(vstate.parameters, updates)

        if timing:
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), vstate.parameters)

        S2_val = float(S2)
        entropy_history.append(S2_val)

        if S2_val > best_S2:
            best_S2 = S2_val
            best_params = vstate.parameters

        if step % freq == 0 and verbose:
            if timing:
                print(f"Step {step:4d} | S₂={S2_val:.6f} | t={time.time()-t0:.3f}s")
            else:
                print(f"Step {step:4d} | S₂={S2_val:.6f}")

    vstate.parameters = best_params

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(entropy_history, color="tab:green", label=r"$S_2$")
        ax.set_xlabel("Step")
        ax.set_ylabel(r"$S_2$", color="tab:green")
        plt.tight_layout()
        plt.show()

    return entropy_history, best_S2, best_params

def renyi_entropy_maximize_ADAM(
    vstate, partition, n_steps=1000,
    verbose=True, freq=50, plot=True,
    learning_rate=None, timing=False,
):
    if learning_rate is None:
        learning_rate = optax.linear_schedule(1e-3, 1e-5, n_steps)

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(vstate.parameters)
    n_samples_full = vstate.n_samples

    entropy_history = []
    best_S2 = float("-inf")
    best_params = None

    for step in range(n_steps):
        if timing:
            t0 = time.time()

        S2, grad_S2 = renyi2_entropy_and_grad_sampled(
            vstate, partition, n_samples_full
        )

        # Negamos el gradiente para maximizar
        neg_grad = jax.tree_util.tree_map(lambda g: -g, grad_S2)
        updates, opt_state = optimizer.update(neg_grad, opt_state)
        vstate.parameters = optax.apply_updates(vstate.parameters, updates)

        if timing:
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), vstate.parameters)

        S2_val = float(S2)
        entropy_history.append(S2_val)

        if S2_val > best_S2:
            best_S2 = S2_val
            best_params = vstate.parameters

        if step % freq == 0 and verbose:
            if timing:
                print(f"Step {step:4d} | S₂={S2_val:.6f} | t={time.time()-t0:.3f}s")
            else:
                print(f"Step {step:4d} | S₂={S2_val:.6f}")

    vstate.parameters = best_params

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(entropy_history, color="tab:green", label=r"$S_2$")
        ax.set_xlabel("Step")
        ax.set_ylabel(r"$S_2$", color="tab:green")
        plt.tight_layout()
        plt.show()

    return entropy_history, best_S2, best_params