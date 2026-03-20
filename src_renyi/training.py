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
from .entropy import renyi2_entropy_sampled, renyi2_entropy_exact


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

        lr = float(learning_rate(step) if callable(learning_rate) else learning_rate)
        vstate.parameters = jax.tree_util.tree_map(
            lambda p, d: p - lr * d, vstate.parameters, delta
        )

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
    E_best = float(vstate.expect(Hamiltonian).mean.real)
    S2_best = renyi2_entropy_sampled(vstate, partition, n_samples_full)
    best_F = E_best - T * S2_best

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(free_energy_history, color="tab:blue", label=r"$F$")
        ax.set_xlabel("Step")
        ax.set_ylabel(r"$F$", color="tab:blue")
        plt.tight_layout()
        plt.show()

    return free_energy_history, best_F, E_best, S2_best


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