import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import optax

from .observables import FreeRenyiEnergyObservable
from .entropy import renyi2_entropy_sampled


def free_energy_minimize_SR_SGD(
    vstate, T, partition, Hamiltonian, n_steps=1000,
    verbose=True, freq=50, plot=True,
    learning_rate=None, diag_shift=None,
    n_samples_sr=4096,
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
        t0 = time.time()

        F_stats, F_grad = vstate.expect_and_grad(free_renyi_op)

        vstate.n_samples = n_samples_sr
        delta = sr(vstate, F_grad, step)
        vstate.n_samples = n_samples_full

        lr = float(learning_rate(step) if callable(learning_rate) else learning_rate)
        vstate.parameters = jax.tree_util.tree_map(
            lambda p, d: p - lr * d, vstate.parameters, delta
        )

        F_val = float(F_stats.mean.real)
        free_energy_history.append(F_val)

        if F_val < best_F:
            best_F = F_val
            best_params = vstate.parameters

        if step % freq == 0 and verbose:
            print(f"Step {step:4d} | F={F_val:.6f} | t={time.time()-t0:.3f}s")

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