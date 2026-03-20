from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np


# ── Utilidades ────────────────────────────────────────────────────────────────

def vstate_to_vector(vstate):
    """
    Convierte un vstate de NetKet a un vector de estado completo.

    Enumera todas las configuraciones de la base computacional y evalúa
    las amplitudes log ψ(x) para construir |ψ⟩ normalizado.

    Parámetros
    ----------
    vstate : MCState de NetKet.

    Devuelve
    -------
    state_vector : Array complejo normalizado de dimensión 2^N.
    all_configs  : Configuraciones de la base en representación {-1, +1}.
    """
    N = vstate.hilbert.size
    all_configs = jnp.array(list(product([-1, 1], repeat=N)))
    log_psi = vstate.log_value(all_configs)
    psi = jnp.exp(log_psi)
    norm = jnp.sqrt(jnp.sum(jnp.abs(psi)**2))
    return psi / norm, all_configs


# ── Rényi-2 exacto ────────────────────────────────────────────────────────────

def renyi2_entropy_exact(vstate, subsystem_sites):
    """
    Calcula S₂ de forma exacta construyendo la matriz densidad reducida ρ_A.

    Escala exponencialmente con N — solo viable para sistemas pequeños.
    Devuelve también ρ_A por si se necesita para diagnóstico.

    S₂ = -ln Tr(ρ_A²)

    Parámetros
    ----------
    vstate          : MCState de NetKet.
    subsystem_sites : Índices de los sitios del subsistema A.

    Devuelve
    -------
    S2   : Entropía de Rényi-2 (escalar real).
    rho_A: Matriz densidad reducida del subsistema A.
    """
    N = vstate.hilbert.size
    subsystem_sites = np.array(subsystem_sites, dtype=int)

    psi, basis = vstate_to_vector(vstate)

    n_A = len(subsystem_sites)
    n_B = N - n_A
    dim_A = 2**n_A
    dim_B = 2**n_B

    all_sites = np.arange(N)
    complement_sites = np.setdiff1d(all_sites, subsystem_sites)

    # Convertir base {-1,+1} → {0,1} y reordenar (A primero, B después)
    basis_01 = ((basis + 1) // 2).astype(int)
    reordered_sites = np.concatenate([subsystem_sites, complement_sites])
    basis_reordered = basis_01[:, reordered_sites]

    powers_A = 2**np.arange(n_A)[::-1]
    powers_B = 2**np.arange(n_B)[::-1]
    idx_A = (basis_reordered[:, :n_A] * powers_A).sum(axis=1)
    idx_B = (basis_reordered[:, n_A:] * powers_B).sum(axis=1)

    # Construir ψ como matriz dim_A × dim_B
    psi_matrix = jnp.zeros((dim_A, dim_B), dtype=complex)
    psi_matrix = psi_matrix.at[idx_A, idx_B].set(psi)

    # ρ_A = Tr_B(|ψ⟩⟨ψ|) = ψ_matrix @ ψ_matrix†
    rho_A = psi_matrix @ jnp.conj(psi_matrix.T)

    purity = jnp.trace(rho_A @ rho_A).real
    S2 = -jnp.log(purity)

    return S2, rho_A


# ── Rényi-2 muestreado ────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("apply_fun",))
def _renyi2_loss_and_grad(apply_fun, params, model_state,
                          samples1, samples2, swapped1, swapped2):
    """
    Calcula S₂ y grad S₂ usando backprop.
    Memoria O(n_params) en vez de O(n_samples × n_params).
    """
    def log_psi(p, s):
        return apply_fun({"params": p, **model_state}, s)

    log_o1 = log_psi(params, samples1)
    log_o2 = log_psi(params, samples2)
    log_s1 = log_psi(params, swapped1)
    log_s2 = log_psi(params, swapped2)
    R = jnp.exp(jnp.real(log_s1 + log_s2 - log_o1 - log_o2))
    S2 = -jnp.log(jnp.abs(jnp.mean(R)))

    def loss_fn(p):
        lo1 = log_psi(p, samples1)
        lo2 = log_psi(p, samples2)
        ls1 = log_psi(p, swapped1)
        ls2 = log_psi(p, swapped2)
        R_ = jnp.exp(jnp.real(ls1 + ls2 - lo1 - lo2))
        R_mean = jax.lax.stop_gradient(jnp.mean(R_))
        w = jax.lax.stop_gradient(R_ / R_mean)
        return (
            -2.0 * jnp.mean(w * jnp.real(ls1 + ls2))
            + 2.0 * jnp.mean(jnp.real(lo1 + lo2))
        )

    grad_S2 = jax.grad(loss_fn)(params)
    return S2, grad_S2


def renyi2_entropy_and_grad_sampled(vstate, subsystem_sites, n_samples, key=0, debug=False):
    """
    Estima S₂ y su gradiente mediante muestreo Monte Carlo (swap trick).

    Parámetros
    ----------
    vstate          : MCState de NetKet.
    subsystem_sites : Índices de los sitios del subsistema A.
    n_samples       : Número de muestras por cada copia.
    key             : Semilla para la permutación aleatoria.
    debug           : Si True, imprime S₂ y norma del gradiente.

    Devuelve
    -------
    S2      : Estimación de la entropía de Rényi-2.
    grad_S2 : Gradiente de S₂ respecto a los parámetros.
    """
    subsystem_sites = jnp.array(subsystem_sites, dtype=int)

    all_samples = vstate.sample(
        n_samples=2 * n_samples, n_discard_per_chain=1000
    ).reshape(-1, vstate.hilbert.size)

    rng = jax.random.PRNGKey(key)
    all_samples = jax.random.permutation(rng, all_samples, axis=0)
    samples1 = all_samples[:n_samples]
    samples2 = all_samples[n_samples:]

    all_sites = jnp.arange(samples1.shape[1])
    complement_sites = jnp.setdiff1d(all_sites, subsystem_sites)
    swapped1 = jnp.concatenate([samples2[:, subsystem_sites], samples1[:, complement_sites]], axis=1)
    swapped2 = jnp.concatenate([samples1[:, subsystem_sites], samples2[:, complement_sites]], axis=1)

    S2, grad_S2 = _renyi2_loss_and_grad(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        samples1, samples2, swapped1, swapped2,
    )

    if debug:
        grad_flat, _ = jax.flatten_util.ravel_pytree(grad_S2)
        print(f"S₂    = {float(S2):.6f}")
        print(f"|∇S₂| = {jnp.linalg.norm(grad_flat):.6f}")

    return S2, grad_S2


def renyi2_entropy_sampled(vstate, partition, n_samples):
    """
    Estima S₂ (escalar) mediante muestreo Monte Carlo.

    Parámetros
    ----------
    vstate    : MCState de NetKet.
    partition : Índices de los sitios del subsistema A.
    n_samples : Número de muestras por copia.

    Devuelve
    -------
    S2 : Estimación escalar de la entropía de Rényi-2.
    """
    S2, _ = renyi2_entropy_and_grad_sampled(vstate, partition, n_samples)
    return float(S2)