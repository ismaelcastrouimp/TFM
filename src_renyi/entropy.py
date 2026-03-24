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

from netket.jax import jacobian
def renyi2_entropy_and_grad_sampled2(vstate, subsystem_sites, n_samples, key=0, debug=False):
    subsystem_sites = jnp.array(subsystem_sites, dtype=int)
    """
    Calcula la entropía de Rényi-2 de un subsistema y su gradiente mediante el *swap trick*
    
    Args:
        vstate: estado variacional de NetKet
        subsystem_sites: índices del subsistema
        n_samples: número de muestras
        key: semilla aleatoria
        debug: modo depuración
    
    Returns:
        S2: entropía de Rényi-2
        grad_S2: gradiente respecto a parámetros (misma estructura que vstate.parameters)
    """
    
    # ============================================================
    # 1) MUESTREO
    # ============================================================
    all_samples = vstate.sample(n_samples=2 * n_samples, n_discard_per_chain=1000).reshape(-1, vstate.hilbert.size)
    key = jax.random.PRNGKey(key)
    all_samples = jax.random.permutation(key, all_samples, axis=0)
    
    samples1 = all_samples[:n_samples]
    samples2 = all_samples[n_samples:2*n_samples]
    
    # ============================================================
    # 2) CONFIGURACIONES SWAP
    # ============================================================
    all_sites = jnp.arange(samples1.shape[1])
    complement_sites = jnp.setdiff1d(all_sites, subsystem_sites)
    
    swapped1 = jnp.concatenate([samples2[:, subsystem_sites], samples1[:, complement_sites]], axis=1)
    swapped2 = jnp.concatenate([samples1[:, subsystem_sites], samples2[:, complement_sites]], axis=1)
    
    # ============================================================
    # 3) LOG-AMPLITUDES
    # ============================================================
    log_o1 = vstate.log_value(samples1)
    log_o2 = vstate.log_value(samples2)
    log_s1 = vstate.log_value(swapped1)
    log_s2 = vstate.log_value(swapped2)
    
    # ============================================================
    # 4) RATIO SWAP Y ENTROPÍA S₂
    # ============================================================
    R = jnp.exp(jnp.real(log_s1 + log_s2 - log_o1 - log_o2))
    R_mean = jnp.mean(R)
    S2 = -jnp.log(jnp.abs(R_mean))
    
    # ============================================================
    # 5) JACOBIANOS O_θ = ∂_θ log ψ
    # ============================================================
    O_o1 = jacobian(vstate._apply_fun, vstate.parameters, samples1,
                    model_state=vstate.model_state, mode="real", dense=True)
    O_o2 = jacobian(vstate._apply_fun, vstate.parameters, samples2,
                    model_state=vstate.model_state, mode="real", dense=True)
    O_s1 = jacobian(vstate._apply_fun, vstate.parameters, swapped1,
                    model_state=vstate.model_state, mode="real", dense=True)
    O_s2 = jacobian(vstate._apply_fun, vstate.parameters, swapped2,
                    model_state=vstate.model_state, mode="real", dense=True)
    
    R_exp = R.reshape(-1, 1)  # (n_samples, 1) para broadcasting
    
    # ============================================================
    # 6) TÉRMINO 1: dependencia explícita de R en θ
    #    ⟨R · ∇log R⟩ = ⟨R · 2(O_s1 + O_s2 - O_o1 - O_o2)⟩
    # ============================================================
    grad_log_R = O_s1 + O_s2 - O_o1 - O_o2  # (n_samples, n_params)
    term1 = 2.0 * jnp.mean(R_exp * grad_log_R, axis=0)  # (n_params,)

    # ============================================================
    # 7) TÉRMINO 2: dependencia de p(x)p(y) en θ — REINFORCE
    #    2⟨(R - ⟨R⟩) · (O_o1 + O_o2)⟩
    #    ∇log p(x)p(y) = 2(O_o1 + O_o2) porque p = |ψ|²
    # ============================================================
    R_centered = (R - R_mean).reshape(-1, 1)
    term2 = 2.0 * jnp.mean(R_centered * (O_o1 + O_o2), axis=0)  # (n_params,)

    # ============================================================
    # 8) ∇S₂ = -(term1 + term2) / ⟨R⟩
    # ============================================================
    grad_S2_flat = -(term1 + term2) / R_mean

    _, unravel = jax.flatten_util.ravel_pytree(vstate.parameters)
    grad_S2 = unravel(grad_S2_flat)

    if debug:
        print(f"⟨R⟩  = {float(R_mean):.6f}")
        print(f"S₂   = {float(S2):.6f}")
        print(f"||term1|| = {jnp.linalg.norm(term1):.6f}  (explícito)")
        print(f"||term2|| = {jnp.linalg.norm(term2):.6f}  (REINFORCE)")
        print(f"|∇S₂| = {jnp.linalg.norm(grad_S2_flat):.6f}")
    
    return S2, grad_S2

def renyi2_entropy_and_grad_lambda_integral(vstate, subsystem_sites, n_samples, n_lambda=10, key=0, debug=False):
    """
    Calcula S₂ y ∇S₂ usando el método de la λ-integral
    basado en Drut & Porter (2022)

    S₂ = - ∫₀¹ dλ ⟨ln R⟩_{P_λ}
    ∇S₂ = - ∫₀¹ [ ⟨∇lnR⟩_{P_λ} + λ·Cov_{P_λ}(lnR, ∇lnR) + 2⟨(lnR-⟨lnR⟩)(O_o1+O_o2)⟩_{P_λ} ] dλ
    """
    subsystem_sites = jnp.array(subsystem_sites, dtype=int)
    key_obj = jax.random.PRNGKey(key)
    lambda_grid = jnp.linspace(0.0, 1.0, n_lambda)

    f_vals = []
    grad_f_vals = []

    for lam in lambda_grid:

        key_obj, subkey = jax.random.split(key_obj)

        # =====================================================
        # 1) MUESTREO
        # =====================================================
        all_samples = (
            vstate.sample(n_samples=2 * n_samples, n_discard_per_chain=500)
            .reshape(-1, vstate.hilbert.size)
        )
        all_samples = jax.random.permutation(subkey, all_samples, axis=0)

        samples1 = all_samples[:n_samples]
        samples2 = all_samples[n_samples:2 * n_samples]

        # =====================================================
        # 2) SWAP CONFIGS
        # =====================================================
        all_sites = jnp.arange(samples1.shape[1])
        complement_sites = jnp.setdiff1d(all_sites, subsystem_sites)

        swapped1 = jnp.concatenate(
            [samples2[:, subsystem_sites], samples1[:, complement_sites]], axis=1
        )
        swapped2 = jnp.concatenate(
            [samples1[:, subsystem_sites], samples2[:, complement_sites]], axis=1
        )

        # =====================================================
        # 3) LOG-AMPLITUDES Y RATIO R
        # =====================================================
        log_o1 = vstate.log_value(samples1)
        log_o2 = vstate.log_value(samples2)
        log_s1 = vstate.log_value(swapped1)
        log_s2 = vstate.log_value(swapped2)

        log_R = jnp.real(log_s1 + log_s2 - log_o1 - log_o2)

        # =====================================================
        # 4) JACOBIANOS O_θ = ∂_θ log ψ
        # =====================================================
        O_o1 = jacobian(vstate._apply_fun, vstate.parameters, samples1,
                        model_state=vstate.model_state, mode="real", dense=True)
        O_o2 = jacobian(vstate._apply_fun, vstate.parameters, samples2,
                        model_state=vstate.model_state, mode="real", dense=True)
        O_s1 = jacobian(vstate._apply_fun, vstate.parameters, swapped1,
                        model_state=vstate.model_state, mode="real", dense=True)
        O_s2 = jacobian(vstate._apply_fun, vstate.parameters, swapped2,
                        model_state=vstate.model_state, mode="real", dense=True)

        # ∇ln R = 2*(O_s1 + O_s2 - O_o1 - O_o2)
        grad_lnR = 2.0 * (O_s1 + O_s2 - O_o1 - O_o2)  # (n_samples, n_params)

        # =====================================================
        # 5) PESOS P_λ ∝ p(x)p(y) R^λ
        # =====================================================
        log_weights = lam * log_R
        log_weights -= jax.nn.logsumexp(log_weights)
        weights = jnp.exp(log_weights)  # (n_samples,)

        w = weights.reshape(-1, 1)  # para broadcasting

        # =====================================================
        # 6) f(λ) = ⟨ln R⟩_λ
        # =====================================================
        f_lam = jnp.sum(weights * log_R)
        f_vals.append(f_lam)

        # =====================================================
        # 7) ⟨∇ln R⟩_λ
        # =====================================================
        mean_grad = jnp.sum(w * grad_lnR, axis=0)  # (n_params,)

        # =====================================================
        # 8) Covarianza: Cov_λ(lnR, ∇lnR)
        #    = ⟨lnR · ∇lnR⟩_λ - ⟨lnR⟩_λ · ⟨∇lnR⟩_λ
        # =====================================================
        lnR_exp = log_R.reshape(-1, 1)
        mean_lnR_grad = jnp.sum(w * lnR_exp * grad_lnR, axis=0)
        cov = mean_lnR_grad - f_lam * mean_grad  # (n_params,)

        # =====================================================
        # 9) TÉRMINO REINFORCE: dependencia de p(x)p(y) en θ
        #    2⟨(lnR - ⟨lnR⟩_λ)(O_o1 + O_o2)⟩_λ
        # =====================================================
        lnR_centered = (log_R - f_lam).reshape(-1, 1)
        grad_log_p = O_o1 + O_o2  # ∇log[p(x)p(y)] / 2, shape (n_samples, n_params)
        reinforce = 2.0 * jnp.sum(w * lnR_centered * grad_log_p, axis=0)  # (n_params,)

        # =====================================================
        # 10) ∇f(λ) = ⟨∇lnR⟩_λ + λ·Cov + REINFORCE
        # =====================================================
        grad_f_flat = mean_grad + lam * cov + reinforce  # (n_params,)
        grad_f_vals.append(grad_f_flat)

        if debug:
            ess = 1.0 / jnp.sum(weights ** 2)
            print(f"λ={float(lam):.3f} | f(λ)={float(f_lam):.6f} | "
                  f"||mean_grad||={jnp.linalg.norm(mean_grad):.4f} | "
                  f"||cov||={jnp.linalg.norm(cov):.4f} | "
                  f"||reinforce||={jnp.linalg.norm(reinforce):.4f} | "
                  f"ESS={float(ess):.1f}")

    # =====================================================
    # 11) INTEGRACIÓN TRAPECIO
    # =====================================================
    f_vals = jnp.array(f_vals)
    dlam = lambda_grid[1] - lambda_grid[0]

    S2 = -(dlam * (0.5 * f_vals[0] + jnp.sum(f_vals[1:-1]) + 0.5 * f_vals[-1]))

    flat_grads = jnp.stack(grad_f_vals)  # (n_lambda, n_params)
    grad_S2_flat = -(dlam * (0.5 * flat_grads[0] + jnp.sum(flat_grads[1:-1], axis=0) + 0.5 * flat_grads[-1]))

    _, unravel = jax.flatten_util.ravel_pytree(vstate.parameters)
    grad_S2 = unravel(grad_S2_flat)

    if debug:
        print(f"\nS₂ = {float(S2):.6f}")
        print(f"||∇S₂|| = {jnp.linalg.norm(grad_S2_flat):.6f}")

    return float(S2), grad_S2

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