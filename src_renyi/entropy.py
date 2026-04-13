from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import netket as nk


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

def renyi2_entropy_and_grad_exact(vstate, subsystem_sites, hi_extended, isFullSum=False):
    """
    Calcula el gradiente exacto de la entropía de Rényi-2 usando
    diferenciación automática sobre la función que calcula S₂ exactamente.
    
    Args:
        vstate: NetKet variational state
        subsystem_sites: lista de sitios del subsistema A
        
    Returns:
        S2: Entropía de Rényi-2
        grad_S2: gradiente exacto (misma estructura que vstate.parameters)
    """
    
    # Definir una función que solo depende de los parámetros
    def S2_func(params):
        # Crear un estado temporal con los parámetros dados
        if isFullSum:
            vstate_tmp = nk.vqs.FullSumState(hi_extended, vstate.model)
        else:
            vstate_tmp = nk.vqs.MCState(sampler=vstate.sampler,model=vstate.model,n_samples=vstate.n_samples,)
        vstate_tmp.parameters = params
        S2, _ = renyi2_entropy_exact(vstate_tmp, subsystem_sites)
        return S2
    
    # Usar value_and_grad de JAX para obtener S₂ y su gradiente
    S2, S2_grad = jax.value_and_grad(S2_func)(vstate.parameters)
    return S2, S2_grad

# ── Rényi-2 muestreado ────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("apply_fun", "chunk_size"))
def _renyi2_forward_jit(apply_fun, params, model_state,
                         samples1, samples2, swapped1, swapped2,
                         chunk_size=128):
    def log_psi(s):
        return apply_fun({"params": params, **model_state}, s)

    n = samples1.shape[0]
    n_chunks = n // chunk_size

    def reshape(x):
        return x.reshape(n_chunks, chunk_size, *x.shape[1:])

    s1_c, s2_c, sw1_c, sw2_c = map(reshape, (samples1, samples2, swapped1, swapped2))

    def renyi_chunk(bs1, bs2, bsw1, bsw2):
        def renyi_single(s1, s2, sw1, sw2):
            lo1 = log_psi(s1[None])[0]
            lo2 = log_psi(s2[None])[0]
            ls1 = log_psi(sw1[None])[0]
            ls2 = log_psi(sw2[None])[0]
            return jnp.real(ls1 + ls2 - lo1 - lo2)
        return jax.vmap(renyi_single)(bs1, bs2, bsw1, bsw2)

    log_R = jax.lax.map(
        lambda x: renyi_chunk(*x), (s1_c, s2_c, sw1_c, sw2_c)
    ).reshape(n)

    return -jnp.log(jnp.abs(jnp.mean(jnp.exp(log_R))))

def renyi2_entropy_sampled(vstate, subsystem_sites, n_samples, key=0, chunk_size=128, debug=False):
    subsystem_sites = jnp.array(subsystem_sites, dtype=int)

    all_samples = vstate.sample(n_samples=2 * n_samples).reshape(-1, vstate.hilbert.size)

    rng = jax.random.PRNGKey(key)
    all_samples = jax.random.permutation(rng, all_samples, axis=0)
    samples1 = all_samples[:n_samples]
    samples2 = all_samples[n_samples:]

    all_sites = jnp.arange(samples1.shape[1])
    complement_sites = jnp.setdiff1d(all_sites, subsystem_sites)
    swapped1 = jnp.concatenate([samples2[:, subsystem_sites], samples1[:, complement_sites]], axis=1)
    swapped2 = jnp.concatenate([samples1[:, subsystem_sites], samples2[:, complement_sites]], axis=1)

    S2 = _renyi2_forward_jit(
        vstate._apply_fun, vstate.parameters, vstate.model_state,
        samples1, samples2, swapped1, swapped2,
        chunk_size=chunk_size,
    )

    if debug:
        print(f"S₂ = {float(S2):.6f}")

    return S2

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
        n_samples=2 * n_samples
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


def _renyi2_lambda_integral_jit(apply_fun, params, model_state,
                                 samples1, samples2, swapped1, swapped2,
                                 subsystem_sites, complement_sites,
                                 n_lambda):

    def log_psi(p, s):
        return apply_fun({"params": p, **model_state}, s)

    lambda_grid = jnp.linspace(0.0, 1.0, n_lambda)

    # Calculados una sola vez, fuera del vmap
    log_o1 = log_psi(params, samples1)
    log_o2 = log_psi(params, samples2)
    log_s1 = log_psi(params, swapped1)
    log_s2 = log_psi(params, swapped2)
    log_R = jnp.real(log_s1 + log_s2 - log_o1 - log_o2)

    def compute_for_lambda(lam):
        log_w = lam * log_R
        log_w -= jax.nn.logsumexp(log_w)
        w = jnp.exp(log_w)
        f_lam = jnp.sum(w * log_R)

        def loss_fn(p):
            lo1 = log_psi(p, samples1)
            lo2 = log_psi(p, samples2)
            ls1 = log_psi(p, swapped1)
            ls2 = log_psi(p, swapped2)

            log_R_ = jnp.real(ls1 + ls2 - lo1 - lo2)

            log_w_ = lam * log_R_
            log_w_ -= jax.nn.logsumexp(log_w_)
            w_ = jnp.exp(log_w_)
            f_ = jnp.sum(w_ * log_R_)

            w_stopped = jax.lax.stop_gradient(w)
            lnR_centered = jax.lax.stop_gradient(log_R - f_lam)
            reinforce = 2.0 * jnp.sum(
                w_stopped * lnR_centered * jnp.real(lo1 + lo2)
            )

            return f_ + reinforce

        grad_f = jax.grad(loss_fn)(params)
        return f_lam, grad_f

    f_vals, grad_vals = jax.vmap(compute_for_lambda)(lambda_grid)

    dlam = lambda_grid[1] - lambda_grid[0]
    trap_w = jnp.ones(n_lambda).at[0].set(0.5).at[-1].set(0.5)

    S2_max = subsystem_sites.shape[0] * jnp.log(2.0)
    S2 = jnp.minimum(-dlam * jnp.sum(trap_w * f_vals), S2_max)

    grad_S2 = jax.tree_util.tree_map(
        lambda g: -dlam * jnp.sum(
            trap_w.reshape((-1,) + (1,) * (g.ndim - 1)) * g, axis=0
        ),
        grad_vals,
    )

    return S2, grad_S2

def renyi2_entropy_and_grad_lambda_integral(vstate, subsystem_sites, n_samples,
                                            n_lambda=10, key=0, debug=False):
    subsystem_sites = jnp.array(subsystem_sites, dtype=int)
    all_sites = jnp.arange(vstate.hilbert.size)
    complement_sites = jnp.setdiff1d(all_sites, subsystem_sites)

    all_samples = vstate.sample(
        n_samples=2 * n_samples
    ).reshape(-1, vstate.hilbert.size)

    rng = jax.random.PRNGKey(key)
    all_samples = jax.random.permutation(rng, all_samples, axis=0)
    samples1 = all_samples[:n_samples]
    samples2 = all_samples[n_samples:]

    swapped1 = jnp.concatenate([samples2[:, subsystem_sites], samples1[:, complement_sites]], axis=1)
    swapped2 = jnp.concatenate([samples1[:, subsystem_sites], samples2[:, complement_sites]], axis=1)

    S2, grad_S2 = _renyi2_lambda_integral_jit(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        samples1, samples2, swapped1, swapped2,
        subsystem_sites, complement_sites,
        n_lambda,
    )

    if debug:
        grad_flat, _ = jax.flatten_util.ravel_pytree(grad_S2)
        print(f"S₂    = {float(S2):.6f}")
        print(f"|∇S₂| = {jnp.linalg.norm(grad_flat):.6f}")

    return float(S2), grad_S2


@partial(jax.jit, static_argnames=("apply_fun",))
def _renyi2_loss_and_grad_cv(apply_fun, params, model_state,
                              samples1, samples2, swapped1, swapped2):
    def log_psi(p, s):
        return apply_fun({"params": p, **model_state}, s)

    # Forward pass para cantidades auxiliares
    log_o1 = log_psi(params, samples1)
    log_o2 = log_psi(params, samples2)
    log_s1 = log_psi(params, swapped1)
    log_s2 = log_psi(params, swapped2)
    R = jnp.exp(jnp.real(log_s1 + log_s2 - log_o1 - log_o2))
    R_mean = jnp.mean(R)
    S2 = -jnp.log(jnp.abs(R_mean))

    # Coeficiente óptimo b* estimado con muestras actuales
    log_p = jnp.real(log_o1 + log_o2)
    log_p_centered = log_p - jnp.mean(log_p)
    R_centered = R - R_mean
    b_star = jnp.clip(
        jnp.mean(R_centered * log_p_centered) / (jnp.mean(log_p_centered**2) + 1e-8),
        -10.0, 10.0
    )

    def loss_fn(p):
        lo1 = log_psi(p, samples1)
        lo2 = log_psi(p, samples2)
        ls1 = log_psi(p, swapped1)
        ls2 = log_psi(p, swapped2)

        R_ = jnp.exp(jnp.real(ls1 + ls2 - lo1 - lo2))
        R_mean_ = jax.lax.stop_gradient(R_mean)
        w = jax.lax.stop_gradient(R_ / R_mean_)

        # Control variate: reemplaza (w - 1) por (w - 1 - b* * log_p_centered)
        # en el término REINFORCE, dejando el término principal intacto
        b = jax.lax.stop_gradient(b_star)
        lp_c = jax.lax.stop_gradient(log_p_centered)

        return (
            -2.0 * jnp.mean(w * jnp.real(ls1 + ls2))
            + 2.0 * jnp.mean((1.0 + b * lp_c) * jnp.real(lo1 + lo2))
        )

    grad_S2 = jax.grad(loss_fn)(params)
    return S2, grad_S2


def renyi2_entropy_and_grad_cv(vstate, subsystem_sites, n_samples, key=0, debug=False, diagnostics=False):
    subsystem_sites = jnp.array(subsystem_sites, dtype=int)

    all_samples = vstate.sample(n_samples=2 * n_samples).reshape(-1, vstate.hilbert.size)

    rng = jax.random.PRNGKey(key)
    all_samples = jax.random.permutation(rng, all_samples, axis=0)
    samples1 = all_samples[:n_samples]
    samples2 = all_samples[n_samples:]

    all_sites = jnp.arange(samples1.shape[1])
    complement_sites = jnp.setdiff1d(all_sites, subsystem_sites)
    swapped1 = jnp.concatenate([samples2[:, subsystem_sites], samples1[:, complement_sites]], axis=1)
    swapped2 = jnp.concatenate([samples1[:, subsystem_sites], samples2[:, complement_sites]], axis=1)

    if diagnostics:
        log_o1 = vstate.log_value(samples1)
        log_o2 = vstate.log_value(samples2)
        log_s1 = vstate.log_value(swapped1)
        log_s2 = vstate.log_value(swapped2)
        log_R = jnp.real(log_s1 + log_s2 - log_o1 - log_o2)
        R = jnp.exp(log_R)
        R_mean = jnp.mean(R)
        log_p = jnp.real(log_o1 + log_o2)
        log_p_centered = log_p - jnp.mean(log_p)
        R_centered = R - R_mean
        log_R_centered = log_R - jnp.mean(log_R)

        corr_logp = jnp.mean(R_centered * log_p_centered) / (
            jnp.std(R_centered) * jnp.std(log_p_centered) + 1e-8
        )
        corr_logR = jnp.mean(R_centered * log_R_centered) / (
            jnp.std(R_centered) * jnp.std(log_R_centered) + 1e-8
        )

        print(f"── Diagnostics ──────────────────────────")
        print(f"  mean(R)      = {float(R_mean):.3e}")
        print(f"  std(R)       = {float(jnp.std(R)):.3e}")
        print(f"  max(R)       = {float(jnp.max(R)):.3e}")
        print(f"  mean(log_R)  = {float(jnp.mean(log_R)):.3f}")
        print(f"  std(log_R)   = {float(jnp.std(log_R)):.3f}")
        print(f"  corr(R,logp) = {float(corr_logp):.3f}  → CV reducción ≈ {100*(1-float(corr_logp)**2):.1f}% NO, {100*float(corr_logp)**2:.1f}% SÍ")
        print(f"  corr(R,logR) = {float(corr_logR):.3f}  → CV reducción ≈ {100*float(corr_logR)**2:.1f}% SÍ")
        print(f"─────────────────────────────────────────")

    S2, grad_S2 = _renyi2_loss_and_grad_cv(
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
