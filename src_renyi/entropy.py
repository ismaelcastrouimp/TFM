from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from netket.jax import jacobian


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

#SWAP TRICK: S₂ = -ln ⟨R⟩, R = ψ(swapped1)ψ(swapped2)/ψ(samples1)ψ(samples2)
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

@partial(jax.jit, static_argnames=("apply_fun", "chunk_size"))
def _renyi2_loss_and_grad(apply_fun, params, model_state,
                          samples1, samples2, swapped1, swapped2,
                          chunk_size=256):
    def log_psi(p, s):
        return apply_fun({"params": p, **model_state}, s)

    n = samples1.shape[0]
    n_chunks = n // chunk_size

    def reshape(x):
        return x.reshape(n_chunks, chunk_size, *x.shape[1:])

    s1_c, s2_c, sw1_c, sw2_c = map(reshape, (samples1, samples2, swapped1, swapped2))

    # ── forward: log_R sin backprop ───────────────────────────────────────────
    def forward_chunk(bs1, bs2, bsw1, bsw2):
        def single(s1, s2, sw1, sw2):
            lo1 = log_psi(params, s1[None])[0]
            lo2 = log_psi(params, s2[None])[0]
            ls1 = log_psi(params, sw1[None])[0]
            ls2 = log_psi(params, sw2[None])[0]
            return jnp.real(ls1 + ls2 - lo1 - lo2)
        return jax.vmap(single)(bs1, bs2, bsw1, bsw2)

    log_R = jax.lax.map(
        lambda x: forward_chunk(*x), (s1_c, s2_c, sw1_c, sw2_c)
    ).reshape(n)
    log_R = jax.lax.stop_gradient(log_R)

    R_mean = jax.lax.stop_gradient(jnp.mean(jnp.exp(log_R)))
    S2 = -jnp.log(jnp.abs(R_mean))

    # ── backward: gradiente chunkeado ─────────────────────────────────────────
    def loss_fn(p):
        log_R_c = log_R.reshape(n_chunks, chunk_size)

        @jax.checkpoint
        def grad_chunk(bs1, bs2, bsw1, bsw2, blog_R):
            def single(s1, s2, sw1, sw2, log_Ri):
                lo1 = log_psi(p, s1[None])[0]
                lo2 = log_psi(p, s2[None])[0]
                ls1 = log_psi(p, sw1[None])[0]
                ls2 = log_psi(p, sw2[None])[0]
                w_i = jax.lax.stop_gradient(jnp.exp(log_Ri) / R_mean)
                return (
                    -2.0 * w_i * jnp.real(ls1 + ls2)
                    + 2.0 * jnp.real(lo1 + lo2)
                )
            return jax.vmap(single)(bs1, bs2, bsw1, bsw2, blog_R)

        contributions = jax.lax.map(
            lambda x: grad_chunk(*x),
            (s1_c, s2_c, sw1_c, sw2_c, log_R_c)
        )
        return jnp.mean(contributions)

    grad_S2 = jax.grad(loss_fn)(params)
    return S2, grad_S2

def renyi2_entropy_and_grad_sampled(vstate, subsystem_sites, n_samples, key=0, debug=False, chunk_size=256):
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
        vstate._apply_fun, vstate.parameters, vstate.model_state,
        samples1, samples2, swapped1, swapped2,
        chunk_size=chunk_size,
    )

    if debug:
        grad_flat, _ = jax.flatten_util.ravel_pytree(grad_S2)
        print(f"S₂    = {float(S2):.6f}")
        print(f"|∇S₂| = {jnp.linalg.norm(grad_flat):.6f}")

    return S2, grad_S2

#Jacobian for gradient computation
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

#Lamda integral method, S₂ = -∫₀¹ dλ ⟨ln R⟩_λ, with ⟨·⟩_λ = ⟨· R^λ⟩ / ⟨R^λ⟩
#Reweighting: ⟨ln R⟩_λ = ⟨w ln R⟩, w = R^λ / ⟨R^λ⟩
def _renyi2_lambda_integral_jit(apply_fun, params, model_state,
                                 samples1, samples2, swapped1, swapped2,
                                 subsystem_sites, complement_sites,
                                 n_lambda):
    def log_psi(p, s):
        return apply_fun({"params": p, **model_state}, s)

    lambda_grid = jnp.linspace(0.0, 1.0, n_lambda)

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

        # ── ESS(λ): pesos SIN normalizar, Kish formula ───────────────
        w_unnorm = jnp.exp(lam * log_R - jnp.max(lam * log_R))  # estabilidad
        ess_lam = (jnp.sum(w_unnorm))**2 / jnp.sum(w_unnorm**2)

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
        return f_lam, grad_f, ess_lam

    f_vals, grad_vals, ess_vals = jax.vmap(compute_for_lambda)(lambda_grid)

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
    return S2, grad_S2, ess_vals, lambda_grid

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

    S2, grad_S2, ess_vals, lambda_grid = _renyi2_lambda_integral_jit(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        samples1, samples2, swapped1, swapped2,
        subsystem_sites, complement_sites,
        n_lambda,
    )

    if debug:
        #grad_flat, _ = jax.flatten_util.ravel_pytree(grad_S2)
        #print(f"S₂    = {float(S2):.6f}")
        #print(f"|∇S₂| = {jnp.linalg.norm(grad_flat):.6f}")
        M = n_samples
        print("── ESS de TI (Kish) ──")
        for lam, ess in zip(lambda_grid, ess_vals):
            print(f"  λ={float(lam):.3f}  ESS/M={float(ess)/M:.4f}  ({int(ess)} de {M})")
        grad_flat, _ = jax.flatten_util.ravel_pytree(grad_S2)
        print(f"S₂    = {float(S2):.6f}")
        print(f"|∇S₂| = {jnp.linalg.norm(grad_flat):.6f}")

    return float(S2), grad_S2

#Metropolis-Hastings sampling for lambda integral method
def _log_target_and_lnR(apply_fun, params, model_state, s1, s2, A, B, lam):
    """log P̃(λ) y ln R, ambos salvo constante independiente de λ."""
    sw1 = jnp.concatenate([s2[A], s1[B]])
    sw2 = jnp.concatenate([s1[A], s2[B]])

    def lpsi(s):
        return jnp.real(apply_fun({"params": params, **model_state}, s[None])[0])

    lp1  = lpsi(s1)
    lp2  = lpsi(s2)
    lsw1 = lpsi(sw1)
    lsw2 = lpsi(sw2)

    lnR    = (lsw1 + lsw2) - (lp1 + lp2)
    log_P  = (2.0 - lam) * (lp1 + lp2) + lam * (lsw1 + lsw2)
    return log_P, lnR

def _lnR_single(apply_fun, params, model_state, s1, s2, A, B):
    """ln R(s1, s2) para un par de configuraciones (sin batch)."""
    sw1 = jnp.concatenate([s2[A], s1[B]])
    sw2 = jnp.concatenate([s1[A], s2[B]])

    def lpsi(s):
        return jnp.real(apply_fun({"params": params, **model_state}, s[None])[0])

    return (lpsi(sw1) + lpsi(sw2)) - (lpsi(s1) + lpsi(s2))

def _metropolis_step(key, s1, s2, A, B, apply_fun, params, model_state,
                     lam, spin_min, spin_max):
    N = s1.shape[0]
    k_site, k_unif = jax.random.split(key)

    j     = jax.random.randint(k_site, (), 0, 2 * N)
    is_s1 = j < N
    site  = jnp.where(is_s1, j, j - N)

    cur_val = jnp.where(is_s1, s1[site], s2[site])
    new_val = ((spin_min + spin_max) - cur_val).astype(s1.dtype)

    s1_prop = jax.lax.cond(is_s1, lambda s: s.at[site].set(new_val), lambda s: s, s1)
    s2_prop = jax.lax.cond(is_s1, lambda s: s, lambda s: s.at[site].set(new_val), s2)

    lp_cur,  _ = _log_target_and_lnR(apply_fun, params, model_state,
                                     s1, s2, A, B, lam)
    lp_prop, _ = _log_target_and_lnR(apply_fun, params, model_state,
                                     s1_prop, s2_prop, A, B, lam)

    accept = jnp.log(jax.random.uniform(k_unif)) < (lp_prop - lp_cur)

    s1_new = jnp.where(accept, s1_prop, s1)
    s2_new = jnp.where(accept, s2_prop, s2)
    return s1_new, s2_new, accept.astype(jnp.float32)

def _sweep_single_chain(key, s1, s2, A, B, apply_fun, params, model_state,
                        lam, n_props, spin_min, spin_max):
    """Versión rápida: n_props propuestas en un solo scan."""
    keys = jax.random.split(key, n_props)

    def body(carry, k):
        s1, s2, acc_sum = carry
        s1, s2, a = _metropolis_step(k, s1, s2, A, B, apply_fun, params,
                                     model_state, lam, spin_min, spin_max)
        return (s1, s2, acc_sum + a), None

    (s1, s2, acc_sum), _ = jax.lax.scan(
        body, (s1, s2, jnp.float32(0.0)), keys
    )
    return s1, s2, acc_sum / n_props

def _make_sweep_batch(spin_min, spin_max):
    def _fn(key, s1, s2, A, B, apply_fun, params, model_state, lam, n_props):
        return _sweep_single_chain(key, s1, s2, A, B, apply_fun, params,
                                   model_state, lam, n_props,
                                   spin_min, spin_max)
    return jax.vmap(
        _fn,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None),
    )

def _sweep_single_chain_with_traj(key, s1, s2, A, B,
                                   apply_fun, params, model_state,
                                   lam, n_sweeps, n_props_per_sweep,
                                   spin_min, spin_max):
    """
    Igual que _sweep_single_chain, pero divide el sweep total en
    n_sweeps bloques de n_props_per_sweep propuestas cada uno, y
    registra:
        - ln R tras cada bloque  (n_sweeps,)
        - acceptance rate por bloque (n_sweeps,)
    """
    keys_outer = jax.random.split(key, n_sweeps)

    def block(carry, k_outer):
        s1, s2, acc_sum = carry
        keys_inner = jax.random.split(k_outer, n_props_per_sweep)

        def single_prop(c, k):
            s1, s2, acc = c
            s1, s2, a = _metropolis_step(k, s1, s2, A, B, apply_fun, params,
                                         model_state, lam, spin_min, spin_max)
            return (s1, s2, acc + a), None

        (s1_new, s2_new, acc_block), _ = jax.lax.scan(
            single_prop, (s1, s2, jnp.float32(0.0)), keys_inner
        )

        lnR_block = _lnR_single(apply_fun, params, model_state,
                                 s1_new, s2_new, A, B)
        acc_rate = acc_block / n_props_per_sweep

        return (s1_new, s2_new, acc_sum + acc_rate), (lnR_block, acc_rate)

    (s1_final, s2_final, acc_total), (lnR_traj, acc_traj) = jax.lax.scan(
        block, (s1, s2, jnp.float32(0.0)), keys_outer
    )
    return s1_final, s2_final, acc_total / n_sweeps, lnR_traj, acc_traj

def _make_sweep_batch_with_traj(spin_min, spin_max):
    def _fn(key, s1, s2, A, B, apply_fun, params, model_state,
            lam, n_sweeps, n_props_per_sweep):
        return _sweep_single_chain_with_traj(
            key, s1, s2, A, B, apply_fun, params, model_state,
            lam, n_sweeps, n_props_per_sweep, spin_min, spin_max
        )
    return jax.vmap(
        _fn,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None),
    )

def _tau_int_per_chain(lnR_traj):
    n_chains, n = lnR_traj.shape
    x = lnR_traj - jnp.mean(lnR_traj, axis=1, keepdims=True)

    # Varianza por cadena
    var = jnp.mean(x**2, axis=1)                # (n_chains,)
    good = var > 1e-10                          # cadenas con varianza no nula

    # Autocovarianza vía FFT
    fft = jnp.fft.rfft(x, n=2 * n, axis=1)
    acov = jnp.fft.irfft(fft * jnp.conj(fft), n=2 * n, axis=1)[:, :n]

    # Normalizar solo donde var>0; usar 'where' para evitar 0/0
    acov_norm = jnp.where(
        var[:, None] > 1e-10,
        acov / jnp.where(acov[:, :1] > 1e-30, acov[:, :1], 1.0),
        1.0,                                     # para cadenas "malas", ponemos rho_0=1
    )

    T = max(1, n // 4)
    tau = 1.0 + 2.0 * jnp.sum(acov_norm[:, 1:T + 1], axis=1)
    tau = jnp.maximum(tau, 1.0)

    # Solo promediar sobre cadenas buenas
    return jnp.where(good, tau, jnp.nan)

def _lnR_batch(apply_fun, params, model_state, s1, s2, A, B):
    sw1 = jnp.concatenate([s2[:, A], s1[:, B]], axis=1)
    sw2 = jnp.concatenate([s1[:, A], s2[:, B]], axis=1)

    def lpsi(s):
        return jnp.real(apply_fun({"params": params, **model_state}, s))

    return (jax.vmap(lpsi)(sw1) + jax.vmap(lpsi)(sw2)) \
         - (jax.vmap(lpsi)(s1) + jax.vmap(lpsi)(s2))

def _renyi2_drut_jit(apply_fun, params, model_state,
                     s1_init, s2_init, A, B,
                     n_lambda, n_sweeps_per_lam, n_props_per_sweep,
                     key, spin_min, spin_max, debug):

    lambda_grid = jnp.linspace(0.0, 1.0, n_lambda)

    sweep_batch_fast = _make_sweep_batch(spin_min, spin_max)
    sweep_batch_traj = _make_sweep_batch_with_traj(spin_min, spin_max)

    def scan_lam(carry, lam):
        s1, s2, k = carry
        k_burn, k_next = jax.random.split(k)
        keys_batch = jax.random.split(k_burn, s1.shape[0])

        if debug:
            # ── Ruta con diagnóstico: registra lnR por sweep, calcula τ_int ─
            s1, s2, acc_mean, lnR_traj, acc_traj = sweep_batch_traj(
                keys_batch, s1, s2, A, B,
                apply_fun, params, model_state,
                lam, n_sweeps_per_lam, n_props_per_sweep,
            )
            #tau = _tau_int_per_chain(lnR_traj)          # (n_chains,)
            n_total = s1.shape[0] * n_sweeps_per_lam
            tau = _tau_int_per_chain(lnR_traj)
            tau_mean = jnp.nanmean(tau)                      # ignora NaN
            ess_lam = n_total / tau_mean
            #ess_lam = n_total / jnp.mean(tau)            # ESS efectivo
            f_lam = jnp.mean(lnR_traj[:, -1])            # lnR del estado final

            jax.debug.print(
                "λ={lam:.3f}  f(λ)={f:.4f}  accept={acc:.3f}  "
                "τ_int={tau:.2f}  ESS/M={ess:.4f}",
                lam=lam,
                f=jnp.mean(f_lam),                    # ← media sobre cadenas
                acc=jnp.mean(acc_mean),               # ← media sobre cadenas
                tau=jnp.mean(tau),
                ess=jnp.mean(ess_lam) / n_total,
            )
        else:
            # ── Ruta rápida sin diagnóstico ────────────────────────────────
            n_props = n_sweeps_per_lam * n_props_per_sweep
            s1, s2, _ = sweep_batch_fast(
                keys_batch, s1, s2, A, B,
                apply_fun, params, model_state, lam, n_props,
            )

        return (s1, s2, k_next), (s1, s2)

    (_, _, _), (s1_stack, s2_stack) = jax.lax.scan(
        scan_lam, (s1_init, s2_init, key), lambda_grid
    )

    lnR_stack = jax.vmap(
        lambda a, b: _lnR_batch(apply_fun, params, model_state, a, b, A, B)
    )(s1_stack, s2_stack)
    f_vals = jnp.mean(lnR_stack, axis=1)

    dlam   = lambda_grid[1] - lambda_grid[0]
    trap_w = jnp.ones(n_lambda).at[0].set(0.5).at[-1].set(0.5)
    S2     = -dlam * jnp.sum(trap_w * f_vals)

    return S2, s1_stack, s2_stack, lambda_grid

def _drut_loss(apply_fun, params, model_state, s1_stack, s2_stack,
               lambda_grid, A, B):
    """loss tal que ∇loss = ∇S₂ estimado (con REINFORCE)."""
    def lpsi(s):
        return jnp.real(apply_fun({"params": params, **model_state}, s))

    def per_lambda(s1_b, s2_b, lam):
        sw1 = jnp.concatenate([s2_b[:, A], s1_b[:, B]], axis=1)
        sw2 = jnp.concatenate([s1_b[:, A], s2_b[:, B]], axis=1)

        lp1  = jax.vmap(lpsi)(s1_b)
        lp2  = jax.vmap(lpsi)(s2_b)
        lsw1 = jax.vmap(lpsi)(sw1)
        lsw2 = jax.vmap(lpsi)(sw2)

        lnR   = (lsw1 + lsw2) - (lp1 + lp2)
        log_P = (2.0 - lam) * (lp1 + lp2) + lam * (lsw1 + lsw2)

        f_lam = jnp.mean(lnR)
        lnR_c = jax.lax.stop_gradient(lnR - f_lam)
        return f_lam + jnp.mean(lnR_c * log_P)

    contribs = jax.vmap(per_lambda)(s1_stack, s2_stack, lambda_grid)
    dlam   = lambda_grid[1] - lambda_grid[0]
    trap_w = jnp.ones_like(lambda_grid).at[0].set(0.5).at[-1].set(0.5)
    return dlam * jnp.sum(trap_w * contribs)

def renyi2_drut_sampling(vstate, subsystem_sites, n_chains,
                         n_sweeps_per_lam=50, n_props_per_sweep=None,
                         n_lambda=10, key=0, debug=False):

    N_sites = vstate.hilbert.size
    A = jnp.array(subsystem_sites, dtype=int)
    B = jnp.setdiff1d(jnp.arange(N_sites), A)

    if n_props_per_sweep is None:
        n_props_per_sweep = 2 * N_sites

    local_states = jnp.array(vstate.hilbert.local_states)
    spin_min = float(local_states.min())
    spin_max = float(local_states.max())
    if debug:
        flip_const = spin_min + spin_max
        print(f"[renyi2_drut] local_states = {local_states.tolist()}  "
              f"→ flip: new = ({spin_min} + {spin_max}) - cur = "
              f"{flip_const:.1f} - cur")

    all_init = vstate.sample(n_samples=2 * n_chains)
    all_init = all_init.reshape(2 * n_chains, N_sites)
    all_init = jax.random.permutation(jax.random.PRNGKey(key), all_init, axis=0)
    s1_init = all_init[:n_chains]
    s2_init = all_init[n_chains:]

    S2, s1_stack, s2_stack, lambda_grid = _renyi2_drut_jit(
        vstate._apply_fun, vstate.parameters, vstate.model_state,
        s1_init, s2_init, A, B,
        n_lambda, n_sweeps_per_lam, n_props_per_sweep,
        jax.random.PRNGKey(key + 1),
        spin_min, spin_max, debug,
    )

    grad_loss = jax.grad(lambda p: _drut_loss(
        vstate._apply_fun, p, vstate.model_state,
        s1_stack, s2_stack, lambda_grid, A, B,
    ))(vstate.parameters)
    grad_S2 = jax.tree_util.tree_map(lambda g: -g, grad_loss)

    if debug:
        gf, _ = jax.flatten_util.ravel_pytree(grad_S2)
        print(f"S₂    = {float(jnp.asarray(S2)):.6f}")
        print(f"|∇S₂| = {jnp.linalg.norm(gf):.6f}")

    return float(S2), grad_S2


#Increment trick, Metropolis-Hastings sampling for log(R_{A^{i+1}} / R_{A^i})
def _swap_A(s, s_other, A_sites):
    """
    Devuelve una copia de s en la que los sitios indicados por A_sites
    se han sustituido por los correspondientes de s_other, manteniendo el
    orden original de los sitios.

    Si A_sites está vacío, devuelve s sin tocar.
    """
    N = s.shape[-1]
    mask = jnp.zeros(N, dtype=bool).at[A_sites].set(True)
    return jnp.where(mask, s_other, s)

def _log_q_i_scalar(apply_fun, params, model_state, s1, s2, A_i):
    """log q_i para un único par (s1, s2). Sin batch."""
    def lpsi(s):
        return jnp.real(apply_fun({"params": params, **model_state}, s[None])[0])

    if A_i.shape[0] == 0:
        return 2.0 * lpsi(s1) + 2.0 * lpsi(s2)

    sw1 = _swap_A(s1, s2, A_i)
    sw2 = _swap_A(s2, s1, A_i)
    return lpsi(s1) + lpsi(s2) + lpsi(sw1) + lpsi(sw2)

def _metropolis_step_q(key, s1, s2, A_i, apply_fun, params, model_state,
                       spin_min, spin_max):
    """Un paso Metropolis: elige un sitio al azar de s1 o s2, lo flipea."""
    N = s1.shape[0]
    k_site, k_unif = jax.random.split(key)

    j     = jax.random.randint(k_site, (), 0, 2 * N)
    is_s1 = j < N
    site  = jnp.where(is_s1, j, j - N)

    cur_val = jnp.where(is_s1, s1[site], s2[site])
    new_val = ((spin_min + spin_max) - cur_val).astype(s1.dtype)

    s1_prop = jax.lax.cond(is_s1, lambda s: s.at[site].set(new_val), lambda s: s, s1)
    s2_prop = jax.lax.cond(is_s1, lambda s: s, lambda s: s.at[site].set(new_val), s2)

    lp_cur  = _log_q_i_scalar(apply_fun, params, model_state, s1,      s2,      A_i)
    lp_prop = _log_q_i_scalar(apply_fun, params, model_state, s1_prop, s2_prop, A_i)

    accept = jnp.log(jax.random.uniform(k_unif)) < (lp_prop - lp_cur)
    s1_new = jnp.where(accept, s1_prop, s1)
    s2_new = jnp.where(accept, s2_prop, s2)
    return s1_new, s2_new, accept.astype(jnp.float32)

def _sweep_q(key, s1, s2, A_i, apply_fun, params, model_state, n_props,
             spin_min, spin_max):
    keys = jax.random.split(key, n_props)

    def body(carry, k):
        s1, s2, acc = carry
        s1, s2, a = _metropolis_step_q(k, s1, s2, A_i, apply_fun, params,
                                       model_state, spin_min, spin_max)
        return (s1, s2, acc + a), None

    (s1, s2, acc), _ = jax.lax.scan(body, (s1, s2, jnp.float32(0.0)), keys)
    return s1, s2, acc / n_props

def _make_sweep_batch_increment(spin_min, spin_max):
    """vmap de _sweep_q sobre las cadenas."""
    def _fn(key, s1, s2, A_i, apply_fun, params, model_state, n_props):
        return _sweep_q(key, s1, s2, A_i, apply_fun, params, model_state,
                        n_props, spin_min, spin_max)
    return jax.vmap(
        _fn,
        in_axes=(0, 0, 0, None, None, None, None, None),
    )

def _log_ratio_single(apply_fun, params, model_state, s1, s2, A_i, A_ip1):
    """
    log(R_{A^{i+1}} / R_{A^i}) = ℓ(s1^(i+1)) + ℓ(s2^(i+1))
                                - ℓ(s1^(i))   - ℓ(s2^(i)).
    """
    def lpsi(s):
        return jnp.real(apply_fun({"params": params, **model_state}, s[None])[0])

    sw1_i   = _swap_A(s1, s2, A_i)
    sw2_i   = _swap_A(s2, s1, A_i)
    sw1_ip1 = _swap_A(s1, s2, A_ip1)
    sw2_ip1 = _swap_A(s2, s1, A_ip1)

    return (lpsi(sw1_ip1) + lpsi(sw2_ip1)) - (lpsi(sw1_i) + lpsi(sw2_i))

def _increment_kernel(apply_fun, params, model_state,
                      s1_init, s2_init, A_list, n_props,
                      key, spin_min, spin_max, debug):
    """
    A_list: lista de arrays [A^0, A^1, ..., A^n], con A^{i+1} = A^i + 1 sitio.
    Devuelve: ratios ⟨R_{A^{i+1}}/R_{A^i}⟩_{q_i}, y los estados finales.

    Nota: el bucle sobre regiones es un `for` Python (no scan), porque
    las A^i tienen shapes distintas y jax.lax.scan exige formas iguales.
    """
    sweep_batch = _make_sweep_batch_increment(spin_min, spin_max)

    s1, s2 = s1_init, s2_init
    k      = key

    ratios   = []
    s1_list  = []
    s2_list  = []

    for i in range(len(A_list) - 1):
        A_i, A_ip1 = A_list[i], A_list[i + 1]

        k_sweep, k = jax.random.split(k)
        keys = jax.random.split(k_sweep, s1.shape[0])

        s1, s2, acc = sweep_batch(keys, s1, s2, A_i,
                                  apply_fun, params, model_state, n_props)

        log_r = jax.vmap(lambda a, b: _log_ratio_single(
            apply_fun, params, model_state, a, b, A_i, A_ip1,
        ))(s1, s2)
        ratio_i = jnp.mean(jnp.exp(log_r))

        if debug:
            jax.debug.print(
                "i: |A^i|={size:2d}  accept={acc:.3f}  ratio={r:.6f}  "
                "-log ratio={lr:.6f}",
                size=A_i.shape[0],
                acc=jnp.mean(acc),
                r=ratio_i,
                lr=-jnp.log(ratio_i),
            )

        ratios.append(ratio_i)
        s1_list.append(s1)
        s2_list.append(s2)

    ratios   = jnp.stack(ratios)     # (n_sub,)
    s1_stack = jnp.stack(s1_list)    # (n_sub, n_chains, N)
    s2_stack = jnp.stack(s2_list)

    return ratios, s1_stack, s2_stack

def _increment_loss(apply_fun, params, model_state, s1_stack, s2_stack, A_list):
    """
    Loss tal que ∇_θ loss = ∇_θ S₂ estimado, con corrección REINFORCE.

    En cada región A^i:
        ratio_i = E_{q_i}[ r_i ],   r_i = R_{A^{i+1}}/R_{A^i}
        ∇_θ ratio_i = E_{q_i}[ r_i · ∇_θ( log r_i + log q_i ) ]
    (porque ∇_θ E_q[f] = E_q[f · ∇_θ log q] + E_q[∇_θ f]
     y ∇_θ log q = ∇_θ log q, mientras que ∇_θ f = f ∇_θ log f).

    La loss es -log(∏ ratio_i), y su gradiente es ∇S₂.
    """
    def lpsi(s):
        return jnp.real(apply_fun({"params": params, **model_state}, s))

    total = 0.0
    for i in range(len(s1_stack)):
        s1, s2 = s1_stack[i], s2_stack[i]
        A_i, A_ip1 = A_list[i], A_list[i + 1]

        # swaps para el peso q_i
        sw1_i = _swap_A(s1, s2, A_i)
        sw2_i = _swap_A(s2, s1, A_i)
        # swaps para el ratio r_i
        sw1_ip1 = _swap_A(s1, s2, A_ip1)
        sw2_ip1 = _swap_A(s2, s1, A_ip1)

        lo1   = jax.vmap(lpsi)(s1)
        lo2   = jax.vmap(lpsi)(s2)
        lsi1  = jax.vmap(lpsi)(sw1_i)
        lsi2  = jax.vmap(lpsi)(sw2_i)
        lsip1 = jax.vmap(lpsi)(sw1_ip1)
        lsip2 = jax.vmap(lpsi)(sw2_ip1)

        log_r = (lsip1 + lsip2) - (lsi1 + lsi2)     # log r_i
        log_q = (lo1 + lo2) + (lsi1 + lsi2)         # log q_i  (salvo cte)

        # stop-gradient del ratio medio, para construir REINFORCE
        ratio_sg = jax.lax.stop_gradient(jnp.mean(jnp.exp(log_r)))
        log_r_c  = jax.lax.stop_gradient(log_r - jnp.log(ratio_sg))
        exp_r_sg = jax.lax.stop_gradient(jnp.exp(log_r))

        # término REINFORCE: E_q[ r · (log r - <log r>) + r · log q ]
        # (el log r - <log r> es stop-grad, y log q es lo diferenciable)
        total = total + ratio_sg + jnp.mean(exp_r_sg * (log_r_c + log_q))

    return -jnp.log(total)

def renyi2_increment_sampling(vstate, subsystem_sites, n_chains,
                              n_sweeps_per_site=50,
                              n_props_per_sweep=None,
                              key=0, debug=False):
    """
    Estima S₂(A) con el increment trick de Hastings et al.

    Parámetros
    ----------
    vstate              : MCState de NetKet.
    subsystem_sites     : lista/array con los sitios del subsistema A.
    n_chains            : número de cadenas Metropolis paralelas.
    n_sweeps_per_site   : propuestas Metropolis totales por sitio nuevo
                          (= n_props_per_sweep × n_sweeps_per_site).
    n_props_per_sweep   : si None, 2*N.
    key                 : semilla.
    debug               : imprime diagnóstico por sitio.

    Devuelve
    --------
    S2       : float, entropía de Rényi-2 de A.
    grad_S2  : gradiente respecto a vstate.parameters (misma estructura).
    ratios   : array (n_sub,), los ⟨r_i⟩_{q_i}.
    A_list   : lista de arrays [A^0, ..., A^n].
    """
    N_sites = vstate.hilbert.size
    A_sites = jnp.array(subsystem_sites, dtype=int)
    n_sub   = len(subsystem_sites)

    if n_props_per_sweep is None:
        n_props_per_sweep = 2 * N_sites

    n_props = n_sweeps_per_site * n_props_per_sweep

    local_states = jnp.array(vstate.hilbert.local_states)
    spin_min = float(local_states.min())
    spin_max = float(local_states.max())

    # Regiones anidadas: A^i = primeros i sitios del subsistema
    A_list = [A_sites[:i] for i in range(n_sub + 1)]

    # Estado inicial: muestras del vstate (que es q_0 = p⊗p)
    all_init = vstate.sample(n_samples=2 * n_chains).reshape(2 * n_chains, N_sites)
    all_init = jax.random.permutation(jax.random.PRNGKey(key), all_init, axis=0)
    s1_init = all_init[:n_chains]
    s2_init = all_init[n_chains:]

    # ── Núcleo: annealing + ratios ────────────────────────────────────────
    ratios, s1_stack, s2_stack = _increment_kernel(
        vstate._apply_fun, vstate.parameters, vstate.model_state,
        s1_init, s2_init, A_list, n_props,
        jax.random.PRNGKey(key + 1),
        spin_min, spin_max, debug,
    )

    purity = jnp.prod(ratios)
    S2     = -jnp.log(purity)

    # ── Gradiente vía REINFORCE sobre la loss ────────────────────────────
    loss_fn = lambda p: _increment_loss(
        vstate._apply_fun, p, vstate.model_state,
        s1_stack, s2_stack, A_list,
    )
    grad_loss = jax.grad(loss_fn)(vstate.parameters)
    grad_S2   = jax.tree_util.tree_map(lambda g: -g, grad_loss)

    if debug:
        gf, _ = jax.flatten_util.ravel_pytree(grad_S2)
        print(f"  S₂ (increment) = {float(S2):.6f}")
        print(f"  Tr(ρ_A²)       = {float(purity):.6e}")
        print(f"  |∇S₂|          = {jnp.linalg.norm(gf):.6f}")

    return float(S2), grad_S2, ratios, A_list