from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import optax
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


def _resolve_chunk(n_samples, chunk_size):
    """
    Resuelve (n_chunks, chunk_size) de forma que n_chunks * chunk_size <= n_samples.

    Evita los fallos de `reshape` cuando n_samples no es múltiplo de
    chunk_size, y el caso degenerado chunk_size > n_samples (n_chunks = 0).
    """
    chunk_size = max(1, min(int(chunk_size), int(n_samples)))
    n_chunks = n_samples // chunk_size
    if n_chunks == 0:
        raise ValueError(
            f"n_samples={n_samples} < chunk_size={chunk_size}: "
            f"no hay chunks completos."
        )
    return n_chunks, chunk_size


# ── Rényi-2, n exacto ─────────────────────────────────────────────────────────

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

def renyin_entropy_exact(vstate, subsystem_sites, n):
    """
    Calcula S_n de forma exacta construyendo la matriz densidad reducida ρ_A.

    S_n = 1/(1-n) · ln Tr[ρ_A^n]

    Escala exponencialmente con N (memoria y tiempo O(4^{N_A})). Solo viable
    para N total pequeño (≲ 12-14), ya que hay que enumerar toda la base.

    Para n=2 reproduce renyi2_entropy_exact (salvo redondeo).

    Parámetros
    ----------
    vstate          : MCState (o FullSumState) de NetKet.
    subsystem_sites : Índices de los sitios del subsistema A.
    n               : orden de Rényi (entero o real > 0, n ≠ 1).

    Devuelve
    --------
    S_n   : Entropía de Rényi-n (escalar real).
    rho_A : Matriz densidad reducida de A (dim_A × dim_A).
    """
    assert n != 1, "n=1 (von Neumann) no aplica a esta fórmula"
    N = vstate.hilbert.size
    subsystem_sites = np.array(subsystem_sites, dtype=int)

    psi, basis = vstate_to_vector(vstate)

    n_A = len(subsystem_sites)
    n_B = N - n_A
    dim_A = 2 ** n_A
    dim_B = 2 ** n_B

    all_sites = np.arange(N)
    complement_sites = np.setdiff1d(all_sites, subsystem_sites)

    # {-1,+1} → {0,1}, reordenar (A primero, B después)
    basis_01 = ((basis + 1) // 2).astype(int)
    reordered_sites = np.concatenate([subsystem_sites, complement_sites])
    basis_reordered = basis_01[:, reordered_sites]

    powers_A = 2 ** np.arange(n_A)[::-1]
    powers_B = 2 ** np.arange(n_B)[::-1]
    idx_A = (basis_reordered[:, :n_A] * powers_A).sum(axis=1)
    idx_B = (basis_reordered[:, n_A:] * powers_B).sum(axis=1)

    psi_matrix = jnp.zeros((dim_A, dim_B), dtype=complex)
    psi_matrix = psi_matrix.at[idx_A, idx_B].set(psi)

    # ρ_A = Tr_B(|ψ⟩⟨ψ|)
    rho_A = psi_matrix @ jnp.conj(psi_matrix.T)

    # Tr[ρ_A^n] = Σ λ_i^n, con λ_i autovalores (reales, ≥ 0)
    eigvals = jnp.linalg.eigvalsh(rho_A)          # (dim_A,)
    eigvals = jnp.clip(eigvals.real, 0.0, None)   # por si hay ruido numérico

    # Trivialmente: Σ λ_i = 1, así que normalizamos por seguridad
    eigvals = eigvals / jnp.sum(eigvals)

    purity_n = jnp.sum(eigvals ** n)
    S_n = jnp.log(purity_n) / (1.0 - n)

    return S_n, rho_A

def renyin_entropy_and_grad_exact(vstate, subsystem_sites, n,
                                  hi_extended=None, isFullSum=False):
    """
    Gradiente exacto de la entropía de Rényi-n por diferenciación automática
    sobre la función que la calcula exactamente.

    Análoga a renyi2_entropy_and_grad_exact, pero para orden n arbitrario.

    Parámetros
    ----------
    vstate          : MCState de NetKet.
    subsystem_sites : Índices del subsistema A.
    n               : Orden de Rényi (n ≥ 2).
    hi_extended     : (Opcional) Hilbert extendido para FullSumState.
                      Si isFullSum=True y no se pasa, se usa vstate.hilbert.
    isFullSum       : Si True, reconstruye vstate_tmp como FullSumState
                      (necesario si vstate.sampler no está disponible).

    Devuelve
    --------
    S_n     : float
    grad_Sn : pytree con la estructura de vstate.parameters.
    """
    assert n >= 2, "n debe ser ≥ 2"

    def S_n_func(params):
        if isFullSum:
            hi = hi_extended if hi_extended is not None else vstate.hilbert
            vstate_tmp = nk.vqs.FullSumState(hi, vstate.model)
        else:
            vstate_tmp = nk.vqs.MCState(
                sampler=vstate.sampler,
                model=vstate.model,
                n_samples=vstate.n_samples,
            )
        vstate_tmp.parameters = params
        S_n, _ = renyin_entropy_exact(vstate_tmp, subsystem_sites, n)
        return S_n

    S_n, grad_Sn = jax.value_and_grad(S_n_func)(vstate.parameters)
    return S_n, grad_Sn

# ── Rényi-2 muestreado ────────────────────────────────────────────────────────

#SWAP TRICK: S₂ = -ln ⟨R⟩, R = ψ(swapped1)ψ(swapped2)/ψ(samples1)ψ(samples2)
@partial(jax.jit, static_argnames=("apply_fun", "chunk_size"))
def _renyi2_forward_jit(apply_fun, params, model_state,
                         samples1, samples2, swapped1, swapped2,
                         chunk_size=128):
    """
    S₂ por swap trick. Asume que `samples1.shape[0]` es múltiplo de
    `chunk_size`. El caller (`renyi2_entropy_sampled`) garantiza esto
    truncando el batch con `_resolve_chunk`.
    """
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

    # ── Truncar a múltiplo de chunk_size (evita fallos de reshape) ────
    n_chunks, chunk_size = _resolve_chunk(n_samples, chunk_size)
    n_used = n_chunks * chunk_size
    samples1  = samples1[:n_used]
    samples2  = samples2[:n_used]
    swapped1  = swapped1[:n_used]
    swapped2  = swapped2[:n_used]

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
    """
    S₂ y su gradiente por swap trick. Asume que `samples1.shape[0]` es
    múltiplo de `chunk_size`. El caller
    (`renyi2_entropy_and_grad_sampled`) garantiza esto truncando el batch.
    """
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

    # ── Truncar a múltiplo de chunk_size (evita fallos de reshape) ────
    n_chunks, chunk_size = _resolve_chunk(n_samples, chunk_size)
    n_used = n_chunks * chunk_size
    samples1  = samples1[:n_used]
    samples2  = samples2[:n_used]
    swapped1  = swapped1[:n_used]
    swapped2  = swapped2[:n_used]

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

def _drut_loss(apply_fun, params, model_state, s1_stack, s2_stack,
               lambda_grid, gl_weights, A, B):
    """loss tal que ∇loss = ∇S₂ estimado (con REINFORCE), integrado con Gauss-Legendre."""
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

    # ← Gauss-Legendre: pesos distintos por nodo, sin el "trap_w"
    return jnp.sum(gl_weights * contribs)

def _block_flip_step_cached(key, s1, s2, log_P_cur,
                             A, B, apply_fun, params, model_state,
                             lam, spin_min, spin_max, K):
    """
    Paso MH con propuesta simétrica: flip de K sitios elegidos al azar
    (sin reemplazo) de la concatenación (s1, s2).

    Si K=1, es equivalente al flip local original.
    """
    N = s1.shape[0]
    N_total = 2 * N
    k_sites, k_unif = jax.random.split(key)

    # K sitios distintos de {0, ..., 2N-1} (más eficiente que choice)
    sites = jax.random.permutation(k_sites, N_total)[:K]

    # Máscaras de qué sitios flipen en cada réplica
    all_mask = jnp.zeros(N_total, dtype=bool).at[sites].set(True)
    mask_s1 = all_mask[:N]
    mask_s2 = all_mask[N:]

    def flip_masked(s, mask):
        flipped = (spin_min + spin_max) - s
        return jnp.where(mask, flipped, s).astype(s.dtype)

    s1_prop = flip_masked(s1, mask_s1)
    s2_prop = flip_masked(s2, mask_s2)

    # Solo evaluamos la PROPUESTA
    lp_prop, _ = _log_target_and_lnR(apply_fun, params, model_state,
                                     s1_prop, s2_prop, A, B, lam)

    accept = jnp.log(jax.random.uniform(k_unif)) < (lp_prop - log_P_cur)

    s1_new    = jnp.where(accept, s1_prop, s1)
    s2_new    = jnp.where(accept, s2_prop, s2)
    log_P_new = jnp.where(accept, lp_prop, log_P_cur)

    return s1_new, s2_new, log_P_new, accept.astype(jnp.float32)

def _sweep_single_chain_cached(key, s1, s2, A, B,
                               apply_fun, params, model_state,
                               lam, n_props, spin_min, spin_max, K):
    log_P_cur, _ = _log_target_and_lnR(apply_fun, params, model_state,
                                       s1, s2, A, B, lam)

    keys = jax.random.split(key, n_props)

    def body(carry, k):
        s1, s2, log_P, acc_sum = carry
        s1, s2, log_P, a = _block_flip_step_cached(
            k, s1, s2, log_P, A, B, apply_fun, params, model_state,
            lam, spin_min, spin_max, K
        )
        return (s1, s2, log_P, acc_sum + a), None

    (s1, s2, _, acc_sum), _ = jax.lax.scan(
        body, (s1, s2, log_P_cur, jnp.float32(0.0)), keys
    )
    return s1, s2, acc_sum / n_props

def _sweep_single_chain_with_traj_cached(key, s1, s2, A, B,
                                          apply_fun, params, model_state,
                                          lam, n_sweeps, n_props_per_sweep,
                                          spin_min, spin_max, K):
    """
    Versión con traza + caching + block moves (K sitios por propuesta).

    log_P_cur se pasa entre bloques. K=1 recupera el comportamiento original.
    """
    log_P_cur, _ = _log_target_and_lnR(apply_fun, params, model_state,
                                       s1, s2, A, B, lam)

    keys_outer = jax.random.split(key, n_sweeps)

    def block(carry, k_outer):
        s1, s2, log_P, acc_sum = carry
        keys_inner = jax.random.split(k_outer, n_props_per_sweep)

        def single_prop(c, k):
            s1, s2, log_P, acc = c
            s1, s2, log_P, a = _block_flip_step_cached(
                k, s1, s2, log_P, A, B, apply_fun, params, model_state,
                lam, spin_min, spin_max, K
            )
            return (s1, s2, log_P, acc + a), None

        (s1_new, s2_new, log_P_new, acc_block), _ = jax.lax.scan(
            single_prop, (s1, s2, log_P, jnp.float32(0.0)), keys_inner
        )

        lnR_block = _lnR_single(apply_fun, params, model_state,
                                 s1_new, s2_new, A, B)
        acc_rate = acc_block / n_props_per_sweep

        return (s1_new, s2_new, log_P_new, acc_sum + acc_rate), (lnR_block, acc_rate)

    (s1_final, s2_final, _, acc_total), (lnR_traj, acc_traj) = jax.lax.scan(
        block, (s1, s2, log_P_cur, jnp.float32(0.0)), keys_outer
    )
    return s1_final, s2_final, acc_total / n_sweeps, lnR_traj, acc_traj

def _make_sweep_batch_cached(spin_min, spin_max, K):
    def _fn(key, s1, s2, A, B, apply_fun, params, model_state, lam, n_props):
        return _sweep_single_chain_cached(
            key, s1, s2, A, B, apply_fun, params, model_state,
            lam, n_props, spin_min, spin_max, K
        )
    return jax.vmap(
        _fn,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None),
    )

def _make_sweep_batch_with_traj_cached(spin_min, spin_max, K):
    def _fn(key, s1, s2, A, B, apply_fun, params, model_state,
            lam, n_sweeps, n_props_per_sweep):
        return _sweep_single_chain_with_traj_cached(
            key, s1, s2, A, B, apply_fun, params, model_state,
            lam, n_sweeps, n_props_per_sweep, spin_min, spin_max, K
        )
    return jax.vmap(
        _fn,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None),
    )

def _gauss_legendre_01(n):
    """
    Nodos y pesos de Gauss-Legendre en [0, 1].

    Devuelve
    --------
    nodes   : (n,) array en [0, 1]
    weights : (n,) array, suman 1
    """
    x, w = np.polynomial.legendre.leggauss(n)  # en [-1, 1]
    nodes   = 0.5 * (x + 1.0)                  # mapear a [0, 1]
    weights = 0.5 * w                          # pesos escalados
    return jnp.array(nodes), jnp.array(weights)

def _renyi2_drut_jit(apply_fun, params, model_state,
                     s1_init, s2_init, A, B,
                     n_lambda, n_sweeps_per_lam, n_props_per_sweep,
                     key, spin_min, spin_max, debug, K):
    lambda_grid, gl_weights = _gauss_legendre_01(n_lambda)

    sweep_batch_fast = _make_sweep_batch_cached(spin_min, spin_max, K)
    sweep_batch_traj = _make_sweep_batch_with_traj_cached(spin_min, spin_max, K)

    def scan_lam(carry, lam):
        s1, s2, k = carry
        k_burn, k_next = jax.random.split(k)
        keys_batch = jax.random.split(k_burn, s1.shape[0])

        if debug:
            s1, s2, acc_mean, lnR_traj, acc_traj = sweep_batch_traj(
                keys_batch, s1, s2, A, B,
                apply_fun, params, model_state,
                lam, n_sweeps_per_lam, n_props_per_sweep,
            )
            n_total = s1.shape[0] * n_sweeps_per_lam
            tau = _tau_int_per_chain(lnR_traj)
            tau_mean = jnp.nanmean(tau)
            ess_lam = n_total / tau_mean
            f_lam = jnp.mean(lnR_traj[:, -1])

            jax.debug.print(
                "λ={lam:.3f}  f(λ)={f:.4f}  accept={acc:.3f}  "
                "τ_int={tau:.2f}  ESS/M={ess:.4f}",
                lam=lam, f=jnp.mean(f_lam), acc=jnp.mean(acc_mean),
                tau=jnp.nanmean(tau), ess=jnp.mean(ess_lam) / n_total,
            )
        else:
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

    # ← Integración de Gauss-Legendre (en lugar de trapezoidal)
    S2 = -jnp.sum(gl_weights * f_vals)
    S2_max = A.shape[0] * jnp.log(2.0)
    S2 = jnp.minimum(S2, S2_max)

    return S2, s1_stack, s2_stack, lambda_grid

def renyi2_drut_sampling(vstate, subsystem_sites, n_chains=256,
                         n_sweeps_per_lam=20, n_props_per_sweep=None,
                         n_lambda=10, key=0, debug=False, K=1):

    N_sites = vstate.hilbert.size
    A = jnp.array(subsystem_sites, dtype=int)
    B = jnp.setdiff1d(jnp.arange(N_sites), A)

    if n_props_per_sweep is None:
        n_props_per_sweep = 2 * N_sites

    local_states = jnp.array(vstate.hilbert.local_states)
    spin_min = float(local_states.min())
    spin_max = float(local_states.max())
    if debug:
        print(f"[renyi2_drut] local_states = {local_states.tolist()}  "
              f"→ flip: new = ({spin_min} + {spin_max}) - cur")

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
        spin_min, spin_max, debug, K,
    )

    # ← Mismos pesos que en el kernel
    _, gl_weights = _gauss_legendre_01(n_lambda)

    grad_loss = jax.grad(lambda p: _drut_loss(
        vstate._apply_fun, p, vstate.model_state,
        s1_stack, s2_stack, lambda_grid, gl_weights, A, B,
    ))(vstate.parameters)
    grad_S2 = jax.tree_util.tree_map(lambda g: -g, grad_loss)

    if debug:
        gf, _ = jax.flatten_util.ravel_pytree(grad_S2)
        print(f"S₂    = {float(jnp.asarray(S2)):.6f}")
        print(f"|∇S₂| = {jnp.linalg.norm(gf):.6f}")

    return float(S2), grad_S2

def _sweep_single_chain_with_samples_cached(key, s1, s2, A, B,
                                              apply_fun, params, model_state,
                                              lam, n_sweeps, n_props_per_sweep,
                                              spin_min, spin_max, K,
                                              sample_every):
    """
    Sweep MCMC que devuelve MÚLTIPLES configuraciones por cadena.

    Corre n_sweeps bloques MCMC y extrae las configuraciones en los
    últimos (n_sweeps // sample_every) checkpoints, espaciados por
    sample_every sweeps.

    Devuelve
    --------
    s1_final, s2_final : (N,) estado final para warm-start del siguiente λ
    s1_samples, s2_samples : (n_checkpoints, N) muestras extraídas
    acc_mean : aceptance media
    """
    log_P_cur, _ = _log_target_and_lnR(apply_fun, params, model_state,
                                        s1, s2, A, B, lam)

    keys_outer = jax.random.split(key, n_sweeps)

    def block(carry, k_outer):
        s1, s2, log_P, acc_sum = carry
        keys_inner = jax.random.split(k_outer, n_props_per_sweep)

        def single_prop(c, k):
            s1, s2, log_P, acc = c
            s1, s2, log_P, a = _block_flip_step_cached(
                k, s1, s2, log_P, A, B, apply_fun, params, model_state,
                lam, spin_min, spin_max, K
            )
            return (s1, s2, log_P, acc + a), None

        (s1_new, s2_new, log_P_new, acc_block), _ = jax.lax.scan(
            single_prop, (s1, s2, log_P, jnp.float32(0.0)), keys_inner
        )
        acc_rate = acc_block / n_props_per_sweep
        return (s1_new, s2_new, log_P_new, acc_sum + acc_rate), (s1_new, s2_new)

    (s1_final, s2_final, _, acc_total), (s1_traj, s2_traj) = jax.lax.scan(
        block, (s1, s2, log_P_cur, jnp.float32(0.0)), keys_outer
    )

    # Extraer muestras: los últimos n_checkpoints checkpoints,
    # espaciados por sample_every sweeps
    n_checkpoints = n_sweeps // sample_every
    idx = n_sweeps - 1 - sample_every * jnp.arange(n_checkpoints)
    idx = jnp.sort(idx)

    s1_samples = s1_traj[idx]  # (n_checkpoints, N)
    s2_samples = s2_traj[idx]

    return s1_final, s2_final, s1_samples, s2_samples, acc_total / n_sweeps

def _make_sweep_batch_with_samples(spin_min, spin_max, K, sample_every):
    def _fn(key, s1, s2, A, B, apply_fun, params, model_state,
            lam, n_sweeps, n_props_per_sweep):
        return _sweep_single_chain_with_samples_cached(
            key, s1, s2, A, B, apply_fun, params, model_state,
            lam, n_sweeps, n_props_per_sweep, spin_min, spin_max, K,
            sample_every
        )
    return jax.vmap(
        _fn,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None, None),
    )

def _renyi2_drut_jit2(apply_fun, params, model_state,
                     s1_init, s2_init, A, B,
                     n_lambda, n_sweeps_per_lam, n_props_per_sweep,
                     key, spin_min, spin_max, debug, K,
                     sample_every=10):
    lambda_grid, gl_weights = _gauss_legendre_01(n_lambda)

    sweep_batch = _make_sweep_batch_with_samples(spin_min, spin_max, K, sample_every)

    def scan_lam(carry, lam):
        s1, s2, k = carry
        k_burn, k_next = jax.random.split(k)
        keys_batch = jax.random.split(k_burn, s1.shape[0])

        s1_final, s2_final, s1_smp, s2_smp, acc_mean = sweep_batch(
            keys_batch, s1, s2, A, B, apply_fun, params, model_state,
            lam, n_sweeps_per_lam, n_props_per_sweep,
        )
        # s1_smp: (n_chains, n_checkpoints, N)

        if debug:
            n_ch, n_cp = s1_smp.shape[0], s1_smp.shape[1]
            jax.debug.print(
                "λ={lam:.3f}  accept={acc:.3f}  "
                "n_chains={nc}  n_samples/chain={np_}  total={tot}",
                lam=lam, acc=jnp.mean(acc_mean),
                nc=n_ch, np_=n_cp, tot=n_ch * n_cp,
            )

        n_ch, n_cp, N_ = s1_smp.shape
        s1_flat = s1_smp.reshape(n_ch * n_cp, N_)
        s2_flat = s2_smp.reshape(n_ch * n_cp, N_)

        return (s1_final, s2_final, k_next), (s1_flat, s2_flat)

    (_, _, _), (s1_stack, s2_stack) = jax.lax.scan(
        scan_lam, (s1_init, s2_init, key), lambda_grid
    )

    lnR_stack = jax.vmap(
        lambda a, b: _lnR_batch(apply_fun, params, model_state, a, b, A, B)
    )(s1_stack, s2_stack)

    f_vals = jnp.mean(lnR_stack, axis=1)  # (n_lambda,)

    S2 = -jnp.sum(gl_weights * f_vals)
    S2_max = A.shape[0] * jnp.log(2.0)
    S2 = jnp.minimum(S2, S2_max)

    return S2, s1_stack, s2_stack, lambda_grid

def renyi2_drut_sampling2(vstate, subsystem_sites, n_chains=256,
                         n_sweeps_per_lam=20, n_props_per_sweep=None,
                         n_lambda=10, key=0, debug=False, K=1):

    N_sites = vstate.hilbert.size
    A = jnp.array(subsystem_sites, dtype=int)
    B = jnp.setdiff1d(jnp.arange(N_sites), A)

    if n_props_per_sweep is None:
        n_props_per_sweep = 2 * N_sites

    local_states = jnp.array(vstate.hilbert.local_states)
    spin_min = float(local_states.min())
    spin_max = float(local_states.max())
    if debug:
        print(f"[renyi2_drut] local_states = {local_states.tolist()}  "
              f"→ flip: new = ({spin_min} + {spin_max}) - cur")

    all_init = vstate.sample(n_samples=2 * n_chains)
    all_init = all_init.reshape(2 * n_chains, N_sites)
    all_init = jax.random.permutation(jax.random.PRNGKey(key), all_init, axis=0)
    s1_init = all_init[:n_chains]
    s2_init = all_init[n_chains:]

    S2, s1_stack, s2_stack, lambda_grid = _renyi2_drut_jit2(
        vstate._apply_fun, vstate.parameters, vstate.model_state,
        s1_init, s2_init, A, B,
        n_lambda, n_sweeps_per_lam, n_props_per_sweep,
        jax.random.PRNGKey(key + 1),
        spin_min, spin_max, debug, K
    )

    _, gl_weights = _gauss_legendre_01(n_lambda)

    grad_loss = jax.grad(lambda p: _drut_loss(
        vstate._apply_fun, p, vstate.model_state,
        s1_stack, s2_stack, lambda_grid, gl_weights, A, B,
    ))(vstate.parameters)
    grad_S2 = jax.tree_util.tree_map(lambda g: -g, grad_loss)

    if debug:
        gf, _ = jax.flatten_util.ravel_pytree(grad_S2)
        print(f"S₂    = {float(jnp.asarray(S2)):.6f}")
        print(f"|∇S₂| = {jnp.linalg.norm(gf):.6f}")

    return float(S2), grad_S2


#Gradiente sin integral
def _renyi2_drut_jit_direct(apply_fun, params, model_state,
                             s1_init, s2_init, A, B,
                             n_lambda, n_sweeps_per_lam, n_props_per_sweep,
                             key, spin_min, spin_max, debug, K):
    """
    Igual que _renyi2_drut_jit pero devuelve SOLO las muestras finales (λ≈1).
    Usado por el estimador directo del gradiente.
    """
    lambda_grid, gl_weights = _gauss_legendre_01(n_lambda)

    sweep_batch_fast = _make_sweep_batch_cached(spin_min, spin_max, K)
    sweep_batch_traj = _make_sweep_batch_with_traj_cached(spin_min, spin_max, K)

    def scan_lam(carry, lam):
        s1, s2, k = carry
        k_burn, k_next = jax.random.split(k)
        keys_batch = jax.random.split(k_burn, s1.shape[0])

        if debug:
            s1, s2, acc_mean, lnR_traj, acc_traj = sweep_batch_traj(
                keys_batch, s1, s2, A, B,
                apply_fun, params, model_state,
                lam, n_sweeps_per_lam, n_props_per_sweep,
            )
            n_total = s1.shape[0] * n_sweeps_per_lam
            tau = _tau_int_per_chain(lnR_traj)
            tau_mean = jnp.nanmean(tau)
            ess_lam = n_total / tau_mean
            f_lam = jnp.mean(lnR_traj[:, -1])

            jax.debug.print(
                "λ={lam:.3f}  f(λ)={f:.4f}  accept={acc:.3f}  "
                "τ_int={tau:.2f}  ESS/M={ess:.4f}",
                lam=lam, f=jnp.mean(f_lam), acc=jnp.mean(acc_mean),
                tau=jnp.nanmean(tau), ess=jnp.mean(ess_lam) / n_total,
            )
        else:
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
    S2 = -jnp.sum(gl_weights * f_vals)
    S2_max = A.shape[0] * jnp.log(2.0)
    S2 = jnp.minimum(S2, S2_max)

    s1_final = s1_stack[-1]
    s2_final = s2_stack[-1]

    return S2, s1_final, s2_final

def _drut_direct_loss(apply_fun, params, model_state, s1_final, s2_final, A, B):
    """
    Loss tal que ∇loss = ∇S₂ estimado directamente desde q₁.

    Identidad: ∇S₂ = -E_{q₁}[∇ log ρ]
    con ρ(s1,s2) = ψ(s1)ψ(s2)ψ(sw1)ψ(sw2) (salvo constante en Z).

    Las muestras s1_final, s2_final se tratan como CONSTANTES (stop_gradient
    implícito): JAX solo diferencia respecto a params, no respecto a ellas.
    """
    def lpsi(s):
        return jnp.real(apply_fun({"params": params, **model_state}, s))

    sw1 = jnp.concatenate([s2_final[:, A], s1_final[:, B]], axis=1)
    sw2 = jnp.concatenate([s1_final[:, A], s2_final[:, B]], axis=1)

    lp1  = jax.vmap(lpsi)(s1_final)
    lp2  = jax.vmap(lpsi)(s2_final)
    lsw1 = jax.vmap(lpsi)(sw1)
    lsw2 = jax.vmap(lpsi)(sw2)

    return jnp.mean(lp1 + lp2 + lsw1 + lsw2)

def renyi2_drut_sampling_direct(vstate, subsystem_sites, n_chains=256,
                                 n_sweeps_per_lam=20, n_props_per_sweep=None,
                                 n_lambda=10, key=0, debug=False, K=1):
    """
    Versión simplificada de renyi2_drut_sampling:

      - S₂  : integración Gauss-Legendre completa (necesita todos los λ)
      - ∇S₂ : fórmula directa desde muestras de q₁ (último λ de la rejilla),
              sin REINFORCE ni suma sobre la rejilla.

    Referencia teórica:
        ∇S₂ = -E_{q₁}[∇ log ρ],  q₁ ∝ p·p·R = ρ/Γ
    """
    N_sites = vstate.hilbert.size
    A = jnp.array(subsystem_sites, dtype=int)
    B = jnp.setdiff1d(jnp.arange(N_sites), A)

    if n_props_per_sweep is None:
        n_props_per_sweep = 2 * N_sites

    local_states = jnp.array(vstate.hilbert.local_states)
    spin_min = float(local_states.min())
    spin_max = float(local_states.max())
    if debug:
        print(f"[renyi2_drut_direct] local_states = {local_states.tolist()}")

    all_init = vstate.sample(n_samples=2 * n_chains)
    all_init = all_init.reshape(2 * n_chains, N_sites)
    all_init = jax.random.permutation(jax.random.PRNGKey(key), all_init, axis=0)
    s1_init = all_init[:n_chains]
    s2_init = all_init[n_chains:]

    S2, s1_final, s2_final = _renyi2_drut_jit_direct(
        vstate._apply_fun, vstate.parameters, vstate.model_state,
        s1_init, s2_init, A, B,
        n_lambda, n_sweeps_per_lam, n_props_per_sweep,
        jax.random.PRNGKey(key + 1),
        spin_min, spin_max, debug, K,
    )

    grad_loss = jax.grad(lambda p: _drut_direct_loss(
        vstate._apply_fun, p, vstate.model_state,
        s1_final, s2_final, A, B,
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

    ratios   = jnp.stack(ratios)
    s1_stack = jnp.stack(s1_list)
    s2_stack = jnp.stack(s2_list)

    return ratios, s1_stack, s2_stack

def _increment_loss(apply_fun, params, model_state, s1_stack, s2_stack, A_list):
    """
    Loss tal que ∇_θ loss = ∇_θ S₂ (con signo correcto).

    S₂ = -Σ_i log(r_i),  r_i = E_{q_i}[f_i],  f_i = R_{A^{i+1}}/R_{A^i}

    La pérdida se construye como suma de:

      log(mean(f_i))                     ← diferenciable en θ (vía f_i)
    + mean(stop_grad(w_i - 1) * log q_i) ← REINFORCE por la dependencia
                                            de q_i en θ (con baseline)
    """
    def lpsi(s):
        return jnp.real(apply_fun({"params": params, **model_state}, s))

    total = 0.0
    for i in range(len(s1_stack)):
        s1, s2 = s1_stack[i], s2_stack[i]
        A_i, A_ip1 = A_list[i], A_list[i + 1]

        sw1_i   = _swap_A(s1, s2, A_i)
        sw2_i   = _swap_A(s2, s1, A_i)
        sw1_ip1 = _swap_A(s1, s2, A_ip1)
        sw2_ip1 = _swap_A(s2, s1, A_ip1)

        lo1   = jax.vmap(lpsi)(s1)
        lo2   = jax.vmap(lpsi)(s2)
        lsi1  = jax.vmap(lpsi)(sw1_i)
        lsi2  = jax.vmap(lpsi)(sw2_i)
        lsip1 = jax.vmap(lpsi)(sw1_ip1)
        lsip2 = jax.vmap(lpsi)(sw2_ip1)

        log_r = (lsip1 + lsip2) - (lsi1 + lsi2)      # log f_i (per-sample)
        log_q = (lo1 + lo2) + (lsi1 + lsi2)          # log q_i (sin normalizar)

        f_i          = jnp.exp(log_r)
        r_i          = jnp.mean(f_i)
        log_r_mean   = jnp.log(r_i)

        w_sg     = jax.lax.stop_gradient(f_i / r_i)
        baseline = jax.lax.stop_gradient(jnp.ones_like(w_sg))
        reinforce = jnp.mean((w_sg - baseline) * log_q)

        total = total + log_r_mean + reinforce

    return -total

def renyi2_increment_sampling(vstate, subsystem_sites, n_chains,
                              n_sweeps_per_site=50,
                              n_props_per_sweep=None,
                              key=0, debug=False):
    """
    Estima S₂(A) con el increment trick de Hastings et al.

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

    A_list = [A_sites[:i] for i in range(n_sub + 1)]

    all_init = vstate.sample(n_samples=2 * n_chains).reshape(2 * n_chains, N_sites)
    all_init = jax.random.permutation(jax.random.PRNGKey(key), all_init, axis=0)
    s1_init = all_init[:n_chains]
    s2_init = all_init[n_chains:]

    ratios, s1_stack, s2_stack = _increment_kernel(
        vstate._apply_fun, vstate.parameters, vstate.model_state,
        s1_init, s2_init, A_list, n_props,
        jax.random.PRNGKey(key + 1),
        spin_min, spin_max, debug,
    )

    purity = jnp.prod(ratios)
    S2     = -jnp.log(purity)

    loss_fn = lambda p: _increment_loss(
        vstate._apply_fun, p, vstate.model_state,
        s1_stack, s2_stack, A_list,
    )
    grad_loss = jax.grad(loss_fn)(vstate.parameters)
    grad_S2   = grad_loss

    if debug:
        gf, _ = jax.flatten_util.ravel_pytree(grad_S2)
        print(f"  S₂ (increment) = {float(S2):.6f}")
        print(f"  Tr(ρ_A²)       = {float(purity):.6e}")
        print(f"  |∇S₂|          = {jnp.linalg.norm(gf):.6f}")

    return float(S2), grad_S2, ratios, A_list


# ══════════════════════════════════════════════════════════════════════════════
# Wang & Davis (2020) — Conditional sampling (con jit interno)
# ══════════════════════════════════════════════════════════════════════════════

def _per_site_log_psi(vstate, s, reverse=False):
    """
    [Legacy] log ψ(s_i | cond_i) por sitio, en orden ORIGINAL de sitios.

    reverse=False → cond_i = {s_j : j < i}   (forward)
    reverse=True  → cond_i = {s_j : j > i}   (reverse)

    El flip es EXPLÍCITO porque NetKet no aplica reorder/inverse_reorder
    dentro de conditionals_log_psi.

    Esta versión se mantiene por compatibilidad. Para código caliente usa
    `_per_site_log_psi_jit`.
    """
    s_in = jnp.flip(s, axis=-1) if reverse else s
    cond = vstate.model.apply(
        {"params": vstate.parameters, **vstate.model_state},
        s_in, method="conditionals_log_psi",
    )
    if reverse:
        cond = jnp.flip(cond, axis=-2)
    return cond


@partial(jax.jit, static_argnames=("model", "reverse"))
def _per_site_log_psi_jit(model, params, model_state, s, reverse):
    """
    Versión JIT de `_per_site_log_psi`.

    Parámetros
    ----------
    model       : flax Module (estático).
                  Flax implementa __hash__/__eq__ ESTRUCTURALES sobre los
                  campos del dataclass, así que dos modelos con los mismos
                  campos (hilbert, layers, features, activation, ...) SÍ
                  comparten entrada de caché. Requiere que todos los campos
                  sean hashables (lo son para los ARNN estándar de NetKet).
    params      : pytree de parámetros (trazado).
    model_state : pytree (típicamente {}).
    s           : (B, N) batch de configuraciones (trazado).
    reverse     : bool (estático) → dos versiones compiladas.
    """
    s_in = jnp.flip(s, axis=-1) if reverse else s
    cond = model.apply(
        {"params": params, **model_state}, s_in,
        method="conditionals_log_psi",
    )
    if reverse:
        cond = jnp.flip(cond, axis=-2)
    return cond


def _value_index(s_sel, local_states):
    """Índice en local_states correspondiente a s_sel (robusto al orden)."""
    diff = jnp.abs(local_states[None, None, :] - s_sel[..., None])
    return jnp.argmin(diff, axis=-1).astype(jnp.int32)

def _log_cond_sum(cond, s_sel, site_indices, local_states):
    """Σ_i log p(s_i | cond) sobre los sitios dados (machine_pow=2)."""
    idx = _value_index(s_sel, local_states)
    cond_sel = cond[:, site_indices, :]
    log_psi_sel = jnp.take_along_axis(
        cond_sel, idx[..., None], axis=-1
    ).squeeze(-1)
    return 2.0 * log_psi_sel.sum(axis=-1)


def _sample_ar_sites(key, vstate, s_init, sites, reverse=False):
    """[Legacy] Muestrea los sitios `sites` de una sola configuración."""
    local_states = jnp.asarray(vstate.hilbert.local_states)
    sites_arr = jnp.asarray(sites, dtype=int)
    keys = jax.random.split(key, sites_arr.shape[0])

    def step(carry, inputs):
        s, _ = carry
        k, i = inputs
        cond_i = _per_site_log_psi(vstate, s[None], reverse=reverse)[0, i]
        log_p_i = 2.0 * cond_i                           # machine_pow = 2
        choice = jax.random.categorical(k, log_p_i)
        return (s.at[i].set(local_states[choice]), None), None

    (s_final, _), _ = jax.lax.scan(step, (s_init, None), (keys, sites_arr))
    return s_final


def _sample_ar_sites_batch(key, vstate, s_init_batch, sites, reverse=False):
    """[Legacy] vmap de `_sample_ar_sites` sobre un batch."""
    keys = jax.random.split(key, s_init_batch.shape[0])
    return jax.vmap(lambda k, s: _sample_ar_sites(k, vstate, s, sites, reverse))(
        keys, s_init_batch
    )


@partial(jax.jit, static_argnames=("model", "reverse"))
def _sample_ar_sites_batch_jit(key, model, params, model_state,
                                s_init_batch, sites, reverse, local_states):
    """
    Versión JIT de `_sample_ar_sites_batch`.

    Sustituye `vstate` por (model, params, model_state, local_states) para
    poder compilar la cadena autoregresiva completa.

    - `model` es estático ⇒ Flax lo hashea ESTRUCTURALMENTE (todos los
      campos del dataclass deben ser hashables; lo son para ARNN estándar).
    - `reverse` es estático ⇒ dos versiones compiladas.
    - `sites` y `local_states` son arrays ⇒ se trazan, pero son pequeños
      y su forma entra en la cache key (3 shapes típicamente por llamada:
      A_fwd, B_fwd, A_rev).
    """
    keys = jax.random.split(key, s_init_batch.shape[0])

    def single(k, s):
        keys_sites = jax.random.split(k, sites.shape[0])

        def step(carry, inputs):
            s, _ = carry
            k_i, i = inputs
            cond_i = _per_site_log_psi_jit(
                model, params, model_state, s[None], reverse
            )[0, i]
            choice = jax.random.categorical(k_i, 2.0 * jnp.real(cond_i))
            return (s.at[i].set(local_states[choice]), None), None

        (s_final, _), _ = jax.lax.scan(
            step, (s, None), (keys_sites, sites)
        )
        return s_final

    return jax.vmap(single)(keys, s_init_batch)


def _wang_sample_quadruples(key, vstate, vstate_R, n_samples, N_S, N_total):
    """
    Sampling con el truco del Apéndice A.2 de Wang & Davis (2020):

        σ_a¹ ~ p(σ_a)                             [forward]
        σ_b¹ ~ p(σ_b | σ_a¹)                       [forward]
        σ_a² ~ p_R(σ_a | σ_b¹)                     [reverse]
        σ_b² ~ ½[p(σ_b | σ_a²) + p(σ_b | σ_a¹)]   [forward, mixture]

    Usa `_sample_ar_sites_batch_jit` (compilado) internamente.
    """
    model_v  = vstate.model
    params_v = vstate.parameters
    mstate_v = vstate.model_state
    model_R  = vstate_R.model
    params_R = vstate_R.parameters
    mstate_R = vstate_R.model_state

    local_states = jnp.asarray(vstate.hilbert.local_states)
    blank = jnp.full((n_samples, N_total), local_states[0],
                     dtype=local_states.dtype)

    A_fwd = jnp.arange(N_S)
    B_fwd = jnp.arange(N_S, N_total)
    A_rev = jnp.arange(N_S - 1, -1, -1)

    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # σ_a¹ ~ p(σ_a)  — forward
    sa1_full = _sample_ar_sites_batch_jit(
        k1, model_v, params_v, mstate_v, blank, A_fwd, False, local_states
    )
    sa1 = sa1_full[:, :N_S]

    # σ_b¹ ~ p(σ_b | σ_a¹)  — forward
    s_init = blank.at[:, :N_S].set(sa1)
    sb1_full = _sample_ar_sites_batch_jit(
        k2, model_v, params_v, mstate_v, s_init, B_fwd, False, local_states
    )
    sb1 = sb1_full[:, N_S:]

    # σ_a² ~ p_R(σ_a | σ_b¹)  — reverse
    s_init = blank.at[:, N_S:].set(sb1)
    sa2_full = _sample_ar_sites_batch_jit(
        k3, model_R, params_R, mstate_R, s_init, A_rev, True, local_states
    )
    sa2 = sa2_full[:, :N_S]

    # σ_b² ~ mezcla: moneda per-muestra ANTES de muestrear, una sola pasada
    coin = jax.random.bernoulli(k5, 0.5, shape=(n_samples,))
    cond_A = jnp.where(coin[:, None], sa2, sa1)
    s_init = blank.at[:, :N_S].set(cond_A)
    sb2_full = _sample_ar_sites_batch_jit(
        k4, model_v, params_v, mstate_v, s_init, B_fwd, False, local_states
    )
    sb2 = sb2_full[:, N_S:]

    return sa1, sb1, sa2, sb2


def renyi2_wang_cs(vstate, vstate_R, subsystem_sites, n_samples,
                   key=0, debug=False, chunk_size=None):
    """
    S₂ = -ln Tr[ρ_A²] y su gradiente vía Wang & Davis (2020),
    con el truco del Apéndice A.2 (mixture en σ_b²).

    Implementación
    --------------
    Delega en `renyin_wang_cs(..., n=2)`, que:
      - usa `_wang_sample_chain` con `_sample_ar_sites_batch_jit` (compilado),
      - evalúa logP con `_per_site_log_psi_jit` (compilado),
      - calcula Tr y gradiente con `_trace_value_and_grad` (jiteada, cacheada
        por `apply_fun` estático),
      - trunca el batch a múltiplo de chunk_size (no falla si no es divisible).

    Firma histórica preservada para compatibilidad. El prefijo del debug
    cambia de "[wang]" a "[renyi_n Wang]", pero los valores son idénticos.
    """
    cs = int(chunk_size) if chunk_size is not None else int(n_samples)
    return renyin_wang_cs(
        vstate, vstate_R, subsystem_sites,
        n=2, n_samples=int(n_samples),
        key=key, chunk_size=cs, debug=debug,
    )


def train_reverse_network(vstate, vstate_R, n_steps=500, batch=1024,
                          lr=1e-3, key=0, verbose=True, freq=50,
                          return_history=False, refresh_every=1):
    """
    Entrena vstate_R (ARNNDense estándar) para modelar p en orden invertido:
        vstate_R(flip(s))  ≈  vstate(s)

    Optimizado con jax.jit: todo el paso (loss + grad + update) se compila
    una sola vez. El sampling se refresca cada `refresh_every` pasos para
    amortizar su coste.

    Parámetros
    ----------
    refresh_every : int
        Cada cuántos pasos se re-muestrea un nuevo batch. 1 = comportamiento
        original (sample cada paso). 10–50 suele dar la misma calidad con
        mucho menos coste, porque el objetivo (la distribución p) es
        estacionario durante el entrenamiento de la red reversa.

    Devuelve
    --------
    vstate_R : MCState con parámetros entrenados.
    dkl_history : list[float] (solo si return_history=True)
        Valor de D_KL en cada paso del entrenamiento, longitud n_steps.
    """
    opt = optax.adam(lr)
    opt_state = opt.init(vstate_R.parameters)

    p_params = vstate.parameters
    p_state  = vstate.model_state
    apply_p  = vstate._apply_fun
    apply_R  = vstate_R._apply_fun

    def loss_fn(params_R, state_R, s):
        log_p  = 2.0 * jnp.real(apply_p({"params": p_params, **p_state}, s))
        log_pR = 2.0 * jnp.real(apply_R(
            {"params": params_R, **state_R},
            jnp.flip(s, axis=-1),
        ))
        return jnp.mean(log_p - log_pR)

    @jax.jit
    def train_step(params_R, state_R, opt_state, s):
        loss_val, grads = jax.value_and_grad(loss_fn, argnums=0)(
            params_R, state_R, s
        )
        updates, opt_state = opt.update(grads, opt_state, params_R)
        params_R = optax.apply_updates(params_R, updates)
        return params_R, opt_state, loss_val

    dkl_history = []
    s = None
    for step in range(n_steps):
        if s is None or step % refresh_every == 0:
            s = vstate.sample(n_samples=batch).reshape(
                -1, vstate.hilbert.size
            )
        vstate_R.parameters, opt_state, loss_val = train_step(
            vstate_R.parameters, vstate_R.model_state, opt_state, s
        )
        dkl_history.append(float(loss_val))
        if verbose and step % freq == 0:
            print(f"[train N_R] step {step:4d}  D_KL ≈ {float(loss_val):.4f}")

    if return_history:
        return vstate_R, dkl_history
    return vstate_R


# ══════════════════════════════════════════════════════════════════════════════
# Rényi de orden n >= 2
# ══════════════════════════════════════════════════════════════════════════════
#
# S_n = 1/(1-n) · ln Tr[ρ_A^n]
#
# Tr[ρ_A^n] = Σ Ω(σa, σb),  Ω = Π_{k=1}^n ψ(σa^k, σb^k) ψ*(σa^{k+1}, σb^k)
#                              con σa^{n+1} ≡ σa^1
#
# Dos estimadores:
#   - renyin_entropy_and_grad_sampled : n muestras iid de p por batch,
#     recombinadas cíclicamente (direct sampling, DS).
#   - renyin_wang_cs : cadena σa¹→σb¹→σa²→…→σbⁿ con la red inversa para
#     los pasos σb→σa, y truco A.2 (mezcla) en σbⁿ (conditional sampling).
#
# Gradiente: Tr[ρ_A^n] = Σ Ω sobre configuraciones FIJAS ⇒
#     ∇Tr = Σ ∇Ω = E_P[ (Ω/P) ∇ln Ω ]   (P tratado como constante)
# Por eso la pérdida es  mean( Re exp( ln Ω − stop_gradient(ln P) ) ).
# No hay término REINFORCE: el estimador "pathwise" es insesgado porque P
# es una distribución de muestreo, no un objeto sobre el que diferenciamos.
# Finalmente, ∇S_n = ∇Tr / ((1 − n) Tr).

def _to_chunks(x, n_chunks, chunk):
    """(n, M, ...) -> (n_chunks, n, chunk, ...) descartando el resto de M."""
    n = x.shape[0]
    x = x[:, : n_chunks * chunk].reshape(n, n_chunks, chunk, *x.shape[2:])
    return jnp.swapaxes(x, 0, 1)

def _chunk_sizes(n_samples, chunk_size):
    chunk = max(1, min(chunk_size, n_samples))
    return n_samples // chunk, chunk

@partial(jax.jit, static_argnames=("apply_fun",))
def _trace_value_and_grad(apply_fun, params, model_state, c_diag, c_off, log_P):
    """
    Tr[ρ_A^n] ≈ mean( Re exp(ln Ω − sg(ln P)) )  y  su gradiente.

    c_diag : (n_chunks, n, chunk, N)   configs (σa^k, σb^k)
    c_off  : (n_chunks, n, chunk, N)   configs (σa^{k+1}, σb^k)
    log_P  : (n_chunks, chunk)         ln P_sampling (tratado como constante)

    El gradiente del estimador es  E_P[(Ω/P) ∇ln Ω] = ∇Tr, sin REINFORCE.
    """
    def lpsi(p, s):
        flat = s.reshape(-1, s.shape[-1])
        out = apply_fun({"params": p, **model_state}, flat)
        return jnp.squeeze(out).reshape(s.shape[:-1])

    n_chunks, _, chunk, _ = c_diag.shape
    M = n_chunks * chunk

    def loss_fn(p):
        @jax.checkpoint
        def one_chunk(x):
            cd, co, lP = x
            log_Omega = jnp.sum(lpsi(p, cd) + jnp.conj(lpsi(p, co)), axis=0)
            return jnp.sum(jnp.real(jnp.exp(log_Omega - lP)))

        return jnp.sum(jax.lax.map(one_chunk, (c_diag, c_off, log_P))) / M

    return jax.value_and_grad(loss_fn)(params)

def _finalize(T, grad_T, n):
    S_n = jnp.log(jnp.abs(T)) / (1.0 - n)
    grad_S = jax.tree_util.tree_map(lambda g: g / ((1.0 - n) * T), grad_T)
    return S_n, grad_S

def _log_omega_det(vstate, c_diag, c_off):
    """ln Ω (complejo) por secuencia, sin diferenciar. c_*: (n, M, N)."""
    n, M, N = c_diag.shape
    ld = vstate.log_value(c_diag.reshape(-1, N)).reshape(n, M)
    lo = vstate.log_value(c_off.reshape(-1, N)).reshape(n, M)
    return jnp.sum(ld + jnp.conj(lo), axis=0)

def _debug_report(name, T, S_n, grad_S, w):
    w = jnp.real(w)
    ess = float(jnp.sum(w) ** 2 / (jnp.sum(w ** 2) + 1e-30))
    gf, _ = jax.flatten_util.ravel_pytree(grad_S)
    print(f"[{name}] Tr[ρ_A^n] = {float(T):.6e}")
    print(f"[{name}] S_n       = {float(S_n):.6f}")
    print(f"[{name}] ESS/M     = {ess / w.shape[0]:.4f}  ({ess:.0f}/{w.shape[0]})")
    print(f"[{name}] |∇S_n|    = {float(jnp.linalg.norm(gf)):.6f}")

# ── Rényi-n por muestreo directo (DS) ────────────────────────────────────────

def renyin_entropy_and_grad_sampled(vstate, subsystem_sites, n, n_samples,
                                    key=0, chunk_size=256, debug=False):
    """
    S_n y ∇S_n por muestreo directo.

    Cada batch son n muestras iid de p = |ψ|², recombinadas cíclicamente:
    (σa^{k+1}, σb^k). Estimador f_DS = Ω / Π_k p(σ^k).

    Parámetros
    ----------
    n         : orden de Rényi (entero >= 2).
                n=2 da el mismo estimador (no bit-a-bit) que
                renyi2_entropy_and_grad_sampled, con menos varianza.
    n_samples : número de batches M (se muestrean n·M configuraciones).

    Devuelve
    --------
    S_n (float), grad_S_n (pytree con la estructura de vstate.parameters).
    """
    assert n >= 2, "n debe ser un entero >= 2"
    N = vstate.hilbert.size
    mask_A = jnp.zeros(N, dtype=bool).at[
        jnp.asarray(subsystem_sites, dtype=int)
    ].set(True)

    total = n * n_samples
    all_samples = vstate.sample(n_samples=total).reshape(-1, N)
    all_samples = jax.random.permutation(
        jax.random.PRNGKey(key), all_samples, axis=0
    )[:total]
    s = all_samples.reshape(n, n_samples, N)

    s_off = jnp.where(mask_A, jnp.roll(s, -1, axis=0), s)

    lp = vstate.log_value(s.reshape(-1, N)).reshape(n, n_samples)
    log_P = 2.0 * jnp.sum(jnp.real(lp), axis=0)

    n_chunks, chunk = _chunk_sizes(n_samples, chunk_size)
    T, grad_T = _trace_value_and_grad(
        vstate._apply_fun, vstate.parameters, vstate.model_state,
        _to_chunks(s, n_chunks, chunk),
        _to_chunks(s_off, n_chunks, chunk),
        log_P[: n_chunks * chunk].reshape(n_chunks, chunk),
    )
    S_n, grad_S = _finalize(T, grad_T, n)

    if debug:
        w = jnp.exp(_log_omega_det(vstate, s, s_off) - log_P)
        _debug_report("renyi_n DS", T, S_n, grad_S, w)

    return float(S_n), grad_S

# ── Rényi-n por conditional sampling (Wang & Davis 2020) ─────────────────────

def _wang_sample_chain(key, vstate, vstate_R, n_samples, n, N_S, N_total):
    """
    Cadena de Wang & Davis generalizada:

        σa¹ ~ p(σa)
        para k = 1..n-1:
            σb^k     ~ p(σb | σa^k)              [forward]
            σa^{k+1} ~ p_R(σa | σb^k)            [reverse]
        σbⁿ ~ ½ [ p(σb | σaⁿ) + p(σb | σa¹) ]   (truco A.2)

    Devuelve sa (n, M, N_S) con sa[k] = σa^{k+1}, sb (n, M, N_B) análogo.

    Usa `_sample_ar_sites_batch_jit` (compilado) internamente.
    """
    model_v  = vstate.model
    params_v = vstate.parameters
    mstate_v = vstate.model_state
    model_R  = vstate_R.model
    params_R = vstate_R.parameters
    mstate_R = vstate_R.model_state

    local_states = jnp.asarray(vstate.hilbert.local_states)
    blank = jnp.full((n_samples, N_total), local_states[0],
                     dtype=local_states.dtype)

    A_fwd = jnp.arange(N_S)
    B_fwd = jnp.arange(N_S, N_total)
    A_rev = jnp.arange(N_S - 1, -1, -1)

    keys = jax.random.split(key, 2 * n + 2)
    sa, sb = [], []

    sa.append(_sample_ar_sites_batch_jit(
        keys[0], model_v, params_v, mstate_v, blank, A_fwd, False, local_states
    )[:, :N_S])

    for k in range(n):
        if k < n - 1:
            cond_A = sa[k]
        else:
            coin = jax.random.bernoulli(keys[2 * n + 1], 0.5,
                                        shape=(n_samples,))
            cond_A = jnp.where(coin[:, None], sa[0], sa[n - 1])

        s_init = blank.at[:, :N_S].set(cond_A)
        sb.append(_sample_ar_sites_batch_jit(
            keys[1 + 2 * k], model_v, params_v, mstate_v,
            s_init, B_fwd, False, local_states
        )[:, N_S:])

        if k < n - 1:
            s_init = blank.at[:, N_S:].set(sb[k])
            sa.append(_sample_ar_sites_batch_jit(
                keys[2 + 2 * k], model_R, params_R, mstate_R,
                s_init, A_rev, True, local_states
            )[:, :N_S])

    return jnp.stack(sa), jnp.stack(sb)

def _wang_sample_chain_chunked(key, vstate, vstate_R, n_samples, n,
                               N_S, N_total, chunk_size):
    """
    Versión troceada de `_wang_sample_chain`.

    Usa `jax.lax.map` (no `for` Python) para que el HLO contenga un
    while_loop: solo un chunk de activaciones vive en VRAM a la vez,
    incluso cuando esta función se traza dentro de un jit superior
    (p.ej. `vstate.expect_and_grad(op)`).

    El troceo es SOLO en batch — la cadena en k es secuencial.

    Devuelve:
        sa: (n, M, N_S)
        sb: (n, M, N_B)
    """
    n_chunks, chunk_size = _chunk_sizes(n_samples, chunk_size)
    keys = jax.random.split(key, n_chunks)

    def _one(k):
        return _wang_sample_chain(k, vstate, vstate_R,
                                  chunk_size, n, N_S, N_total)

    sa_all, sb_all = jax.lax.map(_one, keys)
    # sa_all: (n_chunks, n, chunk_size, N_S)
    # sb_all: (n_chunks, n, chunk_size, N_B)

    N_B = N_total - N_S
    sa = jnp.transpose(sa_all, (1, 0, 2, 3)).reshape(
        n, n_chunks * chunk_size, N_S)
    sb = jnp.transpose(sb_all, (1, 0, 2, 3)).reshape(
        n, n_chunks * chunk_size, N_B)
    return sa, sb


def renyin_wang_cs(vstate, vstate_R, subsystem_sites, n, n_samples,
                   key=0, chunk_size=256, debug=False):
    """
    S_n y ∇S_n vía conditional sampling (Wang & Davis 2020), orden n >= 2,
    con mezcla en σbⁿ (Apéndice A.2).

    n=2 reproduce `renyi2_wang_cs` (que delega aquí).

    TROCEO
    ------
    `chunk_size` controla TODOS los sub-cálculos que escalan con M:
      - el sampling AR (`_wang_sample_chain_chunked`),
      - la evaluación de logP (`_per_site_log_psi_jit` por chunk),
      - el Tr[ρ_A^n] y su gradiente (`_trace_value_and_grad`, ya troceado).
    Si `n_samples` no es múltiplo de `chunk_size`, se descartan las últimas
    `n_samples % chunk_size` muestras. El troceo es siempre sobre el batch
    M — las cadenas σa¹→σb¹→σa²→…→σbⁿ son secuenciales en k y no se
    trocean.
    """
    assert n >= 2, "n debe ser un entero >= 2"
    N_total = vstate.hilbert.size
    N_S = len(subsystem_sites)
    assert list(subsystem_sites) == list(range(N_S)), (
        "Wang CS requiere subsystem_sites = [0, 1, ..., N_S-1]"
    )

    # ── Resolver chunks una sola vez (sampling + logP + Tr usan lo mismo)
    n_chunks, chunk_size = _chunk_sizes(n_samples, chunk_size)
    M = n_chunks * chunk_size

    # ── Primitivas del modelo, extraídas una sola vez
    model_v  = vstate.model
    params_v = vstate.parameters
    mstate_v = vstate.model_state
    model_R  = vstate_R.model
    params_R = vstate_R.parameters
    mstate_R = vstate_R.model_state

    local_states = jnp.asarray(vstate.hilbert.local_states)
    A_sites = jnp.arange(N_S)
    B_sites = jnp.arange(N_S, N_total)

    # ── 1. Sampling troceado
    sa, sb = _wang_sample_chain_chunked(
        jax.random.PRNGKey(key), vstate, vstate_R,
        M, n, N_S, N_total, chunk_size,
    )
    sa = jax.lax.stop_gradient(sa)              # (n, M, N_S)
    sb = jax.lax.stop_gradient(sb)              # (n, M, N_B)
    sa_next = jnp.roll(sa, -1, axis=0)          # (σa^{k+1}), con σa^{n+1}=σa¹

    c_diag = jnp.concatenate([sa, sb], axis=-1)     # (n, M, N_total)
    c_off  = jnp.concatenate([sa_next, sb], axis=-1)

    # ── 2. logP troceado con lax.map ─────────────────────────────────
    # Reorganizar a (n_chunks, n, chunk, ...) para que lax.map itere
    # por chunk. Una sola iteración viva en HLO ⇒ sin desenrollado.
    _cd_c = _to_chunks(c_diag,  n_chunks, chunk_size)   # (n_chunks, n, chunk, N_total)
    _co_c = _to_chunks(c_off,   n_chunks, chunk_size)
    _sa_c = _to_chunks(sa,      n_chunks, chunk_size)   # (n_chunks, n, chunk, N_S)
    _sb_c = _to_chunks(sb,      n_chunks, chunk_size)   # (n_chunks, n, chunk, N_B)
    _sn_c = _to_chunks(sa_next, n_chunks, chunk_size)

    def _logP_one_chunk(args):
        cd, co, sa_, sb_, sn_ = args                    # (n, chunk, N_*)
        flat = lambda x: x.reshape(n * chunk_size, *x.shape[2:])

        # Forward en c_diag → p(σb^k|σa^k) y p(σa^k|prefix)
        cond_fwd_diag = _per_site_log_psi_jit(
            model_v, params_v, mstate_v, flat(cd), False)
        lpA_diag_c = _log_cond_sum(
            cond_fwd_diag, flat(sa_), A_sites, local_states
        ).reshape(n, chunk_size)
        lpB_diag_c = _log_cond_sum(
            cond_fwd_diag, flat(sb_), B_sites, local_states
        ).reshape(n, chunk_size)

        # Reverse para k=1..n-1: p_R(σa^{k+1}|σb^k)
        co_rev = co[:-1].reshape((n - 1) * chunk_size, N_total)
        cond_rev = _per_site_log_psi_jit(
            model_R, params_R, mstate_R, co_rev, True)
        lpR_A_c = _log_cond_sum(
            cond_rev,
            sn_[:-1].reshape((n - 1) * chunk_size, N_S),
            A_sites, local_states,
        ).reshape(n - 1, chunk_size)

        # Mezcla en σbⁿ
        cond_last = _per_site_log_psi_jit(
            model_v, params_v, mstate_v, co[-1], False)
        lpB_last_a1_c = _log_cond_sum(
            cond_last, sb_[-1], B_sites, local_states)
        log_p_b_mix_c = (jnp.logaddexp(lpB_diag_c[-1], lpB_last_a1_c)
                         - jnp.log(2.0))

        return (lpA_diag_c[0]
                + lpB_diag_c[:-1].sum(axis=0)
                + lpR_A_c.sum(axis=0)
                + log_p_b_mix_c)

    log_P = jax.lax.stop_gradient(
        jax.lax.map(_logP_one_chunk, (_cd_c, _co_c, _sa_c, _sb_c, _sn_c))
    )   # (n_chunks, chunk_size)

    # ── 3. Tr[ρ_A^n] y gradiente (ya troceado internamente)
    T, grad_T = _trace_value_and_grad(
        vstate._apply_fun, vstate.parameters, vstate.model_state,
        _to_chunks(c_diag, n_chunks, chunk_size),
        _to_chunks(c_off,  n_chunks, chunk_size),
        log_P,
    )
    S_n, grad_S = _finalize(T, grad_T, n)

    if debug:
        w = jnp.exp(_log_omega_det(vstate, c_diag, c_off) - log_P.reshape(-1))
        _debug_report("renyi_n Wang", T, S_n, grad_S, w)

    return float(S_n), grad_S