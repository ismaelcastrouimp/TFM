"""
bridge.py  (17-11-2026)
==============================
Estimador "puente" de S₂ = -ln Tr ρ_S² (y su gradiente) para la ancilla FÍSICA con red autoregresiva cuyo orden pone
TODO el sistema antes que la ancilla (ARNNDense, ARSpinViT_Causal, ...).  Teoría: notas.md, "Por qué el AR inclinado va
mal con ancilla física".  Con ese orden ψ(s,a) = A(s) φ(a|s) y

    P = Σ_{s1,s2} |S(s1,s2)|²,      S(s1,s2) = Σ_a ψ*(s1,a) ψ(s2,a).

Muestreo (todas las densidades con stop-gradient):
  s1 ~ p(s) (marginal AR del sistema),  s2 ~ q(s2|s1) = mezcla de κ's inclinada hacia s1 SOLO en el sistema,
  a_1..a_n ~ r(a|s1,s2): AR "puente", r(a_k|a_<k) ∝ sqrt(p(a_k|a_<k,s1) p(a_k|a_<k,s2)) (media geométrica).
  W_i = ψ*(s1,a_i) ψ(s2,a_i) / r(a_i)  =>  E[W] = S;   |S|² ≈ (|ΣW|² - Σ|W|²)/(n(n-1))  (U-estadístico, insesgado).
  P̂ = media de  f / (p(s1) q(s2|s1)).
Gradiente: ∇P = Σ ∇F con las densidades congeladas  =>  ∇P̂ = media ∇[f_θ/(p q)] − media b_m ∇ln p_θ(s1_m)
(línea base sin la propia muestra; E[∇ln p(s1)] = 0).  ∇S₂ = −∇P̂/P̂.
Coste por x1 (pasos de condicional, como en s2_noise):  N_S + K·(N_S + 1 + n_a·(2 N_A + 2)).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .entropy import _ar_order, check_ar_order, _tilted_sample, _value_index


def _sys_log_p_and_q(model, params, mstate, x2, x1, log_kappas, log_a, sys_mask, local_states):
    """ln p(s2) (marginal del sistema) y ln q_mix(s2|s1), con una pasada de la red sobre x2 (solo cuentan los sitios
    del sistema: sus condicionales no dependen de la ancilla)."""
    cond = model.apply({"params": params, **mstate}, x2, method="conditionals_log_psi")
    logp = jax.nn.log_softmax(2.0 * jnp.real(cond), axis=-1)
    i2, i1 = _value_index(x2, local_states), _value_index(x1, local_states)
    lp2 = jnp.where(sys_mask[None], jnp.take_along_axis(logp, i2[..., None], -1)[..., 0], 0.0)
    p1 = jnp.exp(jnp.take_along_axis(logp, i1[..., None], -1)[..., 0])
    nd = jnp.sum((i2 != i1) & sys_mask[None], -1)
    k = jnp.exp(log_kappas)[None, None, :]
    logc = jnp.where(sys_mask[None, :, None], jnp.log(p1[..., None] + k * (1.0 - p1[..., None])), 0.0)
    lq = lp2.sum(-1)[:, None] + nd[:, None] * log_kappas[None] - logc.sum(1)
    return lp2.sum(-1), jax.scipy.special.logsumexp(lq + log_a[None], axis=-1), nd


@partial(jax.jit, static_argnames=("model", "nc"))
def _bridge_sample(model, params, mstate, key, xa, xb, order_anc, local_states, nc=1):
    """Muestrea la ancilla del AR puente.  xa = (s1, ·), xb = (s2, ·) (B, N); devuelve (xa, xb) con la MISMA ancilla
    y ln r(a) (B,).  Dos cachés (una por contexto), como ARDirectSampler."""
    B, N = xa.shape
    nloc = local_states.shape[0]

    def chunk(args):
        k, xa, xb = args
        kinit, kscan = jax.random.split(k)
        ca = model.init(kinit, xa, 0, method=model.conditional).get("cache")
        cb = model.init(kinit, xb, 0, method=model.conditional).get("cache")

        def cond(x, c, i):
            v = {"params": params, **mstate}
            if c:
                v = {**v, "cache": c}
            pr, mut = model.apply(v, x, i, method=model.conditional, mutable=["cache"])
            lp = jnp.log(jnp.maximum(pr, 1e-300))
            return lp - jax.scipy.special.logsumexp(lp, -1, keepdims=True), mut.get("cache")

        def step(carry, inp):
            xa, xb, ca, cb, lr = carry
            ki, i = inp
            la, ca = cond(xa, ca, i)
            lb, cb = cond(xb, cb, i)
            lg = 0.5 * (la + lb)
            lg = lg - jax.scipy.special.logsumexp(lg, -1, keepdims=True)
            ch = jax.random.categorical(ki, lg, axis=-1)
            lr = lr + jnp.take_along_axis(lg, ch[:, None], -1)[:, 0]
            v = local_states[ch]
            return (xa.at[:, i].set(v), xb.at[:, i].set(v), ca, cb, lr), None

        n = order_anc.shape[0]
        (xa, xb, _, _, lr), _ = jax.lax.scan(step, (xa, xb, ca, cb, jnp.zeros(xa.shape[0])),
                                            (jax.random.split(kscan, n), order_anc))
        return xa, xb, lr

    sh = lambda x: x.reshape(nc, -1, *x.shape[1:])
    xa, xb, lr = jax.lax.map(chunk, (jax.random.split(key, nc), sh(xa), sh(xb)))
    return xa.reshape(B, N), xb.reshape(B, N), lr.reshape(B)


def _ustat(W):
    """(n_pares, n) complejo -> estimador insesgado de |E W|² por par."""
    n = W.shape[-1]
    return (jnp.abs(jnp.sum(W, -1)) ** 2 - jnp.sum(jnp.abs(W) ** 2, -1)) / (n * (n - 1))


@partial(jax.jit, static_argnames=("model", "nc", "n_a"))
def _bridge_terms(model, params, mstate, X1, X2, XA, XB, lr, log_kappas, log_a, sys_mask, local_states, n_a, nc):
    """Z_pair = f/(p(s1) q(s2|s1)) por par (B,), ln p(s1) (B,), nd."""
    def lpsi(x):
        return model.apply({"params": params, **mstate}, x).astype(jnp.complex128)

    def one(c):
        x1, x2, xa, xb, lrc = c
        lp1, _, _ = _sys_log_p_and_q(model, params, mstate, x1, x1, log_kappas, log_a, sys_mask, local_states)
        _, lq, nd = _sys_log_p_and_q(model, params, mstate, x2, x1, log_kappas, log_a, sys_mask, local_states)
        W = jnp.exp(jnp.conj(lpsi(xa)) + lpsi(xb) - lrc).reshape(-1, n_a)
        return _ustat(W) * jnp.exp(-lp1 - lq), lp1, nd

    sh = lambda x: x.reshape(nc, -1, *x.shape[1:])
    z, lp1, nd = jax.lax.map(one, (sh(X1), sh(X2), sh(XA), sh(XB), sh(lr)))
    return jax.lax.stop_gradient(z.reshape(-1)), lp1.reshape(-1), nd.reshape(-1)


@partial(jax.jit, static_argnames=("model", "nc", "n_a"))
def _bridge_grad_P(model, params, mstate, X1, X2, XA, XB, lr, lp1_sg, lq_sg, b, log_kappas, log_a, sys_mask,
                   local_states, n_a, nc):
    lp1_sg, lq_sg, b, lr = map(jax.lax.stop_gradient, (lp1_sg, lq_sg, b, lr))
    B = X1.shape[0]
    sh = lambda x: x.reshape(nc, -1, *x.shape[1:])

    def loss(p):
        def lpsi(x):
            return model.apply({"params": p, **mstate}, x).astype(jnp.complex128)

        @jax.checkpoint
        def chunk(c):
            x1, x2, xa, xb, lrc, l1, lq, bc = c
            W = jnp.exp(jnp.conj(lpsi(xa)) + lpsi(xb) - lrc).reshape(-1, n_a)
            lp1, _, _ = _sys_log_p_and_q(model, p, mstate, x1, x1, log_kappas, log_a, sys_mask, local_states)
            return jnp.sum(_ustat(W) * jnp.exp(-l1 - lq)) - jnp.sum(bc * lp1)
        return jnp.sum(jax.lax.map(chunk, (sh(X1), sh(X2), sh(XA), sh(XB), sh(lr), sh(lp1_sg), sh(lq_sg),
                                          sh(b)))) / B

    return jax.grad(loss)(params)


def renyi2_bridge(vstate, subsystem_sites, n_samples, K=5, kappas=(0.05, 0.1, 0.2, 0.4, 1.0), n_a=2, key=0,
                  compute_grad=True, chunk_size=None, return_info=False, check_order=True):
    """
    S₂ del sistema (subsystem_sites, que deben ser los PRIMEROS sitios del orden AR) con el estimador puente.
    n_samples: M configuraciones s1;  K compañeros s2 por s1;  n_a >= 2 muestras de ancilla del puente por par.
    chunk_size: nº de s1 por trozo (debe dividir a n_samples).  Devuelve S2, grad[, info].
    """
    model, params, mstate = vstate.model, vstate.parameters, vstate.model_state
    N = vstate.hilbert.size
    local_states = jnp.asarray(vstate.hilbert.local_states)
    sys = np.sort(np.asarray(subsystem_sites, int)); NS = len(sys)
    order = np.asarray(_ar_order(model, params, mstate, N))
    if check_order:
        bad = check_ar_order(model, params, mstate, local_states, N, order)
        if bad:
            raise ValueError(f"{type(model).__name__}: red no causal en su orden ({order}); violación {bad[0]}")
    if set(order[:NS].tolist()) != set(sys.tolist()):
        raise ValueError("renyi2_bridge necesita que el sistema vaya ENTERO antes que la ancilla en el orden AR "
                         f"(orden {order}, sistema {sys}); con orden intercalado ψ no factoriza como A(s)φ(a|s).")
    assert n_a >= 2, "n_a >= 2 (U-estadístico)"
    sys_mask = jnp.zeros(N, bool).at[jnp.asarray(sys)].set(True)
    kap = np.asarray(kappas, float); J = len(kap); kidx = np.arange(K) % J; n_j = np.bincount(kidx, minlength=J)
    assert np.all(n_j > 0) and np.all((kap > 0) & (kap <= 1 + 1e-12))
    log_kappas, log_a = jnp.log(jnp.asarray(kap)), jnp.log(jnp.asarray(n_j / K))
    key = jax.random.PRNGKey(key) if isinstance(key, (int, np.integer)) else key
    k1, k2, k3 = jax.random.split(key, 3)
    cs = n_samples if chunk_size is None else max(1, min(int(chunk_size), n_samples))
    assert n_samples % cs == 0
    nc = n_samples // cs
    o_sys, o_anc = jnp.asarray(order[:NS]), jnp.asarray(order[NS:])
    fill = jnp.full((n_samples, N), local_states[0], local_states.dtype)

    x1 = _tilted_sample(model, params, mstate, k1, fill, jnp.zeros(n_samples), sys_mask, o_sys, local_states, nc=nc)
    X1 = jnp.repeat(x1, K, 0)
    lk_rows = jnp.tile(log_kappas[jnp.asarray(kidx)], n_samples)
    X2 = _tilted_sample(model, params, mstate, k2, X1, lk_rows, sys_mask, o_sys, local_states, nc=nc)
    XA = jnp.repeat(X1, n_a, 0); XB = jnp.repeat(X2, n_a, 0)
    XA, XB, lr = _bridge_sample(model, params, mstate, k3, XA, XB, o_anc, local_states, nc=nc)

    z, lp1, nd = _bridge_terms(model, params, mstate, X1, X2, XA, XB, lr, log_kappas, log_a, sys_mask, local_states,
                               n_a, nc)                                      # XA/XB/lr: n_a filas consecutivas por par
    Zm = jnp.mean(z.reshape(n_samples, K), 1)
    P = jnp.mean(Zm); S2 = -jnp.log(P)
    info = dict(P=float(P), S2_err=float(jnp.std(Zm) / jnp.sqrt(n_samples) / P), cv_Z=float(jnp.std(Zm) / P),
                mean_d=float(jnp.mean(nd)))
    grad = None
    if compute_grad:
        lq = _lq_only(model, params, mstate, X1, X2, log_kappas, log_a, sys_mask, local_states, nc)
        b = jnp.repeat((jnp.sum(Zm) - Zm) / max(n_samples - 1, 1), K)
        gP = _bridge_grad_P(model, params, mstate, X1, X2, XA, XB, lr, lp1, lq, b, log_kappas, log_a, sys_mask,
                            local_states, n_a, nc)
        grad = jax.tree_util.tree_map(lambda g: -g / P, gP)
    return (S2, grad, info) if return_info else (S2, grad)


@partial(jax.jit, static_argnames=("model", "nc"))
def _lq_only(model, params, mstate, X1, X2, log_kappas, log_a, sys_mask, local_states, nc):
    sh = lambda x: x.reshape(nc, -1, *x.shape[1:])
    lq = jax.lax.map(lambda c: _sys_log_p_and_q(model, params, mstate, c[1], c[0], log_kappas, log_a, sys_mask,
                                                local_states)[1], (sh(X1), sh(X2)))
    return lq.reshape(-1)


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Fijar el gauge de la ancilla: penalización de hermiticidad (17-11-2026)
# ψ -> (I⊗U)ψ no cambia ρ_S ni F, pero sí la varianza de los estimadores de S₂ (×100-1000; notas.md).  El gauge
# canónico ψ = ρ^{1/2} (thermofield double) es hermítico como matriz ψ(s,a).  H = Re Σ_x ψ(x) ψ(σx) ∈ [-1, 1],
# σ(s,a) = (a,s), con H = 1 sii ψ hermítica (‖ψ − ψ†‖² = 2(1 − H)).
#   H = Re E_{x~|ψ|²}[ w ],  w = ψ(σx)/ψ*(x);     ∇H = 2 Re E[ w ∇ln ψ(x) ]  (σ es involución)
# Línea base: E[2 Re ∇ln ψ(x)] = ∇Σ|ψ|² = 0  ->  se resta b·2Re∇ln ψ(x) con b = H̄ sin la propia muestra.
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
def hermiticity(vstate, system_sites, ancilla_sites, n_samples, key=0, compute_grad=True, chunk_size=None):
    """H y ∇H (para minimizar F + λ(1 − H): sumar −λ∇H al gradiente).  system_sites[i] <-> ancilla_sites[i]."""
    model, params, mstate = vstate.model, vstate.parameters, vstate.model_state
    N = vstate.hilbert.size
    local_states = jnp.asarray(vstate.hilbert.local_states)
    sy, an = np.asarray(system_sites, int), np.asarray(ancilla_sites, int)
    assert len(sy) == len(an)
    perm = np.arange(N); perm[sy], perm[an] = an, sy                              # x[perm] = σx
    order = jnp.asarray(_ar_order(model, params, mstate, N))
    key = jax.random.PRNGKey(key) if isinstance(key, (int, np.integer)) else key
    cs = n_samples if chunk_size is None else max(1, min(int(chunk_size), n_samples))
    assert n_samples % cs == 0
    nc = n_samples // cs
    x = _tilted_sample(model, params, mstate, key, jnp.full((n_samples, N), local_states[0], local_states.dtype),
                       jnp.zeros(n_samples), jnp.zeros(N, bool), order, local_states, nc=nc)
    return _herm_eval(model, params, mstate, x, jnp.asarray(perm), compute_grad, nc)


@partial(jax.jit, static_argnames=("model", "compute_grad", "nc"))
def _herm_eval(model, params, mstate, x, perm, compute_grad, nc):
    sh = lambda y: y.reshape(nc, -1, *y.shape[1:])

    def lpsi(p, y):
        return model.apply({"params": p, **mstate}, y).astype(jnp.complex128)

    lw = jax.lax.map(lambda c: lpsi(params, c[:, perm]) - jnp.conj(lpsi(params, c)), sh(x)).reshape(-1)
    w = jnp.exp(lw)
    M = w.shape[0]
    H = jnp.mean(jnp.real(w))
    if not compute_grad:
        return H, None
    w = jax.lax.stop_gradient(w)
    b = (jnp.sum(jnp.real(w)) - jnp.real(w)) / max(M - 1, 1)

    def loss(p):
        @jax.checkpoint
        def chunk(c):
            xc, wc, bc = c
            l = lpsi(p, xc)
            return jnp.sum(2.0 * jnp.real(wc * l) - bc * 2.0 * jnp.real(l))
        return jnp.sum(jax.lax.map(chunk, (sh(x), sh(w), sh(b)))) / M

    return H, jax.grad(loss)(params)
