"""
bridge_noise.py  (18-11-2026)
=============================
Estimador MC del ruido que aporta la ancilla al puente (la "J" de notas.md, apartado del gauge, 18-11) y su gradiente,
para penalizarlo durante el entrenamiento (F + μ J) y elegir así un gauge de la ancilla con poco ruido.

Con ψ(s,a) = sqrt(p(s)) φ(a|s) (sistema primero en el orden AR) y el AR puente r(a|s1,s2) ∝ sqrt(p(a|s1) p(a|s2)):
    W(a)  = φ(a|s1) φ*(a|s2) / r(a|s1,s2)                    (E_r W = O(s1,s2))
    S     = E_r |W|²                                        (segundo momento del estimador interno)
    Q     = E_{s1,s2 ~ p} [ S(s1,s2)² ],     J = Q / P − P  (P = Tr ρ²)
Estimador insesgado de S²: dos ancillas independientes a, a' del puente por par:  Q̂ = |W(a)|² |W(a')|².
Gradiente (todas las muestras dependen de θ):
    ∇S  = E_r[ ∇|W|² + |W|² ∇ln r ]               (W con r_θ en el denominador; término de puntuación del muestreo de a)
    ∇Q  = E[ sg(|W'|²) (∇|W|² + sg|W|² ∇ln r) + (a <-> a') ] + E[ (Q̂ − b) ∇(ln p(s1) + ln p(s2)) ]
con línea base b = media de Q̂ sin la propia muestra.  Se implementa con una pérdida sustituta cuyo gradiente es ese.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .entropy import _ar_order, _tilted_sample, _value_index
from .bridge import _bridge_sample


def _logp_sites(model, params, mstate, x, mask, local_states):
    """Σ_{i en mask} ln p(x_i | x_<i)  (B,)."""
    cond = model.apply({"params": params, **mstate}, x, method="conditionals_log_psi")
    logp = jax.nn.log_softmax(2.0 * jnp.real(cond), axis=-1)
    idx = _value_index(x, local_states)
    return jnp.sum(jnp.where(mask[None], jnp.take_along_axis(logp, idx[..., None], -1)[..., 0], 0.0), -1), logp, idx


def _log_r(logp_a, logp_b, idx, anc_mask):
    """ln r(a) del AR puente con las condicionales de los dos contextos (ya en la configuración muestreada)."""
    lg = 0.5 * (logp_a + logp_b)
    lg = lg - jax.scipy.special.logsumexp(lg, -1, keepdims=True)
    return jnp.sum(jnp.where(anc_mask[None], jnp.take_along_axis(lg, idx[..., None], -1)[..., 0], 0.0), -1)


@partial(jax.jit, static_argnames=("model", "nc"))
def _surrogate_grad(model, params, mstate, XA, XB, sys_mask, anc_mask, local_states, nc):
    """XA = (s1, a), XB = (s2, a) con 2 filas consecutivas (a, a') por par.  Devuelve Q̂ por par y ∇Q (sustituta)."""
    B2 = XA.shape[0]
    sh = lambda x: x.reshape(nc, -1, *x.shape[1:])

    def terms(p, xa, xb):
        lpa = model.apply({"params": p, **mstate}, xa).astype(jnp.complex128)
        lpb = model.apply({"params": p, **mstate}, xb).astype(jnp.complex128)
        ls1, la, ia = _logp_sites(model, p, mstate, xa, sys_mask, local_states)
        ls2, lb, _ = _logp_sites(model, p, mstate, xb, sys_mask, local_states)
        lr = _log_r(la, lb, ia, anc_mask)
        # ln|W|² = 2 Re(ln ψ(s1,a) + ln ψ(s2,a)) − ln p(s1) − ln p(s2) − 2 ln r
        lw2 = 2.0 * jnp.real(lpa + lpb) - ls1 - ls2 - 2.0 * lr
        return lw2.reshape(-1, 2), lr.reshape(-1, 2), ls1.reshape(-1, 2)[:, 0], ls2.reshape(-1, 2)[:, 0]

    def qhat(c):
        lw2, _, _, _ = terms(params, *c)
        return jnp.exp(jnp.sum(lw2, -1))
    Qp = jax.lax.map(qhat, (sh(XA), sh(XB))).reshape(-1)
    Qp = jax.lax.stop_gradient(Qp)
    n = Qp.shape[0]
    b = (jnp.sum(Qp) - Qp) / max(n - 1, 1)

    def loss(p):
        @jax.checkpoint
        def chunk(c):
            xa, xb, qc, bc = c
            lw2, lr, ls1, ls2 = terms(p, xa, xb)
            w2 = jnp.exp(lw2)
            w2s = jax.lax.stop_gradient(w2)
            # sg(|W'|²)(|W|² + sg|W|² ln r) + (a <-> a')
            t = w2s[:, 1] * (w2[:, 0] + w2s[:, 0] * lr[:, 0]) + w2s[:, 0] * (w2[:, 1] + w2s[:, 1] * lr[:, 1])
            return jnp.sum(t + (qc - bc) * (ls1 + ls2))
        return jnp.sum(jax.lax.map(chunk, (sh(XA), sh(XB), sh(Qp), sh(b)))) / n

    return Qp, jax.grad(loss)(params)


def bridge_noise(vstate, subsystem_sites, n_pairs, key=0, chunk_size=None, compute_grad=True):
    """Q̂ = E[S²] (media sobre n_pairs pares s1, s2 ~ p), su error y ∇Q.  J = Q/P − P con P del puente."""
    model, params, mstate = vstate.model, vstate.parameters, vstate.model_state
    N = vstate.hilbert.size
    local_states = jnp.asarray(vstate.hilbert.local_states)
    sys = np.sort(np.asarray(subsystem_sites, int)); NS = len(sys)
    order = np.asarray(_ar_order(model, params, mstate, N))
    assert set(order[:NS].tolist()) == set(sys.tolist()), "el sistema debe ir entero antes que la ancilla"
    sys_mask = jnp.zeros(N, bool).at[jnp.asarray(sys)].set(True)
    anc_mask = ~sys_mask
    key = jax.random.PRNGKey(key) if isinstance(key, (int, np.integer)) else key
    k1, k2, k3 = jax.random.split(key, 3)
    cs = n_pairs if chunk_size is None else max(1, min(int(chunk_size), n_pairs))
    assert n_pairs % cs == 0
    nc = n_pairs // cs
    o_sys, o_anc = jnp.asarray(order[:NS]), jnp.asarray(order[NS:])
    fill = jnp.full((n_pairs, N), local_states[0], local_states.dtype)
    zero = jnp.zeros(n_pairs)
    X1 = _tilted_sample(model, params, mstate, k1, fill, zero, sys_mask, o_sys, local_states, nc=nc)   # s1 ~ p
    X2 = _tilted_sample(model, params, mstate, k2, fill, zero, sys_mask, o_sys, local_states, nc=nc)   # s2 ~ p
    XA, XB = jnp.repeat(X1, 2, 0), jnp.repeat(X2, 2, 0)
    XA, XB, _ = _bridge_sample(model, params, mstate, k3, XA, XB, o_anc, local_states, nc=nc)
    Qp, g = _surrogate_grad(model, params, mstate, XA, XB, sys_mask, anc_mask, local_states, nc)
    Q = float(jnp.mean(Qp))
    return Q, float(jnp.std(Qp) / np.sqrt(n_pairs)), (g if compute_grad else None)
