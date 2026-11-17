"""
src_renyi/mps_ancilla/swap_mc.py
================================
PROTOTIPO: gradiente de la pureza muestreando el ENSEMBLE SWAP (idea de Hastings / Drut).

Con la ancilla trazada a mano, el peso de una pareja de configuraciones del sistema es
        w(s,t) = |rho(s,t)|^2 = p(s) p(t) f(s,t),     f(s,t) = |<Phi_t|Phi_s>|^2 / (n2(s) n2(t)),
una distribucion CLASICA positiva sobre parejas, con densidad barata (hasta la normalizacion P = sum w).
Como  grad log P = sum_{s,t} pi(s,t) grad log w(s,t)  con  pi = w / P,
        grad( T log P ) = T * E_{(s,t)~pi} [ grad log w(s,t) ]
no hace falta estimar P (ni dividir por la P^ ruidosa del lote), ni terminos de score: basta derivar la
log-densidad en parejas sorteadas de pi.  Las parejas caen donde esta el peso (t cerca de s), justo lo que
a S2 alta las parejas iid casi nunca encuentran.

Muestreo de pi: cadenas de Metropolis con estado (s, t).  Movimientos (aceptacion exacta, balance detallado):
   flip_t     : invierte un espin de t            acc = min(1, w(s,t')/w(s,t))
   flip_s     : invierte un espin de s            idem
   flip_both  : invierte el MISMO sitio en s y t  idem   (mantiene la distancia)
   indep_t    : t' ~ p (muestra exacta del pool)  acc = min(1, f(s,t')/f(s,t))
   indep_s    : s' ~ p                            idem
Las cadenas arrancan en (s, s) con s ~ p (muestra exacta) y se pueden mantener entre pasos de optimizacion
(cadenas persistentes) devolviendo el estado final.

Solo depende de JAX y de models.py.  No toca mc_loss.py.
"""
import jax
import jax.numpy as jnp

from .models import mps_overlap_aligned

sg = jax.lax.stop_gradient

MOVES = ("flip_t", "flip_s", "flip_both", "indep_t", "indep_s")


def _cfg(model, params, s):
    """(log p(s), tensores MPS, ||Phi_s||^2) de un lote de configuraciones."""
    logA, M = model.apply({"params": params}, s)
    n2 = jnp.real(mps_overlap_aligned(M, M, model.chi))
    return 2.0 * jnp.real(logA), M, n2


def _logf(chi, Ma, n2a, Mb, n2b):
    """log f(a,b) = log |<Phi_b|Phi_a>|^2 / (n2a n2b)  (alineado por parejas)."""
    O = mps_overlap_aligned(Ma, Mb, chi)
    return jnp.log(jnp.maximum(jnp.abs(O) ** 2, 1e-300)) - jnp.log(n2a) - jnp.log(n2b)


def _where(mask, new, old):
    m = mask.reshape(mask.shape + (1,) * (new.ndim - mask.ndim))
    return jnp.where(m, new, old)


def make_swap_chains(model, N, n_steps, moves=MOVES):
    """run(params, key, pool_s, s0=None, t0=None) -> (s, t, acc)

    pool_s: (C, N) muestras EXACTAS de p (propuestas de independencia y arranque (s,s) si s0 es None).
            Si `moves` no contiene indep_*, pool_s no se usa salvo para el arranque.
    s0, t0: estado previo de las cadenas (persistentes); si None arrancan en (pool_s, pool_s).
    n_steps: numero total de pasos de Metropolis; se recorre `moves` en ciclo.  Los ciclos completos van en un
             lax.scan (compila una sola vez, valido para n_steps grande) y el resto se desenrolla.
    acc: tasa de aceptacion media por tipo de movimiento (dict nombre -> escalar)."""
    chi = model.chi
    L = len(moves)
    n_full, rem = divmod(n_steps, L)
    count = {m: n_full + (1 if k < rem else 0) for k, m in enumerate(moves)}
    eyeN = jnp.eye(N)

    def one_move(mv, carry, params, pool_s, pool):
        key, s, t, lp_s, Ms, n2s, lp_t, Mt, n2t, lf, acc = carry
        lp_pool, M_pool, n2_pool = pool
        key, k1, k2, k3 = jax.random.split(key, 4)
        C = s.shape[0]
        logu = jnp.log(jax.random.uniform(k1, (C,)))
        j = jax.random.randint(k2, (C,), 0, N)
        flip = 1.0 - 2.0 * eyeN[j]                                         # (C, N)
        if mv == "flip_t":
            t2 = t * flip
            lp_t2, Mt2, n2t2 = _cfg(model, params, t2)
            lf2 = _logf(chi, Ms, n2s, Mt2, n2t2)
            a = logu < ((lp_t2 + lf2) - (lp_t + lf))
            t, lp_t, Mt, n2t, lf = (_where(a, t2, t), _where(a, lp_t2, lp_t), _where(a, Mt2, Mt),
                                    _where(a, n2t2, n2t), _where(a, lf2, lf))
        elif mv == "flip_s":
            s2 = s * flip
            lp_s2, Ms2, n2s2 = _cfg(model, params, s2)
            lf2 = _logf(chi, Ms2, n2s2, Mt, n2t)
            a = logu < ((lp_s2 + lf2) - (lp_s + lf))
            s, lp_s, Ms, n2s, lf = (_where(a, s2, s), _where(a, lp_s2, lp_s), _where(a, Ms2, Ms),
                                    _where(a, n2s2, n2s), _where(a, lf2, lf))
        elif mv == "flip_both":
            s2, t2 = s * flip, t * flip
            lp_s2, Ms2, n2s2 = _cfg(model, params, s2)
            lp_t2, Mt2, n2t2 = _cfg(model, params, t2)
            lf2 = _logf(chi, Ms2, n2s2, Mt2, n2t2)
            a = logu < ((lp_s2 + lp_t2 + lf2) - (lp_s + lp_t + lf))
            s, lp_s, Ms, n2s = _where(a, s2, s), _where(a, lp_s2, lp_s), _where(a, Ms2, Ms), _where(a, n2s2, n2s)
            t, lp_t, Mt, n2t = _where(a, t2, t), _where(a, lp_t2, lp_t), _where(a, Mt2, Mt), _where(a, n2t2, n2t)
            lf = _where(a, lf2, lf)
        elif mv == "indep_t":
            idx = jax.random.randint(k3, (C,), 0, C)
            t2, lp_t2, Mt2, n2t2 = pool_s[idx], lp_pool[idx], M_pool[idx], n2_pool[idx]
            lf2 = _logf(chi, Ms, n2s, Mt2, n2t2)
            a = logu < (lf2 - lf)
            t, lp_t, Mt, n2t, lf = (_where(a, t2, t), _where(a, lp_t2, lp_t), _where(a, Mt2, Mt),
                                    _where(a, n2t2, n2t), _where(a, lf2, lf))
        elif mv == "indep_s":
            idx = jax.random.randint(k3, (C,), 0, C)
            s2, lp_s2, Ms2, n2s2 = pool_s[idx], lp_pool[idx], M_pool[idx], n2_pool[idx]
            lf2 = _logf(chi, Ms2, n2s2, Mt, n2t)
            a = logu < (lf2 - lf)
            s, lp_s, Ms, n2s, lf = (_where(a, s2, s), _where(a, lp_s2, lp_s), _where(a, Ms2, Ms),
                                    _where(a, n2s2, n2s), _where(a, lf2, lf))
        else:
            raise ValueError(mv)
        acc = {**acc, mv: acc[mv] + jnp.mean(a.astype(jnp.float64))}
        return (key, s, t, lp_s, Ms, n2s, lp_t, Mt, n2t, lf, acc)

    def run(params, key, pool_s, s0=None, t0=None):
        s = pool_s if s0 is None else s0
        t = pool_s if t0 is None else t0
        lp_s, Ms, n2s = _cfg(model, params, s)
        lp_t, Mt, n2t = (lp_s, Ms, n2s) if t0 is None and s0 is None else _cfg(model, params, t)
        lf = _logf(chi, Ms, n2s, Mt, n2t)
        pool = _cfg(model, params, pool_s) if any(m.startswith("indep") for m in moves) else (None, None, None)
        acc = {m: jnp.zeros(()) for m in moves}
        carry = (key, s, t, lp_s, Ms, n2s, lp_t, Mt, n2t, lf, acc)

        def cycle(c, _):
            for mv in moves:
                c = one_move(mv, c, params, pool_s, pool)
            return c, None
        if n_full > 0:
            carry, _ = jax.lax.scan(cycle, carry, None, length=n_full)
        for k in range(rem):
            carry = one_move(moves[k], carry, params, pool_s, pool)
        _, s, t, _, _, _, _, _, _, _, acc = carry
        return s, t, {m: acc[m] / max(count[m], 1) for m in moves}

    return run


def make_swap_loss(model, N, T):
    """L(params, s, t) = T * mean[ log p(s) + log p(t) + log f(s,t) ]  con (s,t) ~ pi fijas (sin gradiente).
    grad L = T * E_pi[grad log w] = T * grad log P = -T * grad S2."""
    chi = model.chi

    def loss(params, s, t):
        s, t = sg(s), sg(t)
        lp_s, Ms, n2s = _cfg(model, params, s)
        lp_t, Mt, n2t = _cfg(model, params, t)
        lf = _logf(chi, Ms, n2s, Mt, n2t)
        return T * jnp.mean(lp_s + lp_t + lf)
    return loss


# ── cadenas PERSISTENTES para entrenar ────────────────────────────────────────

SWAP_TRAIN_MOVES = ("flip_t", "flip_s", "flip_both")      # sin indep_*: aceptan ~0 y exigirian un pool exacto cada paso


def make_swap_stepper(model, N, T, n_chains, group, stride, warm, moves=SWAP_TRAIN_MOVES):
    """Gradiente de T*log P con cadenas persistentes.  El estado es (s, t) de forma (ng, g, N), ng*g = n_chains.

    init(params, key, sampler)  -> (s, t)   arranque (s,s) con s ~ p exacto (sampler(params,key)->(g,N)) y
                                            `warm` pasos de calentamiento.
    grad(params, key, s, t)     -> (grad, s, t, acc)   avanza `stride` pasos y devuelve el gradiente de
                                            T*log P (= -T grad S2) promediado sobre las cadenas.
    Memoria acotada: los grupos se procesan con lax.scan (un grupo en memoria cada vez)."""
    assert n_chains % group == 0, "n_chains debe ser multiplo de swap_group"
    ng = n_chains // group
    run_stride = make_swap_chains(model, N, stride, moves)
    run_warm = make_swap_chains(model, N, warm, moves)
    gfun = jax.grad(make_swap_loss(model, N, T))

    def init(params, key, sampler):
        pools = jnp.stack([sampler(params, jax.random.fold_in(key, k)) for k in range(ng)])

        @jax.jit
        def warm_fn(params, key, pools):
            def body(_, x):
                p, k = x
                s, t, _a = run_warm(params, k, p)
                return None, (s, t)
            return jax.lax.scan(body, None, (pools, jax.random.split(key, ng)))[1]
        if warm > 0:
            return warm_fn(params, jax.random.fold_in(key, 10 ** 6), pools)
        return pools, pools

    def grad(params, key, s, t):
        def body(gacc, x):
            sg_, tg_, k = x
            s2, t2, a = run_stride(params, k, sg_, sg_, tg_)
            gr = gfun(params, s2, t2)
            return jax.tree_util.tree_map(jnp.add, gacc, gr), (s2, t2, a)
        g0 = jax.tree_util.tree_map(jnp.zeros_like, params)
        gacc, (S, Tt, A) = jax.lax.scan(body, g0, (s, t, jax.random.split(key, ng)))
        return (jax.tree_util.tree_map(lambda x: x / ng, gacc), S, Tt,
                jax.tree_util.tree_map(lambda a: jnp.mean(a), A))

    return init, grad