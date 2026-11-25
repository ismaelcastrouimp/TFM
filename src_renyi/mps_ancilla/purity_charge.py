"""
src_renyi/mps_ancilla/purity_charge.py
======================================
Propuesta de parejas en el sector de Sz con "carga transportada" (notas.md, 25-11; B exacto en scripts/charge_B_exact.py):
    q_{κ,c}(t_i | t_<i, s) ∝ p(t_i | t_<i) · κ^[t_i != s_i] · μ^|Q_i − c| · [factible],
    Q_i = Σ_{j<=i} (t_j − s_j)/2   (carga transportada hasta el sitio i; Q_N = 0 por estar en el sector),
mezcla estratificada sobre componentes (κ, c) (la pareja k de cada s usa la componente k mod J; pesos a_j = n_j/K).
μ = 1 y c = 0 es la propuesta u1 de purity_lowvar.  t siempre en el sector de s → compatible con la etiqueta de Sz.
La densidad de la mezcla sale de UNA pasada del ARViT sobre t (condicionales con t_<i fijado).
Usar con purity_lowvar.make_u1_lowvar_purity(shell=False, cv_t=False, ...): solo necesita (t, log q) por pareja.
"""
import numpy as np
import jax
import jax.numpy as jnp

sg = jax.lax.stop_gradient


def _logq_sites(lg, S, t, lk, c, lmu):
    """log q_{κ,c}(t|S) (B,) dado lg = log p(·|t_<i) (B,N,2); lk, c: (B,) o escalares."""
    B, N = S.shape
    vals = jnp.asarray([-1.0, 1.0], S.dtype)
    m = jnp.sum(S, 1, keepdims=True)
    cs = jnp.cumsum(t, 1) - t                                                       # Σ_{j<i} t_j
    Qp = (cs - (jnp.cumsum(S, 1) - S)) / 2.0                                        # Q_{i-1}
    nrem = (N - jnp.arange(N))[None, :].astype(S.dtype)
    fe = jnp.abs((m - cs)[..., None] - vals) <= (nrem[..., None] - 1.0)             # (B,N,2)
    oth = (vals[None, None, :] != S[..., None]).astype(lg.dtype)                    # valor != s_i
    Qv = Qp[..., None] + (vals[None, None, :] - S[..., None]) / 2.0                 # Q_i si t_i = v
    lk = jnp.reshape(lk, (-1, 1, 1)) if jnp.ndim(lk) else lk
    c = jnp.reshape(c, (-1, 1, 1)) if jnp.ndim(c) else c
    lv = jnp.where(fe, lg + lk * oth + lmu * jnp.abs(Qv - c), -jnp.inf)
    i_t = (t > 0).astype(jnp.int32)[..., None]
    return jnp.sum(jnp.take_along_axis(lv, i_t, -1)[..., 0] - jax.scipy.special.logsumexp(lv, axis=-1), 1)


def make_charge_mix_logq(model, comps, mu, K):
    """mix_logq(params, S, t) -> log q_mix(t|S) (B,)."""
    J = len(comps); comp = np.arange(K) % J
    loga = jnp.asarray(np.log(np.bincount(comp, minlength=J) / K))
    lks = jnp.asarray([np.log(k) for k, _ in comps]); cs_ = jnp.asarray([float(c) for _, c in comps]); lmu = float(np.log(mu))

    def mix_logq(params, S, t):
        lg = jax.nn.log_softmax(model.apply({"params": params}, t, only_logits=True), axis=-1)
        lqj = jax.vmap(lambda lk, c: _logq_sites(lg, S, t, lk, c, lmu))(lks, cs_)   # (J,B)
        return jax.scipy.special.logsumexp(lqj + loga[:, None], axis=0)
    return mix_logq


def make_charge_mix_partners(model, N, comps, mu, K):
    """partners(params, s, key) -> (t (M*K,N), log q_mix (M*K,)).  comps: lista de (κ, c); μ común."""
    J = len(comps); comp = np.arange(K) % J
    assert np.all(np.bincount(comp, minlength=J) > 0), "K debe ser >= número de componentes"
    lk_all = np.array([np.log(k) for k, _ in comps]); c_all = np.array([float(c) for _, c in comps]); lmu = float(np.log(mu))
    mix_logq = make_charge_mix_logq(model, comps, mu, K)

    def partners(params, s, key):
        params = sg(params)
        S = jnp.repeat(s, K, axis=0); B = S.shape[0]
        lk = jnp.tile(jnp.asarray(lk_all[comp]), s.shape[0])[:, None]
        cc = jnp.tile(jnp.asarray(c_all[comp]), s.shape[0])[:, None]
        m = jnp.sum(S, 1)
        vals = jnp.asarray([-1.0, 1.0], S.dtype)

        def body(i, carry):
            t, cs, Q, key = carry
            key, sub = jax.random.split(key)
            lg = model.apply({"params": params}, t, only_logits=True)
            li = jax.nn.log_softmax(jax.lax.dynamic_index_in_dim(lg, i, axis=1, keepdims=False), -1)
            si = jax.lax.dynamic_index_in_dim(S, i, axis=1, keepdims=False)
            other = (vals[None, :] != si[:, None]).astype(li.dtype)
            fe = jnp.abs((m - cs)[:, None] - vals) <= (N - i - 1.0)
            Qv = Q[:, None] + (vals[None, :] - si[:, None]) / 2.0
            lt = jnp.where(fe, li + lk * other + lmu * jnp.abs(Qv - cc), -jnp.inf)
            draw = jax.random.categorical(sub, lt, axis=-1)
            v = 2.0 * draw - 1.0
            return t.at[:, i].set(v), cs + v, Q + (v - si) / 2.0, key
        t, _, _, _ = jax.lax.fori_loop(0, N, body, (jnp.zeros_like(S), jnp.zeros(B, S.dtype), jnp.zeros(B, S.dtype), key))
        return t, mix_logq(params, S, t)
    return partners


def charge_comps(kappa=0.5, offsets=(-1, 0, 1)):
    return [(kappa, c) for c in offsets]
