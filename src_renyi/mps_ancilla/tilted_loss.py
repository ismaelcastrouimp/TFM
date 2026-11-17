"""
src_renyi/mps_ancilla/tilted_loss.py
====================================
Termino de pureza con MUESTREO CONDICIONADO DEL COMPANERO ("AR inclinado"; explicacion completa en
.claude/notas.md, apartados 10-16).

  P = sum_s p(s) Z(s),   Z(s) = sum_t p(t) f(s,t),   f(s,t) = |<phi_t|phi_s>|^2 / (|phi_t|^2 |phi_s|^2)
  t ~ q(t|s) = prod_i q_i(t_i | t_<i; s),   q_i ∝ p(t_i|t_<i) * kappa^[t_i != s_i]
  w(s,t) = p(t) f(s,t) / q(t|s)      ->   Z_hat(s) = (1/K) sum_k w(s,t_k),   P_hat = (1/M) sum_s Z_hat(s)
  grad P_hat = (1/M) sum_s [ (Z_hat(s) - b) grad log p(s) + (1/K) sum_k w(s,t_k) grad log( p(t_k) f(s,t_k) ) ]
(b = linea base constante, no sesga porque E_p[grad log p(s)] = 0; q va con stop-gradient.)
Perdida sustituta de pureza:  L_P = T * [ ... ] / sg(P_den),   grad L_P = T grad P_hat / P_den  ( = -T grad S2 ).
"""
import jax
import jax.numpy as jnp

from .models import mps_overlap_aligned

sg = jax.lax.stop_gradient


def make_tilt_sampler(model, N):
    """tilt(params, S, key, logk) -> (t, log q(t|S)).  S: (B, N) objetivos (+-1).  Muestreo AR exacto de q.
    logk: escalar o (B,) (un kappa por fila: para la mezcla defensiva)."""
    def tilt(params, S, key, logk):
        B = S.shape[0]
        lk = jnp.broadcast_to(jnp.asarray(logk, S.dtype), (B,))[:, None]

        def body(i, carry):
            t, lq, key = carry
            key, sub = jax.random.split(key)
            lg = model.apply({"params": params}, t, only_logits=True)            # (B, N, 2): [-1, +1]
            li = jax.lax.dynamic_index_in_dim(lg, i, axis=1, keepdims=False)
            si = jax.lax.dynamic_index_in_dim(S, i, axis=1, keepdims=False)
            other = jnp.stack([si > 0, si < 0], axis=-1).astype(li.dtype)        # 1 en el valor != s_i
            lt = jax.nn.log_softmax(li + lk * other, axis=-1)                    # log q_i
            draw = jax.random.categorical(sub, lt, axis=-1)
            lq = lq + jnp.take_along_axis(lt, draw[:, None], axis=-1)[:, 0]
            return t.at[:, i].set(2.0 * draw - 1.0), lq, key
        t, lq, _ = jax.lax.fori_loop(0, N, body, (jnp.zeros_like(S), jnp.zeros(B, S.dtype), key))
        return t, lq
    return tilt


def make_mix_logq(model):
    """mix_logq(params, S, t, logks (J,), loga (J,)) -> (log q_mix(t|S) (B,), log q_j(t|S) (J,B)).
    q_mix = sum_j a_j q_{kappa_j}.  Todos los q_kappa(t|s) salen de UNA pasada del ARViT sobre t:
      log q_kappa(t|s) = sum_i [ log p(t_i|t_<i) + [t_i != s_i] log kappa - log c_i(kappa) ],
      c_i(kappa) = p(t_i = s_i | t_<i) + kappa p(t_i != s_i | t_<i)."""
    def mix_logq(params, S, t, logks, loga):
        lg = jax.nn.log_softmax(model.apply({"params": params}, t, only_logits=True), axis=-1)   # (B,N,2)
        i_s = (S > 0).astype(jnp.int32)[..., None]; i_t = (t > 0).astype(jnp.int32)[..., None]
        lp_same = jnp.take_along_axis(lg, i_s, -1)[..., 0]
        lp_oth = jnp.take_along_axis(lg, 1 - i_s, -1)[..., 0]
        lp_t = jnp.take_along_axis(lg, i_t, -1)[..., 0].sum(1)                                     # log p(t)
        nfl = jnp.sum(t != S, axis=1).astype(lg.dtype)
        def one(lk):
            return lp_t + nfl * lk - jnp.sum(jnp.logaddexp(lp_same, lk + lp_oth), axis=1)
        lqj = jax.vmap(one)(jnp.asarray(logks))                                                    # (J,B)
        return jax.scipy.special.logsumexp(lqj + jnp.asarray(loga)[:, None], axis=0), lqj
    return mix_logq


def make_mix_partners(model, N, kappas, K):
    """partners(params, s, key) -> (t (M*K,N), log q_mix (M*K,)) con MEZCLA DEFENSIVA estratificada:
    el companero k de cada s usa kappa_{k mod J}; pesos de la mezcla a_j = n_j/K (heuristica de balance de Veach:
    insesgado con la densidad de la mezcla).  Parametros sin gradiente (q va con stop-gradient)."""
    import numpy as np
    J = len(kappas)
    comp = np.arange(K) % J
    a = np.bincount(comp, minlength=J) / K
    assert np.all(a > 0), "K debe ser >= numero de kappas"
    logks = np.log(np.asarray(kappas, float)); loga = np.log(a)
    tilt = make_tilt_sampler(model, N); mix_logq = make_mix_logq(model)

    def partners(params, s, key):
        params = sg(params)
        S = jnp.repeat(s, K, axis=0)
        lk_rows = jnp.tile(jnp.asarray(logks[comp]), s.shape[0])
        t, _ = tilt(params, S, key, lk_rows)
        lq, _ = mix_logq(params, S, t, logks, loga)
        return t, lq
    return partners


def _rep(x, K):
    return jnp.repeat(x, K, axis=0)


def make_pair_logs(model, chi):
    """pair_logs(params, s, t, K) -> (log p(s) (M,), log p(t) (M*K,), log f (M*K,)).  t ordenado como repeat(s, K)."""
    def pair_logs(params, s, t, K):
        lA_s, M_s = model.apply({"params": params}, s)
        lA_t, M_t = model.apply({"params": params}, t)
        n_s = jnp.real(mps_overlap_aligned(M_s, M_s, chi))
        n_t = jnp.real(mps_overlap_aligned(M_t, M_t, chi))
        Ms_r = jax.tree_util.tree_map(lambda x: _rep(x, K), M_s)
        ov = mps_overlap_aligned(M_t, Ms_r, chi)
        logf = jnp.log(jnp.abs(ov) ** 2) - jnp.log(n_t) - _rep(jnp.log(n_s), K)
        return 2.0 * jnp.real(lA_s), 2.0 * jnp.real(lA_t), logf
    return pair_logs


def make_tilted_purity(model, N, chi, T):
    """purity_loss(params, s, t, lq, K, den, base) -> (L_P, (P_hat, ess_frac, d_mean, frac_P_d>=3)).
    s: (M,N), t: (M*K,N) ~ q(.|s) (muestreados con params sin gradiente), lq = log q(t|s).
    den, base: denominador y linea base (floats) o None (= P_hat del lote, sin gradiente)."""
    pair_logs = make_pair_logs(model, chi)

    def purity_loss(params, s, t, lq, K, den=None, base=None):
        M = s.shape[0]
        lps, lpt, logf = pair_logs(params, s, t, K)
        logw = sg(lpt + logf - lq)
        w = jnp.exp(logw)                                                          # (M*K,)
        Zs = w.reshape(M, K).mean(1)
        P_hat = Zs.mean()
        den = sg(P_hat) if den is None else den
        base = sg(P_hat) if base is None else base
        L = (jnp.sum(sg(Zs - base) * lps) / M + jnp.sum(w * (lpt + logf)) / (M * K)) / den
        W = w.reshape(M, K)
        ess = jnp.mean(W.sum(1) ** 2 / (W ** 2).sum(1)) / K
        dd = jnp.sum(t != _rep(s, K), axis=1)
        d = jnp.mean(dd)
        f3 = jnp.sum(w * (dd >= 3)) / jnp.sum(w)                                   # fraccion de P_hat en d >= 3
        return T * L, (P_hat, ess, d, f3)
    return purity_loss
