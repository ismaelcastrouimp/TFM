"""
src_renyi/mps_ancilla/purity_lowvar.py
======================================
Estimador de la pureza (y de su gradiente) con menos varianza que make_tilted_purity, para la ancilla MPS.
Explicacion en .claude/notas.md ("Estimador de pureza de baja varianza", 25-11).

  P = sum_s p(s) Z(s),   Z(s) = sum_t p(t) f(s,t)

Dos piezas sobre el estimador inclinado (t ~ q_mix(.|s), w = p(t) f / q):
  (1) cascaron exacto: los t a distancia de Hamming d(s,t) <= dsh se suman exactamente (1 + N + ... evaluaciones
      deterministas por s); el muestreo solo cubre el resto (w * [d > dsh]).  Insesgado.
          Z(s) = sum_{d<=dsh} p(t) f(s,t) + (1/K) sum_k w_k [d_k > dsh]
  (2) variable de control para el termino de las parejas: E_{q_theta}[grad log q_theta(t|s)] = 0 (q con los mismos
      parametros que p, densidad de la mezcla exacta), asi que se resta beta_k grad log q(t_k|s) con
      beta_k = media "deja-uno-fuera" de los w [d > dsh] del mismo s (independiente de t_k: sigue insesgado).
      Es la linea base del termino de score de t, igual que la de s.
Misma firma que make_tilted_purity: loss(params, s, t, lq, K, den=None, base=None) -> (T*L, (P_hat, ess, d_mean, frac_far)).
"""
import itertools
import numpy as np
import jax
import jax.numpy as jnp

from .tilted_loss import make_pair_logs, make_mix_logq, _rep

sg = jax.lax.stop_gradient


def shell_flips(N, dsh):
    pats = [np.ones(N)]
    for d in range(1, dsh + 1):
        for c in itertools.combinations(range(N), d):
            v = np.ones(N); v[list(c)] = -1.0; pats.append(v)
    return np.array(pats)


def make_lowvar_purity(model, N, chi, T, kappas, K, dsh=1, cv_t=True):
    pair_logs = make_pair_logs(model, chi)
    mix_logq = make_mix_logq(model)
    Fl = jnp.asarray(shell_flips(N, dsh)) if dsh >= 0 else None
    nF = 0 if Fl is None else Fl.shape[0]
    J = len(kappas); comp = np.arange(K) % J
    loga = np.log(np.bincount(comp, minlength=J) / K); logks = np.log(np.asarray(kappas, float))

    def loss(params, s, t, lq, K_=None, den=None, base=None):
        M = s.shape[0]
        lps, lpt, logf = pair_logs(params, s, t, K)
        dd = jnp.sum(t != _rep(s, K), axis=1)
        far = (dd > dsh).astype(lpt.dtype)
        w = jnp.exp(sg(lpt + logf - lq)) * far                                    # (M*K,)
        W = w.reshape(M, K)
        Zs = W.mean(1)
        L = jnp.sum(w * (lpt + logf)) / (M * K)
        if nF:
            tsh = (s[:, None, :] * Fl[None]).reshape(-1, N)
            _, lpt_sh, logf_sh = pair_logs(params, s, tsh, nF)
            wsh = jnp.exp(sg(lpt_sh + logf_sh))                                     # p(t) f, exacto
            Zs = Zs + wsh.reshape(M, nF).sum(1)
            L = L + jnp.sum(wsh * (lpt_sh + logf_sh)) / M
        P_hat = Zs.mean()
        den = sg(P_hat) if den is None else den
        base = sg(P_hat) if base is None else base
        L = L + jnp.sum(sg(Zs - base) * lps) / M
        if cv_t:
            lq_live, _ = mix_logq(params, _rep(s, K), t, logks, loga)              # log q_mix con gradiente
            beta = (W.sum(1, keepdims=True) - W) / (K - 1)                          # deja-uno-fuera
            L = L - jnp.sum(sg(beta).reshape(-1) * lq_live) / (M * K)
        ess = jnp.mean(W.sum(1) ** 2 / jnp.maximum((W ** 2).sum(1), 1e-300)) / K
        frac_far = jnp.mean(W.mean(1)) / P_hat
        return T * L / den, (P_hat, ess, jnp.mean(dd), frac_far)
    return loss


# Forma simétrica (sym=True): como f(s,t) = f(t,s), Σ p(s)p(t) f ∇log p(t) = Σ_s p(s) Z(s) ∇log p(s), así que
#   ∇P = E_s[ 2 (Z(s) − b) ∇log p(s) ] + E_{s,t}[ w ∇log f(s,t) ]
# (todo el score en s).  Insesgada; su score se cancela casi del todo con el de la energía en las mismas s cerca del
# óptimo (e(s) + 2T Z(s)/P ≈ cte).  Con sym la variable de control en t no se usa (no hay score de t).

# ─── version U(1): parejas solo en el sector de Sz de s ────────────────────────
# Si H conserva Sz, rho(s,t) = 0 salvo Sz(t) = Sz(s) (d par, mismas subidas que bajadas).  La propuesta inclinada
# general gasta la mayoria de las parejas fuera del sector.  Aqui q_kappa se restringe al sector: en el sitio i solo se
# permiten los valores con los que todavia se puede llegar a sum(t) = sum(s) con los sitios que quedan (densidad exacta:
# condicionales renormalizados sobre los valores permitidos).  Cascaron exacto: d = 0 y los intercambios d = 2
# (todas las parejas i<j; las de s_i = s_j se anulan con una mascara).

def _feasible(S, t):
    """(B,N,2) bool: valores (-1,+1) permitidos en cada sitio dado el prefijo t_<i, para acabar en sum(t) = sum(s)."""
    N = S.shape[1]
    m = jnp.sum(S, 1, keepdims=True)
    cs = jnp.cumsum(t, 1) - t                                                       # sum_{j<i} t_j
    nrem = (N - jnp.arange(N))[None, :].astype(S.dtype)                             # sitios i..N-1
    r = m - cs
    vals = jnp.asarray([-1.0, 1.0], S.dtype)
    return jnp.abs(r[..., None] - vals) <= (nrem[..., None] - 1.0)


def make_u1_mix_logq(model):
    def mix_logq(params, S, t, logks, loga):
        lg = jax.nn.log_softmax(model.apply({"params": params}, t, only_logits=True), axis=-1)  # (B,N,2)
        fe = _feasible(S, t)
        i_s = (S > 0).astype(jnp.int32)
        oth = jnp.stack([i_s == 1, i_s == 0], -1).astype(lg.dtype)                  # 1 en el valor != s_i
        i_t = (t > 0).astype(jnp.int32)[..., None]

        def one(lk):
            lv = jnp.where(fe, lg + lk * oth, -jnp.inf)
            lc = jax.scipy.special.logsumexp(lv, axis=-1)
            return jnp.sum(jnp.take_along_axis(lv, i_t, -1)[..., 0] - lc, 1)
        lqj = jax.vmap(one)(jnp.asarray(logks))
        return jax.scipy.special.logsumexp(lqj + jnp.asarray(loga)[:, None], axis=0), lqj
    return mix_logq


def make_u1_mix_partners(model, N, kappas, K):
    J = len(kappas); comp = np.arange(K) % J
    a = np.bincount(comp, minlength=J) / K
    assert np.all(a > 0)
    logks = np.log(np.asarray(kappas, float)); loga = np.log(a)
    mix_logq = make_u1_mix_logq(model)

    def partners(params, s, key):
        params = sg(params)
        S = jnp.repeat(s, K, axis=0); B = S.shape[0]
        lk = jnp.tile(jnp.asarray(logks[comp]), s.shape[0])[:, None]
        m = jnp.sum(S, 1)
        vals = jnp.asarray([-1.0, 1.0], S.dtype)

        def body(i, carry):
            t, cs, key = carry
            key, sub = jax.random.split(key)
            lg = model.apply({"params": params}, t, only_logits=True)
            li = jax.nn.log_softmax(jax.lax.dynamic_index_in_dim(lg, i, axis=1, keepdims=False), -1)
            si = jax.lax.dynamic_index_in_dim(S, i, axis=1, keepdims=False)
            other = jnp.stack([si > 0, si < 0], axis=-1).astype(li.dtype)
            fe = jnp.abs((m - cs)[:, None] - vals) <= (N - i - 1.0)
            lt = jnp.where(fe, li + lk * other, -jnp.inf)
            draw = jax.random.categorical(sub, lt, axis=-1)
            v = 2.0 * draw - 1.0
            return t.at[:, i].set(v), cs + v, key
        t, _, _ = jax.lax.fori_loop(0, N, body, (jnp.zeros_like(S), jnp.zeros(B, S.dtype), key))
        lq, _ = mix_logq(params, S, t, logks, loga)
        return t, lq
    return partners


def make_u1_lowvar_purity(model, N, chi, T, kappas, K, shell=True, cv_t=True, sym=False):
    """Como make_lowvar_purity pero con parejas en el sector de Sz de s (usar con make_u1_mix_partners).
    shell=True: d = 0 y d = 2 exactos; el muestreo cubre d >= 4."""
    pair_logs = make_pair_logs(model, chi)
    mix_logq = make_u1_mix_logq(model)
    Fl = jnp.asarray(shell_flips(N, 2)[[0] + list(range(N + 1, N + 1 + N * (N - 1) // 2))]) if shell else None
    nF = 0 if Fl is None else Fl.shape[0]
    pi, pj = np.triu_indices(N, 1)
    J = len(kappas); comp = np.arange(K) % J
    loga = np.log(np.bincount(comp, minlength=J) / K); logks = np.log(np.asarray(kappas, float))
    dsh = 2 if shell else -1

    def loss(params, s, t, lq, K_=None, den=None, base=None):
        M = s.shape[0]
        lps, lpt, logf = pair_logs(params, s, t, K)
        dd = jnp.sum(t != _rep(s, K), axis=1)
        far = (dd > dsh).astype(lpt.dtype)
        w = jnp.exp(sg(lpt + logf - lq)) * far
        W = w.reshape(M, K)
        Zs = W.mean(1)
        L = jnp.sum(w * (logf if sym else lpt + logf)) / (M * K)
        if nF:
            tsh = (s[:, None, :] * Fl[None]).reshape(-1, N)
            _, lpt_sh, logf_sh = pair_logs(params, s, tsh, nF)
            ok = jnp.concatenate([jnp.ones((M, 1), bool), s[:, pi] != s[:, pj]], 1).reshape(-1)
            wsh = jnp.where(ok, jnp.exp(sg(lpt_sh + logf_sh)), 0.0)
            Zs = Zs + wsh.reshape(M, nF).sum(1)
            L = L + jnp.sum(wsh * jnp.where(ok, logf_sh if sym else lpt_sh + logf_sh, 0.0)) / M
        P_hat = Zs.mean()
        den = sg(P_hat) if den is None else den
        base = sg(P_hat) if base is None else base
        L = L + (2.0 if sym else 1.0) * jnp.sum(sg(Zs - base) * lps) / M
        if cv_t and not sym:
            lq_live, _ = mix_logq(params, _rep(s, K), t, logks, loga)
            beta = (W.sum(1, keepdims=True) - W) / (K - 1)
            L = L - jnp.sum(sg(beta).reshape(-1) * lq_live) / (M * K)
        ess = jnp.mean(W.sum(1) ** 2 / jnp.maximum((W ** 2).sum(1), 1e-300)) / K
        return T * L / den, (P_hat, ess, jnp.mean(dd), jnp.mean(W.mean(1)) / P_hat)
    return loss


# ─── cascaron general definido por mascaras de giro (U(1)) ─────────────────────
# shell(s) = { t = s XOR m : m en la lista de mascaras, Sz(t) = Sz(s) }.  Cualquier conjunto que dependa solo de s vale
# (insesgado) si el muestreo excluye exactamente ese conjunto: w * [mask(s,t) no esta en la lista].
#   "ham2": {0} U {todas las parejas i<j}                  (= make_u1_lowvar_purity con shell=True)
#   "loc1": {0} U {(i,i+1)} U {(i,i+2)}  (un movimiento de H J1-J2; PBC)
#   "loc2": XOR de <= 2 mascaras de loc1 (contiene los t a <= 2 movimientos, y alguno mas del sector)
# Para "ham2" las parejas validas (s_i != s_j) se compactan a 1 + floor(N^2/4) evaluaciones por s.

def shell_masks(N, kind):
    pair = lambda i, j: (1 << i) | (1 << j)
    loc = [pair(i, (i + 1) % N) for i in range(N)] + [pair(i, (i + 2) % N) for i in range(N)]
    loc = sorted(set(loc))
    if kind == "ham2":
        ms = {0} | {pair(i, j) for i in range(N) for j in range(i + 1, N)}
    elif kind == "loc1":
        ms = {0} | set(loc)
    elif kind == "loc2":
        ms = {0} | set(loc) | {x ^ y for x in loc for y in loc}
    else:
        raise ValueError(kind)
    return np.array(sorted(ms), dtype=np.int64)


def max_valid(N, ms, chunk=1 << 14):
    """max sobre configuraciones x de #{m : 2 popcount(m & x) = popcount(m)} (mascaras que conservan Sz)."""
    pc = np.bitwise_count(ms).astype(np.int64); best = 0
    for x0 in range(0, 1 << N, chunk):
        x = np.arange(x0, min(x0 + chunk, 1 << N), dtype=np.int64)
        v = (2 * np.bitwise_count(x[:, None] & ms[None, :]) == pc[None, :]).sum(1)
        best = max(best, int(v.max()))
    return best


def make_u1_mask_purity(model, N, chi, T, kappas, K, kind="loc1", cv_t=True, sym=False):
    pair_logs = make_pair_logs(model, chi)
    mix_logq = make_u1_mix_logq(model)
    ms = shell_masks(N, kind); msj = jnp.asarray(ms)
    bits = ((ms[:, None] >> np.arange(N)) & 1).astype(np.float64)                      # (nM, N)
    flipsg = jnp.asarray(1.0 - 2.0 * bits)                                              # multiplicador +-1
    nM = len(ms)
    L_eval = max_valid(N, ms)                                                            # evaluaciones por s
    pw = jnp.asarray(1 << np.arange(N), dtype=jnp.int64)
    J = len(kappas); comp = np.arange(K) % J
    loga = np.log(np.bincount(comp, minlength=J) / K); logks = np.log(np.asarray(kappas, float))

    def loss(params, s, t, lq, K_=None, den=None, base=None):
        M = s.shape[0]
        lps, lpt, logf = pair_logs(params, s, t, K)
        mt = jnp.sum((t != _rep(s, K)).astype(jnp.int64) * pw, 1)                      # mascara de s XOR t
        pos = jnp.clip(jnp.searchsorted(msj, mt), 0, nM - 1)
        far = (msj[pos] != mt).astype(lpt.dtype)
        w = jnp.exp(sg(lpt + logf - lq)) * far
        W = w.reshape(M, K)
        Zs = W.mean(1)
        L = jnp.sum(w * (logf if sym else lpt + logf)) / (M * K)
        # cascaron: todas las mascaras, validas si conservan Sz; se compactan las validas a L_eval columnas
        tall = s[:, None, :] * flipsg[None]                                               # (M, nM, N)
        ok = jnp.sum(tall, -1) == jnp.sum(s, -1)[:, None]
        order = jnp.argsort(~ok, axis=1, stable=True)[:, :L_eval]                         # validas primero
        oks = jnp.take_along_axis(ok, order, 1)
        tsh = jnp.take_along_axis(tall, order[..., None], 1).reshape(-1, N)
        _, lpt_sh, logf_sh = pair_logs(params, s, tsh, L_eval)
        oks = oks.reshape(-1)
        wsh = jnp.where(oks, jnp.exp(sg(lpt_sh + logf_sh)), 0.0)
        Zs = Zs + wsh.reshape(M, L_eval).sum(1)
        L = L + jnp.sum(wsh * jnp.where(oks, logf_sh if sym else lpt_sh + logf_sh, 0.0)) / M
        P_hat = Zs.mean()
        den = sg(P_hat) if den is None else den
        base = sg(P_hat) if base is None else base
        L = L + (2.0 if sym else 1.0) * jnp.sum(sg(Zs - base) * lps) / M
        if cv_t and not sym:
            lq_live, _ = mix_logq(params, _rep(s, K), t, logks, loga)
            beta = (W.sum(1, keepdims=True) - W) / (K - 1)
            L = L - jnp.sum(sg(beta).reshape(-1) * lq_live) / (M * K)
        ess = jnp.mean(W.sum(1) ** 2 / jnp.maximum((W ** 2).sum(1), 1e-300)) / K
        return T * L / den, (P_hat, ess, jnp.mean(far), jnp.mean(W.mean(1)) / P_hat)
    loss.n_eval = L_eval
    return loss
