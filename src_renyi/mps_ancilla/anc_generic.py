"""
src_renyi/mps_ancilla/anc_generic.py
====================================
Energía local, pureza inclinada y ρ exacta para CUALQUIER ancilla cuyo solape se pueda contraer exactamente
(MPS, SBS como cadena, árbol de tensores, ...).  Es lo mismo que generic_eloc.py, tilted_loss.py (make_pair_logs /
make_tilted_purity) y training.make_loss_fns, pero con el solape como argumento:

    ov(M1, M2) -> (P,) complejo,  ov = Σ_a Φ_1(a) Φ_2(a)*  (= ⟨Φ_2|Φ_1⟩),  con M1, M2 pytrees de lote P alineados

(para una MPS: ov = lambda M1, M2: mps_overlap_aligned(M1, M2, chi)).  El modelo devuelve (log A(s), M(s)) con M un
pytree cualquiera (array para MPS, dict para el árbol).

  make_eloc_ov(model, H, ov)                  -> eloc(params, s)
  make_energy_terms_ov(model, H, ov)          -> terms(params, s, Ebar, Mtot) -> (L_E, ΣReE/Mtot)
  make_tilted_purity_ov(model, N, T, ov)      -> purity_loss(params, s, t, lq, K, den=None, base=None)
  make_loss_fns_ov(model, all_s, H, T, ov, chunk) -> (loss, rho_fn)   (ρ completa, N pequeño)
"""
import jax
import jax.numpy as jnp

from .generic_eloc import _sign_map

sg = jax.lax.stop_gradient
tmap = jax.tree_util.tree_map


def _rep(M, K):
    return tmap(lambda x: jnp.repeat(x, K, axis=0), M)


def make_eloc_ov(model, H, ov):
    conv = _sign_map(H)
    Hj = H.to_jax_operator()
    Hj.get_conn_padded(jnp.ones((1, H.hilbert.size)))

    def eloc(params, s):
        B, N = s.shape
        sp, mels = Hj.get_conn_padded(conv(s))
        C = sp.shape[1]
        sp = sp.reshape(B * C, N).astype(s.dtype)
        lA, Mt = model.apply({"params": params}, s)
        lAp, Mp = model.apply({"params": params}, sp)
        n2 = jnp.real(ov(Mt, Mt)); n2p = jnp.real(ov(Mp, Mp))
        O = ov(Mp, _rep(Mt, C))
        G = O / jnp.sqrt(n2p * jnp.repeat(n2, C))
        ratio = jnp.exp(lAp - jnp.repeat(lA, C))
        w = jnp.where(mels.reshape(-1) != 0, mels.reshape(-1) * ratio * G, 0.0)
        return jnp.sum(w.reshape(B, C), axis=1)
    return eloc


def make_energy_terms_ov(model, H, ov):
    eloc = make_eloc_ov(model, H, ov)

    def terms(params, s, Ebar, Mtot):
        s = sg(s)
        ReE = jnp.real(eloc(params, s))
        logp = 2.0 * jnp.real(model.apply({"params": params}, s)[0])
        return jnp.sum(sg(ReE - Ebar) * logp + ReE) / Mtot, jnp.sum(ReE) / Mtot
    return terms


def make_tilted_purity_ov(model, N, T, ov):
    """Igual que tilted_loss.make_tilted_purity (mismas fórmulas y salidas)."""
    def pair_logs(params, s, t, K):
        lA_s, M_s = model.apply({"params": params}, s)
        lA_t, M_t = model.apply({"params": params}, t)
        n_s = jnp.real(ov(M_s, M_s)); n_t = jnp.real(ov(M_t, M_t))
        o = ov(M_t, _rep(M_s, K))
        logf = jnp.log(jnp.abs(o) ** 2) - jnp.log(n_t) - jnp.repeat(jnp.log(n_s), K)
        return 2.0 * jnp.real(lA_s), 2.0 * jnp.real(lA_t), logf

    def purity_loss(params, s, t, lq, K, den=None, base=None):
        M = s.shape[0]
        lps, lpt, logf = pair_logs(params, s, t, K)
        w = jnp.exp(sg(lpt + logf - lq))
        Zs = w.reshape(M, K).mean(1)
        P_hat = Zs.mean()
        den = sg(P_hat) if den is None else den
        base = sg(P_hat) if base is None else base
        L = (jnp.sum(sg(Zs - base) * lps) / M + jnp.sum(w * (lpt + logf)) / (M * K)) / den
        W = w.reshape(M, K)
        ess = jnp.mean(W.sum(1) ** 2 / (W ** 2).sum(1)) / K
        dd = jnp.sum(t != jnp.repeat(s, K, axis=0), axis=1)
        f3 = jnp.sum(w * (dd >= 3)) / jnp.sum(w)
        return T * L, (P_hat, ess, jnp.mean(dd), f3)
    return purity_loss


def compute_rho_ov(model, params, all_s, ov, chunk=16):
    """ρ completa (D x D): filas por trozos (lax.map + checkpoint)."""
    logA, M = model.apply({"params": params}, all_s)
    D = all_s.shape[0]
    A = jnp.exp(logA - jnp.max(jnp.real(logA))); A = A / jnp.linalg.norm(A)
    nrm = jnp.sqrt(jnp.real(ov(M, M)))
    chunk = min(chunk, D); assert D % chunk == 0

    @jax.checkpoint
    def rows(i0):
        Mi = tmap(lambda x: jax.lax.dynamic_slice_in_dim(x, i0, chunk, 0), M)
        Mi = _rep(Mi, D)                                                   # (chunk*D, ...): fila i repetida D veces
        Mj = tmap(lambda x: jnp.tile(x, (chunk,) + (1,) * (x.ndim - 1)), M)
        return ov(Mi, Mj).reshape(chunk, D)                                # ⟨Φ_t|Φ_s⟩ con s = fila, t = columna
    O = jax.lax.map(rows, jnp.arange(0, D, chunk)).reshape(D, D)
    G = O / (nrm[:, None] * nrm[None, :])
    return A[:, None] * jnp.conj(A)[None, :] * G


def make_loss_fns_ov(model, all_s, H_dense, T, ov, chunk=16):
    all_s = jnp.asarray(all_s); H = jnp.asarray(H_dense)

    def loss(params):
        rho = compute_rho_ov(model, params, all_s, ov, chunk)
        E = jnp.real(jnp.sum(H.T * rho)); S2 = -jnp.log(jnp.sum(jnp.abs(rho) ** 2))
        return E - T * S2, (E, S2)
    return loss, jax.jit(lambda p: compute_rho_ov(model, p, all_s, ov, chunk))
