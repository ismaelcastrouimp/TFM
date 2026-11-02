"""
src_renyi/mps_ancilla/mc_loss.py
================================
Perdida sustituta MC (energia local + pureza) del modelo ARViT + MPS ancilla, con dos extras respecto
a la version que estaba en training.py:

  * purity = "iid" | "nbr"   (nbr: suma exacta de t=s y de los N vecinos a un espin + parejas iid con d>=2)
  * SAMPLE CHUNKING EXACTO:  el lote de M muestras se procesa en n_chunks trozos (lax.scan) y el gradiente
    es EXACTAMENTE el del lote completo (no es una aproximacion), porque la perdida es una suma sobre
    muestras (o bloques de pair_block muestras) una vez fijados los estadisticos globales
    Ebar, P_near, P_far.  Dos pasadas:
        1) solo forward por trozos            -> Ebar, P_near, P_far globales
        2) grad por trozos con esos globales  -> suma de gradientes
    La memoria de pico pasa de ~M a ~M/n_chunks.  Sin chunking (n_chunks=1) es el mismo codigo.

Notacion: p(s)=|A(s)|^2, f(s,t)=|G(s,t)|^2, P = sum_{s,t} p(s)p(t) f(s,t), S2 = -log P.
"""
import jax
import jax.numpy as jnp

from .models import mps_overlap_rect, mps_overlap_aligned

sg = jax.lax.stop_gradient


def make_mc_terms(model, N, T, hx, J, chunk=None, pair_block=None, purity="iid"):
    """terms(params, s, glob, Mtot) -> (L_E, L_P, stats).

    s: (M, N) muestras de ESTE trozo; Mtot: numero total de muestras del lote completo.
    glob = (Ebar, P_near, P_far) globales (floats, sin gradiente) o None (= usar los de este lote,
    valido solo si el trozo es el lote completo).  L_E y L_P llegan ya normalizados con Mtot, de modo que
    la suma sobre trozos es la perdida del lote completo.  stats = contribucion del trozo a (Ebar, P_near, P_far).
    """
    assert purity in ("iid", "nbr")
    chi = model.chi

    def terms(params, s, glob, Mtot):
        s = sg(s)
        M = s.shape[0]
        logA, Mt = model.apply({"params": params}, s)
        n2 = jnp.real(mps_overlap_aligned(Mt, Mt, chi))                         # ||Phi_s||^2
        logp = 2.0 * jnp.real(logA)                                             # log p(s)

        # ---- energia local ----
        flip = 1.0 - 2.0 * jnp.eye(N)
        sf = (s[:, None, :] * flip[None]).reshape(M * N, N)                     # s con el espin i invertido
        logAf, Mf = model.apply({"params": params}, sf)
        Mrep = jnp.repeat(Mt, N, axis=0)
        Of = mps_overlap_aligned(Mf, Mrep, chi)                                 # <Phi_s | Phi_s'>
        n2f = jnp.real(mps_overlap_aligned(Mf, Mf, chi))
        Gf = Of / jnp.sqrt(n2f * jnp.repeat(n2, N))
        ratio = jnp.exp(logAf - jnp.repeat(logA, N))
        offdiag = jnp.sum((ratio * Gf).reshape(M, N), axis=1)
        diag = jnp.sum(s * jnp.roll(s, -1, axis=1), axis=1)
        ReE = jnp.real(hx * offdiag + J * diag)

        # ---- pureza: parejas ----
        if pair_block is None:
            assert M == Mtot, "pair_block=None (todos los pares) no admite sample chunking"
            O = mps_overlap_rect(Mt, Mt, chi, chunk)
            f = jnp.abs(O) ** 2 / (n2[:, None] * n2[None, :])
            mask = 1.0 - jnp.eye(M)
            npairs = M * (M - 1)
            lp = logp[:, None] + logp[None, :]
            m2 = (s @ s.T <= N - 4).astype(f.dtype)                             # 1 si d(s_i,s_j) >= 2  (s.s' = N-2d)
        else:
            b = pair_block
            assert M % b == 0, "el trozo debe ser multiplo de pair_block"
            nb = M // b
            Mb = Mt.reshape(nb, b, *Mt.shape[1:])
            O = jax.vmap(lambda m: mps_overlap_rect(m, m, chi))(Mb)             # (nb, b, b)
            n2b = n2.reshape(nb, b)
            f = jnp.abs(O) ** 2 / (n2b[:, :, None] * n2b[:, None, :])
            mask = jnp.broadcast_to(1.0 - jnp.eye(b), (nb, b, b))
            npairs = Mtot * (b - 1)                                             # pares TOTALES del lote completo
            lpb = logp.reshape(nb, b)
            lp = lpb[:, :, None] + lpb[:, None, :]
            sb = s.reshape(nb, b, N)
            m2 = (jnp.einsum("nbk,nck->nbc", sb, sb) <= N - 4).astype(f.dtype)

        zero = jnp.zeros(())
        if purity == "iid":
            h = f
            g = None
            sPn = zero
        else:
            p_s = jnp.exp(logp)
            p_f = jnp.exp(2.0 * jnp.real(logAf))                                # p(s^k)
            g = p_s + jnp.sum((p_f * jnp.abs(Gf) ** 2).reshape(M, N), axis=1)   # g(s) = p(s) + sum_k p(s^k) f(s,s^k)
            h = f * m2
            sPn = jnp.sum(g) / Mtot
        sPf = jnp.sum(mask * h) / npairs
        stats = (jnp.sum(ReE) / Mtot, sPn, sPf)

        if glob is None:
            glob = tuple(sg(x) for x in stats)
        Ebar, Pn, Pf = glob

        L_E = jnp.sum(sg(ReE - Ebar) * logp + ReE) / Mtot
        if purity == "iid":
            Pb = Pf
            L_P = T * (jnp.sum(mask * h) + jnp.sum(mask * sg(h - Pf) * lp)) / npairs / sg(Pb)
        else:
            L_near = jnp.sum(g + sg(g - Pn) * logp) / Mtot                       # camino (grad g) + score de s~p
            L_far = (jnp.sum(mask * h) + jnp.sum(mask * sg(h - Pf) * lp)) / npairs
            Pb = Pn + Pf
            L_P = T * (L_near + L_far) / sg(Pb)
        return L_E, L_P, stats

    return terms


def _purity_from_stats(stats, purity):
    return stats[2] if purity == "iid" else stats[1] + stats[2]


def make_mc_loss(model, N, T, hx, J, chunk=None, pair_block=None, terms="both", purity="iid"):
    """mc_loss(params, s) -> (total, (Ebar, S2, F)).  Mismo contrato que la version de training.py.
    purity="iid": P = E_{s,t~p iid}[f]  |  purity="nbr": suma exacta en d<=1 + parejas iid con d>=2 (insesgado)."""
    f_terms = make_mc_terms(model, N, T, hx, J, chunk, pair_block, purity)

    def mc_loss(params, s):
        M = s.shape[0]
        L_E, L_P, st = f_terms(params, s, None, M)
        S2 = -jnp.log(sg(_purity_from_stats(st, purity)))
        Ebar = st[0]
        total = {"both": L_E + L_P, "E": L_E, "P": L_P}[terms]
        return total, (Ebar, S2, Ebar - T * S2)

    return mc_loss


def make_mc_vg(model, N, T, hx, J, chunk=None, pair_block=None, terms="both", purity="iid", n_chunks=1):
    """vg(params, s) -> ((total, aux), grad), con el mismo resultado que
    jax.value_and_grad(make_mc_loss(...), has_aux=True) pero procesando s en n_chunks trozos."""
    if n_chunks <= 1:
        return jax.value_and_grad(make_mc_loss(model, N, T, hx, J, chunk, pair_block, terms, purity),
                                  has_aux=True)
    assert pair_block is not None, "sample chunking requiere pair_block (todos los pares no se puede trocear)"
    f_terms = make_mc_terms(model, N, T, hx, J, chunk, pair_block, purity)

    def vg(params, s):
        M = s.shape[0]
        assert M % (n_chunks * pair_block) == 0, f"M={M} debe ser multiplo de n_chunks*pair_block"
        sch = s.reshape(n_chunks, M // n_chunks, N)
        dummy = (jnp.zeros(()), jnp.zeros(()), jnp.ones(()))                    # los stats no dependen de glob

        def fwd(acc, sc):                                                       # pasada 1: solo forward
            _, _, st = f_terms(params, sc, dummy, M)
            return tuple(a + x for a, x in zip(acc, st)), None
        stats, _ = jax.lax.scan(fwd, (jnp.zeros(()),) * 3, sch)

        def chunk_loss(p, sc):                                                  # pasada 2: grad con globales fijos
            L_E, L_P, _ = f_terms(p, sc, stats, M)
            return {"both": L_E + L_P, "E": L_E, "P": L_P}[terms]

        def body(carry, sc):
            v, gr = jax.value_and_grad(chunk_loss)(params, sc)
            return (carry[0] + v, jax.tree_util.tree_map(jnp.add, carry[1], gr)), None
        (tot, grad), _ = jax.lax.scan(body, (jnp.zeros(()), jax.tree_util.tree_map(jnp.zeros_like, params)), sch)

        S2 = -jnp.log(_purity_from_stats(stats, purity))
        return (tot, (stats[0], S2, stats[0] - T * S2)), grad
    return vg


def make_mc_stats(model, N, T, hx, J, chunk=None, pair_block=None, purity="iid", n_chunks=1, diag=False):
    """stats(params, s) -> (E, S2) solo forward (evaluacion), por trozos si n_chunks>1.
    diag=True -> (E, S2, P_near, P_far): P = P_near + P_far (purity nbr: parte exacta d<=1 y cola iid d>=2;
    purity iid: P_near=0, P_far=P)."""
    f_terms = make_mc_terms(model, N, T, hx, J, chunk, pair_block, purity)

    def fn(params, s):
        M = s.shape[0]
        if n_chunks <= 1:
            _, _, st = f_terms(params, s, None, M)
        else:
            assert pair_block is not None and M % (n_chunks * pair_block) == 0
            sch = s.reshape(n_chunks, M // n_chunks, N)
            dummy = (jnp.zeros(()), jnp.zeros(()), jnp.ones(()))

            def fwd(acc, sc):
                _, _, st_ = f_terms(params, sc, dummy, M)
                return tuple(a + x for a, x in zip(acc, st_)), None
            st, _ = jax.lax.scan(fwd, (jnp.zeros(()),) * 3, sch)
        S2 = -jnp.log(_purity_from_stats(st, purity))
        return (st[0], S2, st[1], st[2]) if diag else (st[0], S2)
    return fn


# ── cuantos trozos ─────────────────────────────────────────────────────────────

def bytes_per_sample(N, chi, grad=True):
    """Memoria de pico estimada por muestra del lote.  Ajuste lineal (con ~25% de margen) a la
    memory_analysis de XLA para N=8..12, chi=4..8:
        base = N*2*chi^2*16*(N+1)      (tensores MPS de s y sus N vecinos, complex128)
        grad:  10*base + 13e3*N^2      (forward + backward)
        fwd :  3.2*base + 1.35e3*N^2   (solo evaluacion)
    El segundo termino es la red (ARViT), que domina a chi pequeno.  Es una ESTIMACION: si dudas,
    fija sample_chunk a mano."""
    base = N * 2 * chi * chi * 16 * (N + 1)
    return 10.0 * base + 13e3 * N * N if grad else 3.2 * base + 1.35e3 * N * N


def auto_chunks(M, N, chi, pair_block, sample_chunk=None, mem_gb=None, grad=True):
    """Numero de trozos n (divisor de M/pair_block) para que cada trozo tenga <= sample_chunk muestras
    (explicito) o <= mem_gb GiB estimados.  Ambos None -> 1 (sin trocear)."""
    if pair_block is None or (sample_chunk is None and mem_gb is None):
        return 1
    lim = M
    if sample_chunk is not None:
        lim = min(lim, int(sample_chunk))
    if mem_gb is not None:
        lim = min(lim, int(mem_gb * 2 ** 30 / bytes_per_sample(N, chi, grad)))
    lim = max(lim, pair_block)
    if M <= lim:
        return 1
    u = M // pair_block                                     # numero de bloques
    n = -(-M // lim)
    while n < u and u % n:
        n += 1
    return min(n, u)