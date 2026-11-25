"""
src_renyi/mps_ancilla/vmc_energy.py
===================================
Gradiente de la energía en "forma VMC" para ARViT + ancilla MPS (suma exacta sobre la ancilla).  La purificación
Ψ(s,a) = A(s) φ̂_s(a) es un estado puro del espacio doble y E = ⟨Ψ|H⊗1|Ψ⟩/⟨Ψ|Ψ⟩, así que
    ∇E = E_{s~p} [ 2 Re ⟨∇Ψ(s)| ((H − Ē) Ψ)(s)⟩ / ‖Ψ(s)‖² ],      Ψ(s) = vector de ancilla de la fila s,
con ⟨Ψ(s)|(HΨ)(s)⟩ = Σ_{s'} H_{ss'} conj(A(s)) A(s') ⟨φ̂_s|φ̂_{s'}⟩ (los mismos solapes que E_loc).  Sustituta por
muestra (derivada solo en el "bra" de s; todo lo de s' y el "ket" con stop-gradient):
    l(s) = 2 Re[ e^{conj(lA_s) − sg} ( Σ_{s'} H_{ss'} sg(A_{s'}/A_s) ⟨Φ_s|sg Φ_{s'}⟩ / (‖Φ_s‖ sg‖Φ_{s'}‖)
                                       − Ē ⟨Φ_s|sg Φ_s⟩ / (‖Φ_s‖ sg‖Φ_s‖) ) ]
Misma media que make_energy_terms (score + camino), menos varianza por muestra (notas.md, 25-11).  Misma firma:
    terms(params, s, Ebar, Mtot) -> (L_E, Σ Re E_loc / Mtot)
"""
import jax
import jax.numpy as jnp

from .models import mps_overlap_aligned
from .generic_eloc import _sign_map

sg = jax.lax.stop_gradient


def make_vmc_local(model, chi, H):
    """vmc_local(params, s, Ebar) -> (l(s) (B,), Re E_loc(s) (B,))"""
    conv = _sign_map(H)
    Hj = H.to_jax_operator()
    Hj.get_conn_padded(jnp.ones((1, H.hilbert.size)))

    def vmc_local(params, s, Ebar):
        s = sg(s)
        B, N = s.shape
        sp, mels = Hj.get_conn_padded(conv(s))
        C = sp.shape[1]
        sp = sp.reshape(B * C, N).astype(s.dtype)
        lA, Mt = model.apply({"params": params}, s)
        lAp, Mp = model.apply({"params": params}, sp)
        Mp = jax.tree_util.tree_map(sg, Mp); lAp = sg(lAp)
        n2 = jnp.real(mps_overlap_aligned(Mt, Mt, chi))                      # vivo (norma del bra)
        n2p = sg(jnp.real(mps_overlap_aligned(Mp, Mp, chi)))
        Mrep = jax.tree_util.tree_map(lambda x: jnp.repeat(x, C, axis=0), Mt)
        O = mps_overlap_aligned(Mp, Mrep, chi)                                # Σ_a sgΦ_{s'}(a) Φ_s(a)*
        G = O / jnp.sqrt(jnp.repeat(n2, C) * n2p)
        ratio = jnp.exp(lAp - sg(jnp.repeat(lA, C)))
        m = mels.reshape(-1)
        HPsi = jnp.sum(jnp.where(m != 0, m * ratio * G, 0.0).reshape(B, C), axis=1)
        Od = mps_overlap_aligned(jax.tree_util.tree_map(sg, Mt), Mt, chi)    # ⟨Φ_s|sg Φ_s⟩
        Gd = Od / jnp.sqrt(n2 * sg(n2))
        bra = jnp.exp(jnp.conj(lA) - sg(jnp.conj(lA)))
        l = 2.0 * jnp.real(bra * (HPsi - Ebar * Gd))
        return l, sg(jnp.real(HPsi))
    return vmc_local


def make_energy_terms_vmc(model, chi, H):
    vl = make_vmc_local(model, chi, H)

    def terms(params, s, Ebar, Mtot):
        l, ReE = vl(params, s, Ebar)
        return jnp.sum(l) / Mtot, jnp.sum(ReE) / Mtot
    return terms
