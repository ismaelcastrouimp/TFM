"""
src_renyi/mps_ancilla/generic_eloc.py
=====================================
Energía local GENÉRICA para ARViT + ancilla MPS (suma exacta sobre la ancilla), para cualquier hamiltoniano de espines
dado como operador de NetKet:

    E_loc(s) = Σ_{s'} H_{s s'} · A(s')/A(s) · ⟨Φ_s|Φ_{s'}⟩ / (‖Φ_s‖ ‖Φ_{s'}‖)

(con s' = s el término diagonal, G = 1).  Las conexiones (s', H_{ss'}) salen de get_conn_padded del operador jax, así que
sirve para TFIM, XXZ, J1-J2, ... sin escribir los giros a mano.  Convención (la de robust_ed.py): Paulis, s_k = ±1
es el autovalor de Z_k.

  build_hamiltonian(name, N, **kw)  -> operador NetKet (Paulis) en Spin(1/2, N)
  make_eloc(model, chi, H)          -> eloc(params, s) (complejo, (B,))
  make_energy_terms(model, chi, H)  -> terms(params, s, Ebar, Mtot) -> (L_E, sum ReE / Mtot)
        L_E = Σ [sg(ReE − Ebar) ln p(s) + ReE] / Mtot   (mismo estimador que mc_loss: score + camino; la suma sobre
        trozos da la pérdida del lote completo)
"""
import numpy as np
import jax
import jax.numpy as jnp

from .models import mps_overlap_aligned

sg = jax.lax.stop_gradient


def build_hamiltonian(name, N, obc=False, J=None, h=-1.5, hz=-0.5, Delta=1.0, J2=0.5, h_i=None):
    """Mismos modelos que robust_ed.py (Paulis):
      tfim     J Σ Z_i Z_{i+1} + h Σ X_i                   (J = -1 por defecto)
      tfim_hz  tfim + hz Σ Z_i
      tfim_dis J Σ ZZ + Σ h_i X_i  (h_i dado)
      xxz      J Σ (XX + YY + Δ ZZ)                         (J = +1 por defecto)
      j1j2     J Σ_{i,i+1} (XX+YY+ZZ) + J2 J Σ_{i,i+2} (XX+YY+ZZ)
    """
    import netket as nk
    from netket.operator.spin import sigmax, sigmay, sigmaz
    hi = nk.hilbert.Spin(1 / 2, N)
    if J is None:
        J = -1.0 if name.startswith("tfim") else 1.0
    nn1 = [(i, (i + 1) % N) for i in range(N if not obc else N - 1)]
    nn2 = [(i, (i + 2) % N) for i in range(N if not obc else N - 2)]

    def heis(bonds, Jb, D=1.0):
        return sum(Jb * (sigmax(hi, i) @ sigmax(hi, j) + sigmay(hi, i) @ sigmay(hi, j) + D * sigmaz(hi, i) @ sigmaz(hi, j))
                   for i, j in bonds)
    if name in ("tfim", "tfim_hz", "tfim_dis"):
        H = sum(J * sigmaz(hi, i) @ sigmaz(hi, j) for i, j in nn1)
        hs = h_i if name == "tfim_dis" else [h] * N
        H = H + sum(hs[i] * sigmax(hi, i) for i in range(N))
        if name == "tfim_hz":
            H = H + sum(hz * sigmaz(hi, i) for i in range(N))
    elif name == "xxz":
        H = heis(nn1, J, Delta)
    elif name == "j1j2":
        H = heis(nn1, J) + heis(nn2, J2 * J)
    else:
        raise ValueError(name)
    return H


def _sign_map(H):
    """s_k = ±1 (autovalor de Z) -> valores locales de NetKet y vuelta.  NetKet ordena local_states como [1, -1] o [-1, 1]
    según la versión, pero los VALORES son los autovalores de σ^z en ambos casos; se comprueba."""
    import netket as nk
    from netket.operator.spin import sigmaz
    hi = nk.hilbert.Spin(1 / 2, 2)                  # misma convención local; se comprueba en pequeño (2^N denso no cabe)
    assert H.hilbert.local_states is not None and np.allclose(H.hilbert.local_states, hi.local_states)
    z = sigmaz(hi, 0).to_dense()
    st = np.asarray(hi.all_states())
    assert np.allclose(np.diag(z), st[:, 0]), "convención: σ^z debe ser diagonal con el valor local"
    return lambda s: s


def make_eloc(model, chi, H):
    """eloc(params, s): E_loc(s) para s (B, N) con valores ±1 (= autovalores de Z)."""
    conv = _sign_map(H)
    Hj = H.to_jax_operator()
    Hj.get_conn_padded(jnp.ones((1, H.hilbert.size)))   # inicializa la caché interna fuera de jit (si no, fuga de tracer)

    def eloc(params, s):
        B, N = s.shape
        sp, mels = Hj.get_conn_padded(conv(s))                      # (B, C, N), (B, C)
        C = sp.shape[1]
        sp = sp.reshape(B * C, N).astype(s.dtype)
        lA, Mt = model.apply({"params": params}, s)
        lAp, Mp = model.apply({"params": params}, sp)
        n2 = jnp.real(mps_overlap_aligned(Mt, Mt, chi))
        n2p = jnp.real(mps_overlap_aligned(Mp, Mp, chi))
        Mrep = jax.tree_util.tree_map(lambda x: jnp.repeat(x, C, axis=0), Mt)
        O = mps_overlap_aligned(Mp, Mrep, chi)                        # Σ_a Φ_{s'}(a) Φ_s(a)*  = ⟨Φ_s|Φ_{s'}⟩
        G = O / jnp.sqrt(n2p * jnp.repeat(n2, C))
        ratio = jnp.exp(lAp - jnp.repeat(lA, C))
        w = jnp.where(mels.reshape(-1) != 0, mels.reshape(-1) * ratio * G, 0.0)
        return jnp.sum(w.reshape(B, C), axis=1)
    return eloc


def make_energy_terms(model, chi, H):
    eloc = make_eloc(model, chi, H)

    def terms(params, s, Ebar, Mtot):
        s = sg(s)
        ReE = jnp.real(eloc(params, s))
        logp = 2.0 * jnp.real(model.apply({"params": params}, s)[0])
        L_E = jnp.sum(sg(ReE - Ebar) * logp + ReE) / Mtot
        return L_E, jnp.sum(ReE) / Mtot
    return terms
