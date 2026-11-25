"""
src_renyi/mps_ancilla/sz_label.py
=================================
Etiqueta clásica de Sz en la ancilla:  φ_s(a, q) = φ_s(a) · δ_{q, Sz(s)}   ⇒   ⟨φ_t|φ_s⟩ → ⟨φ_t|φ_s⟩ · δ_{Sz(s), Sz(t)}.
ρ(s,t) = A_s A_t* ⟨φ̂_t|φ̂_s⟩ queda diagonal por bloques de Sz POR CONSTRUCCIÓN (el MRE también lo es: no se pierde el
objetivo), y los estimadores de la pureza restringidos al sector (purity_lowvar, pest u1) pasan a ser exactos para esta
familia.  p(s) = |A_s|² (ARViT) no cambia, ni la norma de φ_s (la etiqueta tiene norma 1), ni E (H conserva Sz).

Implementación: la carga q = (Σ_i s_i + N)/2 ∈ {0..N} se escribe en binario en L = ceil(log2(N+1)) sitios extra de la
ancilla, producto de tensores one-hot (2, χ, χ) con el único elemento [bit, 0, 0] = 1.  Así TODOS los solapes del código
(mps_overlap_aligned / rect / ρ exacta) incluyen el δ sin tocar nada.  Mismos parámetros que el modelo de partida.
OJO: con solape exactamente 0, log|⟨φ_t|φ_s⟩|² = −inf.  Solo vale con estimadores cuyas parejas estén siempre en el sector
de s (pest u1 sin cascarón); el inclinado o los cascarones con huecos inválidos darían nan en el gradiente.
"""
import math
import jax.numpy as jnp


class SzLabelModel:
    """Envoltorio de un modelo (ARViT + ancilla MPS): apply devuelve (logA, M) con L sitios de etiqueta añadidos."""

    def __init__(self, model, N):
        self._m = model
        self._N = N
        self.n_label = max(1, math.ceil(math.log2(N + 1)))

    def __getattr__(self, k):
        return getattr(self._m, k)

    def init(self, *a, **kw):
        return self._m.init(*a, **kw)

    def label_tensors(self, s, chi, dtype):
        q = jnp.round((jnp.sum(s, -1) + self._N) / 2).astype(jnp.int32)                  # (B,)
        bits = (q[:, None] >> jnp.arange(self.n_label)) & 1                                # (B, L)
        oh = jnp.stack([1 - bits, bits], -1).astype(dtype)                                 # (B, L, 2)
        T = jnp.zeros(oh.shape + (chi, chi), dtype)
        return T.at[..., 0, 0].set(oh)                                                     # (B, L, 2, χ, χ)

    def apply(self, variables, s, *a, **kw):
        out = self._m.apply(variables, s, *a, **kw)
        if kw.get("only_logits", False) or (a and a[0]):
            return out
        logA, M = out
        lab = self.label_tensors(s, M.shape[-1], M.dtype)
        return logA, jnp.concatenate([M, lab], axis=1)


def maybe_label(model, N, on):
    return SzLabelModel(model, N) if on else model
