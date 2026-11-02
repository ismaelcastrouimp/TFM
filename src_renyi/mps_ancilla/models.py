# src_renyi/mps_ancilla.py
"""
ARViT + MPS ancilla: modelo y helpers de contracción.
No depende de NetKet. Solo JAX / Flax / NumPy.
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

# Aseguramos float64 (idempotente — seguro si se llama varias veces)
jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


class CausalBlock(nn.Module):
    n_heads: int
    embedding_d: int
    n_ffn_layers: int = 1

    @nn.compact
    def __call__(self, x, mask):
        a = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.embedding_d,
            out_features=self.embedding_d, param_dtype=F64)(x, x, mask=mask)
        x = nn.LayerNorm(param_dtype=F64)(x + a)
        f = x
        for _ in range(self.n_ffn_layers):
            f = nn.gelu(nn.Dense(2 * self.embedding_d, param_dtype=F64)(f))
        f = nn.Dense(self.embedding_d, param_dtype=F64)(f)
        return nn.LayerNorm(param_dtype=F64)(x + f)


class ARViT_MPSAncilla(nn.Module):
    N_sys: int
    N_anc: int
    chi: int = 2
    embedding_d: int = 16
    n_heads: int = 4
    n_blocks: int = 2
    complex_mps: bool = False
    phase_init: float = 0.05     # escala de la cabeza de fase del sistema
    mps_init: float = 0.3        # escala de la perturbacion de los tensores respecto de 1/sqrt(2)*I
    copy_init: bool = False      # init "la ancilla copia al sistema": rho = I/2^N (S2 maximo), A(s) uniforme

    @nn.compact
    def __call__(self, s, only_logits=False):
        """s: (B, N_sys) con valores +-1.  Devuelve (logA complejo (B,), M complejo (B,K,2,chi,chi)).
        only_logits=True devuelve solo los logits (B, N_sys, 2) de las condicionales (para muestrear)."""
        B, N, K, chi, d = s.shape[0], self.N_sys, self.N_anc, self.chi, self.embedding_d
        s = s.astype(F64)
        L = N + 1
        tok = jnp.concatenate([jnp.zeros((B, 1), F64), s], axis=1)            # BOS, s_0..s_{N-1}
        x = nn.Dense(d, param_dtype=F64)(tok[..., None])
        x = x + self.param("pos", nn.initializers.normal(0.01), (L, d), F64)
        mask = jnp.tril(jnp.ones((L, L), dtype=bool))[None, None]
        for i in range(self.n_blocks):
            x = CausalBlock(self.n_heads, d, name=f"blk{i}")(x, mask)
        x = nn.LayerNorm(param_dtype=F64)(x)

        # ---- sistema: posicion i predice s_i (ve BOS, s_0..s_{i-1}) ----
        xs = x[:, :N]
        logits = nn.Dense(2, param_dtype=F64, name="amp_head",
                          kernel_init=nn.initializers.zeros if self.copy_init
                          else nn.initializers.lecun_normal())(xs)
        if only_logits:
            return logits
        phase = nn.Dense(2, param_dtype=F64, name="phase_head",
                         kernel_init=nn.initializers.normal(self.phase_init))(xs)
        idx = ((s + 1) / 2).astype(jnp.int32)[..., None]                      # +1 -> 1, -1 -> 0
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits, -1), idx, -1)[..., 0]
        ph = jnp.take_along_axis(phase, idx, -1)[..., 0]
        logA = jax.lax.complex(0.5 * jnp.sum(logp, 1), jnp.sum(ph, 1))        # (B,)

        # ---- ancilla: tensores MPS condicionados al contexto (posicion N, ve todo s) ----
        ctx = x[:, N]
        n_out = K * 2 * chi * chi * (2 if self.complex_mps else 1)
        eye = jnp.broadcast_to(jnp.eye(chi, dtype=F64) * (0.5 if self.copy_init else 1 / np.sqrt(2.0)),
                               (K, 2, chi, chi))
        bias0 = jnp.concatenate([eye.reshape(-1)] + ([jnp.zeros(K * 2 * chi * chi, F64)] if self.complex_mps else []))
        W = nn.Dense(n_out, param_dtype=F64, name="mps_head",
                     kernel_init=nn.initializers.normal(self.mps_init / np.sqrt(d)),
                     bias_init=lambda *_: bias0)(ctx)
        if self.copy_init:
            # M_k^a(s) = (1 + (2a-1) s_k)/2 * I  =  delta(a, s_k) I   ->  Phi_s = |s_1 ... s_K>  (ancilla copia a s)
            kern = np.zeros((N, n_out))
            for k in range(min(K, N)):
                for a in (0, 1):
                    for i in range(chi):
                        kern[k, k * 2 * chi * chi + a * chi * chi + i * chi + i] = 0.5 * (2 * a - 1)
            W = W + s @ self.param("mps_copy", lambda *_: jnp.asarray(kern), (N, n_out), F64)
        if self.complex_mps:
            re, im = jnp.split(W, 2, axis=-1)
            M = jax.lax.complex(re, im).reshape(B, K, 2, chi, chi)
        else:
            M = W.reshape(B, K, 2, chi, chi).astype(jnp.complex128)
        return logA, M
    
# ─── Helpers de contracción MPS ────────────────────────────────────────────

def mps_overlap_rect(M1, M2, chi, chunk=None):
    """M1: (S,K,2,chi,chi), M2: (R,K,2,chi,chi).  O[s,t] = sum_a Phi_s(a) conj(Phi_t(a)) (S,R),
    Phi_s(a) = e0^T M_1^{a1}(s) ... M_K^{aK}(s) e0.  chunk: filas por trozo (lax.map, ahorra memoria)."""
    S, R = M1.shape[0], M2.shape[0]
    Mk2 = jnp.moveaxis(M2, 1, 0)

    def block(M1b):
        Sb = M1b.shape[0]
        Mk1 = jnp.moveaxis(M1b, 1, 0)                                       # (K, Sb, 2, chi, chi)
        v0 = jnp.zeros((Sb, R, chi, chi), jnp.complex128).at[:, :, 0, 0].set(1.0)

        def step(v, ms):
            m1, m2 = ms
            X = jnp.einsum("stim,saij->stajm", v, m1)                       # contrae i con M(s)
            return jnp.einsum("stajm,tamn->stjn", X, jnp.conj(m2)), None    # contrae a,m con conj M(t)

        v, _ = jax.lax.scan(jax.checkpoint(step), v0, (Mk1, Mk2))
        return v[:, :, 0, 0]

    if chunk is None or chunk >= S:
        return block(M1)
    assert S % chunk == 0, "chunk debe dividir a S"
    out = jax.lax.map(jax.checkpoint(block), M1.reshape(S // chunk, chunk, *M1.shape[1:]))
    return out.reshape(S, R)


def mps_overlap_aligned(M1, M2, chi):
    """Solapes por parejas alineadas: O[p] = <Phi_{M2[p]} | Phi_{M1[p]}>-style, es decir
    sum_a Phi_1(a) conj(Phi_2(a)).  M1, M2: (P,K,2,chi,chi).  Devuelve (P,)."""
    P = M1.shape[0]
    v0 = jnp.zeros((P, chi, chi), jnp.complex128).at[:, 0, 0].set(1.0)

    def step(v, ms):
        m1, m2 = ms
        X = jnp.einsum("pim,paij->pajm", v, m1)
        return jnp.einsum("pajm,pamn->pjn", X, jnp.conj(m2)), None

    v, _ = jax.lax.scan(jax.checkpoint(step), v0, (jnp.moveaxis(M1, 1, 0), jnp.moveaxis(M2, 1, 0)))
    return v[:, 0, 0]


def mps_overlap_matrix(M, chi):
    return mps_overlap_rect(M, M, chi)


def count_params(params):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))

