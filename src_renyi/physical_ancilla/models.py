import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
from netket.models import ARNNDense, AbstractARNN
from typing import Any


class ARNN_Z2(AbstractARNN):
    """Wrapper Z2 sobre la PROBABILIDAD, para un worker AR real.

    Construye la distribución simetrizada:

        P_Z2(s) = ½ [ P_w(s) + P_w(-s) ]

    que sigue siendo autoregresiva, normalizada y compatible con
    ARDirectSampler y con Wang CS.

    Solo válido para:
      - ansatz real (fase cero)
      - Hilbert con local_states simétricos bajo negación
        (p. ej. Spin(1/2) con {-1, +1})
    """
    worker: nn.Module
    machine_pow: int = 2

    def conditionals_log_psi(self, inputs):
        """log ψ_Z2(s_i | s_<i) por sitio. Devuelve (B, N, d)."""
        if inputs.ndim == 1:
            inputs = inputs[None]

        idx  = self.hilbert.states_to_local_indices(inputs)      # (B, N)
        idxn = self.hilbert.states_to_local_indices(-inputs)     # (B, N)

        tiny = jnp.finfo(jnp.float64).tiny
        lp   = jnp.log(self.worker.conditionals(inputs)  + tiny)  # log p_w(s_i|s_<i)
        lpn  = jnp.log(self.worker.conditionals(-inputs) + tiny)  # log p_w(-s_i|-s_<i)

        # Prefijos exclusivos (log P(s_<i), log P(-s_<i))
        g1 = jnp.take_along_axis(lp,  idx[..., None],  -1)[..., 0]
        g2 = jnp.take_along_axis(lpn, idxn[..., None], -1)[..., 0]
        pre1 = jnp.cumsum(g1, -1) - g1
        pre2 = jnp.cumsum(g2, -1) - g2

        # Para P(-s_i | -s_<i): invertimos el eje de valores del worker
        lpn_flip = lpn[..., ::-1]

        lognum = jnp.logaddexp(pre1[..., None] + lp,
                                pre2[..., None] + lpn_flip)
        logden = jnp.logaddexp(pre1, pre2)[..., None]

        # log ψ_Z2 condicional (machine_pow = 2 → |ψ|² = P_Z2)
        return 0.5 * (lognum - logden)

    # ── Métodos que el sampler usa explícitamente ─────────────────────────
    # Los sobrescribimos para forzar la delegación al conditionals_log_psi
    # del wrapper. La base AbstractARNN a veces asume estructura interna
    # del worker y no delega correctamente.

    def conditionals(self, inputs):
        """Probabilidades P_Z2(s_i | s_<i). Devuelve (B, N, d)."""
        return jnp.exp(self.machine_pow * self.conditionals_log_psi(inputs))

    def conditional(self, inputs, index):
        """log ψ_Z2(s_i | s_<i) en el sitio `index`. Devuelve (B, d)."""
        return self.conditionals_log_psi(inputs)[..., index, :]
    

class RBM_Z2(nn.Module):
    """
    Wrapper RBM con simetría Z2.

    """

    alpha: int = 1
    trivial_Z2: bool = False

    @nn.compact
    def __call__(self, x):
        worker = nk.models.RBM(alpha=self.alpha, param_dtype=float)

        # Log-amplitudes para x y su inverso
        output_x     = jnp.atleast_1d(worker(x))
        output_inv_x = jnp.atleast_1d(worker(-x))

        z2_stack = jnp.stack([output_x, output_inv_x], axis=0)

        if self.trivial_Z2:
            # log(e^ψ(x) + e^ψ(-x)) — sector simétrico
            res = jax.nn.logsumexp(z2_stack, axis=0)
        else:
            # log(e^ψ(x) - e^ψ(-x)) — sector antisimétrico
            b = jnp.array([1.0, -1.0])[:, None]
            res = jax.nn.logsumexp(z2_stack, b=b, axis=0)

        return res


class MODARNN(ARNNDense):
    """
    ARNNDense con normalización MOD.

    MOD(r_j) = r_j² / Σ_i r_i²

    A diferencia de _normalize de NetKet (que aplica softmax), esta clase
    eleva al cuadrado los logits crudos y normaliza sin exponenciar.
    Jreissaty et al., PRR 8, 013147 (2026), Ec. (10).
    """

    def conditionals_log_psi(self, inputs):
        inputs = self.reshape_inputs(inputs)
        x = jnp.expand_dims(inputs, axis=-1)

        for i in range(len(self._layers)):
            if i > 0 and hasattr(self, "activation"):
                x = self.activation(x)
            x = self._layers[i](x)

        x = x.reshape((x.shape[0], -1, x.shape[-1]))
        # x: (batch, N, 2) logits crudos

        # MOD: cuadrado + normalización
        r_sq = x ** 2
        Z = r_sq.sum(axis=-1, keepdims=True) + 1e-12
        log_p = jnp.log(r_sq + 1e-12) - jnp.log(Z)
        # Devolvemos log ψ = (1/2) log p, para que p = |ψ|²
        return 0.5 * log_p

class InterleavedARNNDense(ARNNDense):
    """
    Reordena la secuencia autoregresiva para intercalar sistema y ancilla:
    s_1, a_1, s_2, a_2, ..., s_N, a_N  en vez de  s_1,...,s_N, a_1,...,a_N.

    Cada sitio sigue siendo individual (dimensión local sin cambios),
    solo cambia el orden en que la MaskedDense1D los procesa.
    """
    def reorder(self, inputs, axis=0):
        # inputs: (..., N_S + N_A, ...) en orden [sistema, ancilla]
        N_S = self.hilbert.size // 2
        s_part = jnp.take(inputs, jnp.arange(N_S), axis=axis)
        a_part = jnp.take(inputs, jnp.arange(N_S, 2 * N_S), axis=axis)
        interleaved = jnp.stack([s_part, a_part], axis=axis + 1)
        return interleaved.reshape(
            inputs.shape[:axis] + (2 * N_S,) + inputs.shape[axis + 1:]
        )

    def inverse_reorder(self, inputs, axis=0):
        N_S = self.hilbert.size // 2
        reshaped = jnp.reshape(
            inputs,
            inputs.shape[:axis] + (N_S, 2) + inputs.shape[axis + 1:]
        )
        s_part = jnp.take(reshaped, 0, axis=axis + 1)
        a_part = jnp.take(reshaped, 1, axis=axis + 1)
        return jnp.concatenate([s_part, a_part], axis=axis)


class CausalTransformerBlock(nn.Module):
    n_heads: int
    n_ffn_layers: int
    embedding_d: int
    param_dtype: Any = jnp.float32 

    @nn.compact
    def __call__(self, x, mask):
        # Multi-Head Attention pasándole la máscara causal
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.embedding_d,
            out_features=self.embedding_d,
            param_dtype=self.param_dtype
        )(x, x, mask=mask)
        
        x = x + attn_out
        x = nn.LayerNorm(param_dtype=self.param_dtype)(x)

        # Feed-Forward Network (MLP)
        ffn = x
        for _ in range(self.n_ffn_layers):
            ffn = nn.Dense(features=self.embedding_d * 2, param_dtype=self.param_dtype)(ffn)
            ffn = nn.gelu(ffn)
        ffn = nn.Dense(features=self.embedding_d, param_dtype=self.param_dtype)(ffn)

        x = x + ffn
        x = nn.LayerNorm(param_dtype=self.param_dtype)(x)
        return x


class ARSpinViT_Causal(nk.models.AbstractARNN):
    embedding_d: int = 8
    n_heads: int = 2
    n_blocks: int = 2
    n_ffn_layers: int = 1
    machine_pow: int = 2
    param_dtype: Any = jnp.float32

    @nn.compact
    def conditionals_log_psi(self, inputs):
        
        batch_size, N = inputs.shape

        # 1. SHIFT CAUSAL
        zeros = jnp.zeros((batch_size, 1), dtype=inputs.dtype)
        x = jnp.concatenate([zeros, inputs[:, :-1]], axis=1)
        x = x.astype(self.param_dtype)[..., None]

        # 2. EMBEDDING INICIAL
        x = nn.Dense(features=self.embedding_d, name="embed", param_dtype=self.param_dtype)(x)
        
        pos_emb = self.param(
            'pos_embedding', 
            nn.initializers.normal(stddev=0.01), 
            (N, self.embedding_d),
            self.param_dtype
        )
        x = x + pos_emb

        # 3. MÁSCARA CAUSAL
        mask = nn.make_causal_mask(jnp.empty((batch_size, N)))

        # 4. PASO POR LOS BLOQUES TRANSFORMER
        for i in range(self.n_blocks):
            x = CausalTransformerBlock(
                n_heads=self.n_heads,
                n_ffn_layers=self.n_ffn_layers,
                embedding_d=self.embedding_d,
                param_dtype=self.param_dtype,
                name=f"causal_block_{i}" 
            )(x, mask)

        x = nn.LayerNorm(param_dtype=self.param_dtype)(x)

        logits = nn.Dense(
            features=2, 
            name="final_dense",
            kernel_init=nn.initializers.zeros, 
            param_dtype=self.param_dtype
        )(x)
        
        # 6. CONVERSIÓN A LOG-AMPLITUDES CUÁNTICAS
        log_probs = jax.nn.log_softmax(logits, axis=-1)            # real
        phase = nn.Dense(2, name="phase_dense",
                        kernel_init=nn.initializers.normal(0.05),   # NO zeros
                        param_dtype=self.param_dtype)(x)            # real
        return jax.lax.complex(0.5 * log_probs, phase)

class ARSpinViT_Causal_Interleaved(ARSpinViT_Causal):
    """
    Versión intercalada del Transformer causal.
    Procesa la secuencia como [s_0, a_0, s_1, a_1, ..., s_N, a_N]
    """
    def reorder(self, inputs, axis=0):
        N_S = self.hilbert.size // 2
        s_part = jnp.take(inputs, jnp.arange(N_S), axis=axis)
        a_part = jnp.take(inputs, jnp.arange(N_S, 2 * N_S), axis=axis)
        interleaved = jnp.stack([s_part, a_part], axis=axis + 1)
        return interleaved.reshape(
            inputs.shape[:axis] + (2 * N_S,) + inputs.shape[axis + 1:]
        )

    def inverse_reorder(self, inputs, axis=0):
        N_S = self.hilbert.size // 2
        reshaped = jnp.reshape(
            inputs,
            inputs.shape[:axis] + (N_S, 2) + inputs.shape[axis + 1:]
        )
        s_part = jnp.take(reshaped, 0, axis=axis + 1)
        a_part = jnp.take(reshaped, 1, axis=axis + 1)
        return jnp.concatenate([s_part, a_part], axis=axis)

    @nn.compact
    def conditionals_log_psi(self, inputs):
        # 1. REORDENAMOS LA ENTRADA
        inputs = self.reorder(inputs, axis=1)  # (batch, 2N)
        
        batch_size, N = inputs.shape

        # 2. SHIFT CAUSAL (Copia exacta del original)
        zeros = jnp.zeros((batch_size, 1), dtype=inputs.dtype)
        x = jnp.concatenate([zeros, inputs[:, :-1]], axis=1)
        x = x.astype(self.param_dtype)[..., None] 

        # 3. EMBEDDING INICIAL
        x = nn.Dense(features=self.embedding_d, name="embed", param_dtype=self.param_dtype)(x)
        
        pos_emb = self.param(
            'pos_embedding', 
            nn.initializers.normal(stddev=0.01), 
            (N, self.embedding_d),
            self.param_dtype
        )
        x = x + pos_emb

        # 4. MÁSCARA CAUSAL
        mask = nn.make_causal_mask(jnp.empty((batch_size, N)))

        # 5. BLOQUES TRANSFORMER
        for i in range(self.n_blocks):
            x = CausalTransformerBlock(
                n_heads=self.n_heads,
                n_ffn_layers=self.n_ffn_layers,
                embedding_d=self.embedding_d,
                param_dtype=self.param_dtype,
                name=f"causal_block_{i}" 
            )(x, mask)

        x = nn.LayerNorm(param_dtype=self.param_dtype)(x)

        logits = nn.Dense(
            features=2, 
            name="final_dense",
            kernel_init=nn.initializers.zeros, 
            param_dtype=self.param_dtype
        )(x)
        
        # log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        # 6. INVERTIMOS EL REORDENAMIENTO PARA LA SALIDA
        # out = jnp.asarray(0.5 * log_probs, dtype=self.param_dtype)
        # return self.inverse_reorder(out, axis=1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        phase = nn.Dense(2, name="phase_dense",
                        kernel_init=nn.initializers.normal(0.05),
                        param_dtype=self.param_dtype)(x)
        out = jax.lax.complex(0.5 * log_probs, phase)
        return self.inverse_reorder(out, axis=1)