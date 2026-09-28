import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk
from netket.models import ARNNDense

class ARNN_Z2(nn.Module):
    """
    Wrapper autoregresivo con simetría Z2.
    Toma un módulo ARNN (e.g. ARNNDense) como worker y proyecta
    la función de onda sobre el sector simétrico (trivial_Z2=True)
    o antisimétrico (trivial_Z2=False).
    """

    worker: nn.Module
    trivial_Z2: bool = False

    def conditional(self, inputs, index):
        return self.worker.conditional(inputs, index)

    def reorder(self, inputs, axis=-1):
        return self.worker.reorder(inputs, axis=axis)

    def inverse_reorder(self, inputs, axis=-1):
        return self.worker.inverse_reorder(inputs, axis=axis)

    def __call__(self, x):
        output_x = jnp.atleast_1d(self.worker(x))
        output_inv_x = jnp.atleast_1d(self.worker(-x))

        z2_stack = jnp.stack([output_x, output_inv_x], axis=0)

        if self.trivial_Z2:
            res = jax.nn.logsumexp(z2_stack, axis=0)
        else:
            b = jnp.array([1.0, -1.0])[:, None]
            res = jax.nn.logsumexp(z2_stack, b=b, axis=0)

        return res
    

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

    @nn.compact
    def __call__(self, x, mask):
        # Multi-Head Attention pasándole la máscara causal
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.embedding_d,
            out_features=self.embedding_d,
            param_dtype=jnp.float64
        )(x, x, mask=mask)
        
        x = x + attn_out
        x = nn.LayerNorm(param_dtype=jnp.float64)(x)

        # Feed-Forward Network (MLP)
        ffn = x
        for _ in range(self.n_ffn_layers):
            ffn = nn.Dense(features=self.embedding_d * 2, param_dtype=jnp.float64)(ffn)
            ffn = nn.gelu(ffn)
        ffn = nn.Dense(features=self.embedding_d, param_dtype=jnp.float64)(ffn)

        x = x + ffn
        x = nn.LayerNorm(param_dtype=jnp.float64)(x)
        return x

class ARSpinViT_Causal(nk.models.AbstractARNN):
    embedding_d: int = 8
    n_heads: int = 2
    n_blocks: int = 2
    n_ffn_layers: int = 1
    machine_pow: int = 2

    @nn.compact
    def conditionals_log_psi(self, inputs):
        
        batch_size, N = inputs.shape

        # 1. SHIFT CAUSAL
        zeros = jnp.zeros((batch_size, 1), dtype=inputs.dtype)
        x = jnp.concatenate([zeros, inputs[:, :-1]], axis=1)
        x = x.astype(jnp.float64)[..., None] 

        # 2. EMBEDDING INICIAL
        x = nn.Dense(features=self.embedding_d, name="embed", param_dtype=jnp.float64)(x)
        
        pos_emb = self.param(
            'pos_embedding', 
            nn.initializers.normal(stddev=0.01), 
            (N, self.embedding_d),
            jnp.float64
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
                name=f"causal_block_{i}" 
            )(x, mask)

        x = nn.LayerNorm(param_dtype=jnp.float64)(x)

        logits = nn.Dense(
            features=2, 
            name="final_dense",
            kernel_init=nn.initializers.zeros, 
            param_dtype=jnp.float64
        )(x)
        
        # 6. CONVERSIÓN A LOG-AMPLITUDES CUÁNTICAS
        # La red devuelve P(x). La amplitud cuántica es sqrt(P(x)).
        # En el espacio logarítmico: log(sqrt(P)) = 0.5 * log(P)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return jnp.asarray(0.5 * log_probs, dtype=jnp.float64)