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