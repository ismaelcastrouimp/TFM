import jax
import jax.numpy as jnp
import flax.linen as nn


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