import jax
import jax.numpy as jnp
import flax.linen as nn
import netket as nk


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