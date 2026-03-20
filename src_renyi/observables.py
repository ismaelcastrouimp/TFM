from functools import partial

import jax
import jax.numpy as jnp
import netket as nk
from netket.experimental.observable import AbstractObservable


class FreeRenyiEnergyObservable(AbstractObservable):
    """
    Observable F = E - T·S₂ cuyo gradiente se computa de forma fusionada
    en un único jit, reutilizando las muestras del vstate.
    """

    def __init__(self, hilbert, H, partition, T):
        super().__init__(hilbert)
        self.H = H
        self.partition = partition
        self.T = T

    @property
    def dtype(self):
        return float


@partial(jax.jit, static_argnames=("logpsi",))
def _free_renyi_grad_jit(logpsi, params, model_state,
                          samples1, samples2, sigma_p, mels,
                          subsystem_sites, complement_sites, T):
    def log_psi(p, s):
        return logpsi({"params": p, **model_state}, s)

    swapped1 = jnp.concatenate([samples2[:, subsystem_sites], samples1[:, complement_sites]], axis=1)
    swapped2 = jnp.concatenate([samples1[:, subsystem_sites], samples2[:, complement_sites]], axis=1)

    def loss_fn(p):
        # Energía local
        @partial(jax.vmap, in_axes=(0, 0, 0))
        def e_loc_p(sigma, eta, mel):
            return jnp.sum(mel * jnp.exp(log_psi(p, eta) - log_psi(p, sigma)))

        E_loc = e_loc_p(samples1, sigma_p, mels)
        E_mean = jax.lax.stop_gradient(jnp.mean(E_loc))
        loss_E = 2.0 * jnp.mean(
            jax.lax.stop_gradient(E_loc - E_mean) * jnp.real(log_psi(p, samples1))
        )

        # Rényi-2
        lo1 = log_psi(p, samples1)
        lo2 = log_psi(p, samples2)
        ls1 = log_psi(p, swapped1)
        ls2 = log_psi(p, swapped2)
        R = jnp.exp(jnp.real(ls1 + ls2 - lo1 - lo2))
        R_mean = jax.lax.stop_gradient(jnp.mean(R))
        w = jax.lax.stop_gradient(R / R_mean)
        loss_S2 = (
            -2.0 * jnp.mean(w * jnp.real(ls1 + ls2))
            + 2.0 * jnp.mean(jnp.real(lo1 + lo2))
        )

        S2 = jax.lax.stop_gradient(-jnp.log(jnp.abs(R_mean)))
        return loss_E - T * loss_S2, (E_mean, S2)

    (_, (E_mean, S2)), grad_F = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return E_mean, S2, grad_F


@nk.vqs.expect_and_grad.dispatch
def expect_and_grad_free_renyi(
    vstate: nk.vqs.MCState,
    op: FreeRenyiEnergyObservable,
    chunk_size,
    **kwargs,
):
    subsystem_sites = jnp.array(op.partition, dtype=int)
    all_sites = jnp.arange(vstate.hilbert.size)
    complement_sites = jnp.setdiff1d(all_sites, subsystem_sites)

    samples1 = vstate.samples.reshape(-1, vstate.hilbert.size)
    samples2 = vstate.sample(n_samples=vstate.n_samples).reshape(-1, vstate.hilbert.size)
    sigma_p, mels = op.H.get_conn_padded(samples1)

    E_mean, S2, grad_F = _free_renyi_grad_jit(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        samples1, samples2, sigma_p, mels,
        subsystem_sites, complement_sites,
        op.T,
    )

    F = float(E_mean) - op.T * float(S2)
    F_stats = nk.stats.statistics(jnp.array([[F]]))
    return F_stats, grad_F