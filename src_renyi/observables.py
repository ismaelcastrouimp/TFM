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

    def __init__(self, hilbert, H, partition, T, chunk_size=128):
        super().__init__(hilbert)
        self.H = H
        self.partition = partition
        self.T = T
        self.chunk_size = chunk_size

    @property
    def dtype(self):
        return float


@partial(jax.jit, static_argnames=("logpsi", "chunk_size"))
def _compute_E_loc_jit(logpsi, params, model_state, samples1, sigma_p, mels, chunk_size=128):
    def log_psi(p, s):
        return logpsi({"params": p, **model_state}, s)

    def e_loc_chunk(batch_sigma, batch_eta, batch_mel):
        # vmap dentro del chunk → paralelo en GPU
        def e_loc_single(sigma, eta, mel):
            lp_sigma = log_psi(params, sigma[None])[0]
            lp_eta = jax.vmap(lambda e: log_psi(params, e[None])[0])(eta)
            return jnp.sum(mel * jnp.exp(lp_eta - lp_sigma))
        return jax.vmap(e_loc_single)(batch_sigma, batch_eta, batch_mel)

    n = samples1.shape[0]
    n_chunks = n // chunk_size
    # reshape a (n_chunks, chunk_size, ...)
    s1_c    = samples1.reshape(n_chunks, chunk_size, -1)
    sp_c    = sigma_p.reshape(n_chunks, chunk_size, *sigma_p.shape[1:])
    mels_c  = mels.reshape(n_chunks, chunk_size, -1)

    E_loc_chunked = jax.lax.map(
        lambda x: e_loc_chunk(*x), (s1_c, sp_c, mels_c)
    )
    return E_loc_chunked.reshape(n)


@partial(jax.jit, static_argnames=("logpsi", "chunk_size"))
def _free_renyi_grad_jit(logpsi, params, model_state,
                          samples1, samples2, E_loc_sg,
                          swapped1, swapped2, T, chunk_size=128):
    def log_psi(p, s):
        return logpsi({"params": p, **model_state}, s)

    def loss_fn(p):
        E_mean = jnp.mean(E_loc_sg)
        lp_s1 = jax.vmap(lambda s: log_psi(p, s[None])[0])(samples1)
        loss_E = 2.0 * jnp.mean((E_loc_sg - E_mean) * jnp.real(lp_s1))

        n = samples1.shape[0]
        n_chunks = n // chunk_size
        def reshape(x):
            return x.reshape(n_chunks, chunk_size, *x.shape[1:])

        s1_c, s2_c, sw1_c, sw2_c = map(reshape, (samples1, samples2, swapped1, swapped2))

        @jax.checkpoint
        def renyi_chunk(bs1, bs2, bsw1, bsw2):
            def renyi_single(s1, s2, sw1, sw2):
                lo1 = log_psi(p, s1[None])[0]
                lo2 = log_psi(p, s2[None])[0]
                ls1 = log_psi(p, sw1[None])[0]
                ls2 = log_psi(p, sw2[None])[0]
                return lo1, lo2, ls1, ls2
            return jax.vmap(renyi_single)(bs1, bs2, bsw1, bsw2)

        lo1, lo2, ls1, ls2 = jax.lax.map(
            lambda x: renyi_chunk(*x), (s1_c, s2_c, sw1_c, sw2_c)
        )
        lo1, lo2, ls1, ls2 = [x.reshape(n) for x in (lo1, lo2, ls1, ls2)]

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
def expect_and_grad_free_renyi(vstate, op, chunk_size, **kwargs):
    subsystem_sites = jnp.array(op.partition, dtype=int)
    all_sites = jnp.arange(vstate.hilbert.size)
    complement_sites = jnp.setdiff1d(all_sites, subsystem_sites)

    all_samples = vstate.samples.reshape(-1, vstate.hilbert.size)
    n = all_samples.shape[0] // 2
    samples1, samples2 = all_samples[:n], all_samples[n:]

    swapped1 = jnp.concatenate([samples2[:, subsystem_sites], samples1[:, complement_sites]], axis=1)
    swapped2 = jnp.concatenate([samples1[:, subsystem_sites], samples2[:, complement_sites]], axis=1)

    # E_loc fuera del JIT principal para no materializar sigma_p en el grafo
    sigma_p, mels = op.H.get_conn_padded(samples1)
    E_loc = _compute_E_loc_jit(
        vstate._apply_fun, vstate.parameters, vstate.model_state,
        samples1, sigma_p, mels, op.chunk_size
    )
    E_loc_sg = jax.lax.stop_gradient(E_loc)

    E_mean, S2, grad_F = _free_renyi_grad_jit(
        vstate._apply_fun, vstate.parameters, vstate.model_state,
        samples1, samples2, E_loc_sg,
        swapped1, swapped2, op.T, op.chunk_size
    )

    F = float(E_mean) - op.T * float(S2)
    F_stats = nk.stats.statistics(jnp.array([[F]]))
    return F_stats, grad_F