from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from netket.experimental.observable import AbstractObservable

from .entropy import renyi2_drut_sampling, renyi2_wang_cs, renyin_wang_cs
from .extrapolation import extrapolate_renyi_to_vN, svn_grad_from_Sn_estimators


class FreeRenyiEnergyObservable(AbstractObservable):
    """
    Observable F = E - T·S₂.

    Parámetros
    ----------
    chunk_size : int
        Chunk para la parte de ENTROPÍA (Wang / Drut / swap).
        NO se usa para la energía.
    chunk_size_energy : int or None
        Chunk específico para la energía. Si None, se usa `chunk_size`.
        Motivo: la energía evaluada en H.get_conn_padded produce un tensor
        grande (n_samples × nº_conexiones × N_sites). Ese tensor crece
        mucho más rápido que los de Wang/Drut, así que necesita un chunk
        más pequeño. Con chunk_size_energy controlas ese pico sin tocar
        el chunk de la entropía.

    method : "swap" | "drut" | "wang"
    """

    def __init__(self, hilbert, H, partition, T, chunk_size=128,
                 chunk_size_energy=None,
                 method="swap", drut_kwargs=None, drut_seed=0,
                 wang_kwargs=None, wang_seed=0, vstate_R=None):
        super().__init__(hilbert)
        if method not in ("swap", "drut", "wang"):
            raise ValueError(f"method must be 'swap', 'drut' or 'wang', got {method!r}")
        self.H = H
        self.partition = partition
        self.T = T
        self.chunk_size = int(chunk_size)
        self.chunk_size_energy = int(chunk_size_energy) if chunk_size_energy is not None else self.chunk_size
        self.method = method
        self.drut_kwargs = dict(drut_kwargs or {})
        self.drut_seed = drut_seed
        self._drut_step = 0
        self.wang_kwargs = dict(wang_kwargs or {})
        self.wang_seed = wang_seed
        self.vstate_R = vstate_R
        self._wang_step = 0


class FreeEnergyObservable(AbstractObservable):
    """
    Observable F_vN = E - T·S_vN.

    S_vN se estima a partir de S_2, S_3, ..., S_{n_max} vía extrapolación
    (por defecto cuadrática en n). El gradiente se obtiene por chain rule
    con diferencias finitas sobre los coeficientes de la extrapolación.

    Parámetros
    ----------
    H, partition, T : como en FreeRenyiEnergyObservable.
    chunk_size : int
        Chunk para la parte de entropía (Wang).
    chunk_size_energy : int or None
        Chunk para la energía.
    n_orders : list of int
        Órdenes de Rényi usados. Ejemplos:
          - [2, 3]          → extrapolación lineal en n (2 puntos)
          - [2, 3, 4]       → cuadrática en n (recomendado para gradiente)
          - [2, 3, 4, 5, 6] → Padé [2/2] posible
        Cuanto más largo, más preciso el valor pero más ruidoso el gradiente.
    n_samples_wang : int
        Muestras por cada S_n de Wang. OJO: se llama `renyi2_wang_cs`
        n_orders veces (una por orden), así que el coste por step es
        n_orders × n_samples_wang.
    vstate_R : MCState
        Red reversa (compartida con la fase Wang de Rényi-2, si la hay).
    wang_kwargs : dict
        Argumentos extra para `renyi2_wang_cs`.
    extrapolation_method : str
        Método de extrapolación a S_vN. 'quadratic_n' por defecto.
        Otros: 'linear_n', 'pade_2_2', etc.

    Notas
    -----
    - El GRADIENTE usa `quadratic_n` internamente por defecto. Cambiar a
      Padé para el gradiente degrada la calidad (‖c‖₁ grande → ruido).
    - El VALOR final reportado puede calcularse por separado con Padé
      para más precisión (ver `final_evaluation` abajo).
    """

    def __init__(self, hilbert, H, partition, T,
                 chunk_size=128, chunk_size_energy=None,
                 n_orders=None, n_samples_wang=2**15,
                 vstate_R=None, wang_kwargs=None, wang_seed=0,
                 extrapolation_method="quadratic_n"):
        super().__init__(hilbert)
        self.H = H
        self.partition = partition
        self.T = T
        self.chunk_size = int(chunk_size)
        self.chunk_size_energy = (
            int(chunk_size_energy) if chunk_size_energy is not None
            else self.chunk_size
        )
        self.n_orders = list(n_orders) if n_orders is not None else [2, 3, 4]
        self.n_samples_wang = int(n_samples_wang)
        self.wang_kwargs = dict(wang_kwargs or {})
        self.wang_seed = wang_seed
        self.vstate_R = vstate_R
        self.extrapolation_method = extrapolation_method
        self._wang_step = 0

# ── helpers chunked ───────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("logpsi", "chunk_size"))
def _compute_E_loc_jit(logpsi, params, model_state, samples, sigma_p, mels,
                       chunk_size=128):
    """E_loc(σ) para un batch, troceado en chunks de `chunk_size`."""
    def log_psi(p, s):
        return logpsi({"params": p, **model_state}, s)

    def e_loc_chunk(batch_sigma, batch_eta, batch_mel):
        def e_loc_single(sigma, eta, mel):
            lp_sigma = log_psi(params, sigma[None])[0]
            lp_eta = log_psi(params, eta)
            return jnp.sum(mel * jnp.exp(lp_eta - lp_sigma))
        return jax.vmap(e_loc_single)(batch_sigma, batch_eta, batch_mel)

    n = samples.shape[0]
    n_chunks = n // chunk_size
    s_c   = samples.reshape(n_chunks, chunk_size, -1)
    sp_c  = sigma_p.reshape(n_chunks, chunk_size, *sigma_p.shape[1:])
    mel_c = mels.reshape(n_chunks, chunk_size, -1)

    E_loc_chunked = jax.lax.map(
        lambda x: e_loc_chunk(*x), (s_c, sp_c, mel_c)
    )
    return E_loc_chunked.reshape(n)


def _energy_and_grad_chunked(vstate, op, chunk_size):
    """
    <H> y su gradiente, usando `chunk_size` explícitamente.
    """
    H = op.H
    all_s = vstate.samples.reshape(-1, vstate.hilbert.size)
    n = all_s.shape[0]
    cs = min(chunk_size, n)
    n_chunks = n // cs

    def _E_loc_chunk(s_batch):
        sigma_p, mels = H.get_conn_padded(s_batch)
        return _compute_E_loc_jit(
            vstate._apply_fun, vstate.parameters, vstate.model_state,
            s_batch, sigma_p, mels, chunk_size=cs,
        )

    E_loc = jnp.concatenate([
        _E_loc_chunk(all_s[i*cs:(i+1)*cs]) for i in range(n_chunks)
    ])
    E_loc_sg = jax.lax.stop_gradient(E_loc)

    # ── FIX: la energía reportada es la media de E_loc, NO el valor de la loss
    E_mean_true = float(jnp.mean(E_loc_sg))

    def loss_E(params):
        lp_s = jax.vmap(
            lambda s: vstate._apply_fun(
                {"params": params, **vstate.model_state}, s[None]
            )[0]
        )(all_s)
        E_avg = jnp.mean(E_loc_sg)                          # ← renombrado
        return 2.0 * jnp.mean((E_loc_sg - E_avg) * jnp.real(lp_s))

    _, grad_E = jax.value_and_grad(loss_E)(vstate.parameters)
    E_stats = nk.stats.statistics(jnp.array([[E_mean_true]]))
    return E_stats, grad_E


# ── rama swap (fusionada) ─────────────────────────────────────────────────────

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

        s1_c, s2_c, sw1_c, sw2_c = map(
            reshape, (samples1, samples2, swapped1, swapped2)
        )

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


# ── dispatch ──────────────────────────────────────────────────────────────────

@nk.vqs.expect_and_grad.dispatch
def expect_and_grad_free_renyi(vstate, op, chunk_size, **kwargs):
    """
    Dispatch unificado para FreeRenyiEnergyObservable (F = E - T·S₂)
    y FreeEnergyObservable (F_vN = E - T·S_vN).

    `chunk_size` posicional viene de NetKet y se ignora; la observable
    controla op.chunk_size y op.chunk_size_energy.
    """

    # ════════════════════════ GIBBS (F_vN) ════════════════════════
    if isinstance(op, FreeEnergyObservable):
        if op.vstate_R is None:
            raise RuntimeError("FreeEnergyObservable requires `vstate_R`.")

        E_stats, grad_E = _energy_and_grad_chunked(
            vstate, op, op.chunk_size_energy
        )

        S_n_list, grad_Sn_list = [], []
        for i, n in enumerate(op.n_orders):
            key = op.wang_seed + op._wang_step + i
            gibbs_wang_kwargs = {
                **op.wang_kwargs,
                "n_samples": op.n_samples_wang,
                "key": key,
                "chunk_size": op.chunk_size,
            }
            S_n, grad_S_n = renyin_wang_cs(
                vstate, op.vstate_R, op.partition, n, **gibbs_wang_kwargs
            )
            S_n_list.append(float(S_n))
            grad_Sn_list.append(grad_S_n)
        op._wang_step += len(op.n_orders)

        grad_SvN, _coefs = svn_grad_from_Sn_estimators(
            S_n_list, grad_Sn_list,
            method=op.extrapolation_method,
            n_orders=op.n_orders,
        )
        S_vN = extrapolate_renyi_to_vN(
            op.n_orders, np.array(S_n_list), op.extrapolation_method
        )

        grad_F = jax.tree_util.tree_map(
            lambda ge, gs: ge - op.T * gs, grad_E, grad_SvN
        )
        F = float(E_stats.mean.real) - op.T * S_vN
        F_stats = nk.stats.statistics(jnp.array([[F]]))
        return F_stats, grad_F

    # ════════════════════════ WANG (F_R = E - T·S₂) ════════════════════════
    if op.method == "wang":
        if op.vstate_R is None:
            raise RuntimeError("Wang method requires `vstate_R` in the observable.")

        E_stats, grad_E = _energy_and_grad_chunked(
            vstate, op, op.chunk_size_energy
        )
        wang_kwargs = {
            **op.wang_kwargs,
            "key": op.wang_seed + op._wang_step,
            "chunk_size": op.chunk_size,
        }
        S2, grad_S2 = renyi2_wang_cs(
            vstate, op.vstate_R, op.partition, **wang_kwargs
        )
        op._wang_step += 1

        grad_F = jax.tree_util.tree_map(
            lambda ge, gs: ge - op.T * gs, grad_E, grad_S2
        )
        F = float(E_stats.mean.real) - op.T * float(S2)
        F_stats = nk.stats.statistics(jnp.array([[F]]))
        return F_stats, grad_F

    # ════════════════════════ DRUT ════════════════════════
    if op.method == "drut":
        E_stats, grad_E = _energy_and_grad_chunked(
            vstate, op, op.chunk_size_energy
        )
        drut_kwargs = {**op.drut_kwargs, "key": op.drut_seed + op._drut_step}
        S2, grad_S2 = renyi2_drut_sampling(vstate, op.partition, **drut_kwargs)
        op._drut_step += 1

        grad_F = jax.tree_util.tree_map(
            lambda ge, gs: ge - op.T * gs, grad_E, grad_S2
        )
        F = float(E_stats.mean.real) - op.T * float(S2)
        F_stats = nk.stats.statistics(jnp.array([[F]]))
        return F_stats, grad_F

    
    # ════════════════════════ SWAP ════════════════════════
    if op.method == "swap":
        subsystem_sites = jnp.array(op.partition, dtype=int)
        all_sites = jnp.arange(vstate.hilbert.size)
        complement_sites = jnp.setdiff1d(all_sites, subsystem_sites)

        all_samples = vstate.samples.reshape(-1, vstate.hilbert.size)
        n = all_samples.shape[0] // 2
        samples1, samples2 = all_samples[:n], all_samples[n:]

        swapped1 = jnp.concatenate(
            [samples2[:, subsystem_sites], samples1[:, complement_sites]], axis=1
        )
        swapped2 = jnp.concatenate(
            [samples1[:, subsystem_sites], samples2[:, complement_sites]], axis=1
        )

        # Energía con chunk_size_energy; S₂ (swap) con chunk_size
        cs_E = min(op.chunk_size_energy, n)
        n_chunks_E = n // cs_E

        def compute_E_loc_chunk(samples1_chunk):
            sigma_p_chunk, mels_chunk = op.H.get_conn_padded(samples1_chunk)
            return _compute_E_loc_jit(
                vstate._apply_fun, vstate.parameters, vstate.model_state,
                samples1_chunk, sigma_p_chunk, mels_chunk, chunk_size=cs_E,
            )

        E_loc = jnp.concatenate([
            compute_E_loc_chunk(samples1[i*cs_E:(i+1)*cs_E])
            for i in range(n_chunks_E)
        ])
        E_loc_sg = jax.lax.stop_gradient(E_loc)

        cs_S = min(op.chunk_size, n)
        E_mean, S2, grad_F = _free_renyi_grad_jit(
            vstate._apply_fun, vstate.parameters, vstate.model_state,
            samples1[:n_chunks_E*cs_E], samples2[:n_chunks_E*cs_E],
            E_loc_sg,
            swapped1[:n_chunks_E*cs_E], swapped2[:n_chunks_E*cs_E],
            op.T, cs_S,
        )

        F = float(E_mean) - op.T * float(S2)
        F_stats = nk.stats.statistics(jnp.array([[F]]))
        return F_stats, grad_F

    raise ValueError(
        f"expect_and_grad_free_renyi: método no soportado "
        f"({getattr(op, 'method', type(op).__name__)!r})"
    )
