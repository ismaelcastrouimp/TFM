"""
fast_ed.py
==========
Núcleo rápido para minimizar F = E - T*S2 con gradiente EXACTO (ED), sin NetKet.

Idea: el sistema son los N primeros espines y la ancilla los N últimos.
Con el orden de NetKet (primer espín = bit más significativo), psi.reshape(2^N, 2^N)
es la matriz Psi[sistema, ancilla]. Entonces

    rho_A = Psi Psi^†
    E     = Tr(H_sys rho_A)
    Tr rho^2 = sum |rho_ij|^2          (rho_A hermítica)
    F     = E - T * (-log Tr rho^2) = E + T log Tr rho^2

Todo sale de UNA evaluación del modelo sobre las 2^(2N) configuraciones y se
compila con jit (valor + gradiente + paso de optimizador).

Solo depende de jax / optax / numpy: no importa NetKet.
Requiere parámetros REALES (la salida del modelo puede ser compleja).
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)


def make_loss_fns(model, configs, H_sys, T, N, model_state=None, idx_sys=None, idx_anc=None,
                   chunk_size=None):
    """
    model    : módulo flax/NetKet con .apply({'params': ...}, configs) -> log psi (complejo o real)
    configs  : (2^(2N), 2N) todas las configuraciones (±1), en el orden que sea
    H_sys    : (2^N, 2^N) Hamiltoniano denso del sistema (N espines), con la numeración
               de estados del sistema que corresponda a idx_sys
    chunk_size : si no es None, evalúa el modelo por trozos con jax.lax.map + jax.checkpoint
               (reduce mucho la memoria de la GPU; el nº de configuraciones debe ser múltiplo).
    idx_sys, idx_anc : (2^(2N),) enteros: para cada configuración, índice del estado del
               sistema (configs[:, :N]) y de la ancilla (configs[:, N:]). Si se dan, Psi se
               construye con un scatter (no depende del orden de `configs`). Si no, se supone
               orden tipo itertools.product y se usa reshape.
    Devuelve (loss, rho_fn): loss(params) -> (F, (E, S2)); rho_fn(params) -> (rho_A, logpsi)
    """
    configs = jnp.asarray(configs)
    H_sys = jnp.asarray(H_sys)
    dA = dB = 2 ** N
    extra = dict(model_state) if model_state else {}
    use_scatter = idx_sys is not None
    if use_scatter:
        idx_sys, idx_anc = jnp.asarray(idx_sys), jnp.asarray(idx_anc)

    def _to_matrix(psi):
        if use_scatter:
            return jnp.zeros((dA, dB), dtype=psi.dtype).at[idx_sys, idx_anc].set(psi)
        return psi.reshape(dA, dB)

    def _logpsi(params):
        if chunk_size is None:
            lp = model.apply({"params": params, **extra}, configs)
        else:
            n = configs.shape[0]
            assert n % chunk_size == 0, "chunk_size debe dividir a 2^(2N)"
            cch = configs.reshape(n // chunk_size, chunk_size, -1)
            f = jax.checkpoint(lambda p, c: model.apply({"params": p, **extra}, c))
            lp = jax.lax.map(lambda c: f(params, c), cch).reshape(-1)
        return lp.astype(jnp.complex128)       # reducciones siempre en doble precisión

    def _psi(params):
        logpsi = _logpsi(params)
        # estabilidad numérica: restamos el máximo de la parte real antes de exponenciar
        psi = jnp.exp(logpsi - jnp.max(jnp.real(logpsi)))
        psi = psi / jnp.linalg.norm(psi)
        return psi, logpsi

    def _rho(params):
        psi, logpsi = _psi(params)
        Psi = _to_matrix(psi)                  # filas: sistema, columnas: ancilla
        rho = Psi @ jnp.conj(Psi).T
        return rho, logpsi

    def loss(params):
        rho, _ = _rho(params)
        E = jnp.real(jnp.sum(H_sys.T * rho))   # Tr(H rho)
        purity = jnp.sum(jnp.abs(rho) ** 2)    # Tr rho^2
        S2 = -jnp.log(purity)
        return E - T * S2, (E, S2)

    return loss, jax.jit(_rho)


def make_step(loss, tx):
    @jax.jit
    def step(params, opt_state):
        (F, (E, S2)), g = jax.value_and_grad(loss, has_aux=True)(params)
        updates, opt_state = tx.update(g, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, F, E, S2
    return step


def count_params(params):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def train(loss, params, n_steps, lr, T, *, freq=50, warmup=100, end_frac=1e-2,
          clip=100.0, diag_fn=None, verbose=True, tag=""):
    """
    Adam + clip global + warmup/coseno. Devuelve dict con el mejor estado y el historial.
    diag_fn(params) -> str  (texto extra cada `freq` pasos: fidelidad, signos, ...)
    """
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, n_steps, lr * end_frac)
    tx = optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched))
    opt_state = tx.init(params)
    step = make_step(loss, tx)

    hist = {"F": [], "E": [], "S2": []}
    best_F, best_params = np.inf, params
    t_start = time.time()
    for it in range(n_steps):
        cur = params                              # F, E, S2 corresponden a estos parámetros
        params, opt_state, F, E, S2 = step(params, opt_state)
        hist["F"].append(F); hist["E"].append(E); hist["S2"].append(S2)
        if verbose and it % freq == 0:
            Fv = float(F)
            msg = f"{tag}Step {it:5d} | F={Fv:.6f} | E={float(E):.6f} | S2={float(S2):.6f}"
            if diag_fn is not None:
                msg += " | " + diag_fn(cur)
            print(msg, flush=True)
        # mejor F: se resuelve al final con el historial (evita sincronizar cada paso)
        if it % 10 == 0 or it == n_steps - 1:
            Fv = float(F)
            if Fv < best_F:
                best_F, best_params = Fv, cur
    jax.block_until_ready(params)
    dt = time.time() - t_start

    # evaluación final exacta del mejor estado
    _, (E_b, S2_b) = loss(best_params)
    E_b, S2_b = float(E_b), float(S2_b)
    out = {
        "params_best": best_params, "params_last": params,
        "F": E_b - T * S2_b,
        "E": E_b, "S2": S2_b,
        "hist": {k: np.asarray(jnp.stack(v)) for k, v in hist.items()},
        "time": dt, "s_per_step": dt / n_steps,
    }
    return out