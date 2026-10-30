"""
src_renyi/mps_ancilla/ladder.py
===============================
Utilidades para la ESCALERA DE chi del modelo ARViT + MPS ancilla:

  * build_model / init_params     -> construyen el modelo a un chi dado (siempre copy_init=True)
  * pad_chi                       -> amplia los parametros de chi a chi2 (ceros + ruido en bloques nuevos);
                                     con ruido=0 el estado (y F) no cambia
  * save_params / load_params     -> msgpack de flax (escritura atomica: un job que espera nunca lee a medias)
  * make_evaluator / evaluate     -> F, E, S2 por MC con barra de error (lotes independientes)
  * update_results_json_locked    -> JSON de resultados con lock entre procesos (varios jobs SLURM a la vez)

Solo depende de JAX / Flax / NumPy (+ models.py y training.py de este paquete).
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization as serialization
from flax.core import unfreeze

try:                                   # fcntl no existe en Windows
    import fcntl
except ImportError:                    # pragma: no cover
    fcntl = None

from .models import ARViT_MPSAncilla
from .training import make_sampler, make_mc_loss


# ── modelo ─────────────────────────────────────────────────────────────────────

def build_model(N, chi, d=16, heads=4, blocks=2, mps_init=0.05):
    """copy_init=True SIEMPRE: asi la estructura de parametros es la misma para todos los chi
    (incluye 'mps_copy') y la escalera puede ampliar de un chi al siguiente."""
    return ARViT_MPSAncilla(N_sys=N, N_anc=N, chi=chi, embedding_d=d, n_heads=heads,
                            n_blocks=blocks, copy_init=True, mps_init=mps_init)


def init_params(model, N, seed=0):
    return model.init(jax.random.PRNGKey(seed), jnp.ones((1, N)))["params"]


# ── ampliar chi ────────────────────────────────────────────────────────────────

def pad_chi(params, chi, chi2, K, noise_std=0.0, seed=0):
    """Amplia los parametros que generan los tensores del MPS de chi a chi2.
    Los bloques nuevos son 0 (bias y mps_copy) y 0 + ruido N(0, noise_std) (kernel de mps_head).
    Con noise_std=0 el estado es EXACTAMENTE el mismo => F no cambia.  K = N_anc (sitios del MPS)."""
    assert chi2 >= chi
    rng = np.random.default_rng(seed)
    new_block = np.ones((chi2, chi2), bool)
    new_block[:chi, :chi] = False

    def pad_last(a, add_noise):
        a = np.asarray(a)
        lead = a.shape[:-1]
        a4 = a.reshape(lead + (K, 2, chi, chi))
        out = np.zeros(lead + (K, 2, chi2, chi2), a.dtype)
        out[..., :chi, :chi] = a4
        if add_noise and noise_std > 0:
            out = out + rng.normal(0.0, noise_std, size=out.shape) * new_block
        return out.reshape(lead + (K * 2 * chi2 * chi2,))

    p = jax.tree_util.tree_map(np.asarray, unfreeze(params))
    p["mps_head"]["kernel"] = pad_last(p["mps_head"]["kernel"], True)
    p["mps_head"]["bias"] = pad_last(p["mps_head"]["bias"], False)
    if "mps_copy" in p:
        p["mps_copy"] = pad_last(p["mps_copy"], False)
    return jax.tree_util.tree_map(jnp.asarray, p)


# ── guardar / cargar ───────────────────────────────────────────────────────────

def save_params(path, params):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(serialization.to_bytes(params))
    os.replace(tmp, path)              # atomico


def load_params(path, model, N):
    target = init_params(model, N)
    with open(path, "rb") as f:
        return serialization.from_bytes(target, f.read())


# ── evaluacion MC final ────────────────────────────────────────────────────────

def make_evaluator(model, N, T, hx, J, M, pair_block):
    """one(params, key) -> (E, S2) de UN lote de M muestras (sin gradiente)."""
    sampler = make_sampler(model, N, M)
    mc_loss = make_mc_loss(model, N, T, hx, J, None, pair_block)

    @jax.jit
    def one(params, key):
        s = sampler(params, key)
        _, (E, S2, _F) = mc_loss(params, s)
        return E, S2
    return one


def evaluate(one, params, T, n_batches, seed=0):
    """Promedia n_batches lotes independientes.  S2 = -log(media de P) (P es el estimador insesgado;
    -log P de un lote suelto esta sesgado hacia arriba).  El error de F es std de los lotes / sqrt(n)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n_batches)
    Es, Ss = [], []
    for k in keys:
        E, S2 = one(params, k)
        Es.append(float(E)); Ss.append(float(S2))
    Es, Ss = np.array(Es), np.array(Ss)
    E = Es.mean()
    S2 = -np.log(np.mean(np.exp(-Ss)))
    Fb = Es - T * Ss
    sq = np.sqrt(max(n_batches, 1))
    ddof = 1 if n_batches > 1 else 0
    return {"energy": float(E), "entropy": float(S2), "free_energy": float(E - T * S2),
            "energy_err": float(Es.std(ddof=ddof) / sq), "entropy_err": float(Ss.std(ddof=ddof) / sq),
            "free_energy_err": float(Fb.std(ddof=ddof) / sq)}


# ── JSON de resultados (con lock) ──────────────────────────────────────────────

def _extend(lst, n):
    while len(lst) < n:
        lst.append(None)


def read_results(results_file):
    if not os.path.exists(results_file):
        return None
    try:
        with open(results_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_result(data, chi, key, idx):
    """data['by_chi'][str(chi)][key][idx] o None si no existe."""
    try:
        return data["by_chi"][str(chi)][key][idx]
    except (TypeError, KeyError, IndexError):
        return None


def update_results_json_locked(results_file, idx, T, chi, entry, N=None):
    """Read-modify-write con lock entre procesos.  Estructura:
       {"N":..., "T":[...], "by_chi": {"4": {"energy":[...], "free_energy":[...], ...}, "8": {...}}}
    (listas indexadas por idx, igual que tu results_N*_vs_T.json)."""
    os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
    with open(results_file + ".lock", "w") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = read_results(results_file) or {"N": N, "T": [], "by_chi": {}}
            _extend(data["T"], idx + 1)
            data["T"][idx] = float(T)
            blk = data["by_chi"].setdefault(str(chi), {})
            for k, v in entry.items():
                lst = blk.setdefault(k, [])
                _extend(lst, idx + 1)
                lst[idx] = v
            tmp = results_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, results_file)
        finally:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)