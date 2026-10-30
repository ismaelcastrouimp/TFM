"""
fast_ed_sweep.py
================
Barrido de tamaño del ViT interleaved con ED rápida (ver fast_ed.py).
Mide F, gap vs MRE exacto, gap vs suelo positivo, fidelidad, fracción de
elementos negativos de rho_A y desviación típica de la fase.
"""
import copy, json, os, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # antes de importar jax
from itertools import product

import numpy as np
import jax

# ── Parche JAX >= 0.10 (pvary) ──
if not hasattr(jax.lax, 'pvary'):
    def _pvary(x, axis_name):
        return jax.lax.pcast(x, axis_name, to='varying')
    jax.lax.pvary = _pvary

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import netket as nk
from netket.operator.spin import sigmax, sigmaz
from scipy.optimize import brentq
from scipy.linalg import eigh

import fast_ed as fe
from src_renyi.physical_ancilla.models import ARSpinViT_Causal_Interleaved

# ══════════════════════════════════════════════════════════════════════════════
# Configuración
# ══════════════════════════════════════════════════════════════════════════════
N, T = 8, 1.5
J_ZZ, J_XX, h_x, h_z = -1.0, 0.0, -1.5, 0.0

N_STEPS = 3000
LR      = 1e-2
FREQ    = 100
SEEDS   = [1, 2, 3]

# (embedding_d, n_heads, n_blocks)  — embedding_d debe ser divisible por n_heads
MODEL_CONFIGS = [(4, 2, 1), (8, 2, 1), (16, 4, 2), (32, 4, 2)]

CHUNK_SIZE = 2**13   # configuraciones por trozo (baja si hay OOM, sube si va lento); None = sin trozos
VALIDATE = True     # compara una vez con tu pipeline lento (renyi2_entropy_and_grad_exact)

# Suelo con amplitudes >= 0 calculado con L-BFGS para N=8, T=1.5, J=1, hx=1.5
# (NO vale para otros parámetros; poner None si cambias algo)
F_POS_FLOOR = -14.337 if (N, T, J_ZZ, h_x) == (8, 1.5, -1.0, -1.5) else None

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots", "fast_ed")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Hilbert, Hamiltonianos, referencias exactas
# ══════════════════════════════════════════════════════════════════════════════
hi     = nk.hilbert.Spin(s=1/2, N=2 * N)
hi_sys = nk.hilbert.Spin(s=1/2, N=N)
H_ext, H_sys_op = 0, 0
for i in range(N):
    H_ext    += h_x * sigmax(hi, i) + h_z * sigmaz(hi, i) \
              + J_ZZ * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N) \
              + J_XX * sigmax(hi, i) @ sigmax(hi, (i + 1) % N)
    H_sys_op += h_x * sigmax(hi_sys, i) + h_z * sigmaz(hi_sys, i) \
              + J_ZZ * sigmaz(hi_sys, i) @ sigmaz(hi_sys, (i + 1) % N) \
              + J_XX * sigmax(hi_sys, i) @ sigmax(hi_sys, (i + 1) % N)
partition = list(range(N))
H_dense = np.array(H_sys_op.to_dense())

E_vals, V = eigh(H_dense)
def _mre(T):
    def sc(c):
        w = np.maximum(c - E_vals, 0); return c - (w * E_vals).sum() / w.sum() - 2 * T
    c = brentq(sc, E_vals.min() + 1e-6, E_vals.max() + 20 * T + 10, xtol=1e-12)
    w = np.maximum(c - E_vals, 0); p = w / w.sum()
    return p, (p * E_vals).sum(), -np.log((p ** 2).sum())
p_mre, E_EX, S2_EX = _mre(T)
F_EX = E_EX - T * S2_EX
rho_MRE = (V * p_mre) @ V.conj().T
print(f"MRE exacto: E={E_EX:.4f}  S2={S2_EX:.4f}  F={F_EX:.4f}   (suelo positivo: {F_POS_FLOOR})")


def fidelity(rho, sigma):
    """(Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2"""
    rho = np.array(rho); rho = rho / np.trace(rho)
    sigma = np.array(sigma); sigma = sigma / np.trace(sigma)
    ev, U = np.linalg.eigh(rho)
    sq = (U * np.sqrt(np.clip(ev, 0, None))) @ U.conj().T
    evm = np.linalg.eigvalsh(sq @ sigma @ sq)
    return float(np.sum(np.sqrt(np.clip(evm, 0, None))) ** 2)


# Orden de configuraciones: NO suponemos ninguno. Índices (sistema, ancilla) con la numeración de
# NetKet para el sistema, de modo que rho_A queda en el mismo orden que H_dense.
configs = np.asarray(hi.all_states())
IDX_SYS = np.asarray(hi_sys.states_to_numbers(configs[:, :N]))
IDX_ANC = np.asarray(hi_sys.states_to_numbers(configs[:, N:]))
assert IDX_SYS.max() == 2**N - 1 and IDX_ANC.max() == 2**N - 1
_same = np.array_equal(configs, np.array(list(product([-1, 1], repeat=2 * N)), dtype=configs.dtype))
print(f"hi.all_states() con orden tipo itertools.product: {_same}  (se usa scatter, da igual)")


def make_model(d, heads, blocks):
    return ARSpinViT_Causal_Interleaved(
        hilbert=hi, embedding_d=d, n_heads=heads, n_blocks=blocks,
        n_ffn_layers=1, machine_pow=2, param_dtype=jnp.float64)


def init_params(model, seed):
    vs = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model, n_samples=1, seed=seed * 1000 + 42)
    return copy.deepcopy(vs.parameters), copy.deepcopy(getattr(vs, "model_state", {}) or {})


# ══════════════════════════════════════════════════════════════════════════════
# Validación contra el pipeline lento (una vez)
# ══════════════════════════════════════════════════════════════════════════════
def validate(d, heads, blocks):
    from src_renyi.physical_ancilla.entropy import renyi2_entropy_and_grad_exact
    model = make_model(d, heads, blocks)
    params, mstate = init_params(model, 1)
    loss, _ = fe.make_loss_fns(model, configs, H_dense, T, N, model_state=mstate,
                                     idx_sys=IDX_SYS, idx_anc=IDX_ANC, chunk_size=CHUNK_SIZE)
    (F, (E, S2)), g = jax.value_and_grad(loss, has_aux=True)(params)

    vs = nk.vqs.FullSumState(hi, model); vs.parameters = copy.deepcopy(params)
    E_stats, gE = vs.expect_and_grad(H_ext)
    S2_s, gS2 = renyi2_entropy_and_grad_exact(vs, partition, hi, isFullSum=True)
    g_slow = jax.tree_util.tree_map(lambda a, b: a - T * b, gE, gS2)

    print("VALIDACIÓN  (rápido vs lento)")
    print(f"  E : {float(E):.10f} vs {float(E_stats.mean.real):.10f}")
    print(f"  S2: {float(S2):.10f} vs {float(S2_s):.10f}")
    fa, _ = ravel_pytree(g)
    fb, _ = ravel_pytree(jax.tree_util.tree_map(lambda x: jnp.real(x), g_slow))
    print(f"  max|grad_rapido - grad_lento| = {float(jnp.max(jnp.abs(fa - fb))):.3e}"
          f"   (|grad| max = {float(jnp.max(jnp.abs(fb))):.3e})")


# ══════════════════════════════════════════════════════════════════════════════
# Un run
# ══════════════════════════════════════════════════════════════════════════════
def run(d, heads, blocks, seed):
    model = make_model(d, heads, blocks)
    params, mstate = init_params(model, seed)
    loss, rho_fn = fe.make_loss_fns(model, configs, H_dense, T, N, model_state=mstate,
                                     idx_sys=IDX_SYS, idx_anc=IDX_ANC, chunk_size=CHUNK_SIZE)

    def diag(p):
        rho, logpsi = rho_fn(p)
        rho = np.array(rho)
        neg = float((rho.real < -1e-12).mean())
        im = float(np.linalg.norm(rho.imag) / np.linalg.norm(rho))
        return f"Fid={fidelity(rho, rho_MRE):.4f} | neg={neg:.3f} | |Im rho|/|rho|={im:.4f}"

    tag = f"[d={d},h={heads},b={blocks},s={seed}] "
    out = fe.train(loss, params, N_STEPS, LR, T, freq=FREQ, diag_fn=diag, tag=tag)
    rho, logpsi = rho_fn(out["params_best"]); rho = np.array(rho)
    res = dict(
        d=d, heads=heads, blocks=blocks, seed=seed, n_params=fe.count_params(params),
        F=out["F"], E=out["E"], S2=out["S2"],
        gap_exact=out["F"] - F_EX,
        gap_floor=None if F_POS_FLOOR is None else out["F"] - F_POS_FLOOR,
        fid=fidelity(rho, rho_MRE), neg_frac=float((rho.real < -1e-12).mean()),
        im_ratio=float(np.linalg.norm(rho.imag) / np.linalg.norm(rho)), s_per_step=out["s_per_step"])
    print(f"{tag}FINAL F={res['F']:.5f} gap_exact={res['gap_exact']:+.4f} "
          f"fid={res['fid']:.4f} neg={res['neg_frac']:.3f}  ({res['s_per_step']*1e3:.1f} ms/paso)\n", flush=True)
    return res


if __name__ == "__main__":
    if VALIDATE:
        validate(*MODEL_CONFIGS[0])
    RES_PATH = os.path.join(OUT_DIR, "results.json")
    results = json.load(open(RES_PATH)) if os.path.exists(RES_PATH) else []   # reanuda: salta lo ya hecho
    done = {(r["d"], r["heads"], r["blocks"], r["seed"]) for r in results}
    for (d, heads, blocks) in MODEL_CONFIGS:
        for seed in SEEDS:
            if (d, heads, blocks, seed) in done:
                print(f"[saltado, ya en results.json] d={d} h={heads} b={blocks} s={seed}")
                continue
            results.append(run(d, heads, blocks, seed))
            with open(RES_PATH, "w") as f:
                json.dump(results, f, indent=1)

    print("=" * 96)
    print(f"{'d':>3} {'h':>2} {'b':>2} {'seed':>4} {'#par':>7} {'F':>10} {'gap_ex':>8} "
          f"{'gap_floor':>10} {'E':>9} {'S2':>7} {'fid':>6} {'neg':>5} {'Im/rho':>7}")
    for r in results:
        gf = "   n/d" if r["gap_floor"] is None else f"{r['gap_floor']:+.4f}"
        print(f"{r['d']:>3} {r['heads']:>2} {r['blocks']:>2} {r['seed']:>4} {r['n_params']:>7} "
              f"{r['F']:>10.4f} {r['gap_exact']:>+8.4f} {gf:>10} {r['E']:>9.4f} {r['S2']:>7.4f} "
              f"{r['fid']:>6.3f} {r['neg_frac']:>5.2f} {r.get('im_ratio', float('nan')):>7.4f}")