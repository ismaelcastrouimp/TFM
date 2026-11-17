"""
pad_chi_finish.py
=================
Test decisivo "¿limita chi?": carga los parametros (EMA) de un run MC con chi=CHI0, los AMPLIA a chi=CHI1
rellenando los tensores del MPS con ceros (+ ruido pequeno en los bloques nuevos) y continua con gradiente
EXACTO (ED).  El estado inicial es el mismo (se comprueba que F coincide con ruido=0); si la F baja por
debajo de lo que daba el ED con chi=CHI0, el limite era chi.  Es tambien la receta "inicializar un modelo
grande desde uno pequeno" para MC.

Uso:  python pad_chi_finish.py N CHI0 CHI1 [STEPS=300] [NOISE=0.02]
Lee   mc_N{N}_chi{CHI0}_seed1.pkl  (lo escribe mc_train.py).  Solo N<=10 (ED).
"""
import itertools, os, pickle, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import brentq
from src_renyi.mps_ancilla.training import make_loss_fns, train
from src_renyi.mps_ancilla.models import ARViT_MPSAncilla

N    = int(sys.argv[1]); CHI0 = int(sys.argv[2]); CHI1 = int(sys.argv[3])
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 300
NOISE = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02
SEED = 1
T, HX, JJ = 1.5, -1.5, -1.0
D, HEADS, BLOCKS = 16, 4, 2
LR = 3e-3
PKL = f"mc_N{N}_chi{CHI0}_seed{SEED}.pkl"

def tfim(N, hx, J):
    sx = np.array([[0, 1], [1, 0]], float); sz = np.diag([1., -1.]); I = np.eye(2)
    def op(o, i):
        m = np.array([[1.]])
        for k in range(N):
            m = np.kron(m, o if k == i else I)
        return m
    return sum(hx * op(sx, i) + J * op(sz, i) @ op(sz, (i + 1) % N) for i in range(N))

H = tfim(N, HX, JJ)
E_vals = np.linalg.eigvalsh(H)
def _sc(c):
    w = np.maximum(c - E_vals, 0); return c - (w * E_vals).sum() / w.sum() - 2 * T
c = brentq(_sc, E_vals.min() + 1e-6, E_vals.max() + 20 * T + 10, xtol=1e-12)
w = np.maximum(c - E_vals, 0); p = w / w.sum()
E_EX = float((p * E_vals).sum()); S2_EX = float(-np.log((p ** 2).sum())); F_EX = E_EX - T * S2_EX
print(f"BASELINE exacto N={N}: F={F_EX:.5f}  E={E_EX:.5f}  S2={S2_EX:.4f}", flush=True)

def pad_chi(params, chi, chi2, K, noise_std, seed=0):
    """Amplia los parametros que generan los tensores del MPS de chi a chi2 (ceros + ruido en bloques nuevos)."""
    rng = np.random.default_rng(seed)
    new_block = np.ones((chi2, chi2), bool); new_block[:chi, :chi] = False

    def pad_last(a, add_noise):
        a = np.asarray(a)
        lead = a.shape[:-1]
        a4 = a.reshape(lead + (K, 2, chi, chi))
        out = np.zeros(lead + (K, 2, chi2, chi2))
        out[..., :chi, :chi] = a4
        if add_noise and noise_std > 0:
            out = out + rng.normal(0.0, noise_std, size=out.shape) * new_block
        return out.reshape(lead + (K * 2 * chi2 * chi2,))

    p = jax.tree_util.tree_map(np.asarray, params)
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in p.items()}
    p["mps_head"]["kernel"] = pad_last(p["mps_head"]["kernel"], True)
    p["mps_head"]["bias"] = pad_last(p["mps_head"]["bias"], False)
    if "mps_copy" in p:
        p["mps_copy"] = pad_last(p["mps_copy"], False)
    return jax.tree_util.tree_map(jnp.asarray, p)

with open(PKL, "rb") as f:
    saved = pickle.load(f)
params0 = jax.tree_util.tree_map(jnp.asarray, saved["params_ema"])

all_s = jnp.asarray(np.array(list(itertools.product([1., -1.], repeat=N))))
chunk = 128 if N >= 10 else None
mk = lambda chi: ARViT_MPSAncilla(N_sys=N, N_anc=N, chi=chi, embedding_d=D, n_heads=HEADS, n_blocks=BLOCKS,
                                     copy_init=True, mps_init=0.05)
model0, model1 = mk(CHI0), mk(CHI1)
loss0, _ = make_loss_fns(model0, all_s, H, T, chunk=chunk)
loss1, _ = make_loss_fns(model1, all_s, H, T, chunk=chunk)

F0, (E0, S0) = loss0(params0)
Fp, _ = loss1(pad_chi(params0, CHI0, CHI1, N, 0.0))
print(f"chi={CHI0}: F={float(F0):.5f} (gap {float(F0) - F_EX:+.4f})   |  ampliado a chi={CHI1} sin ruido: "
      f"F={float(Fp):.5f}   (|dF|={abs(float(F0) - float(Fp)):.1e}: debe ser ~0)", flush=True)

params1 = pad_chi(params0, CHI0, CHI1, N, NOISE / np.sqrt(D), seed=0)
out = train(loss1, params1, STEPS, LR, T, freq=max(1, STEPS // 10), warmup=20, tag=f"[ED chi={CHI1}] ")
print(f"FINAL chi={CHI1} (desde chi={CHI0}): F={out['F']:.5f}  E={out['E']:.5f}  S2={out['S2']:.4f}   "
      f"gap F = {out['F'] - F_EX:+.4f}   (dE={out['E'] - E_EX:+.3f}, dS2={out['S2'] - S2_EX:+.3f})", flush=True)