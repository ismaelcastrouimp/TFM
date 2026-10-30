"""
mc_train.py
===========
Entrena ARViT+MPS con el gradiente MONTE CARLO (mc_mps_ancilla.py).  Imprime el BASELINE exacto del
MRE (F, E, S2) y, para N pequeno, monitoriza F, E, S2 exactos (ED) de los parametros actuales y de su EMA.
No necesita NetKet.  Uso:  python mc_train.py [chi] [M] [steps] [N]
"""
import itertools, os, pickle, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # antes de importar jax
# os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")  # si sigue habiendo OOM: libera memoria al momento (mas lento)
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import brentq
# import src_renyi.mps_ancilla.models.fast_ed_mps_ancilla as fm
# import mc_mps_ancilla as mc
import src_renyi.mps_ancilla.models   as models
import src_renyi.mps_ancilla.training as training

CHI   = int(sys.argv[1]) if len(sys.argv) > 1 else 4
M     = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
N     = int(sys.argv[4]) if len(sys.argv) > 4 else 8
T, HX, JJ = 1.5, -1.5, -1.0
D, HEADS, BLOCKS = 16, 4, 2
LR, SEED = 5e-3, 1
PAIR_BLOCK = 8                 # pares dentro de bloques de 8 muestras (coste M*8); None = todos los pares (M^2)
CHUNK = None                   # solo si PAIR_BLOCK=None y M grande

# Inicializacion.  COPY_INIT=True: la ancilla copia al sistema => rho = I/2^N, S2 = N ln2, E = 0.
# MPS_INIT es la escala de la perturbacion aleatoria sobre esa copia:
#   0.01-0.05 -> arranca (casi) exactamente en S2 maximo y E ~ 0
#   0.3       -> arranca en un punto intermedio (la perturbacion crea coherencias entre configuraciones: baja S2, E<0)
COPY_INIT = True
MPS_INIT  = 0.05

# Fase final de pulido (lotes mas grandes, lr constante).  0 = desactivada.
POLISH_STEPS = 1000
POLISH_M     = (4 if N <= 8 else 2) * M     # a N=10 con 4*M faltaba memoria (buffer de 4.56 GiB)
POLISH_LR    = 1e-3

def tfim(N, hx, J):
    sx = np.array([[0, 1], [1, 0]], float); sz = np.diag([1., -1.]); I = np.eye(2)
    def op(o, i):
        m = np.array([[1.]])
        for k in range(N):
            m = np.kron(m, o if k == i else I)
        return m
    return sum(hx * op(sx, i) + J * op(sz, i) @ op(sz, (i + 1) % N) for i in range(N))

# ---- baseline exacto (MRE): rho ~ (c - H)_+ con c - <H> = 2T ----
H = tfim(N, HX, JJ)
E_vals = np.linalg.eigvalsh(H)
def _sc(c):
    w = np.maximum(c - E_vals, 0); return c - (w * E_vals).sum() / w.sum() - 2 * T
c = brentq(_sc, E_vals.min() + 1e-6, E_vals.max() + 20 * T + 10, xtol=1e-12)
w = np.maximum(c - E_vals, 0); p = w / w.sum()
E_EX = float((p * E_vals).sum()); S2_EX = float(-np.log((p ** 2).sum())); F_EX = E_EX - T * S2_EX
print(f"BASELINE exacto N={N}, T={T}: F={F_EX:.5f}  E={E_EX:.5f}  S2={S2_EX:.4f}   "
      f"(estado inicial 'copia': E=0, S2={N*np.log(2):.4f}, F={-T*N*np.log(2):.4f})", flush=True)

model = models.ARViT_MPSAncilla(N_sys=N, N_anc=N, chi=CHI, embedding_d=D, n_heads=HEADS, n_blocks=BLOCKS,
                            copy_init=COPY_INIT, mps_init=MPS_INIT)
params = model.init(jax.random.PRNGKey(SEED * 1000 + 42), jnp.ones((1, N)))["params"]

exact_fn = None
if N <= 10:
    all_s = jnp.asarray(np.array(list(itertools.product([1., -1.], repeat=N))))
    ED_CHUNK = 128 if N >= 10 else None       # evita OOM en el gradiente exacto a N>=10
    loss_ed, _ = training.make_loss_fns(model, all_s, H, T, chunk=ED_CHUNK)
    _eval = jax.jit(loss_ed)       # compilado UNA vez (eager recompilaba en cada llamada y llenaba la GPU)
    exact_fn = lambda p_: (lambda r: (r[0], r[1][0], r[1][1]))(_eval(p_))

# Con --finish se reutilizan los parametros guardados de un run MC anterior (salta el entrenamiento MC).
SAVE = f"mc_N{N}_chi{CHI}_seed{SEED}.pkl"
if "--finish" in sys.argv and os.path.exists(SAVE):
    with open(SAVE, "rb") as f_:
        d_ = pickle.load(f_)
    out = {k: jax.tree_util.tree_map(jnp.asarray, d_[k]) for k in ("params", "params_ema")}
    print(f"cargado {SAVE}: se salta el entrenamiento MC", flush=True)
else:
    def _ckpt(p_, e_):                         # checkpoint periodico: si el run se cae, --finish usa el ultimo
        with open(SAVE, "wb") as f_:
            pickle.dump({"params": jax.tree_util.tree_map(np.asarray, p_),
                         "params_ema": jax.tree_util.tree_map(np.asarray, e_)}, f_)
    out = training.train_mc(model, params, N, T, HX, JJ, M=M, n_steps=STEPS, lr=LR, seed=SEED, chunk=CHUNK,
                      pair_block=PAIR_BLOCK, freq=max(1, STEPS // 30), exact_fn=exact_fn,
                      ref=(F_EX, E_EX, S2_EX), polish_steps=POLISH_STEPS, polish_M=POLISH_M, polish_lr=POLISH_LR,
                      ckpt_fn=_ckpt, ckpt_every=250, tag=f"[chi={CHI},M={M}] ")
    print(f"{out['s_per_step']*1e3:.0f} ms/paso")
    with open(SAVE, "wb") as f_:
        pickle.dump({k: jax.tree_util.tree_map(np.asarray, out[k]) for k in ("params", "params_ema")}, f_)
    print(f"guardado {SAVE}  (relanzar solo el diagnostico ED con:  python mc_train.py {CHI} {M} {STEPS} {N} --finish)",
          flush=True)

# ---- diagnostico (solo N<=10): continuar desde la EMA con gradiente EXACTO ----
# Si baja hasta ~+0.002, lo que falta en el run MC es optimizacion/ruido (chi y la cuenca ya valen).
# Si se queda donde esta, el limite es chi / el ViT.  0 = desactivado.
ED_FINISH = 1000 if N <= 8 else 500
if exact_fn is not None and ED_FINISH > 0:
    print("--- diagnostico: continuar con gradiente exacto (ED) desde la EMA ---", flush=True)
    o2 = training.train(loss_ed, out["params_ema"], ED_FINISH, 3e-3, T, freq=max(1, ED_FINISH // 10),
                  warmup=20, tag="[ED] ")
    print(f"FINAL tras ED: F={o2['F']:.5f}  E={o2['E']:.5f}  S2={o2['S2']:.4f}   gap F = {o2['F'] - F_EX:+.4f}"
          f"   (dE={o2['E'] - E_EX:+.3f}, dS2={o2['S2'] - S2_EX:+.3f})", flush=True)