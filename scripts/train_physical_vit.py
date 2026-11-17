"""
train_physical_vit.py
=====================
Entrenamiento con el pipeline ANTIGUO de ancilla fisica (src_renyi/physical_ancilla) usando la ARViT con modulo y
fase (ARSpinViT_Causal_Interleaved, psi compleja; ver notas_gap.md "Tarea extra 2").  Basado en training_single_T.py
(fases swap + wang con red inversa), pero con CLI, salida en data/gap_arch/physical/ y corte por tiempo (--deadline).

Uso:
  python scripts/train_physical_vit.py --N 8 --T 3.0 --label phys_vit_N8_T3
  python scripts/train_physical_vit.py --N 25 --T 3.0 --label phys_vit_N25_T3 --deadline "2026-11-09 08:00"
"""
import argparse, json, os, sys, time, datetime
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cfg")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax
import flax.serialization as serialization

import netket.jax._utils_tree as _utils_tree           # mismo parche que training_single_T.py
def _robust_is_scalar(x):
    if hasattr(x, "ndim"):
        return x.ndim == 0
    return isinstance(x, (int, float, complex))
_utils_tree.is_scalar = _robust_is_scalar

from src_renyi.physical_ancilla import free_energy_minimize_phases, ARSpinViT_Causal_Interleaved
from src_renyi.physical_ancilla.entropy import renyi2_entropy_exact

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--label", required=True)
ap.add_argument("--n_samples", type=int, default=2 ** 12)
ap.add_argument("--steps_swap", type=int, default=200)
ap.add_argument("--steps_wang", type=int, default=3000)
ap.add_argument("--init_steps", type=int, default=1000, help="entrenamiento inicial de la red inversa")
ap.add_argument("--embedding_d", type=int, default=16)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--chunk", type=int, default=0, help="chunk_size (0 -> n_samples/2)")
ap.add_argument("--deadline", default=None, help='"YYYY-MM-DD HH:MM": corta el entrenamiento a esa hora')
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out_dir", default="data/gap_arch/physical")
args = ap.parse_args()
assert "/data/0/" not in os.path.abspath(args.out_dir) + "/", "no escribir en data/0"
os.makedirs(args.out_dir, exist_ok=True)

N, N_A, T = args.N, args.N, args.T
J_ZZ, h_x = -1.0, -1.5
NS = args.n_samples
chunk = args.chunk or NS // 2
deadline = (datetime.datetime.strptime(args.deadline, "%Y-%m-%d %H:%M").timestamp() if args.deadline else None)

hi = nk.hilbert.Spin(s=1 / 2, N=N + N_A)
H_ext = 0
for i in range(N):
    H_ext += h_x * sigmax(hi, i)
    H_ext += J_ZZ * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)

model = ARSpinViT_Causal_Interleaved(hilbert=hi, embedding_d=args.embedding_d, n_heads=args.heads,
                                     n_blocks=args.blocks, param_dtype=jnp.float64)
vstate = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model, n_samples=NS, seed=args.seed)
model_R = nk.models.ARNNDense(hilbert=hi, layers=1, features=16)
vstate_R = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model_R, n_samples=NS, seed=args.seed + 1)
partition = list(range(N))

phases = [
    {"method": "swap", "n_steps": args.steps_swap,
     "optimizer": optax.sgd(optax.linear_schedule(0.05, 0.005, max(args.steps_swap, 1))),
     "chunk_size_energy": 4096},
    {"method": "wang", "n_steps": args.steps_wang,
     "optimizer": optax.adam(optax.linear_schedule(1e-3, 5e-4, max(args.steps_wang, 1))),
     "vstate_R": vstate_R,
     "initial_train": {"n_steps": args.init_steps, "batch": NS, "lr": 3e-3},
     "warm_start": {"every": 30, "n_steps": 15, "lr": 5e-4, "batch": NS},
     "wang_kwargs": {"n_samples": NS},
     "chunk_size_energy": 4096},
]
phases = [p for p in phases if p["n_steps"] > 0]

n_par = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(vstate.parameters))
print(f"[{args.label}] N={N}+{N_A} T={T} ARSpinViT_Causal_Interleaved (complejo) d={args.embedding_d} "
      f"heads={args.heads} blocks={args.blocks} #par={n_par}  fases: "
      + ", ".join(f"{p['method']}x{p['n_steps']}" for p in phases)
      + (f"  deadline {args.deadline}" if deadline else ""), flush=True)

t0 = time.time()
hist, F_best, E_best, S2_best = free_energy_minimize_phases(
    vstate, T, partition, H_ext, phases, chunk_size=chunk, verbose=True, freq=10, plot=False,
    timing=True, history_index=None, deadline=deadline)
dt = time.time() - t0
res = {"label": args.label, "N": N, "T": T, "model": "ARSpinViT_Causal_Interleaved", "complex": True,
       "embedding_d": args.embedding_d, "heads": args.heads, "blocks": args.blocks, "n_params": n_par,
       "n_samples": NS, "phases": [(p["method"], p["n_steps"]) for p in phases], "steps_done": len(hist),
       "energy": E_best, "entropy": S2_best, "free_energy": F_best, "time_s": dt}

# ED del estado entrenado (solo N pequeno): E y S2 exactos de rho_sys
if N <= 10:
    S2_ex_state, rho = renyi2_entropy_exact(vstate, partition)
    rho = np.asarray(rho)
    hs = nk.hilbert.Spin(s=1 / 2, N=N)
    Hs = 0
    for i in range(N):
        Hs += h_x * sigmax(hs, i)
        Hs += J_ZZ * sigmaz(hs, i) @ sigmaz(hs, (i + 1) % N)
    Hd = np.asarray(Hs.to_dense())
    E_ex_state = float(np.real(np.trace(rho @ Hd)) / np.real(np.trace(rho)))
    ev = np.linalg.eigvalsh(Hd)
    # MRE exacto: rho ∝ (c - H)_+ con c tal que minimiza F2 = E - T S2  (barrido en c)
    best = None
    for c in np.linspace(ev.min() + 1e-9, ev.max() + 50 * T, 20001):
        w = np.clip(c - ev, 0, None); w = w / w.sum()
        F2 = float(w @ ev + T * np.log((w ** 2).sum()))
        if best is None or F2 < best[0]:
            best = (F2, float(w @ ev), float(-np.log((w ** 2).sum())))
    res.update({"ED_state": {"E": E_ex_state, "S2": float(S2_ex_state), "F": E_ex_state - T * float(S2_ex_state)},
                "ED_MRE": {"F": best[0], "E": best[1], "S2": best[2]}})
    print(f"[{args.label}] ED estado entrenado: F={E_ex_state - T * float(S2_ex_state):.5f} E={E_ex_state:.5f} "
          f"S2={float(S2_ex_state):.5f} | MRE exacto F={best[0]:.5f} E={best[1]:.5f} S2={best[2]:.5f} "
          f"-> dF={E_ex_state - T * float(S2_ex_state) - best[0]:+.4f}", flush=True)

print(f"[{args.label}] RESULTADO: F={F_best:.5f} E={E_best:.5f} S2={S2_best:.5f} | {len(hist)} pasos, {dt / 3600:.2f} h",
      flush=True)
with open(os.path.join(args.out_dir, f"params_{args.label}.msgpack"), "wb") as f:
    f.write(serialization.to_bytes(vstate.parameters))
rf = os.path.join(args.out_dir, "results.json")
allr = json.load(open(rf)) if os.path.exists(rf) else []
allr = [r for r in allr if r["label"] != args.label] + [res]
json.dump(allr, open(rf, "w"), indent=1, default=float)
