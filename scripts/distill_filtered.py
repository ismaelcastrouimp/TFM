"""
distill_filtered.py
===================
¿Puede una red SIN filtro (ARViT conv + ancilla MPS chi) representar el estado filtrado rho* de un modelo astep?
Se entrena por ajuste supervisado  L = ||rho - rho*||_F^2 / ||rho*||_F^2  (rho normalizados a traza 1) en vez de F.
  - si el alumno llega a dF ~ +0.002 (lo que dan las ancillas truncadas a chi'=16): la familia SI lo representa y el
    problema del entrenamiento con F es de paisaje de optimizacion;
  - si no: la limitacion es de la parametrizacion red -> M(s) (expresividad), no solo del chi.
Profesor: params guardados por filter_joint_exact.py --save_params (astep).  Exacto, N <= 10, CPU.

Uso: python scripts/distill_filtered.py --teacher data/gap_arch/filter/joint_anc_N10_save_astep_s1.msgpack \
        --N 10 --chi_t 4 --chi 16 --out data/gap_arch/filter/distill_N10_chi16.json
"""
import argparse, json, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "..")); sys.path.insert(0, script_dir)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import serialization

from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla import training
from mre_teff import MRE

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True); ap.add_argument("--N", type=int, required=True)
ap.add_argument("--chi_t", type=int, default=4); ap.add_argument("--chi", type=int, default=16)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--arch_t", default="vit:conv:r=1"); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--blocks", type=int, default=2); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--finetune_steps", type=int, default=1500)
ap.add_argument("--out", required=True)
a = ap.parse_args()
N, T = a.N, a.T; assert N <= 10 and "/data/0/" not in os.path.abspath(a.out)

idx = np.arange(2 ** N)
all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
H = np.zeros((2 ** N, 2 ** N))
H[idx, idx] = -1.0 * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
for i in range(N):
    H[idx, idx ^ (1 << i)] += -1.5
ev, V = np.linalg.eigh(H)
Ex, Sx, Fx = MRE(ev)(T)
Hj, all_sj = jnp.asarray(H), jnp.asarray(all_s)


def ancilla_vectors(M):
    S = M.shape[0]
    v = jnp.zeros((S, 1, M.shape[-1]), M.dtype).at[:, 0, 0].set(1.0)
    for k in range(M.shape[1]):
        v = jnp.einsum("sai,sbij->sbaj", v, M[:, k]).reshape(S, -1, M.shape[-1])
    return v[:, :, 0]


def rho_of(model, net, W=None):
    logA, M = model.apply({"params": net}, all_sj)
    A = jnp.exp(logA - jnp.max(jnp.real(logA))); A = A / jnp.linalg.norm(A)
    Phi = ancilla_vectors(M)
    n = jnp.real(jnp.sum(jnp.abs(Phi) ** 2, axis=1))
    O = Phi @ W.astype(Phi.dtype) @ jnp.conj(Phi).T if W is not None else Phi @ jnp.conj(Phi).T
    r = A[:, None] * jnp.conj(A)[None, :] * O / jnp.sqrt(n[:, None] * n[None, :])
    return r / jnp.real(jnp.trace(r))


def FES(r):
    E = jnp.real(jnp.sum(Hj.T * r)); S2 = -jnp.log(jnp.sum(jnp.abs(r) ** 2))
    return E - T * S2, E, S2


mt = build_alt_model(N, a.chi_t, a.arch_t, a.d, 4, a.blocks)
pp = serialization.from_bytes({"net": init_params(mt, N, 1), "g": jnp.zeros(2)}, open(a.teacher, "rb").read())
g = np.asarray(pp["g"])
W = jnp.asarray((V * (1 / (1 + np.exp(-(g[0] - ev) / np.exp(g[1]))))) @ V.T)
rt = rho_of(mt, pp["net"], W)
Ft, Et, St = map(float, FES(rt))
nt2 = float(jnp.sum(jnp.abs(rt) ** 2))
print(f"profesor (astep chi={a.chi_t}): dF={Ft - Fx:+.5f}", flush=True)

ms = build_alt_model(N, a.chi, a.arch, a.d, 4, a.blocks)
res = []


def report(stage, net):
    r = rho_of(ms, net); F, E, S2 = map(float, FES(r))
    dist = float(jnp.sum(jnp.abs(r - rt) ** 2)) / nt2
    fid = float(jnp.real(jnp.sum(jnp.conj(r) * rt))) / np.sqrt(nt2 * float(jnp.sum(jnp.abs(r) ** 2)))
    res.append(dict(stage=stage, chi=a.chi, dF=F - Fx, dE=E - Ex, dS2=S2 - Sx, rel_dist=dist, hs_overlap=fid,
                    teacher_dF=Ft - Fx))
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"  >> {stage}: dF={F - Fx:+.5f} dE={E - Ex:+.4f} dS2={S2 - Sx:+.4f}  ||rho-rho*||^2/||rho*||^2={dist:.2e}"
          f"  HS-overlap={fid:.6f}", flush=True)


def Ldist(net):
    r = rho_of(ms, net); F, E, S2 = FES(r)
    return jnp.sum(jnp.abs(r - rt) ** 2) / nt2, (E, S2)      # el log muestra "F" = distancia relativa


net = init_params(ms, N, a.seed)
out = training.train(Ldist, net, a.steps, a.lr, T, freq=a.steps // 10, tag=f"[distill chi={a.chi}] ")
net = out["params_best"]; report("destilado", net)
if a.finetune_steps > 0:
    o = training.train(lambda nt: (lambda r: (FES(r)[0], (FES(r)[1], FES(r)[2])))(rho_of(ms, nt)), net,
                       a.finetune_steps, 1e-3, T, freq=a.finetune_steps // 5, tag="[finetune F] ")
    report("destilado+finetune_F", o["params_best"])
print("FIN", flush=True)
