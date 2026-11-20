"""
ladder_chi_exact.py
===================
Escalera en chi con gradiente EXACTO (ρ completa, N <= 10) para xxz / j1j2 (PBC): ¿hereda la red ampliada el mínimo
de la de chi pequeño?  Cada escalón: se amplía chi con pad_chi_mlp (estado idéntico + ruido pequeño en los bloques
nuevos) y se entrena con training.train (warmup + coseno, se queda con params_best).  Tras cada escalón: F, D(ρ, MRE),
energía por enlace J1/J2 y correlaciones C(r) = (1/N) Σ_i ⟨σ_i·σ_{i+r}⟩ (Paulis), frente al MRE.
Opcional --scratch_chi: además, entrenamiento desde cero a ese chi (referencia), con --scratch_steps / --scratch_lr.
Uso:
  python scripts/ladder_chi_exact.py --model j1j2 --J2 0.5 --N 10 --T 1 --arch vit:mlp \
      --chis 4 8 16 --steps 3000 1500 1500 --lrs 5e-3 2e-3 2e-3 --noise 1e-3 --label L --out_dir data/gap_arch/ladder
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="j1j2", choices=["j1j2", "xxz"])
ap.add_argument("--N", type=int, default=10); ap.add_argument("--T", type=float, required=True)
ap.add_argument("--J", type=float, default=1.0); ap.add_argument("--J2", type=float, default=0.5)
ap.add_argument("--Delta", type=float, default=1.0)
ap.add_argument("--arch", default="vit:mlp"); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--chis", type=int, nargs="+", required=True)
ap.add_argument("--steps", type=int, nargs="+", required=True); ap.add_argument("--lrs", type=float, nargs="+", required=True)
ap.add_argument("--noise", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--init", default=None, help="params del primer escalón (si no, desde cero)")
ap.add_argument("--scratch_chi", type=int, default=None)
ap.add_argument("--scratch_steps", type=int, default=3000); ap.add_argument("--scratch_lr", type=float, default=5e-3)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--label", required=True); ap.add_argument("--out_dir", default="data/gap_arch/ladder")
a = ap.parse_args()
assert len(a.chis) == len(a.steps) == len(a.lrs)
assert "/data/0" not in os.path.abspath(a.out_dir)
os.makedirs(a.out_dir, exist_ok=True)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from src_renyi.mps_ancilla import training
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params, load_params, save_params
from src_renyi.mps_ancilla.block_ed import mre as mre_fn
from pad_chi_mlp import pad_chi_mlp

N, T = a.N, a.T; D = 2 ** N; tag = f"[{a.label}] "
idx = np.arange(D); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)


def pair_ops(r):
    zz = np.zeros(D); xy = np.zeros((D, D))
    for i in range(N):
        j = (i + r) % N; si, sj = all_s[:, i], all_s[:, j]
        zz += si * sj / N
        f = si != sj; xy[idx[f], (idx ^ (1 << i) ^ (1 << j))[f]] += 2.0 / N
    return zz, xy


ops = {r: pair_ops(r) for r in range(1, N // 2 + 1)}
coup, Dl = ({1: a.J, 2: a.J2 * a.J}, {1: 1.0, 2: 1.0}) if a.model == "j1j2" else ({1: a.J}, {1: a.Delta})
H = sum(coup[r] * N * (Dl[r] * np.diag(ops[r][0]) + ops[r][1]) for r in coup)
ev, V = np.linalg.eigh(H)
m = mre_fn(ev, T)
pw = np.maximum(m["c"] - ev, 0); pw /= pw.sum(); rho_mre = (V * pw) @ V.T


def observables(rho):
    rho = 0.5 * (rho + rho.conj().T)
    E = float(np.real(np.sum(H.T * rho))); S2 = float(-np.log(np.sum(np.abs(rho) ** 2)))
    pd = np.real(np.diagonal(rho))
    C = [float(np.sum(ops[r][0] * pd) + np.real(np.sum(ops[r][1].T * rho))) for r in ops]
    Eb = {str(r): float(coup[r] * N * (Dl[r] * np.sum(ops[r][0] * pd) + np.real(np.sum(ops[r][1].T * rho)))) for r in coup}
    return dict(E=E, S2=S2, F=E - T * S2, dF_per_site=(E - T * S2 - m["F"]) / N,
                D_mre=float(0.5 * np.abs(np.linalg.eigvalsh(rho - rho_mre)).sum()), C=C, E_bond=Eb)


def report(name, o, extra=""):
    print(f"{tag}{name:>18}: dF/N={o['dF_per_site']:+.6f} E={o['E']:.4f} S2={o['S2']:.4f} D={o['D_mre']:.4f} | "
          + " ".join(f"E_J{r}={o['E_bond'][str(r)]:.3f}" for r in coup) + " | C(r)=" + " ".join(f"{c:+.3f}" for c in o["C"])
          + extra, flush=True)


res = dict(args=vars(a), mre=m, stages=[])
report("MRE", observables(rho_mre))
p = None
for i, (chi, st, lr) in enumerate(zip(a.chis, a.steps, a.lrs)):
    model = build_alt_model(N, chi, a.arch, a.d)
    if i == 0:
        p = load_params(a.init, model, N) if a.init else init_params(model, N, a.seed)
    else:
        p = pad_chi_mlp(p, init_params(model, N, a.seed + i), N, a.chis[i - 1], chi, a.noise, a.seed + i)
    loss, rho_fn = training.make_loss_fns(model, all_s, H, T, chunk=a.chunk)
    o0 = observables(np.asarray(rho_fn(p)))
    if i > 0:
        report(f"chi={chi} (ampliado)", o0)
    out = training.train(loss, p, st, lr, T, freq=max(1, st // 5), tag=f"{tag}[chi={chi}] ")
    p = out["params_best"]
    o = observables(np.asarray(rho_fn(p)))
    report(f"chi={chi} entrenado", o, f"  ({1e3 * out['s_per_step']:.0f} ms/paso)")
    f = os.path.join(a.out_dir, f"{a.label}_chi{chi}_params.msgpack"); save_params(f, p)
    res["stages"].append(dict(chi=chi, steps=st, lr=lr, start=o0, end=o, params=f, s_per_step=out["s_per_step"]))
    json.dump(res, open(os.path.join(a.out_dir, a.label + ".json"), "w"), indent=1)

if a.scratch_chi:
    chi = a.scratch_chi
    model = build_alt_model(N, chi, a.arch, a.d)
    loss, rho_fn = training.make_loss_fns(model, all_s, H, T, chunk=a.chunk)
    out = training.train(loss, init_params(model, N, a.seed), a.scratch_steps, a.scratch_lr, T,
                         freq=max(1, a.scratch_steps // 5), tag=f"{tag}[desde cero chi={chi}] ")
    o = observables(np.asarray(rho_fn(out["params_best"])))
    report(f"cero chi={chi}", o, f"  ({1e3 * out['s_per_step']:.0f} ms/paso)")
    f = os.path.join(a.out_dir, f"{a.label}_scratch_chi{chi}_params.msgpack"); save_params(f, out["params_best"])
    res["scratch"] = dict(chi=chi, steps=a.scratch_steps, lr=a.scratch_lr, end=o, params=f)
    json.dump(res, open(os.path.join(a.out_dir, a.label + ".json"), "w"), indent=1)
print(f"{tag}FIN", flush=True)
