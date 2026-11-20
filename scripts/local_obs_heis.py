"""
local_obs_heis.py
=================
Observables locales EXACTOS (ρ completa, N <= 10) de redes ARViT + ancilla MPS entrenadas en xxz / j1j2 (PBC),
frente al MRE exacto:
  E, S2, F, dF/N, D(ρ, ρ_MRE) (distancia de traza)
  energía por tipo de enlace: J1 (r=1) y J2 (r=2)
  correlaciones C(r) = (1/N) Σ_i ⟨σ_i·σ_{i+r}⟩ (Paulis), r = 1..N/2, separadas en ZZ y (XX+YY)/2
  (en un estado SU(2)-invariante ⟨ZZ⟩ = ⟨XX+YY⟩/2: la diferencia mide la ruptura de SU(2))
Cada red: --nets etiqueta=ruta_params:arch:chi  (arch con ':' -> escribir con '/' y se traduce, p.ej. vit/mlp).
Uso:
  JAX_PLATFORMS=cpu python scripts/local_obs_heis.py --model j1j2 --J2 0.5 --N 10 --T 3 \
      --nets mc=data/gap_arch/generic/j1j2_mc_N10_J20.5_T3_chi8_mlp_params_ema.msgpack:vit/mlp:8 --out X.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="j1j2", choices=["j1j2", "xxz"])
ap.add_argument("--N", type=int, default=10); ap.add_argument("--T", type=float, required=True)
ap.add_argument("--J", type=float, default=1.0); ap.add_argument("--J2", type=float, default=0.5)
ap.add_argument("--Delta", type=float, default=1.0); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--nets", nargs="+", required=True)
ap.add_argument("--chunk", type=int, default=64); ap.add_argument("--out", default=None)
a = ap.parse_args()
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import load_params
from src_renyi.mps_ancilla.training import make_loss_fns
from src_renyi.mps_ancilla.block_ed import mre as mre_fn

N, T = a.N, a.T; D = 2 ** N
idx = np.arange(D); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)


def pair_ops(r):
    """(1/N) Σ_i Z_i Z_{i+r} (diagonal) y (1/N) Σ_i (X X + Y Y)_{i,i+r} (giros)."""
    zz = np.zeros(D); xy = np.zeros((D, D))
    for i in range(N):
        j = (i + r) % N; si, sj = all_s[:, i], all_s[:, j]
        zz += si * sj / N
        f = si != sj
        xy[idx[f], (idx ^ (1 << i) ^ (1 << j))[f]] += 2.0 / N
    return zz, xy


rmax = N // 2
ops = {r: pair_ops(r) for r in range(1, rmax + 1)}
if a.model == "j1j2":
    coup = {1: a.J, 2: a.J2 * a.J}; Dl = {1: 1.0, 2: 1.0}
else:
    coup = {1: a.J}; Dl = {1: a.Delta}
H = sum(coup[r] * N * (Dl[r] * np.diag(ops[r][0]) + ops[r][1]) for r in coup)
ev, V = np.linalg.eigh(H)
m = mre_fn(ev, T)
pw = np.maximum(m["c"] - ev, 0); pw /= pw.sum()
rho_mre = (V * pw) @ V.T


def observables(rho):
    rho = 0.5 * (rho + rho.conj().T)
    E = float(np.real(np.sum(H.T * rho))); S2 = float(-np.log(np.sum(np.abs(rho) ** 2)))
    o = dict(E=E, S2=S2, F=E - T * S2, dF_per_site=(E - T * S2 - m["F"]) / N,
             D_mre=float(0.5 * np.abs(np.linalg.eigvalsh(rho - rho_mre)).sum()))
    pd = np.real(np.diagonal(rho))
    o["Czz"] = [float(np.sum(ops[r][0] * pd)) for r in ops]
    o["Cxy_half"] = [float(np.real(np.sum(ops[r][1].T * rho)) / 2) for r in ops]
    o["C"] = [z + 2 * x for z, x in zip(o["Czz"], o["Cxy_half"])]
    o["E_bond"] = {str(r): float(coup[r] * N * (Dl[r] * o["Czz"][r - 1] + 2 * o["Cxy_half"][r - 1])) for r in coup}
    return o


res = dict(args=vars(a), mre=m, nets={})
res["nets"]["MRE"] = observables(rho_mre)
for spec in a.nets:
    lab, rest = spec.split("=", 1)
    path, arch, chi = rest.rsplit(":", 2); chi = int(chi); arch = arch.replace("/", ":")
    model = build_alt_model(N, chi, arch, a.d)
    p = load_params(path, model, N)
    _, rho_fn = make_loss_fns(model, all_s, H, T, chunk=a.chunk)
    res["nets"][lab] = observables(np.asarray(rho_fn(p))); res["nets"][lab]["params"] = path

print(f"{a.model} N={N} T={T}  MRE: E={m['E']:.5f} S2={m['S2']:.5f} F={m['F']:.5f}")
hdr = f"{'red':>22} {'dF/N':>9} {'E':>9} {'S2':>7} {'D':>6} | " + " ".join(f"E_J{r}".rjust(8) for r in coup) + " | " + \
      " ".join(f"C({r})".rjust(7) for r in ops) + " | " + " ".join(f"ΔSU2({r})".rjust(8) for r in ops)
print(hdr)
for lab, o in res["nets"].items():
    print(f"{lab:>22} {o['dF_per_site']:+9.5f} {o['E']:9.4f} {o['S2']:7.4f} {o['D_mre']:6.4f} | "
          + " ".join(f"{o['E_bond'][str(r)]:8.4f}" for r in coup) + " | "
          + " ".join(f"{c:7.4f}" for c in o["C"]) + " | "
          + " ".join(f"{z - x:+8.4f}" for z, x in zip(o["Czz"], o["Cxy_half"])), flush=True)
if a.out:
    assert "/data/0" not in os.path.abspath(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
