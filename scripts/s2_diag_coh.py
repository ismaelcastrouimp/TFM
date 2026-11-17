"""
s2_diag_coh.py
==============
Separa la pureza Tr rho^2 = D + C en parte DIAGONAL D = sum_s rho(s,s)^2 y COHERENCIAS C = sum_{s!=s'} |rho(s,s')|^2,
y C por distancia de Hamming d(s,s') = 1, 2, >=3, para modelos (exacto, N <= 12) y para el MRE de referencia:
  - MRE a T (el objetivo), y MRE con la MISMA E que el modelo (el MRE maximiza S2 a E fija -> deficit "a igual E").
Reparto del deficit de pureza: P_mod - P_ref = (D_mod - D_ref) + (C_mod - C_ref); dS2 = -log(P_mod/P_ref).
Uso:
  python scripts/s2_diag_coh.py --N 10 --out data/gap_arch/diagcoh/N10.json \
      --run mc_conv:data/gap_arch/mcN10/params_mc_conv_s1.msgpack:16:vit:conv:r=1 ...
  (--run label:params:chi:arch ; arch con ':' se admite: todo lo que sigue al 3er ':' es arch)
"""
import argparse, json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import brentq
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import load_params
from src_renyi.mps_ancilla.training import compute_rho

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--h_x", type=float, default=-1.5); ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--run", nargs="*", default=[]); ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert a.N <= 14 and "/data/0/" not in os.path.abspath(a.out)
N, T = a.N, a.T

# mismo H y misma base que gap_mc.py / make_loss_fns
idx = np.arange(2 ** N); bits = (idx[:, None] >> np.arange(N)[None, :]) & 1
all_s = (2 * bits - 1).astype(np.float64)
H = np.zeros((2 ** N, 2 ** N))
H[idx, idx] = a.J_ZZ * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
for i in range(N):
    H[idx, idx ^ (1 << i)] += a.h_x
ev, V = np.linalg.eigh(H)
ham = np.array([bin(x).count("1") for x in range(2 ** N)], dtype=np.int8)
dist = ham[idx[:, None] ^ idx[None, :]]                                   # (2^N, 2^N) distancia de Hamming (int8)
masks = {"d0": None, "d1": None, "d2": None, "d3+": None}               # solo las claves; mascaras al vuelo (memoria N=14)


def split(rho2):  # rho2 = |rho|^2 (real)
    sp = {"d0": float(np.trace(rho2)), "d1": float(rho2[dist == 1].sum()), "d2": float(rho2[dist == 2].sum())}
    sp["d3+"] = float(rho2.sum()) - sum(sp.values())
    return sp


def mre_w(c):
    w = np.maximum(c - ev, 0); return w / w.sum()


def mre_at_T():
    c = brentq(lambda c: c - (mre_w(c) * ev).sum() - 2 * T, ev.min() + 1e-9, ev.max() + 20 * T + 10, xtol=1e-12)
    return c


def mre_at_E(E):
    return brentq(lambda c: (mre_w(c) * ev).sum() - E, ev.min() + 1e-9, ev.max() * 50 + 1e3, xtol=1e-12)


def rec(label, rho):
    rho2 = np.abs(rho) ** 2
    sp = split(rho2); P = sum(sp.values())
    E = float(np.real(np.sum(H.T * rho)))
    r = dict(label=label, E=E, S2=-np.log(P), F=E - T * (-np.log(P)), P=P, **sp,
             S2_diag=-np.log(sp["d0"]), frac_diag=sp["d0"] / P)
    return r


def rho_mre(c):
    w = mre_w(c); return (V * w) @ V.T


out = {"N": N, "T": T, "rows": []}
cT = mre_at_T(); ref = rec("MRE_T", rho_mre(cT)); out["rows"].append(ref)
print(f"N={N} T={T}  MRE: E={ref['E']:.5f} S2={ref['S2']:.5f} F={ref['F']:.5f}  D/P={ref['frac_diag']:.4f}", flush=True)


def show(r, base):
    dP = r["P"] - base["P"]
    parts = {k: r[k] - base[k] for k in masks}
    s = " ".join(f"{k}:{v / dP:+.2f}" for k, v in parts.items()) if abs(dP) > 0 else ""
    return (f"  vs {base['label']:>8}: dS2={r['S2'] - base['S2']:+.5f}  dP/P={dP / base['P']:+.4f}  "
            f"reparto dP [{s}]  dS2_diag={r['S2_diag'] - base['S2_diag']:+.4f}")


for spec in a.run:
    label, path, chi, arch = spec.split(":", 3)
    chi = int(chi)
    model = build_alt_model(N, chi, arch)
    params = load_params(path, model, N); params = params.get("params", params)
    rho = np.asarray(compute_rho(model, params, jnp.asarray(all_s), a.chunk))
    r = rec(label, rho); r.update(params=path, chi=chi, arch=arch)
    rE = rec(f"MRE_E[{label}]", rho_mre(mre_at_E(r["E"])))
    r["dF"] = r["F"] - ref["F"]
    out["rows"] += [r, rE]
    print(f"{label}: E={r['E']:.5f} S2={r['S2']:.5f} dF={r['dF']:+.5f}  D/P={r['frac_diag']:.4f}  "
          f"(MRE a su E: S2={rE['S2']:.5f}, D/P={rE['frac_diag']:.4f})", flush=True)
    print(show(r, ref)); print(show(r, rE), flush=True)
    # fracciones de P por distancia (para comparar con nbr2 a N=25)
    print("  fracciones P  modelo: " + " ".join(f"{k}={r[k] / r['P']:.3f}" for k in masks)
          + "  | MRE_E: " + " ".join(f"{k}={rE[k] / rE['P']:.3f}" for k in masks), flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print("guardado", a.out)
