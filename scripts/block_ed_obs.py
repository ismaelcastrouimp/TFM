"""
block_ed_obs.py
===============
Espectro + observables locales por autoestado (bloques Sz + k) de xxz / j1j2 con PBC, y el MRE en una rejilla de T:
E, S2, F, C(r) = (1/N) Σ_i ⟨σ_i·σ_{i+r}⟩ (Paulis), E_J1, E_J2 y número de niveles en el soporte.
Guarda <tag>_obs.npz (ev, O, rs) y <tag>_mre_obs.json (rejilla de T), para comparar observables de redes a N grande y
detectar saltos del MRE (S2(T) no suave).
Uso: python scripts/block_ed_obs.py --model j1j2 --J2 0.5 --N 20 --out_dir data/gap_arch/generic/spectra
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from src_renyi.mps_ancilla.block_ed import spectrum_obs, mre_obs
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="j1j2", choices=["j1j2", "xxz"]); ap.add_argument("--N", type=int, required=True)
ap.add_argument("--J", type=float, default=1.0); ap.add_argument("--J2", type=float, default=0.5)
ap.add_argument("--Delta", type=float, default=1.0)
ap.add_argument("--Tgrid", type=float, nargs="+", default=None, help="por defecto 0.5..4 en pasos de 0.02")
ap.add_argument("--out_dir", default="data/gap_arch/generic/spectra")
a = ap.parse_args()
assert "/data/0" not in os.path.abspath(a.out_dir)
os.makedirs(a.out_dir, exist_ok=True)
tag = f"{a.model}_J2{a.J2}_N{a.N}" if a.model == "j1j2" else f"{a.model}_D{a.Delta}_N{a.N}"
t0 = time.time()
ev, O, rs = spectrum_obs(a.model, a.N, J=a.J, J2=a.J2, Delta=a.Delta, verbose=True)
print(f"[{tag}] espectro + observables: {time.time() - t0:.0f} s", flush=True)
np.savez(os.path.join(a.out_dir, tag + "_obs.npz"), ev=ev, O=O, rs=np.array(rs))
Ts = a.Tgrid or list(np.round(np.arange(0.5, 4.0001, 0.02), 4))
coup = {1: a.J, 2: a.J2 * a.J} if a.model == "j1j2" else {1: a.J}
res = dict(args=vars(a), rs=rs, grid=[])
for T in Ts:
    m = mre_obs(ev, O, T, rs)
    m["T"] = float(T); m["n_support"] = int((ev < m["c"]).sum())
    if a.model == "j1j2":
        m["E_bond"] = {str(r): float(coup[r] * a.N * m["C"][r - 1]) for r in coup}
    res["grid"].append(m)
json.dump(res, open(os.path.join(a.out_dir, tag + "_mre_obs.json"), "w"), indent=1)
for m in res["grid"]:
    if abs(m["T"] - 1) < 1e-9 or abs(m["T"] - 3) < 1e-9:
        print(f"[{tag}] T={m['T']}: E/N={m['E'] / a.N:.6f} S2/N={m['S2'] / a.N:.5f} soporte={m['n_support']} "
              f"C(r)={np.round(m['C'], 4).tolist()}", flush=True)
S = np.array([m["S2"] / a.N for m in res["grid"]]); T = np.array([m["T"] for m in res["grid"]])
d = np.diff(S) / np.diff(T); med = np.median(d)
jumps = [(float(T[i]), float(T[i + 1]), float(d[i])) for i in range(len(d)) if d[i] > 4 * max(med, 1e-6)]
print(f"[{tag}] dS2/N/dT mediana {med:.4f}; tramos con pendiente > 4x mediana (posibles saltos): {jumps[:10]}", flush=True)
