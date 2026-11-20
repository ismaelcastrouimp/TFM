"""
block_ed_spectrum.py
====================
Espectro completo (bloques Sz + k) de xxz / j1j2 con PBC y MRE a varias T; guarda el espectro (.npy) y un json.
Uso: python scripts/block_ed_spectrum.py --model j1j2 --J2 0.5 --N 16 --T 1 3 --out_dir data/gap_arch/generic/spectra
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from src_renyi.mps_ancilla.block_ed import spectrum, mre
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="j1j2", choices=["j1j2", "xxz"]); ap.add_argument("--N", type=int, required=True)
ap.add_argument("--J", type=float, default=1.0); ap.add_argument("--J2", type=float, default=0.5)
ap.add_argument("--Delta", type=float, default=1.0); ap.add_argument("--T", type=float, nargs="+", default=[1.0, 3.0])
ap.add_argument("--out_dir", default="data/gap_arch/generic/spectra")
a = ap.parse_args()
assert "/data/0" not in os.path.abspath(a.out_dir)
os.makedirs(a.out_dir, exist_ok=True)
tag = f"{a.model}_J2{a.J2}_N{a.N}" if a.model == "j1j2" else f"{a.model}_D{a.Delta}_N{a.N}"
t0 = time.time()
ev = spectrum(a.model, a.N, J=a.J, J2=a.J2, Delta=a.Delta, verbose=True)
dt = time.time() - t0
np.save(os.path.join(a.out_dir, tag + ".npy"), ev)
res = dict(args=vars(a), time_s=dt, E0=float(ev[0]), E0_per_site=float(ev[0] / a.N), mre={})
print(f"[{tag}] {dt:.1f} s  E0/N = {ev[0] / a.N:.8f}", flush=True)
for T in a.T:
    r = mre(ev, T); res["mre"][str(T)] = r
    print(f"[{tag}] T={T}: MRE E/N={r['E'] / a.N:.6f} S2={r['S2']:.5f} F={r['F']:.6f} S2/N={r['S2'] / a.N:.5f}", flush=True)
json.dump(res, open(os.path.join(a.out_dir, tag + ".json"), "w"), indent=1)
