"""
n50_summary.py
==============
Tabla del barrido N=50 (data/cesga/n50) frente al MRE exacto (data/gap_arch/n50/mre_N50.json).
E/N = -(szz1 + 1.5 sx) de obs_<label>.json (32768 muestras); S2 de s2_tilted_<label>.json (mezcla); F = E - T S2.
Escribe data/gap_arch/n50/summary.txt (y .json).  Uso: python3 scripts/n50_summary.py
"""
import json, os, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOC, N = os.path.join(ROOT, "data/cesga/n50"), 50
mre = {round(r["T"], 1): r for r in json.load(open(os.path.join(ROOT, "data/gap_arch/n50/mre_N50.json")))["results"]}
rows = []
for f in sorted(glob.glob(os.path.join(LOC, "obs_*.json")) + glob.glob(os.path.join(ROOT, "data/cesga/n50_low/obs_*.json"))):
    L = os.path.basename(f)[4:-5]; LOCf = os.path.dirname(f); m, T = L.split("_T"); T = round(float(T), 1)
    o = json.load(open(f)); o = o[0] if isinstance(o, list) else o
    e = -(o["szz1"] + 1.5 * o["sx"]); ee = o.get("E_per_site_err", np.hypot(o["szz1_err"], 1.5 * o["sx_err"]))  # con covarianza szz1-sx (verificador 16-11)
    s2f = os.path.join(LOCf, f"s2_tilted_{L}.json")
    S2, S2e = (json.load(open(s2f))["S2"], json.load(open(s2f))["S2_err"]) if os.path.exists(s2f) else (np.nan, np.nan)
    F = N * e - T * S2; Fe = np.hypot(N * ee, T * S2e); r = mre[T]
    rows.append(dict(label=L, m=m, T=T, E_per_site=e, E_err=ee, S2=S2, S2_err=S2e, F=F, F_err=Fe, dF=F - r["F"],
                     dF_per_site=(F - r["F"]) / N, szz1=o["szz1"], sx=o["sx"], mre_E_per_site=r["E_per_site"],
                     mre_S2=r["S2"], mre_szz1=r["szz1"], mre_sx=r["sx"]))
rows.sort(key=lambda r: (r["T"], r["m"]))
with open(os.path.join(ROOT, "data/gap_arch/n50/summary.txt"), "w") as f:
    f.write("N=50: modelo (obs 32768 + S2 mezcla) vs MRE exacto (mre_dos.py)\n")
    f.write(f"{'label':10s} {'T':>4s} {'E/N':>9s} {'MRE':>9s} {'S2':>8s} {'MRE':>8s} {'dF':>8s} {'err':>6s} {'dF/N':>8s}"
            f" {'szz1':>7s} {'MRE':>7s} {'sx':>7s} {'MRE':>7s}\n")
    for r in rows:
        f.write(f"{r['label']:10s} {r['T']:4.1f} {r['E_per_site']:9.5f} {r['mre_E_per_site']:9.5f} {r['S2']:8.3f} "
                f"{r['mre_S2']:8.3f} {r['dF']:+8.3f} {r['F_err']:6.3f} {r['dF_per_site']:+8.5f} {r['szz1']:7.4f} "
                f"{r['mre_szz1']:7.4f} {r['sx']:7.4f} {r['mre_sx']:7.4f}\n")
json.dump(rows, open(os.path.join(ROOT, "data/gap_arch/n50/summary.json"), "w"), indent=1)
print(open(os.path.join(ROOT, "data/gap_arch/n50/summary.txt")).read())
import subprocess
subprocess.run([os.path.join(ROOT, ".venv/bin/python"), os.path.join(ROOT, "scripts/n50_plots.py")], timeout=300)  # figuras
