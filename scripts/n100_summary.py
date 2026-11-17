"""
n100_summary.py
===============
Barrido N=100 (inclinado, data/cesga/n100) frente al MRE exacto N=100 (data/gap_arch/n100/mre_N100_fine.json), y
comparacion del gap por sitio con N=50 (data/gap_arch/n50/summary.json, mejor de tl/tlD por T).
Escribe data/gap_arch/n100/summary.{txt,json} y figuras n100_thermo_vs_T.{png,pdf}, n100_obs_vs_E.{png,pdf}.
Uso: .venv/bin/python scripts/n100_summary.py
"""
import json, os, glob
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", os.path.expanduser("~/.cache/claude/mpl"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); N = 100
LOC, OUT = os.path.join(ROOT, "data/cesga/n100"), os.path.join(ROOT, "data/gap_arch/n100")
mre = json.load(open(os.path.join(OUT, "mre_N100_fine.json")))["results"]
mT = {round(r["T"], 2): r for r in mre}
rows = []
for f in glob.glob(os.path.join(LOC, "obs_tl_T*.json")):
    L = os.path.basename(f)[4:-5]; T = round(float(L.split("_T")[1]), 2)
    o = json.load(open(f)); o = o[0] if isinstance(o, list) else o
    s = json.load(open(os.path.join(LOC, f"s2_tilted_{L}.json"))); r = mT[T]
    e = -(o["szz1"] + 1.5 * o["sx"]); ee = o.get("E_per_site_err", np.hypot(o["szz1_err"], 1.5 * o["sx_err"]))  # con covarianza szz1-sx (verificador 16-11)
    F = N * e - T * s["S2"]; Fe = np.hypot(N * ee, T * s["S2_err"])
    rows.append(dict(label=L, T=T, E_per_site=e, E_err=ee, S2=s["S2"], S2_err=s["S2_err"], F=F, F_err=Fe,
                     dF=F - r["F"], dF_per_site=(F - r["F"]) / N, szz1=o["szz1"], sx=o["sx"],
                     szz1_bnd=o.get("szz1_bnd", np.nan), mre_E_per_site=r["E_per_site"], mre_S2=r["S2"],
                     mre_szz1=r["szz1"], mre_sx=r["sx"]))
rows.sort(key=lambda r: r["T"])
s50 = {}
for r in json.load(open(os.path.join(ROOT, "data/gap_arch/n50/summary.json"))):
    if r["m"] in ("tl", "tlD") and np.isfinite(r["dF_per_site"]):
        s50[r["T"]] = min(s50.get(r["T"], np.inf), r["dF_per_site"])
with open(os.path.join(OUT, "summary.txt"), "w") as f:
    f.write("N=100 inclinado (obs 32768 + S2 mezcla) vs MRE exacto N=100; dF/N de N=50 (mejor tl/tlD) para comparar\n")
    f.write(f"{'T':>4s} {'E/N':>9s} {'MRE':>9s} {'S2':>8s} {'MRE':>8s} {'dF':>7s} {'err':>6s} {'dF/N':>8s} {'N50':>8s}"
            f" {'szz1':>7s} {'MRE':>7s} {'sx':>7s} {'MRE':>7s} {'zz_bnd':>7s}\n")
    for r in rows:
        f.write(f"{r['T']:4.1f} {r['E_per_site']:9.5f} {r['mre_E_per_site']:9.5f} {r['S2']:8.3f} {r['mre_S2']:8.3f} "
                f"{r['dF']:+7.3f} {r['F_err']:6.3f} {r['dF_per_site']:+8.5f} {s50.get(r['T'], np.nan):+8.5f} "
                f"{r['szz1']:7.4f} {r['mre_szz1']:7.4f} {r['sx']:7.4f} {r['mre_sx']:7.4f} {r['szz1_bnd']:7.3f}\n")
json.dump(rows, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print(open(os.path.join(OUT, "summary.txt")).read())

# ── figuras ────────────────────────────────────────────────────────────────────
T = np.array([r["T"] for r in mre]); g = lambda k: np.array([r[k] for r in mre], float)
E, S2, F, zz, sx = g("E") / N, g("S2") / N, g("F") / N, g("szz1"), g("sx")
brk = np.r_[False, (np.abs(S2[:-1]) < 1e-9) & (S2[1:] > 1e-9)]
cut = lambda x: np.where(brk, np.nan, x)
REF = dict(color="#333333", lw=1.5, label="MRE exacto")
C1, C2 = "#2a6fdb", "#2b9a66"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#dddddd", "grid.linewidth": 0.6})
x = np.array([r["T"] for r in rows]); v = lambda k: np.array([r[k] for r in rows])
fig, ax = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
for a, y, yk, ek, sc, lab in ((ax[0, 0], F, "F", "F_err", 1 / N, r"$F_2/N$"), (ax[0, 1], E, "E_per_site", "E_err", 1, r"$E/N$"),
                              (ax[1, 0], S2, "S2", "S2_err", 1 / N, r"$S_2/N$")):
    a.plot(T, cut(y), **REF)
    a.errorbar(x, v(yk) * sc, yerr=v(ek) * sc, fmt="o", ms=5, mfc="none", mew=1.2, color=C1, capsize=2,
               label=f"inclinado N=100 ({len(rows)})", zorder=3)
    a.set_ylabel(lab)
a = ax[1, 1]; a.axhline(0, color="#333333", lw=1.5, label="MRE (dF = 0)")
a.errorbar(x, v("dF_per_site"), yerr=v("F_err") / N, fmt="o", ms=5, mfc="none", mew=1.2, color=C1, capsize=2, label="N=100")
t50 = sorted(s50); a.plot(t50, [s50[t] for t in t50], "^", ms=5, mfc="none", mew=1.2, color=C2, label="N=50 (mejor)")
a.set_ylabel(r"$(F_2 - F_2^{\rm MRE})/N$"); a.legend(frameon=False)
for a in ax[1]: a.set_xlabel(r"$T$")
ax[0, 0].legend(frameon=False)
fig.suptitle(r"TFIM PBC $N=100$, $h_x=-1.5$, $J_{zz}=-1$: ARViT + ancilla MPS ($\chi=8$, inclinado) vs MRE exacto", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"n100_thermo_vs_T.{ext}"), dpi=150)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for a, y, k, lab in ((ax[0], zz, "szz1", r"$\langle Z_iZ_{i+1}\rangle$"), (ax[1], sx, "sx", r"$\langle X_i\rangle$")):
    a.plot(cut(E), cut(y), **REF)
    a.plot(v("E_per_site"), v(k), "o", ms=5, mfc="none", mew=1.2, color=C1, label=f"inclinado N=100 ({len(rows)})", zorder=3)
    a.set_xlabel(r"$E/N$"); a.set_ylabel(lab)
ax[0].legend(frameon=False)
fig.suptitle(r"Observables locales vs energia por sitio ($N=100$; curva MRE parametrica en $T\in[0,4]$)", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"n100_obs_vs_E.{ext}"), dpi=150)
print("figuras en", OUT)
