"""
n50_plots.py
============
Figuras del barrido N=50 (data/gap_arch/n50/summary.json) frente al MRE exacto (mre_N50_fine.json, rejilla dT=0.02):
  n50_thermo_vs_T.png/.pdf : F/N, E/N, S2/N y (F - F_MRE)/N vs T
  n50_obs_vs_E.png/.pdf    : szz1 y sx vs E/N (curva MRE parametrica en T)
Uso: .venv/bin/python scripts/n50_plots.py   (lo llama n50_summary.py)
"""
import json, os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", os.path.expanduser("~/.cache/claude/mpl"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(ROOT, "data/gap_arch/n50"); N = 50
rows = json.load(open(os.path.join(D, "summary.json")))
mre = json.load(open(os.path.join(D, "mre_N50_fine.json")))["results"]
T = np.array([r["T"] for r in mre]); g = lambda k: np.array([r[k] for r in mre], float)
E, S2, F, zz, sx = g("E") / N, g("S2") / N, g("F") / N, g("szz1"), g("sx")
brk = np.r_[False, (S2[:-1] == 0) & (S2[1:] > 0)]       # salto puro -> mixto (T ~ 0.45): no unir con linea
def cut(*a): return [np.where(brk, np.nan, x) for x in a]
STY = {"tl": dict(c="#2a6fdb", m="o", lab="inclinado (mezcla)"), "sw": dict(c="#e8590c", m="s", lab="swap MCMC"),
       "tlD": dict(c="#2b9a66", m="^", lab="inclinado, init T=1")}
REF = dict(color="#333333", lw=1.5, label="MRE exacto")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#dddddd", "grid.linewidth": 0.6})

def pts(ax, xk, yk, ek=None, scale=1.0):
    for m, s in STY.items():
        R = [r for r in rows if r["m"] == m and np.isfinite(r[yk])]
        if not R: continue
        x = np.array([r[xk] for r in R]); y = np.array([r[yk] for r in R]) * scale
        e = np.array([r[ek] for r in R]) * scale if ek else None
        ax.errorbar(x, y, yerr=e, fmt=s["m"], ms=5, mfc="none", mew=1.2, color=s["c"], elinewidth=1, capsize=2,
                    label=f"{s['lab']} ({len(R)})", zorder=3)

fig, ax = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
for a, y, xk, yk, ek, sc, lab in ((ax[0, 0], F, "T", "F", "F_err", 1 / N, r"$F_2/N$"),
                                  (ax[0, 1], E, "T", "E_per_site", "E_err", 1, r"$E/N$"),
                                  (ax[1, 0], S2, "T", "S2", "S2_err", 1 / N, r"$S_2/N$")):
    a.plot(T, cut(y)[0], **REF); pts(a, xk, yk, ek, sc); a.set_ylabel(lab)
a = ax[1, 1]; a.axhline(0, **{**REF, "label": "MRE (dF = 0)"}); pts(a, "T", "dF", "F_err", 1 / N)
a.set_ylabel(r"$(F_2 - F_2^{\rm MRE})/N$")
for a in ax[1]: a.set_xlabel(r"$T$")
ax[0, 0].legend(frameon=False)
fig.suptitle(r"TFIM PBC $N=50$, $h_x=-1.5$, $J_{zz}=-1$: ARViT + ancilla MPS ($\chi=8$) vs MRE exacto", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(os.path.join(D, f"n50_thermo_vs_T.{ext}"), dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
Ec, zzc, sxc = cut(E, zz, sx)
for a, y, k, lab in ((ax[0], zzc, "szz1", r"$\langle Z_iZ_{i+1}\rangle$"), (ax[1], sxc, "sx", r"$\langle X_i\rangle$")):
    a.plot(Ec, y, **REF)
    a.plot(E[0], {"szz1": zz, "sx": sx}[k][0], "o", color="#333333", ms=6, zorder=4, label=r"MRE $T\leq 0.42$ (fundamental puro)")
    for m, s in STY.items():
        R = [r for r in rows if r["m"] == m]
        if R: a.plot([r["E_per_site"] for r in R], [r[k] for r in R], s["m"], ms=5, mfc="none", mew=1.2, color=s["c"],
                     label=f"{s['lab']} ({len(R)})", zorder=3)
    a.set_xlabel(r"$E/N$"); a.set_ylabel(lab)
ax[0].legend(frameon=False)
fig.suptitle(r"Observables locales vs energia por sitio ($N=50$; curva MRE parametrica en $T\in[0,4]$)", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(os.path.join(D, f"n50_obs_vs_E.{ext}"), dpi=150)
print("figuras en", D)
