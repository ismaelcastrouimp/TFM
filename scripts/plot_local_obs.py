"""
plot_local_obs.py
=================
Observables locales frente a E/N a N=25 (como Figs. 3 y 5 de Giudice et al. 2021, en nuestra convencion):
MRE y Gibbs exactos (fermiones libres, ff_local_obs.py en una rejilla de E) frente a los modelos
(data/0 ARViT+MPS con contexto a chi=4 y 16 en el barrido de T; X5/X6 conv a T=3).  Solo pares del bulk.
Uso: python scripts/plot_local_obs.py --out plots/local_obs/local_obs_N25.png
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="data/gap_arch/obs/ff_N25_grid.json")
ap.add_argument("--data0", default="data/gap_arch/obs/model_data0_N25.json")
ap.add_argument("--conv", default="data/gap_arch/obs/model_N25.json")
ap.add_argument("--out", default="plots/local_obs/local_obs_N25.png")
a = ap.parse_args()

C = dict(mre="#2a78d6", gibbs="#eb6834", d4="#1baf7a", d16="#eda100", conv="#e87ba4")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
g = json.load(open(a.grid))
d0 = json.load(open(a.data0)); cv = json.load(open(a.conv))
panels = [("sx", "sx", r"$\langle\sigma^x\rangle$"),
          ("szz1", "szz1_bulk", r"$\langle\sigma^z_i\sigma^z_{i+1}\rangle$"),
          ("szz2", "szz2_bulk", r"$\langle\sigma^z_i\sigma^z_{i+2}\rangle$"),
          ("szz3", "szz3_bulk", r"$\langle\sigma^z_i\sigma^z_{i+3}\rangle$"),
          ("sxx", "sxx_bulk", r"$\langle\sigma^x_i\sigma^x_{i+1}\rangle$"),
          ("sxx_conn", "sxx_conn_bulk", r"$\langle\sigma^x_i\sigma^x_{i+1}\rangle-\langle\sigma^x\rangle^2$")]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="#fcfcfb")
for ax, (kf, km, lab) in zip(axes.flat, panels):
    ax.set_facecolor("#fcfcfb"); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for kind, name in (("mre", "MRE exacto"), ("gibbs", "Gibbs exacto")):
        rr = sorted([x for x in g if x["kind"] == kind], key=lambda x: x["E_per_site"])
        ax.plot([x["E_per_site"] for x in rr], [x[kf] for x in rr], color=C[kind], lw=2, label=name)
    for chi, key, mk in ((4, "d4", "o"), (16, "d16", "s")):
        rr = [x for x in d0 if x["chi"] == chi]
        ax.errorbar([x["E_per_site"] for x in rr], [x[km] for x in rr], yerr=[x.get(km + "_err", 0) for x in rr],
                    fmt=mk, ms=5, color=C[key], mec="#fcfcfb", mew=0.8, elinewidth=1, capsize=0,
                    label=f"data/0 contexto, $\\chi$={chi} (T=0.5–4)")
    ax.errorbar([x["E_per_site"] for x in cv], [x[km] for x in cv], yerr=[x.get(km + "_err", 0) for x in cv],
                fmt="D", ms=6, color=C["conv"], mec="#fcfcfb", mew=0.8, elinewidth=1, capsize=0,
                label="X5/X6 conv, T=3")
    ax.set_xlabel(r"$\langle H\rangle/N$"); ax.set_ylabel(lab)
axes.flat[0].legend(fontsize=8, frameon=False, labelcolor=INK)
fig.suptitle("Observables locales a la misma energía, N=25 (pares del bulk)", color=INK)
fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
fig.savefig(a.out, dpi=140, facecolor=fig.get_facecolor())
print("guardado", a.out)
