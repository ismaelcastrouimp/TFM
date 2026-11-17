"""
plot_qE_hist.py
===============
Histograma de la distribucion de energia del modelo, q(E) = sum_n <n|rho|n> en cada bin, frente al MRE exacto a la
misma energia (T_E), a partir de los *_qE.npz que guarda scripts/s2_deficit_split.py.
Panel izquierdo: escala lineal. Panel derecho: escala log (se ve la cola por encima del corte c, donde el MRE es 0).

Uso: python scripts/plot_qE_hist.py data/gap_arch/varh/split_S_mc_N10_qE.npz [...] --out_dir plots/histograms
"""
import argparse, os, sys
script_dir = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, script_dir)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_MODEL, C_MRE, C_CUT = "#2a78d6", "#eb6834", "#52514e"

ap = argparse.ArgumentParser()
ap.add_argument("npz", nargs="+")
ap.add_argument("--bins", type=int, default=40)
ap.add_argument("--out_dir", default="plots/histograms")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

for path in a.npz:
    d = np.load(path)
    ev, q, c, TE, N, label = d["ev"], d["q"], float(d["c"]), float(d["T_E"]), int(d["N"]), str(d["label"])
    p = np.maximum(c - ev, 0); p /= p.sum()
    edges = np.linspace(ev.min(), ev.max(), a.bins + 1)
    hq, _ = np.histogram(ev, edges, weights=q)
    hp, _ = np.histogram(ev, edges, weights=p)
    w_above = q[ev > c].sum()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, logy in zip(axes, (False, True)):
        ax.stairs(hp, edges, color=C_MRE, lw=2, label=f"MRE exacto (T={TE:.3f})")
        ax.stairs(hq, edges, color=C_MODEL, lw=2, label="modelo")
        ax.axvline(c, color=C_CUT, lw=1, ls="--")
        ax.text(c, 0.97, "  corte c", transform=ax.get_xaxis_transform(), va="top", ha="left",
                color=C_CUT, fontsize=9)
        ax.set_xlabel("E")
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if logy:
            ax.set_yscale("log"); ax.set_ylim(max(hq[hq > 0].min(), 1e-8) / 3, 1)
            ax.set_title(f"escala log: peso del modelo por encima de c = {w_above:.3%}", fontsize=10)
        else:
            ax.set_ylabel("peso por bin")
            ax.set_title("escala lineal", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Distribucion de energia q(E) del modelo frente al MRE  -  {label}, N={N}", fontsize=11)
    out = os.path.join(a.out_dir, f"qE_{label}.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(out)
