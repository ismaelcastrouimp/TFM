"""
j1j2_sweep_plot.py
==================
Barrido J1-J2 (N=10 PBC, exacto, χ=4; data/gap_arch/robust/j1j2*.json): dF/N, ΔE y ΔS2 (globales, modelo - MRE) y distancia D(ρ, ρ_MRE) frente a J2 para
las ancillas conv / convg / mlp, T = 1 y 3.  Salida: data/gap_arch/robust/j1j2_sweep.{png,pdf} y j1j2_sweep.txt.
"""
import glob, json, os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", os.path.expanduser("~/.cache/claude/mpl"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data/gap_arch/robust")
rows = [r for f in glob.glob(os.path.join(D, "j1j2*.json")) for r in json.load(open(f))]
archs = {"vit:conv:r=1": ("conv", "#2a6fdb", "o"), "vit:convg:r=1": ("convg", "#e08a2c", "s"), "vit:mlp": ("mlp", "#2b9a66", "^")}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#dddddd", "grid.linewidth": 0.6})
for r in rows:
    r["dE"], r["dS2"] = r["E"] - r["E_mre"], r["S2"] - r["S2_mre"]
keys = (("dF_per_site", r"$(F_2-F_2^{\rm MRE})/N$"), ("dE", r"$E-E_{\rm MRE}$"), ("dS2", r"$S_2-S_2^{\rm MRE}$"),
        ("D_mre", r"$D(\rho, \rho_{\rm MRE})$"))
fig, ax = plt.subplots(len(keys), 2, figsize=(10, 12), sharex=True)
lines = []
for j, T in enumerate((1.0, 3.0)):
    for arch, (lab, c, mk) in archs.items():
        sel = [r for r in rows if r["T"] == T and r["arch"] == arch]
        J2s = sorted({r["J2"] for r in sel})
        for i, (key, _) in enumerate(keys):
            a_ = ax[i, j]
            m = [np.mean([r[key] for r in sel if r["J2"] == x]) for x in J2s]
            a_.plot(J2s, m, "-", color=c, lw=1.2, marker=mk, ms=5, mfc="none", label=lab)
            for r in sel:                                       # semillas individuales
                a_.plot(r["J2"], r[key], mk, color=c, ms=3, alpha=0.5)
        for x in J2s:
            rs = [r for r in sel if r["J2"] == x]
            lines.append(f"T={T} {lab:5s} J2={x:<5} dF/N={np.mean([r['dF_per_site'] for r in rs]):.2e} "
                         f"dE={np.mean([r['dE'] for r in rs]):+.3f} dS2={np.mean([r['dS2'] for r in rs]):+.3f} "
                         f"D={np.mean([r['D_mre'] for r in rs]):.3f} S2={np.mean([r['S2'] for r in rs]):.3f} "
                         f"(MRE {rs[0]['S2_mre']:.3f}) n_seeds={len(rs)}")
    ax[0, j].set_yscale("log"); ax[0, j].set_title(f"T = {T:g}", fontsize=10)
    ax[-1, j].set_xlabel(r"$J_2$")
    for i in (1, 2): ax[i, j].axhline(0, color="#333", lw=1)
    for a_ in ax[:, j]: a_.axvline(0.5, color="#999", lw=0.8, ls=":")
for i, (_, lab) in enumerate(keys): ax[i, 0].set_ylabel(lab)
ax[0, 0].legend(frameon=False, title="ancilla (χ=4)")
fig.suptitle("J1-J2 PBC N=10 (exacto): ARViT + ancilla MPS frente al MRE  (línea de puntos: Majumdar-Ghosh)", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(os.path.join(D, f"j1j2_sweep.{ext}"), dpi=150)
open(os.path.join(D, "j1j2_sweep.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines)); print("figura:", os.path.join(D, "j1j2_sweep.png"))
