"""
plot_mre_jumps_j1j2.py
======================
Saltos del MRE a N pequeño en la cadena J1-J2 (PBC, Paulis), por qué aparecen y dónde caen las redes.
Figuras (en --out_dir):
  mre_vs_T_N10.png      F, E, S2, nº de niveles del soporte, C(1), C(2), E_J2, C(3) del MRE frente a T (N=10), con las
                        redes entrenadas a T=1 (ED de garmonegro χ=4/8/16 y MC χ=8) como puntos
  mre_vs_T_sizes.png    E/N, S2/N, C(2) y soporte frente a T para N = 10, 12, 14, 16 (los saltos se suavizan)
  mre_loop_N10.png      T(c) de la autoconsistencia c − ⟨E⟩ = 2T: tramo no monótono al entrar los multipletes, y
                        F a T=1 de la familia MRE(T'): dos ramas casi degeneradas
Uso: MPLCONFIGDIR=.jobs/mpl python scripts/plot_mre_jumps_j1j2.py --J2 0.5 --out_dir plots/j1j2_ED
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src_renyi.mps_ancilla.block_ed import spectrum_obs, mre_obs

ap = argparse.ArgumentParser()
ap.add_argument("--J2", type=float, default=0.5)
ap.add_argument("--sizes", type=int, nargs="+", default=[10, 12, 14, 16])
ap.add_argument("--Tmin", type=float, default=0.01); ap.add_argument("--Tmax", type=float, default=2.5)
ap.add_argument("--dT", type=float, default=0.002)
ap.add_argument("--out_dir", default="plots/j1j2_ED")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
Ts = np.arange(a.Tmin, a.Tmax + 1e-9, a.dT)
C_NET = {"ED χ=4": "#eb6834", "ED χ=8": "#eda100", "ED χ=16": "#1baf7a", "MC χ=8": "#e87ba4"}


def load_obs(N):
    """espectro + ⟨n|O_r|n⟩, con caché en data/gap_arch/generic/spectra/j1j2_J2<J2>_N<N>_obs.npz"""
    f = f"data/gap_arch/generic/spectra/j1j2_J2{a.J2}_N{N}_obs.npz"
    if os.path.exists(f):
        z = np.load(f); return z["ev"], z["O"], list(z["rs"])
    ev, O, rs = spectrum_obs("j1j2", N, J2=a.J2)
    np.savez(f, ev=ev, O=O, rs=np.array(rs)); return ev, O, rs


def gibbs(ev, O, T):
    """Gibbs a T: E, S2 (Rényi-2 de Gibbs), F2 = E − T S2 (≥ F2 del MRE), C(r), e^{S2}."""
    w = np.exp(-(ev - ev.min()) / T); p = w / w.sum()
    E = float(p @ ev); S2 = float(-np.log(np.sum(p ** 2)))
    return (T, E - T * S2, E, S2, float(np.exp(S2)), 0.0, *[float(x) for x in (O @ p)[:3]])


def scan(N):
    ev, O, rs = load_obs(N)
    rows, grows = [], []
    for T in Ts:
        m = mre_obs(ev, O, T)
        rows.append((T, m["F"], m["E"], m["S2"], int((ev < m["c"]).sum()), m["c"], *m["C"][:3]))
        grows.append(gibbs(ev, O, T))
    return ev, O, np.array(rows), np.array(grows)


data = {N: scan(N) for N in a.sizes}
ev10, O10, r10, g10 = data[10]
T, F, E, S2, ns, cc, C1, C2, C3 = r10.T
EJ2 = a.J2 * 10 * C2
_, gF, gE, gS2, gneff, _, gC1, gC2, gC3 = g10.T
gEJ2 = a.J2 * 10 * gC2

# redes a N=10, T=1 (J2 = 0.5): ED de garmonegro (E, S2, zz -> C(1) = 3 zz, SU(2)) y MC (observables exactos)
nets = {}
if a.J2 == 0.5:
    for f in sorted(glob.glob("data/garmonegro/robust/j1j2chi_J20.5_s1_N10.json")):
        for r in json.load(open(f)):
            if r["T"] == 1.0:
                c1 = 3 * r["zz"]; nets[f"ED χ={r['chi']}"] = dict(E=r["E"], S2=r["S2"], F=r["F2"], C1=c1,
                                                                    EJ2=r["E"] - 10 * c1)
    fo = "data/gap_arch/generic/obs/obs_N10_T1_mc_lr.json"
    if os.path.exists(fo):
        o = json.load(open(fo))["nets"]["mc_lr5e-3"]
        nets["MC χ=8"] = dict(E=o["E"], S2=o["S2"], F=o["F"], C1=o["C"][0], C2=o["C"][1], C3=o["C"][2],
                              EJ2=o["E_bond"]["2"])
    for k in nets:
        nets[k].setdefault("C2", nets[k]["EJ2"] / (a.J2 * 10))

# ── figura 1: N = 10 frente a T ──
fig, axs = plt.subplots(2, 4, figsize=(17, 7.6), constrained_layout=True)
panels = [("F = E − T·S2", F, None, gF), ("E", E, "E", gE), ("S2", S2, "S2", gS2),
          ("nº de estados: soporte del MRE / e^{S2}", ns, None, gneff),
          ("C(1) = ⟨σ_i·σ_{i+1}⟩", C1, "C1", gC1), ("C(2) = ⟨σ_i·σ_{i+2}⟩", C2, "C2", gC2),
          ("E_J2 (energía de enlaces J2)", EJ2, "EJ2", gEJ2), ("C(3)", C3, "C3", gC3)]
for ax, (lab, y, key, yg) in zip(axs.ravel(), panels):
    ax.plot(T, yg, color="#2a78d6", lw=1.4, ls="--", label="Gibbs N=10")
    if lab.startswith("nº"):
        ax.plot(T, np.exp(S2), color="0.15", lw=1.0, ls=":", label="e^{S2} del MRE")
    ax.plot(T, y, color="0.15", lw=1.6, label="MRE exacto N=10")
    ax.axvline(1.0, color="0.6", ls=":", lw=1)
    if key:
        for k, v in nets.items():
            if key in v:
                ax.plot(1.0, v[key], "o", ms=8, mfc="white", mew=2, color=C_NET[k], label=k)
    ax.set_title(lab, loc="left", fontsize=11); ax.set_xlabel("T"); ax.grid(alpha=0.25)
axs[0, 3].set_yscale("log")
axs[0, 1].legend(frameon=False, fontsize=8, loc="upper left"); axs[0, 3].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle(f"J1-J2 (J2={a.J2}, N=10, PBC, Paulis): MRE (negro) y Gibbs (azul, discontinua) frente a T; puntos: redes a T=1",
             fontsize=12)
fig.savefig(os.path.join(a.out_dir, "mre_vs_T_N10.png"), dpi=140); plt.close(fig)

# ── figura 2: varios N ──
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100"]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
for (N, (_, _, r, g)), c in zip(data.items(), cols):
    t, f_, e_, s_, n_, _, c1, c2, c3 = r.T
    axs[0].plot(t, e_ / N, color=c, lw=1.5, label=f"N={N}")
    axs[1].plot(t, s_ / N, color=c, lw=1.5)
    axs[2].plot(t, c2, color=c, lw=1.5)
    axs[3].plot(t, n_ / 2 ** N, color=c, lw=1.5)
    axs[0].plot(t, g[:, 2] / N, color=c, lw=1.0, ls="--")
    axs[1].plot(t, g[:, 3] / N, color=c, lw=1.0, ls="--")
    axs[2].plot(t, g[:, 7], color=c, lw=1.0, ls="--")
axs[0].plot([], [], color="0.4", ls="--", lw=1.0, label="Gibbs (discontinua)")
for ax, lab in zip(axs, ["E/N", "S2/N", "C(2)", "fracción del espectro en el soporte"]):
    ax.set_title(lab, loc="left"); ax.set_xlabel("T"); ax.grid(alpha=0.25); ax.axvline(1.0, color="0.6", ls=":", lw=1)
axs[3].set_yscale("log"); axs[0].legend(frameon=False, fontsize=9)
fig.suptitle(f"J1-J2 (J2={a.J2}) frente a T para varios N: MRE (continua) y Gibbs (discontinua); los saltos del MRE se suavizan con N", fontsize=12)
fig.savefig(os.path.join(a.out_dir, "mre_vs_T_sizes.png"), dpi=140); plt.close(fig)

# ── figura 3: el lazo T(c) y las dos ramas ──
lv, deg = np.unique(np.round(ev10, 8), return_counts=True)
cs = np.linspace(-14.5, -9.5, 6001)


def T_of_c(c):
    w = np.maximum(c - ev10, 0); return 0.5 * (c - (w * ev10).sum() / w.sum())


Tc = np.array([T_of_c(c) for c in cs])
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
ax = axs[0]
ax.plot(cs, Tc, color="0.15", lw=1.6)
for e, g in zip(lv, deg):
    if cs[0] < e < cs[-1]:
        ax.axvline(e, color="#2a78d6", alpha=min(1, 0.15 + 0.06 * g), lw=0.6 + 0.25 * g)
ax.axhline(1.0, color="0.6", ls=":")
ax.set_xlabel("c (corte del MRE)"); ax.set_ylabel("T(c) = (c − ⟨E⟩)/2")
ax.set_title("autoconsistencia: T(c); líneas azules = niveles (grosor ∝ degeneración)", loc="left", fontsize=10)
ax.grid(alpha=0.25)
ax = axs[1]
z = (cs > -12.0) & (cs < -10.6)
ax.plot(cs[z], Tc[z], color="0.15", lw=1.8)
for e, g in zip(lv, deg):
    if -12.0 < e < -10.6:
        ax.axvline(e, color="#2a78d6", alpha=0.5, lw=0.5 + 0.3 * g); ax.text(e, ax.get_ylim()[1] if False else 1.004, f"{g}",
                                                                          color="#2a78d6", fontsize=8, ha="center")
ax.axhline(1.0, color="0.6", ls=":")
ax.set_xlabel("c"); ax.set_title("zoom: T(c) retrocede al entrar 6+6 estados en E≈−11.43", loc="left", fontsize=10)
ax.grid(alpha=0.25)
ax = axs[2]
F1 = E - 1.0 * S2                                 # F a T=1 de la familia MRE(T')
ax.plot(T, F1, color="0.15", lw=1.6, label="F_{T=1}[MRE(T')]")
i1 = np.argmin(np.abs(T - 1.0)); ax.axhline(F1[i1], color="0.6", ls=":")
for k, v in nets.items():
    ax.plot(1.0, v["F"], "o", ms=8, mfc="white", mew=2, color=C_NET[k], label=k)
ax.set_xlim(0.85, 1.15); ax.set_ylim(F1[i1] - 0.005, F1[i1] + 0.06)
ax.set_xlabel("T' (temperatura del MRE que se evalúa)"); ax.set_title("F a T=1 de los MRE de otras T': mínimo plano y rama casi degenerada",
                                                                     loc="left", fontsize=10)
ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(a.out_dir, "mre_loop_N10.png"), dpi=140); plt.close(fig)

# ── figura 4: MRE frente a Gibbs a IGUAL ENERGÍA (eje x = E/N), varios N ──
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
for (N, (_, _, r, g)), c in zip(data.items(), cols):
    for arr, ls, lw in ((r, "-", 1.5), (g, "--", 1.0)):
        e = arr[:, 2] / N
        axs[0].plot(e, arr[:, 3] / N, color=c, ls=ls, lw=lw, label=f"N={N}" if ls == "-" else None)
        for j, ax in zip((6, 7, 8), axs[1:]):
            ax.plot(e, arr[:, j], color=c, ls=ls, lw=lw)
axs[0].plot([], [], color="0.4", ls="--", lw=1.0, label="Gibbs (discontinua)")
for ax, lab in zip(axs, ["S2/N", "C(1)", "C(2)", "C(3)"]):
    ax.set_title(lab, loc="left"); ax.set_xlabel("E/N"); ax.grid(alpha=0.25)
axs[0].legend(frameon=False, fontsize=9)
fig.suptitle(f"J1-J2 (J2={a.J2}): MRE (continua) frente a Gibbs (discontinua) a IGUAL ENERGÍA por sitio", fontsize=12)
fig.savefig(os.path.join(a.out_dir, "mre_vs_gibbs_equalE.png"), dpi=140); plt.close(fig)
print("guardado en", a.out_dir)
