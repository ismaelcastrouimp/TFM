"""
obs_convergence_N.py
====================
Convergencia de los observables locales a Gibbs al crecer N (N = 12, 25, 50, 100; barridos inclinados ARViT + MPS).
Referencia de Gibbs en el límite termodinámico (fermiones libres, H = -J Σ ZZ - h Σ X, J=1, h=1.5):
    e(T) = -(1/π) ∫_0^π (ε_k/2) tanh(ε_k/2T) dk,   ε_k = 2 sqrt(J² + h² - 2 J h cos k)
    f(T) = -(T/π) ∫_0^π ln(2 cosh(ε_k/2T)) dk,     sx = -∂f/∂h,  szz1 = -∂f/∂J   (diferencias centrales)
Compara a IGUAL densidad de energía (equivalencia de colectividades): δO(E) = O_modelo(E) - O_Gibbs∞(E), y lo mismo para
la curva MRE de cada N (lo que se espera que tienda a 0 al crecer N).
Entradas: data/gap_arch/n{12,25,50,100}/summary.json y mre_N*_fine.json.   Salida: data/gap_arch/convergence_N/.
"""
import json, os
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", os.path.expanduser("~/.cache/claude/mpl"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data/gap_arch/convergence_N"); os.makedirs(OUT, exist_ok=True)
k = np.linspace(0, np.pi, 4001)


def gibbs_inf(T, J=1.0, h=1.5):
    def f(J, h):
        eps = 2 * np.sqrt(J ** 2 + h ** 2 - 2 * J * h * np.cos(k))
        x = eps / (2 * T)
        return -(T / np.pi) * np.trapezoid(x + np.log1p(np.exp(-2 * x)), k)   # ln(2cosh x) = x + ln(1+e^{-2x})
    eps = 2 * np.sqrt(J ** 2 + h ** 2 - 2 * J * h * np.cos(k))
    e = -(1 / np.pi) * np.trapezoid(eps / 2 * np.tanh(eps / (2 * T)), k)
    d = 1e-5
    sx = -(f(J, h + d) - f(J, h - d)) / (2 * d)
    zz = -(f(J + d, h) - f(J - d, h)) / (2 * d)
    return e, zz, sx


Tg = np.linspace(0.05, 6, 600)
G = np.array([gibbs_inf(t) for t in Tg])                       # columnas: e, szz1, sx
Eg = G[:, 0]
gib = lambda E, col: np.interp(E, Eg, G[:, col])               # e(T) monótona creciente
assert abs(gibbs_inf(1e-3)[0] - (-(1 / np.pi) * np.trapezoid(np.sqrt(1 + 2.25 - 3 * np.cos(k)), k))) < 1e-6


def model_rows(N):
    p = os.path.join(ROOT, f"data/gap_arch/n{N}/summary.json")
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    if N == 50:                                                  # n50: varias corridas por T (tl/tlD): la de menor dF
        best = {}
        for r in rows:
            if r.get("m") in ("tl", "tlD") and np.isfinite(r.get("dF_per_site", np.nan)):
                if r["T"] not in best or r["dF_per_site"] < best[r["T"]]["dF_per_site"]:
                    best[r["T"]] = r
        rows = [best[t] for t in sorted(best)]
    return rows


def mre_curve(N):
    p = os.path.join(ROOT, f"data/gap_arch/n{N}/mre_N{N}_fine.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))["results"]
    return np.array([[r["T"], r["E"] / N, r["szz1"], r["sx"], r["S2"]] for r in m])


Ns = [12, 25, 50, 100]
cols = {12: "#e08a2c", 25: "#2b9a66", 50: "#2a6fdb", 100: "#8a3ffc"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#dddddd", "grid.linewidth": 0.6})
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
table = []
for N in Ns:
    rows, mc = model_rows(N), mre_curve(N)
    c = cols[N]
    if mc is not None:
        ok = mc[:, 4] > 1e-9                                        # solo la parte mixta (T por encima del salto)
        for j, (col, a_) in enumerate(((2, ax[0, 0]), (3, ax[0, 1]))):
            a_.plot(mc[ok, 1], mc[ok, col] - gib(mc[ok, 1], 1 + j), "-", color=c, lw=1.2, alpha=0.7, label=f"MRE N={N}")
    if rows:
        E = np.array([r["E_per_site"] for r in rows]); T = np.array([r["T"] for r in rows])
        zz = np.array([r["szz1"] for r in rows]); sx = np.array([r["sx"] for r in rows])
        mix = np.array([r["S2"] for r in rows]) > 0.05
        ax[0, 0].plot(E[mix], (zz - gib(E, 1))[mix], "o", ms=4, mfc="none", mew=1.1, color=c, label=f"modelo N={N}")
        ax[0, 1].plot(E[mix], (sx - gib(E, 2))[mix], "o", ms=4, mfc="none", mew=1.1, color=c, label=f"modelo N={N}")
        ax[1, 0].plot(T, [r["dF_per_site"] for r in rows], "o-", ms=4, mfc="none", color=c, lw=0.8, label=f"N={N}")
        for r in rows:
            if r["S2"] > 0.05:
                table.append(dict(N=N, T=r["T"], E=r["E_per_site"], dzz=r["szz1"] - gib(r["E_per_site"], 1),
                                  dsx=r["sx"] - gib(r["E_per_site"], 2), dF_per_site=r["dF_per_site"]))
ax[0, 0].set_ylabel(r"$\langle Z_iZ_{i+1}\rangle - \langle Z_iZ_{i+1}\rangle_{\rm Gibbs,\infty}$ (misma $E/N$)")
ax[0, 1].set_ylabel(r"$\langle X_i\rangle - \langle X_i\rangle_{\rm Gibbs,\infty}$ (misma $E/N$)")
for a_ in ax[0]: a_.set_xlabel(r"$E/N$"); a_.axhline(0, color="#333", lw=1)
ax[0, 0].legend(frameon=False, fontsize=8, ncol=2)
ax[1, 0].axhline(0, color="#333", lw=1); ax[1, 0].set_xlabel(r"$T$"); ax[1, 0].set_ylabel(r"$(F_2-F_2^{\rm MRE})/N$")
ax[1, 0].legend(frameon=False, fontsize=8); ax[1, 0].set_title("Gap por sitio frente al MRE de cada N", fontsize=10)
# |δ| medio en T >= 1 frente a N
a_ = ax[1, 1]
for key, mk, lab in (("dzz", "o", r"$|\delta\langle ZZ\rangle|$"), ("dsx", "s", r"$|\delta\langle X\rangle|$")):
    xs, ys = [], []
    for N in Ns:
        v = [abs(t[key]) for t in table if t["N"] == N and t["T"] >= 1.0]
        if v: xs.append(N); ys.append(np.mean(v))
    a_.loglog(xs, ys, mk + "-", label=lab + " modelo (media T≥1)")
for col, lab in ((2, r"$|\delta\langle ZZ\rangle|$ MRE"), (3, r"$|\delta\langle X\rangle|$ MRE")):
    xs, ys = [], []
    for N in Ns:
        mc = mre_curve(N)
        if mc is None: continue
        sel = (mc[:, 0] >= 1.0) & (mc[:, 0] <= 4.0)
        xs.append(N); ys.append(np.mean(np.abs(mc[sel, col] - gib(mc[sel, 1], col - 1))))
    a_.loglog(xs, ys, "--", color="#555" if col == 2 else "#999", label=lab + " (media T∈[1,4])")
a_.set_xlabel("N"); a_.set_ylabel("desviación media respecto a Gibbs$_\\infty$"); a_.legend(frameon=False, fontsize=8)
a_.set_title("Convergencia a Gibbs al crecer N", fontsize=10)
fig.suptitle("TFIM PBC, $h_x=-1.5$: observables locales del modelo y del MRE frente a Gibbs (N=∞) a igual energía", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(os.path.join(OUT, f"obs_convergence_N.{ext}"), dpi=150)
json.dump(table, open(os.path.join(OUT, "convergence_table.json"), "w"), indent=1)
for N in Ns:
    v = [t for t in table if t["N"] == N and t["T"] >= 1.0]
    if v:
        print(f"N={N:3d}: media T>=1  |dZZ|={np.mean([abs(t['dzz']) for t in v]):.4f}  |dX|={np.mean([abs(t['dsx']) for t in v]):.4f}"
              f"  dF/N={np.mean([t['dF_per_site'] for t in v]):.5f}  ({len(v)} T)")
print("figura:", os.path.join(OUT, "obs_convergence_N.png"))
