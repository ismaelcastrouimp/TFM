"""
plot_obs_vs_T_N25.py
====================
Observables locales frente a T a N=25: modelos antiguos de data/0 (arch con contexto, chi4 y chi16, barrido en T) y los
modelos actuales a T=3 (X5/X6 conv transferidos, H0/H1, A2 desde cero), frente al Renyi-2 (MRE) EXACTO a N=25
(fermiones libres, scripts/ff_local_obs.py).  Panel inferior: modelo - MRE.
Entradas: data/gap_arch/obs/model_data0_N25.json, model_N25.json, model_N25_hyb.json, model_N25_scratch.json
          data/0/mps_N25/results_mps_N25_vs_T.json (solo lectura, para la T de cada indice)
Salida:   data/gap_arch/obs/mre_N25_vsT.json (cache del MRE), plots/local_obs/obs_N25_old_vs_new.png
Uso: python scripts/plot_obs_vs_T_N25.py [--samples 20000]
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ff_local_obs as ff

ap = argparse.ArgumentParser()
ap.add_argument("--samples", type=int, default=20000); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OBS = os.path.join(ROOT, "data/gap_arch/obs")
N = 25
KEYS = [("E_per_site", r"$\langle H\rangle/N$"), ("sx", r"$\langle\sigma^x\rangle$"),
        ("szz1", r"$\langle\sigma^z_i\sigma^z_{i+1}\rangle$"), ("sxx", r"$\langle\sigma^x_i\sigma^x_{i+1}\rangle$")]

T_all = json.load(open(os.path.join(ROOT, "data/0/mps_N25/results_mps_N25_vs_T.json")))["T"]
old = {}
for r in json.load(open(os.path.join(OBS, "model_data0_N25.json"))):
    m = re.match(r"d0_chi(\d+)_i(\d+)", r["label"]); chi, i = int(m.group(1)), int(m.group(2))
    old.setdefault(chi, []).append((float(T_all[i]), r))
new = []
for f in ("model_N25.json", "model_N25_hyb.json", "model_N25_scratch.json"):
    new += json.load(open(os.path.join(OBS, f)))

Ts = sorted({T for rows in old.values() for T, _ in rows} | {3.0})
cache = os.path.join(OBS, "mre_N25_vsT.json")
mre = {}
if os.path.exists(cache):
    mre = {float(k): v for k, v in json.load(open(cache)).items()}
todo = [T for T in Ts if T not in mre]
if todo:
    secs, E, ids, sid = ff.build(N)
    rng = np.random.default_rng(a.seed)
    for T in todo:
        w, _ = ff.weights("mre", E, T=T)
        mre[T] = ff.ensemble_obs(secs, E, ids, sid, w, N, 1, a.samples, rng)
        print(f"MRE N={N} T={T:.2f}: E/N={mre[T]['E_per_site']:+.4f} sx={mre[T]['sx']:+.4f} "
              f"szz1={mre[T]['szz1']:+.4f} sxx={mre[T]['sxx']:+.4f}", flush=True)
    json.dump({str(k): v for k, v in mre.items()}, open(cache, "w"), indent=1)


def ref(T, k):
    return mre[T][k], mre[T].get(k + "_err", 0.0)


fig, ax = plt.subplots(2, 4, figsize=(19, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
TT = np.array(sorted(mre))
colors_old = {4: "tab:blue", 16: "tab:cyan"}
mk_new = {"X5": ("tab:green", "s"), "X6": ("darkgreen", "D"), "H0": ("tab:olive", "^"), "H1": ("tab:pink", "v"),
          "A2": ("tab:gray", "o")}
for c, (k, lab) in enumerate(KEYS):
    a0, a1 = ax[0, c], ax[1, c]
    a0.plot(TT, [mre[T][k] for T in TT], "r--", lw=1.8, label="Rényi-2 exacto, N=25")
    for chi, rows in sorted(old.items()):
        rows = sorted(rows, key=lambda x: x[0])
        T = np.array([t for t, _ in rows]); y = np.array([r[k] for _, r in rows])
        e = np.array([r.get(k + "_err", 0.0) for _, r in rows])
        yr = np.array([ref(t, k)[0] for t in T]); er = np.array([ref(t, k)[1] for t in T])
        a0.errorbar(T, y, yerr=e, fmt="o", ms=3.5, color=colors_old[chi], label=f"data/0 (ctx), chi={chi}")
        a1.errorbar(T, y - yr, yerr=np.sqrt(e ** 2 + er ** 2), fmt="o", ms=3, color=colors_old[chi])
    seen = set()
    for i, r in enumerate(new):
        fam = r["label"][:2]
        col, m = mk_new.get(fam, ("k", "x"))
        x = 3.0 + 0.04 * (i - len(new) / 2)                              # pequeno desplazamiento para que se vean todos
        yr, er = ref(3.0, k)
        lab_ = f"{fam} (T=3, conv/hib.)" if fam not in seen else None; seen.add(fam)
        a0.errorbar(x, r[k], yerr=r.get(k + "_err", 0.0), fmt=m, ms=6, color=col, mec="k", mew=0.5, label=lab_)
        a1.errorbar(x, r[k] - yr, yerr=np.hypot(r.get(k + "_err", 0.0), er), fmt=m, ms=5, color=col, mec="k", mew=0.5)
    a0.set_ylabel(lab); a0.grid(alpha=0.3)
    a1.axhline(0, color="k", lw=0.8); a1.set_ylabel("modelo − Rényi"); a1.set_xlabel("T"); a1.grid(alpha=0.3)
ax[0, 0].legend(fontsize=7.5)
fig.suptitle("N=25: observables locales frente a T. Modelos antiguos (data/0) frente a los actuales a T=3, "
             "contra el Rényi-2 exacto a N=25")
plt.tight_layout()
out = os.path.join(ROOT, "plots/local_obs/obs_N25_old_vs_new.png")
plt.savefig(out, dpi=110)
print("guardado", out)
print("T=3, modelo - MRE:")
yr = {k: ref(3.0, k)[0] for k, _ in KEYS}
for chi, rows in sorted(old.items()):
    for T, r in rows:
        if abs(T - 3.0) < 1e-9:
            print(f"  data/0 chi{chi}: " + "  ".join(f"{k}={r[k] - yr[k]:+.4f}" for k, _ in KEYS))
for r in new:
    print(f"  {r['label']}: " + "  ".join(f"{k}={r[k] - yr[k]:+.4f}" for k, _ in KEYS))
