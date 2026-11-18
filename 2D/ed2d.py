"""
2D/ed2d.py
==========
TFIM 2D (red Lx x Ly, PBC): H = J sum_<ij> Z_i Z_j + h sum_i X_i  (J=-1, h=-1.5 por defecto, como en 1D).
Entrena ARViT + ancilla MPS con gradiente EXACTO (rho completa; N = Lx*Ly <= ~12) minimizando F2 = E - T S2 y compara
con el MRE exacto (minimo de F2, espectro completo) y con el Gibbs.  Mismo bucle que scripts/gap_ed_small.py.
Reporta dF = F2 - F2_MRE, distancia de traza al MRE, <ZZ> de vecinos, <X>, y F_vN - F_Gibbs.
arch "vit:sbs:...": ancilla de varias strings (2D/models_sbs.py).
Uso:
  python 2D/ed2d.py --Lx 3 --Ly 3 --T 1 3 --archs vit:conv2d:r=1,win=plus vit:conv:r=1 --chis 4 8 --out data/2D/ed_3x3.json
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sd = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(sd, "..")); sys.path.insert(0, sd)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from src_renyi.mps_ancilla import training
from src_renyi.mps_ancilla.ladder import init_params
from models2d import build_model_2d
from models_sbs import build_model_sbs, make_loss_fns_sbs

ap = argparse.ArgumentParser()
ap.add_argument("--Lx", type=int, default=3); ap.add_argument("--Ly", type=int, default=3)
ap.add_argument("--h_x", type=float, default=-1.5); ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--T", type=float, nargs="+", default=[1.0, 3.0])
ap.add_argument("--archs", nargs="+", default=["vit:conv2d:r=1,win=plus"])
ap.add_argument("--chis", type=int, nargs="+", default=[4]); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--seeds", type=int, nargs="+", default=[1]); ap.add_argument("--chunk", type=int, default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
Lx, Ly = a.Lx, a.Ly; N = Lx * Ly; D = 2 ** N
assert Lx >= 3 and (Ly == 1 or Ly >= 3), "L=2 con PBC duplica enlaces"
assert "/data/0/" not in os.path.abspath(a.out)

# ── H en la base all_s (s_k = +-1 <-> bit k; k = x + Lx*y) ────────────────────
idx = np.arange(D); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
bonds = [(x + Lx * y, (x + 1) % Lx + Lx * y) for y in range(Ly) for x in range(Lx)]
if Ly > 1: bonds += [(x + Lx * y, x + Lx * ((y + 1) % Ly)) for y in range(Ly) for x in range(Lx)]
zz = sum(all_s[:, i] * all_s[:, j] for i, j in bonds) / len(bonds)          # <ZZ> medio por enlace (diagonal)
H = np.zeros((D, D)); H[idx, idx] = a.J_ZZ * len(bonds) * zz
X = np.zeros((D, D))
for i in range(N): X[idx, idx ^ (1 << i)] += 1.0 / N                          # <X> medio por sitio
H += a.h_x * N * X
ev, V = np.linalg.eigh(H)
print(f"TFIM 2D {Lx}x{Ly} PBC  N={N}  enlaces={len(bonds)}  J={a.J_ZZ} h={a.h_x}  E0/N={ev[0] / N:.6f}  "
      f"dispositivos={jax.devices()}", flush=True)


def obs(rho):
    rho = np.asarray(rho)
    return dict(E=float(np.real(np.sum(H.T * rho))), S2=float(-np.log(np.sum(np.abs(rho) ** 2))),
                zz=float(np.real(np.sum(zz * np.diagonal(rho)))), sx=float(np.real(np.sum(X.T * rho))),
                Svn=float(-sum(l * np.log(l) for l in np.clip(np.linalg.eigvalsh(rho), 1e-300, None))))


def mre(T):
    """p ∝ (c - E)_+ con c - <E> = 2T.  Soporte = k estados mas bajos, c en (e_{k-1}, e_k]:
    k c^2 - 2(S1 + T k) c + (S2 + 2 T S1) = 0.  Con huecos puede haber varias raices: la de minima F2 (o el puro)."""
    best = (ev[0], 0.0, np.r_[1.0, np.zeros(D - 1)])
    S1c, S2c = np.cumsum(ev), np.cumsum(ev ** 2)
    for k in range(2, D + 1):
        S1, S2_ = S1c[k - 1], S2c[k - 1]
        disc = (S1 + T * k) ** 2 - k * (S2_ + 2 * T * S1)
        if disc < 0: continue
        hi = ev[k] if k < D else np.inf
        for c in ((S1 + T * k) + np.sqrt(disc)) / k, ((S1 + T * k) - np.sqrt(disc)) / k:
            if not (ev[k - 1] < c <= hi + 1e-12): continue
            w = np.maximum(c - ev, 0); p = w / w.sum()
            Em, S2m = (p * ev).sum(), -np.log((p ** 2).sum())
            if Em - T * S2m < best[0] - T * best[1]: best = (Em, S2m, p)
    return best


res = json.load(open(a.out)) if os.path.exists(a.out) else []
done = {(r["arch"], r["chi"], r["T"], r["seed"], r["steps"], r["lr"], r["h_x"]) for r in res}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
for T in a.T:
    Em, S2m, pm = mre(T); rhoM = (V * pm) @ V.T; oM = obs(rhoM); FM = Em - T * S2m
    pG = np.exp(-(ev - ev[0]) / T); pG /= pG.sum(); rhoG = (V * pG) @ V.T; oG = obs(rhoG)
    FvG = oG["E"] - T * oG["Svn"]
    print(f"T={T}: MRE E/N={Em / N:.5f} S2={S2m:.4f} F2={FM:.5f} zz={oM['zz']:.4f} sx={oM['sx']:.4f} | "
          f"Gibbs E/N={oG['E'] / N:.5f} S2={oG['S2']:.4f} Svn={oG['Svn']:.4f} zz={oG['zz']:.4f} sx={oG['sx']:.4f}", flush=True)
    for arch in a.archs:
        for chi in a.chis:
            for seed in a.seeds:
                if (arch, chi, T, seed, a.steps, a.lr, a.h_x) in done: continue
                sbs = arch.split(":")[1] == "sbs"          # ancilla string-bond (2D/models_sbs.py)
                model = (build_model_sbs if sbs else build_model_2d)(Lx, Ly, chi, arch, a.d)
                params = init_params(model, N, seed)
                loss, rho_fn = (make_loss_fns_sbs if sbs else training.make_loss_fns)(model, all_s, H, T, chunk=a.chunk)
                tag = f"[{arch} chi={chi} T={T} s={seed}] "
                out = training.train(loss, params, a.steps, a.lr, T, freq=max(1, a.steps // 5), tag=tag)
                rho = np.asarray(rho_fn(out["params_best"])); rho = 0.5 * (rho + rho.conj().T); o = obs(rho)
                F = o["E"] - T * o["S2"]
                r = dict(Lx=Lx, Ly=Ly, N=N, h_x=a.h_x, J_ZZ=a.J_ZZ, T=T, arch=arch, chi=chi, d=a.d, seed=seed,
                         steps=a.steps, lr=a.lr, n_params=training.count_params(params), **o, F2=F,
                         E_mre=Em, S2_mre=S2m, F2_mre=FM, zz_mre=oM["zz"], sx_mre=oM["sx"],
                         E_G=oG["E"], Svn_G=oG["Svn"], zz_G=oG["zz"], sx_G=oG["sx"],
                         dF=F - FM, dF_per_site=(F - FM) / N,
                         D_mre=float(0.5 * np.abs(np.linalg.eigvalsh(rho - rhoM)).sum()),
                         dFvn_G=(o["E"] - T * o["Svn"]) - FvG, s_per_step=out["s_per_step"])
                print(f"{tag}=> dF={r['dF']:+.5f} ({r['dF_per_site']:+.5f}/sitio)  E={o['E']:.4f} (MRE {Em:.4f})  "
                      f"S2={o['S2']:.4f} (MRE {S2m:.4f})  zz={o['zz']:.4f} ({oM['zz']:.4f})  sx={o['sx']:.4f} "
                      f"({oM['sx']:.4f})  D(rho,MRE)={r['D_mre']:.4f}  {1e3 * out['s_per_step']:.1f} ms/paso", flush=True)
                res.append(r); json.dump(res, open(a.out, "w"), indent=1)
