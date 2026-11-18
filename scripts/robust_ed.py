"""
robust_ed.py  (16-11-2026)
=================================
Bateria de robustez del sesgo inductivo de la ancilla (conv = local + equivariante) fuera del TFIM homogeneo.
Cadena 1D (PBC salvo --obc), gradiente EXACTO (rho completa, N <= 10), F2 = E - T S2 frente al MRE exacto.
Modelos (--model):
  tfim_hz   H = J sum ZZ + h sum X + hz sum Z                  (no integrable, sigue estocastico)
  tfim_dis  H = J sum ZZ + sum h_i X_i,  h_i = h * U(0.3, 1.7)  (desorden: rompe la traslacion; semilla fija)
  xxz       H = J sum (XX + YY + Delta ZZ), J > 0              (U(1), no estocastico en base Z)
  j1j2      H = J1 sum S.S(i,i+1) + J2 sum S.S(i,i+2)          (frustrado; sin regla de Marshall para J2 > 0)
Compara arquitecturas de ancilla (--archs, p.ej. vit:conv:r=1 vit:convg:r=1 vit:mlp) con el mismo sistema ARViT.
Uso (CPU):
  systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 env JAX_PLATFORMS=cpu .venv/bin/python -u \
    scripts/robust_ed.py --model xxz --N 10 --T 0.5 1.5 --archs vit:conv:r=1 vit:convg:r=1 vit:mlp \
    --out data/gap_arch/robust/xxz_N10.json
"""
import argparse, json, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from src_renyi.mps_ancilla import training
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla.models_alt import build_alt_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["tfim_hz", "tfim_dis", "xxz", "j1j2"])
ap.add_argument("--N", type=int, default=10); ap.add_argument("--obc", action="store_true")
ap.add_argument("--J", type=float, default=None); ap.add_argument("--h", type=float, default=-1.5)
ap.add_argument("--hz", type=float, default=-0.5); ap.add_argument("--Delta", type=float, default=1.0)
ap.add_argument("--J2", type=float, default=0.5); ap.add_argument("--dis_seed", type=int, default=0)
ap.add_argument("--T", type=float, nargs="+", default=[1.0])
ap.add_argument("--archs", nargs="+", default=["vit:conv:r=1", "vit:convg:r=1", "vit:mlp"])
ap.add_argument("--chis", type=int, nargs="+", default=[4]); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--seeds", type=int, nargs="+", default=[1]); ap.add_argument("--chunk", type=int, default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
N = a.N; D = 2 ** N
assert "/data/0/" not in os.path.abspath(a.out)
J = a.J if a.J is not None else (-1.0 if a.model.startswith("tfim") else 1.0)

# ── H en la base all_s (s_k = +-1 <-> bit k) ─────────────────────────────────
idx = np.arange(D); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
nn1 = [(i, (i + 1) % N) for i in range(N if not a.obc else N - 1)]
nn2 = [(i, (i + 2) % N) for i in range(N if not a.obc else N - 2)]
zz = sum(all_s[:, i] * all_s[:, j] for i, j in nn1) / len(nn1)               # <Z_i Z_i+1> medio (diagonal)
X = np.zeros((D, D))
for i in range(N): X[idx, idx ^ (1 << i)] += 1.0 / N                          # <X> medio


def heis(bonds, Jb, Delta=1.0):
    """J sum (XX + YY + Delta ZZ) en Paulis; XX+YY = 2 (S+S- + h.c.): conecta espines opuestos con elemento 2."""
    Hb = np.zeros((D, D))
    for i, j in bonds:
        si, sj = all_s[:, i], all_s[:, j]
        Hb[idx, idx] += Jb * Delta * si * sj
        flip = si != sj
        Hb[idx[flip], (idx ^ (1 << i) ^ (1 << j))[flip]] += 2 * Jb
    return Hb


info = dict(model=a.model, N=N, obc=a.obc, J=J)
if a.model == "tfim_hz":
    H = np.diag(J * len(nn1) * zz + a.hz * all_s.sum(1)) + a.h * N * X; info.update(h=a.h, hz=a.hz)
elif a.model == "tfim_dis":
    hi = a.h * np.random.default_rng(a.dis_seed).uniform(0.3, 1.7, N)
    H = np.diag(J * len(nn1) * zz)
    for i in range(N): H[idx, idx ^ (1 << i)] += hi[i]
    info.update(h=a.h, h_i=hi.tolist(), dis_seed=a.dis_seed)
elif a.model == "xxz":
    H = heis(nn1, J, a.Delta); info.update(Delta=a.Delta)
else:
    H = heis(nn1, J) + heis(nn2, a.J2 * J); info.update(J2=a.J2)
assert np.allclose(H, H.T)
ev, V = np.linalg.eigh(H)
print(f"{a.model} N={N} {'OBC' if a.obc else 'PBC'} {info}  E0/N={ev[0] / N:.6f}  gap={ev[1] - ev[0]:.4f}  "
      f"dispositivos={jax.devices()}", flush=True)


def obs(rho):
    rho = np.asarray(rho)
    return dict(E=float(np.real(np.sum(H.T * rho))), S2=float(-np.log(np.sum(np.abs(rho) ** 2))),
                zz=float(np.real(np.sum(zz * np.diagonal(rho)))), sx=float(np.real(np.sum(X.T * rho))),
                Svn=float(-sum(l * np.log(l) for l in np.clip(np.linalg.eigvalsh(rho), 1e-300, None))))


def mre(T):
    """Igual que 2D/ed2d.py: p ∝ (c - E)_+, c - <E> = 2T; con huecos, la raiz de minima F2."""
    best = (ev[0], 0.0, np.r_[1.0, np.zeros(D - 1)])
    S1c, S2c = np.cumsum(ev), np.cumsum(ev ** 2)
    for k in range(2, D + 1):
        S1, S2_ = S1c[k - 1], S2c[k - 1]
        disc = (S1 + T * k) ** 2 - k * (S2_ + 2 * T * S1)
        if disc < 0: continue
        hi_ = ev[k] if k < D else np.inf
        for c in ((S1 + T * k) + np.sqrt(disc)) / k, ((S1 + T * k) - np.sqrt(disc)) / k:
            if not (ev[k - 1] < c <= hi_ + 1e-12): continue
            w = np.maximum(c - ev, 0); p = w / w.sum()
            Em, S2m = (p * ev).sum(), -np.log((p ** 2).sum())
            if Em - T * S2m < best[0] - T * best[1]: best = (Em, S2m, p)
    return best


res = json.load(open(a.out)) if os.path.exists(a.out) else []
done = {(r["arch"], r["chi"], r["T"], r["seed"], r["steps"], r["lr"]) for r in res}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
for T in a.T:
    Em, S2m, pm = mre(T); rhoM = (V * pm) @ V.T; oM = obs(rhoM); FM = Em - T * S2m
    print(f"T={T}: MRE E/N={Em / N:.5f} S2={S2m:.4f} F2={FM:.5f} zz={oM['zz']:.4f} sx={oM['sx']:.4f}", flush=True)
    for arch in a.archs:
        for chi in a.chis:
            for seed in a.seeds:
                if (arch, chi, T, seed, a.steps, a.lr) in done: continue
                model = build_alt_model(N, chi, arch, a.d)
                params = init_params(model, N, seed)
                loss, rho_fn = training.make_loss_fns(model, all_s, H, T, chunk=a.chunk)
                tag = f"[{arch} chi={chi} T={T} s={seed}] "
                out = training.train(loss, params, a.steps, a.lr, T, freq=max(1, a.steps // 5), tag=tag)
                rho = np.asarray(rho_fn(out["params_best"])); rho = 0.5 * (rho + rho.conj().T); o = obs(rho)
                F = o["E"] - T * o["S2"]
                r = dict(**info, T=T, arch=arch, chi=chi, d=a.d, seed=seed, steps=a.steps, lr=a.lr,
                         n_params=training.count_params(params), **o, F2=F, E_mre=Em, S2_mre=S2m, F2_mre=FM,
                         zz_mre=oM["zz"], sx_mre=oM["sx"], dF=F - FM, dF_per_site=(F - FM) / N,
                         D_mre=float(0.5 * np.abs(np.linalg.eigvalsh(rho - rhoM)).sum()),
                         s_per_step=out["s_per_step"])
                print(f"{tag}=> dF={r['dF']:+.5f} ({r['dF_per_site']:+.5f}/sitio)  E={o['E']:.4f} (MRE {Em:.4f})  "
                      f"S2={o['S2']:.4f} (MRE {S2m:.4f})  D(rho,MRE)={r['D_mre']:.4f}  n_par={r['n_params']}  "
                      f"{1e3 * out['s_per_step']:.1f} ms/paso", flush=True)
                res.append(r); json.dump(res, open(a.out, "w"), indent=1)
