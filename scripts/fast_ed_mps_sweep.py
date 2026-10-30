"""
fast_ed_mps_sweep.py
====================
Barrido de la bond dimension chi del MPS de ancilla (ARViT + MPS, traza exacta), N=8, T=1.5.
Requiere netket solo para H y all_states.

Referencias (tus resultados):
  MRE exacto                      F = -14.4255
  suelo amplitudes >= 0           F = -14.337
  mean-field (d=64)               F = -14.2916  (gap +0.134, fid 0.742, igual en 3 seeds)
  ancilla fisica ViT d=16 / d=32  mejor F = -14.361 / -14.396 (seed 3), seeds 1-2 ~ -14.31 / -14.32
chi=1 con complex=True debe reproducir la meseta del mean-field (comprobacion de coherencia).
"""
import json, os
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmaz
from scipy.optimize import brentq
from scipy.linalg import eigh

# import fast_ed_mps_ancilla as fm
import src_renyi.mps_ancilla.models as models
import src_renyi.mps_ancilla.training as training

# ── Configuracion ─────────────────────────────────────────────────────────────
N, T = 8, 1.5
J_ZZ, J_XX, h_x, h_z = -1.0, 0.0, -1.5, 0.0
N_SYS, N_ANC = N, N          # con MPS no hace falta mas ancilla que N espines (chi limita)

N_STEPS = 3000
LR      = 1e-2
FREQ    = 100
SEEDS   = [1, 2, 3]
TRANSFORMER = (16, 4, 2)     # (embedding_d, heads, blocks) del sistema; subir a (32,4,2) si hace falta
# (chi, complex_mps)
VARIANTS = [(1, True), (2, False), (4, False), (8, False), (2, True), (4, True)]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots", "fast_ed_mps")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Referencia exacta ─────────────────────────────────────────────────────────
hi_sys = nk.hilbert.Spin(s=1/2, N=N_SYS)
H_op = 0
for i in range(N_SYS):
    H_op += h_x * sigmax(hi_sys, i) + h_z * sigmaz(hi_sys, i) \
          + J_ZZ * sigmaz(hi_sys, i) @ sigmaz(hi_sys, (i + 1) % N_SYS) \
          + J_XX * sigmax(hi_sys, i) @ sigmax(hi_sys, (i + 1) % N_SYS)
H_dense = np.array(H_op.to_dense())
E_vals, V = eigh(H_dense)

def _mre(T):
    def sc(c):
        w = np.maximum(c - E_vals, 0); return c - (w * E_vals).sum() / w.sum() - 2 * T
    c = brentq(sc, E_vals.min() + 1e-6, E_vals.max() + 20 * T + 10, xtol=1e-12)
    w = np.maximum(c - E_vals, 0); p = w / w.sum()
    return p, (p * E_vals).sum(), -np.log((p ** 2).sum())

p_mre, E_EX, S2_EX = _mre(T)
F_EX = E_EX - T * S2_EX
rho_MRE = (V * p_mre) @ V.conj().T
F_POS_FLOOR = -14.337 if (N, T, J_ZZ, h_x) == (8, 1.5, -1.0, -1.5) else None
print(f"MRE exacto: F={F_EX:.4f}  (suelo positivo: {F_POS_FLOOR})")

def fidelity(rho, sigma):
    rho = np.array(rho); tr = np.trace(rho)
    if not np.isfinite(tr) or abs(tr) < 1e-30:
        return 0.0
    rho = rho / tr
    sigma = np.array(sigma); sigma = sigma / np.trace(sigma)
    ev, U = np.linalg.eigh(rho)
    sq = (U * np.sqrt(np.clip(ev, 0, None))) @ U.conj().T
    evm = np.linalg.eigvalsh(sq @ sigma @ sq)
    return float(np.sum(np.sqrt(np.clip(evm, 0, None))) ** 2)

# mismo orden que H_dense (numeracion de NetKet del sistema)
all_s = np.asarray(hi_sys.all_states())

def init_params(model, seed):
    return model.init(jax.random.PRNGKey(seed * 1000 + 42), jnp.asarray(all_s))["params"]

# ── Run ───────────────────────────────────────────────────────────────────────
def run(chi, cplx, seed):
    d, heads, blocks = TRANSFORMER
    model = models.ARViT_MPSAncilla(N_sys=N_SYS, N_anc=N_ANC, chi=chi, embedding_d=d, n_heads=heads,
                                n_blocks=blocks, complex_mps=cplx)
    params = init_params(model, seed)
    loss, rho_fn = training.make_loss_fns(model, all_s, H_dense, T)

    def diag(p):
        rho = np.array(rho_fn(p))
        neg = float((rho.real < -1e-12).mean())
        im = float(np.linalg.norm(rho.imag) / np.linalg.norm(rho))
        return f"Fid={fidelity(rho, rho_MRE):.4f} | neg={neg:.3f} | |Im rho|/|rho|={im:.4f}"

    tag = f"[chi={chi},cplx={int(cplx)},s={seed}] "
    out = training.train(loss, params, N_STEPS, LR, T, freq=FREQ, diag_fn=diag, tag=tag)
    rho = np.array(rho_fn(out["params_best"]))
    res = dict(chi=chi, cplx=cplx, d=d, seed=seed, n_params=models.count_params(params),
               F=out["F"], E=out["E"], S2=out["S2"],
               gap_exact=out["F"] - F_EX,
               gap_floor=None if F_POS_FLOOR is None else out["F"] - F_POS_FLOOR,
               fid=fidelity(rho, rho_MRE), neg_frac=float((rho.real < -1e-12).mean()),
               im_ratio=float(np.linalg.norm(rho.imag) / np.linalg.norm(rho)),
               s_per_step=out["s_per_step"])
    print(f"{tag}FINAL F={res['F']:.5f} gap_exact={res['gap_exact']:+.4f} fid={res['fid']:.4f} "
          f"neg={res['neg_frac']:.3f} ({res['s_per_step']*1e3:.1f} ms/paso)\n", flush=True)
    return res

if __name__ == "__main__":
    RES_PATH = os.path.join(OUT_DIR, "results.json")
    results = json.load(open(RES_PATH)) if os.path.exists(RES_PATH) else []
    done = {(r["chi"], r["cplx"], r["d"], r["seed"]) for r in results}
    for (chi, cplx) in VARIANTS:
        for seed in SEEDS:
            if (chi, cplx, TRANSFORMER[0], seed) in done:
                print(f"[saltado] chi={chi} cplx={cplx} s={seed}")
                continue
            results.append(run(chi, cplx, seed))
            with open(RES_PATH, "w") as f:
                json.dump(results, f, indent=1)

    print("=" * 100)
    print(f"{'chi':>3} {'cplx':>4} {'d':>3} {'seed':>4} {'#par':>6} {'F':>10} {'gap_ex':>8} "
          f"{'gap_fl':>8} {'E':>9} {'S2':>7} {'fid':>6} {'neg':>5} {'Im/rho':>7}")
    for r in results:
        gf = "   n/d" if r["gap_floor"] is None else f"{r['gap_floor']:+.4f}"
        print(f"{r['chi']:>3} {int(r['cplx']):>4} {r['d']:>3} {r['seed']:>4} {r['n_params']:>6} "
              f"{r['F']:>10.4f} {r['gap_exact']:>+8.4f} {gf:>8} {r['E']:>9.4f} {r['S2']:>7.4f} "
              f"{r['fid']:>6.3f} {r['neg_frac']:>5.2f} {r['im_ratio']:>7.4f}")