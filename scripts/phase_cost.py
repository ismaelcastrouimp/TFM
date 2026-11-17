"""
phase_cost.py
=============
Cuanta fase tiene el sistema (ARViT phase_head) y cuanto cuesta en F.
TFIM con h_x<0, J<0 es estocastico: rho_MRE >= 0 elemento a elemento, la fase optima es 0.
La pureza sum|rho(s,s')|^2 NO depende de la fase (|A(s)A*(s')G|^2), y ZZ tampoco: la fase solo entra en E por el
termino transverso.  Quitar la fase (theta -> 0) cambia F exactamente en
    dF_fase = h_x * N * (<X>_sin_fase - <X>_con_fase)      (<X> = E_s[(1/N) sum_i Re r(s->s^i)])
con r(s->t) = |A_t/A_s| e^{i(theta_t-theta_s)} <phi_s|phi_t>/(|phi_s||phi_t|)  (como model_local_obs.py).
Tambien da: std de theta(s) y <cos(dtheta)> pesado por |r| en los flips de un espin.
Uso: python scripts/phase_cost.py --N 25 --samples 16384 --run X5_s1:data/gap_arch/params_X5...msgpack:8[:arch] ...
"""
import argparse, json, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.models import mps_overlap_aligned
from src_renyi.mps_ancilla.ladder import load_params
from src_renyi.mps_ancilla.training import make_sampler

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=25); ap.add_argument("--h_x", type=float, default=-1.5)
ap.add_argument("--samples", type=int, default=16384); ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--seed", type=int, default=7); ap.add_argument("--run", nargs="+", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()
N = a.N
single = np.ones((N, N)); single[np.arange(N), np.arange(N)] = -1; single = jnp.asarray(single)
rows = []
for spec in a.run:
    p = spec.split(":"); lab, pth, chi = p[0], p[1], int(p[2]); arch = ":".join(p[3:]) or "vit:conv:r=1"
    model = build_alt_model(N, chi, arch); params = load_params(pth, model, N)
    sampler = make_sampler(model, N, a.batch)

    @jax.jit
    def est(params, s):
        B = s.shape[0]
        logA, M = model.apply({"params": params}, s)
        n2 = jnp.real(mps_overlap_aligned(M, M, chi))
        f = (s[:, None, :] * single[None]).reshape(B * N, N)
        logAf, Mf = model.apply({"params": params}, f)
        ov = mps_overlap_aligned(Mf, jnp.repeat(M, N, axis=0), chi)
        ov = ov / jnp.sqrt(jnp.real(mps_overlap_aligned(Mf, Mf, chi)) * jnp.repeat(n2, N) + 1e-300)
        d = logAf - jnp.repeat(logA, N)
        amp = jnp.exp(jnp.real(d))
        r_ph = jnp.real(amp * jnp.exp(1j * jnp.imag(d)) * ov)       # con fase (el estimador de verdad)
        r_no = jnp.real(amp * ov)                                    # fase del sistema puesta a 0
        w = jnp.abs(amp * ov)
        return (r_ph.reshape(B, N).mean(1), r_no.reshape(B, N).mean(1),
                (w * jnp.cos(jnp.imag(d))).reshape(B, N).sum(1), w.reshape(B, N).sum(1),
                jnp.imag(logA), jnp.abs(jnp.imag(ov)).reshape(B, N).mean(1))

    key = jax.random.PRNGKey(a.seed); acc = [[] for _ in range(6)]
    for _ in range(a.samples // a.batch):
        key, sub = jax.random.split(key)
        for L, v in zip(acc, est(params, sampler(params, sub))):
            L.append(np.asarray(v))
    xph, xno, wc, w, th, imov = [np.concatenate(L) for L in acc]
    dx = xno - xph
    r = dict(label=lab, params=pth, chi=chi, arch=arch, N=N, samples=len(xph),
             sx=float(xph.mean()), sx_nophase=float(xno.mean()),
             dsx=float(dx.mean()), dsx_err=float(dx.std() / np.sqrt(len(dx))),
             dF_phase=float(a.h_x * N * dx.mean()), dF_phase_err=float(abs(a.h_x) * N * dx.std() / np.sqrt(len(dx))),
             theta_std=float(np.std(th)), cos_dtheta_w=float(wc.sum() / w.sum()), imag_ov=float(imov.mean()))
    rows.append(r)
    print(f"{lab}: std(theta)={r['theta_std']:.4f}  <cos dtheta>_w={r['cos_dtheta_w']:.6f}  "
          f"sx={r['sx']:.5f} sx_sin_fase={r['sx_nophase']:.5f}  dF_fase={r['dF_phase']:+.5f}+-{r['dF_phase_err']:.5f}  "
          f"|Im ov|={r['imag_ov']:.1e}", flush=True)
if a.out:
    assert "/data/0/" not in os.path.abspath(a.out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(rows, open(a.out, "w"), indent=1)
