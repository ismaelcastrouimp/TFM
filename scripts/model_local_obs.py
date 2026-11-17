"""
model_local_obs.py
==================
Observables locales del estado variacional (ARViT + ancilla MPS, models_alt) por MC exacto (muestreo autorregresivo):
  <X>          = E_s[ (1/N) sum_i r(s -> s^i) ]                    (estimador no diagonal, NO deducido de E)
  <X_i X_i+1>  = E_s[ (1/N) sum_i r(s -> s^{i,i+1}) ]
  <Z_i Z_i+r>  = E_s[ (1/N) sum_i s_i s_{i+r} ],  r = 1..R          (PBC, todas las posiciones)
  conexa X     = <X_i X_i+1> - <X>^2
  E/N          = (J_ZZ <ZZ_1> + h_x <X>) (comprobacion frente a la evaluacion precisa)
r(s->t) = exp(logA_t - logA_s) <phi_s|phi_t> / (|phi_s| |phi_t|)   (mismo que local_energy del notebook / mc_loss).
Comparar con scripts/ff_local_obs.py (Gibbs y MRE exactos a la MISMA energia).

Uso: python scripts/model_local_obs.py --N 25 --samples 16384 --out data/gap_arch/obs/model_N25.json \
        --run X5_s1:data/gap_arch/params_X5_conv_chi8_from_N16.msgpack:8 ...   (etiqueta:params:chi[:arch])
     arch "orig" = ARViT_MPSAncilla original (models.py / ladder.build_model, ancilla con contexto; p.ej. data/0)
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "..")); sys.path.insert(0, script_dir)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.models import mps_overlap_aligned
from src_renyi.mps_ancilla.ladder import load_params, build_model
from src_renyi.mps_ancilla.training import make_sampler

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=25); ap.add_argument("--R", type=int, default=5)
ap.add_argument("--h_x", type=float, default=-1.5); ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--d", type=int, default=16); ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--samples", type=int, default=16384); ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--chunk", type=int, default=64); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--run", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
N, R = a.N, a.R
assert "/data/0/" not in os.path.abspath(a.out)
print(f"N={N} dispositivos={jax.devices()}", flush=True)

single = np.ones((N, N)); single[np.arange(N), np.arange(N)] = -1
pair = np.ones((N, N)); pair[np.arange(N), np.arange(N)] = -1; pair[np.arange(N), (np.arange(N) + 1) % N] = -1
single, pair = jnp.asarray(single), jnp.asarray(pair)

res = json.load(open(a.out)) if os.path.exists(a.out) else []
for spec in a.run:
    parts = spec.split(":"); lab, pth, chi = parts[0], parts[1], int(parts[2])
    arch = ":".join(parts[3:]) if len(parts) > 3 else "vit:conv:r=1"
    model = build_model(N, chi, a.d, 4, a.blocks) if arch == "orig" else build_alt_model(N, chi, arch, a.d, 4, a.blocks)
    params = load_params(pth, model, N)
    sampler = make_sampler(model, N, a.batch)

    @jax.jit
    def estimate(params, s):
        B = s.shape[0]
        logA, M = model.apply({"params": params}, s)
        n2 = jnp.real(mps_overlap_aligned(M, M, chi))

        def ratios(masks):
            K = masks.shape[0]
            f = (s[:, None, :] * masks[None]).reshape(B * K, N)
            logAf, Mf = model.apply({"params": params}, f)
            ov = mps_overlap_aligned(Mf, jnp.repeat(M, K, axis=0), chi)
            n2f = jnp.real(mps_overlap_aligned(Mf, Mf, chi))
            ov = ov / jnp.sqrt(n2f * jnp.repeat(n2, K) + 1e-300)
            return jnp.real(jnp.exp(logAf - jnp.repeat(logA, K)) * ov).reshape(B, K)

        out = {"sx": ratios(single), "sxx": ratios(pair)}           # por sitio (B, N); par i <-> (i, i+r mod N)
        for r in range(1, R + 1):
            out[f"szz{r}"] = s * jnp.roll(s, -r, axis=1)
        return out

    t0 = time.time(); vals = {}
    key = jax.random.PRNGKey(a.seed)
    for b in range(a.samples // a.batch):
        key, sub = jax.random.split(key)
        s = sampler(params, sub)
        for st in range(0, a.batch, a.chunk):
            o = estimate(params, s[st:st + a.chunk])
            for k, v in o.items():
                vals.setdefault(k, []).append(np.asarray(v))
    prof = {k: np.concatenate(v) for k, v in vals.items()}          # (ns, N)
    vals = {k: v.mean(1) for k, v in prof.items()}                     # promedio sobre todos los pares (PBC)
    rlag = {"sx": 0, "sxx": 1, **{f"szz{q}": q for q in range(1, R + 1)}}
    for k, v in prof.items():                                          # bulk = pares que no cruzan la frontera N-1 -> 0
        q = rlag[k]
        vals[k + "_bulk"] = v[:, :N - q].mean(1) if q > 0 else v.mean(1)
        if q > 0: vals[k + "_bnd"] = v[:, N - q:].mean(1)
    ns = len(vals["sx"])
    r = dict(label=lab, params=pth, chi=chi, arch=arch, N=N, samples=ns, secs=time.time() - t0)
    for k, v in vals.items():
        r[k] = float(v.mean()); r[k + "_err"] = float(v.std() / np.sqrt(ns))
    # conexa: <XX> - <X>^2 ; error por propagacion (sx y sxx de las mismas muestras -> covarianza)
    cov = np.cov(vals["sxx"], vals["sx"])
    r["sxx_conn"] = r["sxx"] - r["sx"] ** 2
    r["sxx_conn_bulk"] = r["sxx_bulk"] - r["sx"] ** 2
    r["profiles"] = {k: v.mean(0).tolist() for k, v in prof.items()}
    gvec = np.array([1.0, -2 * r["sx"]])
    r["sxx_conn_err"] = float(np.sqrt(gvec @ cov @ gvec / ns))
    eloc = a.J_ZZ * vals["szz1"] + a.h_x * vals["sx"]
    r["E_per_site"] = float(eloc.mean()); r["E_per_site_err"] = float(eloc.std() / np.sqrt(ns))
    print(f"[{lab}] chi={chi} E/N={r['E_per_site']:+.5f}+-{r['E_per_site_err']:.5f} (E={N * r['E_per_site']:.3f}) "
          + " ".join(f"{k}={r[k]:+.5f}+-{r[k + '_err']:.5f}" for k in ["sx", "sxx_bulk", "sxx_bnd"] + [f"szz{q}_bulk" for q in range(1, R + 1)] + ["szz1_bnd"])
          + f" conn_bulk={r['sxx_conn_bulk']:+.5f}"
          + f"  ({r['secs']:.0f} s)", flush=True)
    res.append(r); json.dump(res, open(a.out, "w"), indent=1)
print("FIN", flush=True)
