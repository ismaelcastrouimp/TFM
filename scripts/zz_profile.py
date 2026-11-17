"""
zz_profile.py
=============
Perfil enlace a enlace <s_i s_{i+1}> (enlace i = (i, i+1 mod N)) de estados entrenados, frente al exacto
(invariante traslacional: <ZZ>/N por Hellmann-Feynman, 0.267 a N=25 T=3).  Sirve para ver si el deficit de ZZ
del bulk se concentra cerca de los extremos del orden autorregresivo (efecto de borde) o es plano.
Solo depende de p(s) (muestreo autorregresivo exacto).

Uso:
  python scripts/zz_profile.py --run label=params.msgpack[@arch[@d[@blocks[@chi]]]] ... --N 25 --chi 4 --out data/gap_arch/zz_profile.json
  (arch "orig" = modelo original de models.py)
"""
import argparse, json, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from src_renyi.mps_ancilla.ladder import build_model, load_params
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.training import make_sampler

ZZ_EX = {8: 0.261, 16: 0.263, 25: 0.267}

ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs="+", required=True, help="label=params[@arch[@d[@blocks[@chi]]]]")
ap.add_argument("--N", type=int, default=25)
ap.add_argument("--samples", type=int, default=65536)
ap.add_argument("--out", required=True)
args = ap.parse_args()
assert "/data/0/" not in os.path.abspath(args.out) + "/", "no escribir en data/0"
N = args.N
res = {}
for spec in args.run:
    label, rest = spec.split("=", 1)
    parts = rest.split("@")
    path, arch = parts[0], (parts[1] if len(parts) > 1 else "orig")
    d = int(parts[2]) if len(parts) > 2 else 16
    blocks = int(parts[3]) if len(parts) > 3 else 2
    chi = int(parts[4]) if len(parts) > 4 else 4
    m = build_model(N, chi, d, 4, blocks) if arch == "orig" else build_alt_model(N, chi, arch, d, 4, blocks)
    p = load_params(path, m, N)
    B = 4096
    samp = make_sampler(m, N, B)
    S = np.concatenate([np.asarray(samp(p, jax.random.PRNGKey(300 + k))) for k in range(args.samples // B)])
    zz = S * np.roll(S, -1, 1)
    b = zz.mean(0); e = zz.std(0) / np.sqrt(len(S))
    res[label] = {"zz": b.tolist(), "err": e.tolist(), "mag": S.mean(0).tolist()}
    print(f"{label:22s} " + " ".join(f"{x:.2f}" for x in b) + f"   (err ~{e.mean():.3f})", flush=True)
print(f"{'exacto':22s} " + " ".join(f"{ZZ_EX[N]:.2f}" for _ in range(N)))
# resumen: primeros 3, centro, ultimos 3 enlaces abiertos, periodico
print("\n| run | enlaces 0-2 | 3-20 | 21-23 | periodico (24,0) | deficit total (sum zz_ex - zz) | de ello en 0-2 y 21-24 |")
print("|---|---|---|---|---|---|---|")
for label, r in res.items():
    b = np.array(r["zz"]); dfc = ZZ_EX[N] - b
    edge = dfc[:3].sum() + dfc[21:].sum()
    print(f"| {label} | {b[:3].mean():.3f} | {b[3:21].mean():.3f} | {b[21:24].mean():.3f} | {b[24]:.3f} | "
          f"{dfc.sum():+.2f} | {edge:+.2f} |")
json.dump(res, open(args.out, "w"), indent=1)
