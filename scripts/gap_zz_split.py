"""
gap_zz_split.py
===============
Tabla de diagnostico del hueco (notas_gap.md) para runs de gap_mc.py:
  F, dF, dE, dS2 frente al MRE exacto; separacion dE = dE_ZZ + dE_X; <s_i s_i+1> del bulk y del enlace
  periodico (N-1,0); numero de parametros; ms/paso.

  dE_ZZ = -J_ZZ ... con J_ZZ=-1:  E_ZZ = -sum_i <s_i s_i+1>,  dE_ZZ = N zz_ex - sum_i <s_i s_i+1>_modelo
  dE_X  = dE - dE_ZZ          (zz_ex por Hellmann-Feynman sobre la DOS; error ~0.006/sitio, ver notas)

Uso:
  python scripts/gap_zz_split.py --results data/gap_arch/results.json [--labels a b ...]
  python scripts/gap_zz_split.py --params data/0/mps_N25/params/params_0030_chi4.msgpack --N 25 --chi 4 \
      --E -19.29 --S2 13.78 --F -60.64 --label data0_chi4        (estado suelto, sin entrada en results.json)
"""
import argparse, json, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from src_renyi.mps_ancilla.ladder import build_model, load_params
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.models import count_params
from src_renyi.mps_ancilla.training import make_sampler

# MRE exacto a T=3 (h_x=-1.5, J=-1, PBC): E, S2, F, <ZZ>/N (HF)
EXACT = {8: (-6.6355, 4.3204, -19.5967, 0.261), 16: (-13.8280, 8.6196, -39.6868, 0.263),
         25: (-22.0678, 13.4697, -62.4770, 0.267)}

ap = argparse.ArgumentParser()
ap.add_argument("--results")
ap.add_argument("--labels", nargs="*")
ap.add_argument("--params"); ap.add_argument("--N", type=int); ap.add_argument("--chi", type=int)
ap.add_argument("--d", type=int, default=16); ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--arch", default=None)
ap.add_argument("--E", type=float); ap.add_argument("--S2", type=float); ap.add_argument("--F", type=float)
ap.add_argument("--Ferr", type=float, default=float("nan"))
ap.add_argument("--label", default="estado")
ap.add_argument("--samples", type=int, default=65536)
ap.add_argument("--T", type=float, default=3.0)
args = ap.parse_args()

rows = []
if args.results:
    for e in json.load(open(args.results)):
        if args.labels and e["label"] not in args.labels:
            continue
        rows.append(dict(label=e["label"], arch=e.get("arch"), N=e["N"], chi=e["chi"], d=e["d"],
                         blocks=e["blocks"], params=e["params"], E=e["energy"], S2=e["entropy"],
                         F=e["free_energy"], Ferr=e["free_energy_err"], ms=1e3 * e["s_per_step"]))
if args.params:
    rows.append(dict(label=args.label, arch=args.arch, N=args.N, chi=args.chi, d=args.d, blocks=args.blocks,
                     params=args.params, E=args.E, S2=args.S2, F=args.F, Ferr=args.Ferr, ms=float("nan")))

print("| run | arch | chi | #par | F | dF | dE | dS2 | dE_ZZ | dE_X | ZZ bulk | ZZ periodico | ms/paso |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    N, chi = r["N"], r["chi"]
    m = (build_alt_model(N, chi, r["arch"], r["d"], 4, r["blocks"]) if r["arch"]
         else build_model(N, chi, r["d"], 4, r["blocks"]))
    p = load_params(r["params"], m, N)
    B = 4096
    samp = make_sampler(m, N, B)
    S = np.concatenate([np.asarray(samp(p, jax.random.PRNGKey(100 + k))) for k in range(args.samples // B)])
    b = (S * np.roll(S, -1, 1)).mean(0)
    Eex, Sex, Fex, zzex = EXACT[N]
    dE = r["E"] - Eex
    dEzz = N * zzex - b.sum()
    print(f"| {r['label']} | {r['arch'] or 'orig'} | {chi} | {count_params(p)} | {r['F']:.3f} +- {r['Ferr']:.3f} | "
          f"{r['F'] - Fex:+.2f} | {dE:+.2f} | {r['S2'] - Sex:+.2f} | {dEzz:+.2f} | {dE - dEzz:+.2f} | "
          f"{b[:-1].mean():.3f} | {b[-1]:.3f} | {r['ms']:.0f} |", flush=True)
