"""
obs_vs_N.py
===========
Desviacion de los observables locales del modelo respecto al MRE (y Gibbs) EXACTOS del mismo N, a la MISMA E/N
(interpolacion en la rejilla data/gap_arch/obs/ff_N{N}_grid.json de scripts/ff_local_obs.py).  Pares del bulk.
Uso: python scripts/obs_vs_N.py data/gap_arch/obs/model_Nsweep.json data/gap_arch/obs/model_N25.json ...
"""
import json, os, sys
import numpy as np

KEYS = [("sx", "sx"), ("szz1", "szz1_bulk"), ("szz2", "szz2_bulk"), ("szz3", "szz3_bulk"), ("sxx_conn", "sxx_conn_bulk")]
refs = {}


def ref(N, kind):
    if (N, kind) not in refs:
        p = f"data/gap_arch/obs/ff_N{N}_grid.json"
        if not os.path.exists(p): return None
        rr = sorted([x for x in json.load(open(p)) if x["kind"] == kind and x["target"].startswith("g")],
                    key=lambda x: x["E_per_site"])
        e = np.array([x["E_per_site"] for x in rr])
        refs[(N, kind)] = {k: (e, np.array([x[k] for x in rr])) for k, _ in KEYS}
    return refs[(N, kind)]


print(f"{'modelo':22s} {'N':>3s} {'chi':>3s} {'E/N':>7s} | dX(MRE) dZZ1    dZZ2    dZZ3    dconnXX | dZZ1(Gibbs)  err_ZZ1")
for f in sys.argv[1:]:
    for x in json.load(open(f)):
        N = x["N"]; m, g = ref(N, "mre"), ref(N, "gibbs")
        if m is None:
            print(f"{x['label']:22s} {N:3d}  (sin referencia)"); continue
        e = x["E_per_site"]
        d = [x[km] - np.interp(e, *m[kf]) for kf, km in KEYS]
        dg = x["szz1_bulk"] - np.interp(e, *g["szz1"])
        print(f"{x['label']:22s} {N:3d} {x['chi']:3d} {e:+.3f} | " + " ".join(f"{v:+.4f}" for v in d)
              + f" | {dg:+.4f}     {x['szz1_bulk_err']:.4f}")
