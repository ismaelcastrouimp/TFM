"""
eval_alt.py: evaluacion MC (E, S2 nbr1) de parametros de models_alt, sin entrenar.  La S2 buena se saca con
s2_bias_check.py (nbr2); aqui sobre todo interesa E.
  python scripts/eval_alt.py --params P.msgpack --arch vit:conv:r=1 --N 25 --chi 4 --out data/gap_arch/eval_X.json
"""
import argparse, json, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax
jax.config.update("jax_enable_x64", True)
from src_renyi.mps_ancilla.ladder import load_params, make_evaluator, evaluate
from src_renyi.mps_ancilla.models_alt import build_alt_model
ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--arch", required=True)
ap.add_argument("--N", type=int, required=True); ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--T", type=float, default=3.0); ap.add_argument("--M", type=int, default=4096)
ap.add_argument("--batches", type=int, default=32); ap.add_argument("--pair_block", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert "/data/0/" not in os.path.abspath(a.out), "no escribir en data/0"
m = build_alt_model(a.N, a.chi, a.arch)
p = load_params(a.params, m, a.N)
one = make_evaluator(m, a.N, a.T, -1.5, -1.0, a.M, a.pair_block, purity="nbr")
r = evaluate(one, p, a.T, a.batches, seed=4242)
r.update(params=a.params, arch=a.arch, N=a.N, chi=a.chi)
print(json.dumps(r, indent=1))
json.dump(r, open(a.out, "w"), indent=1)
