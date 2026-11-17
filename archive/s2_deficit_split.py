"""
s2_deficit_split.py
===================
Descompone EXACTAMENTE (N <= 12) el deficit de S2 a energia fija, S2_MRE(E) - S2(rho), en tres partes, usando la
base propia de H (|n>, E_n) y la rho completa del modelo:

  rho                      -> S2(rho)
  rho_d   = diag_H(rho)    -> S2(rho_d)   >= S2(rho)    [coherencias entre autoestados]
  rho_sh  = media en capas -> S2(rho_sh)  >= S2(rho_d)  [peso no uniforme dentro de cada capa de energia degenerada]
  MRE(T_E)                 -> S2_MRE(E)   >= S2(rho_sh) (para E fija el MRE es el maximo de S2 en distribuciones
                                                        diagonales; la diferencia es la FORMA de q(E))
  deficit = [S2_MRE(E) - S2(rho_sh)]  +  [S2(rho_sh) - S2(rho_d)]  +  [S2(rho_d) - S2(rho)]
              forma de q(E)              no uniforme en capa         coherencias
Capas = autovalores degenerados (|dE| < tol).  Tambien se da la cota de varianza (s2_bound_var.py) si se pide.

Uso:  python scripts/s2_deficit_split.py --params P --arch vit:conv:r=1 --N 8 --chi 4 [--T 3] [--chunk 128]

Memoria (9 nov, tras OOM a N=12): sin trozos, mps_overlap_rect crea intermedios (S,S,2,chi,chi) complejos:
S=4096, chi=4 -> ~8.6 GB por array (+ v de 4.3 GB) -> pico ~19 GB.  Ahora rho se calcula por bloques de filas
(--chunk, por defecto 128): intermedio ~ chunk*S*2*chi^2*16 B = 0.27 GB a N=12, chi=4.  Las matrices densas que
quedan (H, V reales; rho, V^T rho V complejas, 4096^2) suman ~1 GB.  Pico total estimado < 2.5 GB a N=12.
"""
import argparse, json, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "..")); sys.path.insert(0, script_dir)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import brentq

from src_renyi.mps_ancilla.ladder import build_model, load_params
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.training import compute_rho
from mre_teff import MRE

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True); ap.add_argument("--arch", default=None)
ap.add_argument("--N", type=int, required=True); ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--d", type=int, default=16); ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--h_x", type=float, default=-1.5); ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--tol", type=float, default=1e-8)
ap.add_argument("--chunk", type=int, default=128, help="filas por bloque al construir rho (memoria)")
ap.add_argument("--label", default=None); ap.add_argument("--out", default=None)
a = ap.parse_args()
N = a.N; assert N <= 12
if a.out: assert "/data/0/" not in os.path.abspath(a.out)

model = (build_alt_model(N, a.chi, a.arch, a.d, 4, a.blocks) if a.arch else build_model(N, a.chi, a.d, 4, a.blocks))
params = load_params(a.params, model, N)

idx = np.arange(2 ** N)
bits = (idx[:, None] >> np.arange(N)[None, :]) & 1
all_s = (2 * bits - 1).astype(np.float64)
H = np.zeros((2 ** N, 2 ** N))
H[idx, idx] = a.J_ZZ * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
for i in range(N):
    H[idx, idx ^ (1 << i)] += a.h_x
ev, V = np.linalg.eigh(H)

del bits, idx
chunk = min(a.chunk, 2 ** N)
rho = np.asarray(jax.jit(lambda p, s: compute_rho(model, p, s, chunk))(params, jnp.asarray(all_s)))
rho = rho / np.trace(rho).real
jax.clear_caches()
S2 = lambda r: float(-np.log(np.real(np.sum(np.abs(r) ** 2))))
E = float(np.real(np.sum(H.T * rho)))           # Tr(rho H) sin producto matricial
del H
rH = V.T @ (rho @ V)                            # rho en la base propia de H (V real)
q = np.real(np.diag(rH))                        # q_n = <n|rho|n>
S2_rho = S2(rho)
del rho
S2_d = float(-np.log(np.sum(q ** 2)))
# capas degeneradas
order = np.argsort(ev); evs = ev[order]; qs = q[order]
shell = np.concatenate([[0], np.cumsum(np.diff(evs) > a.tol)])
nsh = shell.max() + 1
Qsh = np.bincount(shell, qs, nsh); gsh = np.bincount(shell, minlength=nsh)
S2_sh = float(-np.log(np.sum(Qsh ** 2 / gsh)))
var = float(np.sum(q * ev ** 2) - E ** 2)

mre = MRE(ev)
TE = brentq(lambda t: mre(t)[0] - E, 0.05, 50)
S2_mre_E = mre(TE)[1]
Ex, Sx, Fx = mre(a.T)
F = E - a.T * S2_rho
out = dict(label=a.label, N=N, chi=a.chi, arch=a.arch, T=a.T, E=E, S2=S2_rho, F=F, dF=F - Fx, dE=E - Ex,
           dS2=S2_rho - Sx, T_E=TE, VarH=var, S2_MRE_at_E=S2_mre_E,
           deficit_total=S2_mre_E - S2_rho,
           deficit_forma_qE=S2_mre_E - S2_sh,
           deficit_capa_no_uniforme=S2_sh - S2_d,
           deficit_coherencias=S2_d - S2_rho,
           n_capas=int(nsh))
tot = out["deficit_total"]
print(f"[{a.label}] N={N} chi={a.chi}: dF={out['dF']:+.4f} dE={out['dE']:+.4f} dS2={out['dS2']:+.4f} T_E={TE:.3f}")
print(f"  deficit S2 a E fija = {tot:+.5f}  =  forma q(E) {out['deficit_forma_qE']:+.5f} ({out['deficit_forma_qE']/tot:.0%})"
      f"  +  no uniforme en capa {out['deficit_capa_no_uniforme']:+.5f} ({out['deficit_capa_no_uniforme']/tot:.0%})"
      f"  +  coherencias {out['deficit_coherencias']:+.5f} ({out['deficit_coherencias']/tot:.0%})")
print(f"  T*deficit = {a.T*tot:+.4f};  termino de T equivocada = dF - T*deficit = {out['dF'] - a.T*tot:+.4f}")
f = lambda c: c - mre._sums(c)[1] / mre._sums(c)[0] - 2 * TE
c_cut = brentq(f, ev.min() + 1e-9, ev.max() + 20 * TE + 10, xtol=1e-13)
out.update(corte_c=float(c_cut), peso_sobre_corte=float(q[ev > c_cut].sum()))
print(f"  corte del MRE(T_E): c = {c_cut:.4f};  peso del modelo por encima del corte = {out['peso_sobre_corte']:.4f}")
if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    # q_n del modelo y autovalores (para scripts/plot_qE_hist.py)
    np.savez(os.path.splitext(a.out)[0] + "_qE.npz", ev=ev, q=q, c=c_cut, T_E=TE, label=str(a.label), N=N)
