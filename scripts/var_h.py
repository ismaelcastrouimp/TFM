"""
var_h.py
========
Varianza de la energia Var(H) = Tr(rho H^2) - Tr(rho H)^2 del estado ARViT + ancilla MPS, por MC, y comparacion
con el MRE exacto (scripts/mre_teff.py) y con Gibbs a la misma T.

Motivacion (notas_gap.md, Lu2025): el MRE (p ~ (c-E)_+) tiene Var(H) subextensiva; Gibbs, extensiva. Si el modelo
sale con Var(H) tipo Gibbs, el deficit de S2 a E fija es estructural del ansatz local.

Estimador.  H = D + hx sum_i X_i, con D(s) = J sum_i s_i s_{i+1}.  Con r(s->t) la misma razon que usa la energia local
de mc_loss.py (rho(s,t)/rho(s,s) salvo conjugacion; se toma parte real al final):
    E_loc(s)  = D(s) + hx sum_i r(s->s^i)
    E2_loc(s) = D(s)^2 + N hx^2 + hx sum_i (D(s) + D(s^i)) r(s->s^i) + 2 hx^2 sum_{i<j} r(s->s^{ij})
    Tr(rho H^n) = E_{s~p}[En_loc(s)].
Var(H) por lote = <E2_loc> - <E_loc>^2 + var(E_loc)/M (correccion de sesgo), promediado sobre lotes
(error = std/sqrt(n)).  Los json del 9 nov (data/gap_arch/varh) usaban M/(M-1): ~+0.03 de mas a N=25, despreciable.

Validacion: --check_exact (N <= 10) construye rho exacta (training.compute_rho) y compara.

Uso:
  python scripts/var_h.py --params P.msgpack --arch vit:conv:r=1 --N 25 --chi 8 --M 512 --batches 32 \
      --out data/gap_arch/varh/varh_X.json
  (--arch omitido = ARViT original de ladder.build_model, p.ej. data/0)
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
sys.path.insert(0, script_dir)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src_renyi.mps_ancilla.ladder import build_model, load_params
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.models import mps_overlap_aligned
from src_renyi.mps_ancilla.training import make_sampler
from mre_teff import spectrum, MRE

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True)
ap.add_argument("--arch", default=None)
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--d", type=int, default=16)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--h_x", type=float, default=-1.5)
ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--M", type=int, default=512)
ap.add_argument("--batches", type=int, default=32)
ap.add_argument("--chunk", type=int, default=16, help="muestras por trozo en los dobles flips (memoria)")
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--check_exact", action="store_true")
ap.add_argument("--label", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert "/data/0/" not in os.path.abspath(a.out), "no escribir en data/0"
N, hx, J = a.N, a.h_x, a.J_ZZ
assert a.M % a.chunk == 0

model = (build_alt_model(N, a.chi, a.arch, a.d, 4, a.blocks) if a.arch else build_model(N, a.chi, a.d, 4, a.blocks))
params = load_params(a.params, model, N)
chi = model.chi

# indices de los dobles flips i<j
I, Jj = np.triu_indices(N, 1)
F2 = np.ones((len(I), N)); F2[np.arange(len(I)), I] = -1; F2[np.arange(len(I)), Jj] = -1
F1 = 1.0 - 2.0 * np.eye(N)
F1, F2 = jnp.asarray(F1), jnp.asarray(F2)


def D_of(s):
    return J * jnp.sum(s * jnp.roll(s, -1, axis=-1), axis=-1)


def ratios(params, s, logA, Mt, n2, flips):
    """r(s -> s*flip) para cada fila de flips.  s: (B,N) -> (B, K)."""
    B, K = s.shape[0], flips.shape[0]
    sf = (s[:, None, :] * flips[None]).reshape(B * K, N)
    logAf, Mf = model.apply({"params": params}, sf)
    Mrep = jnp.repeat(Mt, K, axis=0)
    Of = mps_overlap_aligned(Mf, Mrep, chi)                    # mismo orden que mc_loss.py
    n2f = jnp.real(mps_overlap_aligned(Mf, Mf, chi))
    G = Of / jnp.sqrt(n2f * jnp.repeat(n2, K))
    r = jnp.exp(logAf - jnp.repeat(logA, K)) * G
    return r.reshape(B, K), sf.reshape(B, K, N)


@jax.jit
def local_E12(params, s):
    logA, Mt = model.apply({"params": params}, s)
    n2 = jnp.real(mps_overlap_aligned(Mt, Mt, chi))
    D = D_of(s)
    r1, s1 = ratios(params, s, logA, Mt, n2, F1)
    r2, _ = ratios(params, s, logA, Mt, n2, F2)
    D1 = D_of(s1)                                              # (B, N)
    E1 = jnp.real(D + hx * jnp.sum(r1, axis=1))
    E2 = jnp.real(D ** 2 + N * hx ** 2 + hx * jnp.sum((D[:, None] + D1) * r1, axis=1)
                  + 2 * hx ** 2 * jnp.sum(r2, axis=1))
    return E1, E2


sampler = make_sampler(model, N, a.M)


def one_batch(key):
    s = sampler(params, key)
    E1s, E2s = [], []
    for k in range(0, a.M, a.chunk):
        e1, e2 = local_E12(params, s[k:k + a.chunk])
        E1s.append(np.asarray(e1)); E2s.append(np.asarray(e2))
    E1, E2 = np.concatenate(E1s), np.concatenate(E2s)
    return E1.mean(), E2.mean(), E1.var(ddof=1)


t0 = time.time()
keys = jax.random.split(jax.random.PRNGKey(a.seed), a.batches)
rows = np.array([one_batch(k) for k in keys])                  # (n, 3)
Eb, E2b = rows[:, 0], rows[:, 1]
# sesgo por lote: E[Eb^2] = E^2 + Var_p(E_loc)/M (muestras iid) -> se suma var(E_loc)/M a cada lote
# (verificador, 9 nov: antes se multiplicaba por M/(M-1), que sobrecorrige porque Var_p(E_loc) << Var(H))
Vb = E2b - Eb ** 2 + rows[:, 2] / a.M
sq = np.sqrt(a.batches)
E, Var = Eb.mean(), Vb.mean()
E_err, V_err = Eb.std(ddof=1) / sq, Vb.std(ddof=1) / sq
Var_c = Var

# referencias exactas
mre = MRE(spectrum(N))
Ex, Sx, Fx = mre(a.T)


def var_mre(T):
    from scipy.optimize import brentq
    f = lambda c: c - mre._sums(c)[1] / mre._sums(c)[0] - 2 * T
    c = brentq(f, mre.E[0] + 1e-9, mre.E[-1] + 20 * T + 10, xtol=1e-13)
    w = np.maximum(c - mre.E, 0); p = w / w.sum()
    m = (p * mre.E).sum()
    return float((p * mre.E ** 2).sum() - m ** 2)


def var_gibbs(T):
    lw = -(mre.E - mre.E.min()) / T; g = np.exp(lw); g /= g.sum()
    m = (g * mre.E).sum()
    return float((g * mre.E ** 2).sum() - m ** 2)


from scipy.optimize import brentq
TE = brentq(lambda t: mre(t)[0] - E, 0.05, 50)
out = dict(label=a.label, params=a.params, arch=a.arch, N=N, chi=a.chi, T=a.T, M=a.M, batches=a.batches,
           E=float(E), E_err=float(E_err), VarH=float(Var_c), VarH_err=float(V_err),
           E_MRE=float(Ex), VarH_MRE=var_mre(a.T), T_E=float(TE), VarH_MRE_at_TE=var_mre(TE),
           VarH_Gibbs=var_gibbs(a.T), time_s=time.time() - t0)

if a.check_exact:
    assert N <= 10
    from src_renyi.mps_ancilla.training import compute_rho
    idx = np.arange(2 ** N)
    bits = (idx[:, None] >> np.arange(N)[None, :]) & 1
    all_s = (2 * bits - 1).astype(np.float64)
    H = np.zeros((2 ** N, 2 ** N))
    H[idx, idx] = J * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
    for i in range(N):
        H[idx, idx ^ (1 << i)] += hx
    rho = np.asarray(compute_rho(model, params, jnp.asarray(all_s)))
    rho = rho / np.trace(rho).real
    Eex = float(np.real(np.trace(rho @ H)))
    E2ex = float(np.real(np.trace(rho @ H @ H)))
    # el mismo estimador local evaluado EXACTAMENTE (suma sobre todas las s con p(s))
    e1, e2 = local_E12(params, jnp.asarray(all_s))
    p = np.real(np.diag(rho))
    out.update(exact_E=Eex, exact_VarH=E2ex - Eex ** 2,
               exact_local_E=float((p * np.asarray(e1)).sum()),
               exact_local_VarH=float((p * np.asarray(e2)).sum() - (p * np.asarray(e1)).sum() ** 2))

print(json.dumps(out, indent=1))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
