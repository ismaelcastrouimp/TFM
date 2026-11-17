"""
cheb_anc_exact.py
=================
Filtro de Chebyshev en el lado de la ANCILLA, exacto, N <= 12.  ¿Como crece con N el grado K necesario?
    rho(s,t) = A_s A_t^* <phi_t| W |phi_s> / sqrt(n_s n_t),   W = P(H_anc)^2,  P(E) = sum_k g_k T_k(y(E))
    y = (E - Emid)/(Ehalf*1.0001) en [-1,1]   (misma convencion que acheb en filter_joint_exact.py)
En la base propia de H (H = V diag(E) V^T):  rho = sum_e w_e |u_e><u_e|,  u_e(s) = A_s (Phi V)_{s,e} / sqrt(n_s),
w_e = P(E_e)^2 >= 0.  Entonces, sin construir W ni rho:
    Z = w.z,  <H> Z = w.h,  Tr rho^2 Z^2 = w^T Q w,  z_e = ||u_e||^2, h_e = <u_e|H|u_e>, Q = |U^dag U|^2.
Modos:
  --posthoc : red congelada (modelo ya entrenado sin filtro).  Para cada K: (1) escalon de 2 parametros a posteriori,
              (2) su raiz proyectada sobre Chebyshev de grado K como arranque, (3) L-BFGS sobre los K+1 coeficientes;
              tambien desde P=1.  Ademas el TECHO: w_e arbitrario >= 0 (2^N parametros).
  --finetune K : red + coeficientes juntos, --steps pasos (Adam, lr --lr), arrancando de los g a posteriori
              (lee --posthoc_json).  K=0 = control: mismo afinado sin filtro (P=1 fijo).
Uso:
  python scripts/cheb_anc_exact.py --N 12 --params data/gap_arch/scaling/params_S_mc_N12_cpu.msgpack --posthoc \
      --Ks 4 8 12 16 24 32 48 --out data/gap_arch/filter/cheb_post_N12.json
  python scripts/cheb_anc_exact.py --N 12 --params ... --finetune 16 --steps 400 --posthoc_json data/gap_arch/filter/cheb_post_N12.json \
      --out data/gap_arch/filter/cheb_ft_N12.json
"""
import argparse, json, os, sys, time, resource
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "..")); sys.path.insert(0, script_dir)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize

from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import load_params
from src_renyi.mps_ancilla import training
from mre_teff import MRE

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--params", required=True); ap.add_argument("--chi", type=int, default=4)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--d", type=int, default=16)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--posthoc", action="store_true"); ap.add_argument("--Ks", type=int, nargs="*", default=[4, 8, 16, 32])
ap.add_argument("--finetune", type=int, default=None); ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--posthoc_json", default=None)
ap.add_argument("--row_chunk", type=int, default=0, help="U por bloques de filas con jax.checkpoint (memoria, N=12)")
ap.add_argument("--g_l1", type=float, default=0.0, help="penalizacion lam*sum|g_k| (k>=1) en el ajuste a posteriori: coeficientes acotados -> recurrencia MPS estable")
ap.add_argument("--no_ceiling", action="store_true", help="no calcular el techo w_e libre (posthoc)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
N, T = a.N, a.T; assert N <= 12 and "/data/0/" not in os.path.abspath(a.out)
D = 2 ** N
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # GB (ru_maxrss en kB)

idx = np.arange(D)
all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
diagH = -1.0 * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
H = np.zeros((D, D)); H[idx, idx] = diagH
for i in range(N):
    H[idx, idx ^ (1 << i)] += -1.5
ev, V = np.linalg.eigh(H); del H
Ex, Sx, Fx = MRE(ev)(T)
y = (ev - 0.5 * (ev[0] + ev[-1])) / (0.5 * (ev[-1] - ev[0]) * 1.0001)
flips = jnp.asarray(np.stack([idx ^ (1 << i) for i in range(N)]))
diagHj, Vj, yj = jnp.asarray(diagH), jnp.asarray(V), jnp.asarray(y)
print(f"N={N} T={T} F_MRE={Fx:.5f}  E in [{ev[0]:.3f},{ev[-1]:.3f}]  RSS {rss():.2f} GB", flush=True)


def Hmul(U):                                   # H @ U con H dispersa (diag + h sum_i X_i); sin tensor (N, D, D)
    out = diagHj[:, None] * U
    for i in range(N):
        out = out - 1.5 * U[flips[i]]
    return out


def ancilla_vectors(M):
    S = M.shape[0]
    v = jnp.zeros((S, 1, M.shape[-1]), M.dtype).at[:, 0, 0].set(1.0)
    for k in range(M.shape[1]):
        v = jnp.einsum("sai,sbij->sbaj", v, M[:, k]).reshape(S, -1, M.shape[-1])
    return v[:, :, 0]


model = build_alt_model(N, a.chi, a.arch, a.d, 4, a.blocks)
net0 = load_params(a.params, model, N)
all_sj = jnp.asarray(all_s)


def U_of(net):
    logA, M = model.apply({"params": net}, all_sj)
    A = jnp.exp(logA - jnp.max(jnp.real(logA)))
    Phi = ancilla_vectors(M)
    n = jnp.real(jnp.sum(jnp.abs(Phi) ** 2, 1))
    return (A / jnp.sqrt(n))[:, None] * (Phi @ Vj.astype(Phi.dtype))


def _U_rows(net, s_rows):                      # A sin restar el max: el ARViT esta normalizado (|A| <= 1), F es invariante de escala
    logA, M = model.apply({"params": net}, s_rows)
    Phi = ancilla_vectors(M)
    n = jnp.real(jnp.sum(jnp.abs(Phi) ** 2, 1))
    PV = jax.lax.complex(jnp.real(Phi) @ Vj, jnp.imag(Phi) @ Vj)      # sin copia compleja de V
    return (jnp.exp(logA) / jnp.sqrt(n))[:, None] * PV


_U_rows_ck = jax.checkpoint(_U_rows)
if a.row_chunk:
    def U_of(net):
        rows = all_sj.reshape(-1, a.row_chunk, N)
        return jax.lax.map(lambda sr: _U_rows_ck(net, sr), rows).reshape(D, D)


def stats(U):
    z = jnp.real(jnp.sum(jnp.abs(U) ** 2, 0))
    h = jnp.real(jnp.sum(jnp.conj(U) * Hmul(U), 0))
    Q = jnp.abs(jnp.conj(U).T @ U) ** 2
    return z, h, Q


if a.row_chunk:                                # backward: solo U como residuo (recalcula U^dag U y los U[flips])
    stats = jax.checkpoint(stats)


def F_of_w(w, z, h, Q):
    Z = w @ z; E = (w @ h) / Z; S2 = -jnp.log(w @ (Q @ w) / Z ** 2)
    return E - T * S2, E, S2


def cheb_P(g, yv):
    Tm, Tk = jnp.ones_like(yv), yv; P = g[0] * Tm + (g[1] * Tk if len(g) > 1 else 0)
    for k in range(2, len(g)):
        Tm, Tk = Tk, 2 * yv * Tk - Tm; P = P + g[k] * Tk
    return P


def lbfgs(f, x0, maxiter=4000):
    fg = jax.jit(jax.value_and_grad(f))
    r = minimize(lambda x: tuple(np.asarray(v, np.float64) for v in fg(jnp.asarray(x))), np.asarray(x0, np.float64),
                 jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, maxcor=50, ftol=1e-15, gtol=1e-11))
    return r.x, float(r.fun), r


res = json.load(open(a.out)) if os.path.exists(a.out) else []


def save(r):
    res.append(r); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if a.posthoc:
    t0 = time.time()
    z, h, Q = stats(U_of(net0))
    print(f"estadisticas U listas ({time.time() - t0:.0f} s), RSS {rss():.2f} GB", flush=True)
    F0 = float(F_of_w(jnp.ones(D), z, h, Q)[0]); dF0 = F0 - Fx
    print(f"sin filtro: dF={dF0:+.5f}", flush=True)
    base = dict(N=N, T=T, params=a.params, chi=a.chi, F_MRE=Fx, dF_none=dF0, mode="posthoc")
    # escalon (2 parametros)
    evj = jnp.asarray(ev)
    fstep = lambda x: F_of_w(jax.nn.sigmoid((x[0] - evj) / jnp.exp(x[1])), z, h, Q)[0]
    best = None
    for c0 in np.linspace(Ex, Ex + 3 * np.sqrt(N), 7):
        for lt in (np.log(0.3), np.log(1.0), np.log(3.0)):
            x, f, _ = lbfgs(fstep, [c0, lt], 500)
            if best is None or f < best[1]: best = (x, f)
    xs, Fs = best
    print(f"escalon a posteriori: dF={Fs - Fx:+.5f} ({1 - (Fs - Fx) / dF0:.1%})  c={xs[0]:.3f} tau={np.exp(xs[1]):.3f}", flush=True)
    save(dict(base, filter="step", K=None, dF=Fs - Fx, closed=1 - (Fs - Fx) / dF0, c=float(xs[0]), tau=float(np.exp(xs[1]))))
    # techo: w arbitrario >= 0
    if not a.no_ceiling:
        x, f, _ = lbfgs(lambda u: F_of_w(jnp.exp(u), z, h, Q)[0], np.zeros(D), 4000)
        print(f"techo (w_e libre): dF={f - Fx:+.5f} ({1 - (f - Fx) / dF0:.1%})", flush=True)
        save(dict(base, filter="ceiling", K=None, dF=f - Fx, closed=1 - (f - Fx) / dF0))
    sq_step = lambda yy: np.sqrt(1 / (1 + np.exp(-(xs[0] - (yy * 0.5 * (ev[-1] - ev[0]) * 1.0001 + 0.5 * (ev[0] + ev[-1])))
                                                 / np.exp(xs[1]))))
    for K in a.Ks:
        g_step = np.polynomial.chebyshev.chebinterpolate(sq_step, K)
        fK = lambda g: F_of_w(cheb_P(g, yj) ** 2, z, h, Q)[0]
        pen = lambda g: a.g_l1 * jnp.sum(jnp.sqrt(g[1:] ** 2 + 1e-10))
        Fproj = float(fK(jnp.asarray(g_step)))
        cands = []
        for lab, g0 in (("from_step", g_step), ("from_one", np.r_[1.0, np.zeros(K)])):
            g, fp, r = lbfgs(lambda g: fK(g) + pen(g), g0, 4000)
            cands.append((fp, g, lab, int(r.nit)))
        fp, g, lab, nit = min(cands, key=lambda c: c[0])
        f = float(fK(jnp.asarray(g)))
        print(f"K={K:3d}: proyeccion del escalon dF={Fproj - Fx:+.5f} | optimizado dF={f - Fx:+.5f} "
              f"({1 - (f - Fx) / dF0:.1%}, mejor {lab}; otros {[round(c[0] - Fx, 5) for c in cands]})", flush=True)
        save(dict(base, filter="cheb", K=K, dF=f - Fx, closed=1 - (f - Fx) / dF0, dF_step_proj=Fproj - Fx,
                  start=lab, nit=nit, g=np.asarray(g).tolist(), all_starts={c[2]: c[0] - Fx for c in cands},
                  g_l1=a.g_l1, sum_abs_g=float(np.abs(g).sum())))
    print(f"FIN posthoc ({time.time() - t0:.0f} s, RSS max {rss():.2f} GB)", flush=True)

if a.finetune is not None:
    K = a.finetune
    if K > 0:
        ph = [r for r in json.load(open(a.posthoc_json)) if r.get("filter") == "cheb" and r["K"] == K]
        assert ph, f"no hay K={K} en {a.posthoc_json}"
        g0 = jnp.asarray(ph[-1]["g"]); dF_post = ph[-1]["dF"]
    else:
        g0, dF_post = jnp.ones(1), None

    def loss(pp):
        U = U_of(pp["net"]); z, h, Q = stats(U)
        w = cheb_P(pp["g"], yj) ** 2 if K > 0 else jnp.ones(D)
        F, E, S2 = F_of_w(w, z, h, Q); return F, (E, S2)
    params = {"net": net0, "g": g0}
    F_init = float(loss(params)[0])
    print(f"[ft K={K}] dF inicial={F_init - Fx:+.5f}  RSS {rss():.2f} GB", flush=True)
    out = training.train(loss, params, a.steps, a.lr, T, freq=max(1, a.steps // 10), warmup=min(100, a.steps // 4), tag=f"[ft K={K}] ")
    F, E, S2 = map(float, [out["F"], out["E"], out["S2"]])
    r = dict(N=N, T=T, params=a.params, chi=a.chi, F_MRE=Fx, mode="finetune", filter="cheb" if K else "none", K=K,
             steps=a.steps, lr=a.lr, dF_init=F_init - Fx, dF_posthoc=dF_post, dF=F - Fx, dE=E - Ex, dS2=S2 - Sx,
             g=np.asarray(out["params_best"]["g"]).tolist(), s_per_step=out["s_per_step"], rss_gb=rss())
    print(f"[ft K={K}] FINAL dF={F - Fx:+.5f} (inicial {F_init - Fx:+.5f}, a posteriori {dF_post})  "
          f"{out['s_per_step']:.1f} s/paso  RSS max {rss():.2f} GB", flush=True)
    save(r)
