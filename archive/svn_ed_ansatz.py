"""
svn_ed_ansatz.py
================
Prueba con el ANSATZ REAL (ARViT + ancilla MPS, models_alt) y gradiente EXACTO (rho completa, N<=10) de minimizar
F~ = E - T*S~ con S~ = extrapolacion a n->1 de S_n exactas (Tr rho^n), frente a los controles:
  s2 (MRE), vn (S_vN exacta por autovalores: el objetivo "ideal", debe dar Gibbs).
Esquemas como en svn_extrap_theory.py (poly_n:2,3,4, poly_inv:..., sn:k).  Reporta E, S2, S_vN verdadera, S~ y
F_vN - F_Gibbs (>= 0; = 0 solo en el Gibbs) y la distancia de traza a Gibbs.
Uso: python scripts/svn_ed_ansatz.py --N 8 --chi 4 --T 1 3 --schemes s2 vn poly_n:2,3 poly_n:2,3,4 --steps 3000
"""
import argparse, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sd = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(sd, "..")); sys.path.insert(0, sd)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla.training import compute_rho
from svn_extrap_theory import scheme_coefs

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=8); ap.add_argument("--chi", type=int, default=4)
ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--T", type=float, nargs="+", default=[1.0, 3.0])
ap.add_argument("--schemes", nargs="+", default=["s2", "vn", "poly_n:2,3", "poly_n:2,3,4"])
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=5e-3)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", required=True)
a = ap.parse_args(); N = a.N
idx = np.arange(2 ** N); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
H = np.zeros((2 ** N, 2 ** N)); H[idx, idx] = -1.0 * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
for i in range(N): H[idx, idx ^ (1 << i)] += -1.5
ev, V = np.linalg.eigh(H); Hj = jnp.asarray(H); all_sj = jnp.asarray(all_s)
model = build_alt_model(N, a.chi, a.arch, 16, 4, 2)


def stats(rho):
    lam = jnp.clip(jnp.linalg.eigvalsh(rho), 1e-300, None)
    E = jnp.real(jnp.sum(Hj.T * rho))
    Sn = lambda n: jnp.log(jnp.sum(lam ** n)) / (1 - n)
    Svn = -jnp.sum(lam * jnp.log(lam))
    return E, Svn, Sn


def make_loss(T, scheme):
    sc = scheme_coefs(scheme)
    def loss(p):
        rho = compute_rho(model, p, all_sj); rho = 0.5 * (rho + rho.conj().T)
        E, Svn, Sn = stats(rho)
        S = Svn if scheme == "vn" else (Sn(2.0) if scheme == "s2" else sum(c * Sn(float(n)) for n, c in zip(*sc)))
        return E - T * S, (E, Svn, Sn(2.0), S)
    return loss


res = json.load(open(a.out)) if os.path.exists(a.out) else []
for T in a.T:
    pG = np.exp(-(ev - ev[0]) / T); pG /= pG.sum(); rhoG = (V * pG) @ V.T
    EG = (pG * ev).sum(); SG = -(pG * np.log(pG)).sum(); FG = EG - T * SG
    print(f"N={N} T={T}: Gibbs E={EG:.5f} Svn={SG:.5f} F={FG:.5f}", flush=True)
    for s in a.schemes:
        loss = make_loss(T, s); params = init_params(model, N, a.seed)
        sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, 100, a.steps, a.lr * 1e-2)
        tx = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(sched)); st = tx.init(params)
        @jax.jit
        def step(p, st):
            (f, aux), g = jax.value_and_grad(loss, has_aux=True)(p); u, st = tx.update(g, st, p)
            return optax.apply_updates(p, u), st, f, aux
        t0 = time.time(); best = (np.inf, params)
        for it in range(a.steps):
            cur = params; params, st, f, aux = step(params, st)
            if np.isfinite(float(f)) and float(f) < best[0]: best = (float(f), cur)
            if it % (a.steps // 6) == 0:
                print(f"  [{s}] {it:5d} F~={float(f):.5f} E={float(aux[0]):.4f} Svn={float(aux[1]):.4f} S2={float(aux[2]):.4f} S~={float(aux[3]):.4f}", flush=True)
        f, (E, Svn, S2, St) = loss(best[1])
        rho = np.asarray(compute_rho(model, best[1], all_sj)); rho = 0.5 * (rho + rho.conj().T)
        tdist = 0.5 * np.abs(np.linalg.eigvalsh(rho - rhoG)).sum()
        r = dict(N=N, T=T, chi=a.chi, scheme=s, steps=a.steps, seed=a.seed, Ft=float(f), E=float(E), Svn=float(Svn),
                 S2=float(S2), S_tilde=float(St), F_vn=float(E - T * Svn), E_G=EG, Svn_G=SG, F_G=FG,
                 dF_vn=float(E - T * Svn - FG), trace_dist=float(tdist), secs=time.time() - t0)
        print(f"  => {s:14s} E={r['E']:.4f} (G {EG:.4f})  Svn={r['Svn']:.4f} (G {SG:.4f})  S~={r['S_tilde']:.4f}  "
              f"F_vN-F_G={r['dF_vn']:+.4f}  D(rho,rhoG)={tdist:.4f}", flush=True)
        res.append(r); json.dump(res, open(a.out, "w"), indent=1)
