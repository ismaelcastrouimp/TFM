"""
sym_purity_eval.py
==================
Evaluación a posteriori de la simetrización de una purificación entrenada (gap_mc_generic 1D o 2D/gap_mc_2d):

    ρ_sym = (1/|S|) Σ_{g∈S} U_g ρ U_g†,     U_g |s> = |g(s)>  (traslaciones, inversión global de espines)

Si cada U_g conmuta con H, E(ρ_sym) = E(ρ) y, por convexidad de la pureza, S2(ρ_sym) >= S2(ρ): F_sym <= F.
Con S un grupo,  Tr ρ_sym² = (1/|G|) Σ_g X_g,  y para un solo elemento u,  Tr[((ρ + uρu†)/2)²] = (P + X_u)/2,  con

    X_g = Tr(ρ U_g ρ U_g†) = Σ_{s,t} ρ(s,t) ρ(σt, σs),   σ = g^{-1}
    ρ(s,t) = A(s) A(t)* G(s,t),   G(s,t) = <Φ_t|Φ_s> / (|Φ_s||Φ_t|),   p(s) = |A(s)|²

Estimador con las mismas muestras que la pureza inclinada (s ~ p, t ~ q(t|s), mezcla defensiva de kappas):
    X_hat = < p(t)/q(t|s) · exp[(lA(σt) − lA(t)) + conj(lA(σs) − lA(s))] · G(s,t) · G(σt,σs) >      (parte real)
(con σ = identidad es exactamente el estimador de la pureza).  Los X_g de todo el grupo salen del mismo lote, así que
ΔS2 = −ln(Tr ρ_sym² / P) tiene mucho menos ruido que S2.  Error por jackknife sobre lotes.

Grupo: traslaciones (Z_N en 1D, Z_Lx × Z_Ly en 2D, site = x + Lx y) × {1, inversión de espines}.
Salida por red: P, S2, X_g / P (r_g) y S2 / dF/N para {1,T1}, {1,P}, traslaciones y traslaciones × P.
Con N <= 10 y --exact, compara con la ρ completa.
Uso:
  python scripts/sym_purity_eval.py --runs "data/gap_arch/j1j2_N20/J20.5_s15000/*T0.[1-9]*.json" --out X.json
"""
import argparse, glob, json, os, sys, time
sd = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(sd, "..")); sys.path.insert(0, os.path.join(sd, "..", "2D"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=None); ap.add_argument("--which", default="ema", choices=["ema", "last"])
ap.add_argument("--M", type=int, default=256); ap.add_argument("--K", type=int, default=5)
ap.add_argument("--kappas", type=float, nargs="*", default=[0.05, 0.1, 0.2, 0.4, 1.0])
ap.add_argument("--n_batch", type=int, default=64); ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--no_flip", action="store_true"); ap.add_argument("--exact", action="store_true"); ap.add_argument("--rho_chunk", type=int, default=16)
ap.add_argument("--test_random", default=None, help="prueba sin red entrenada: 'j1j2:N' o 'tfim2d:LxxLy' (parámetros aleatorios)")
ap.add_argument("--out", required=True)
ap.add_argument("--refl", action="store_true", help="1D: añade la reflexión espacial i -> -i (conjuntos R = {1,R} y trasl_R = diédrico)")
ap.add_argument("--sz_label", type=int, default=0, help="1: etiqueta de Sz en la ancilla (sz_label.py; solo 1D xxz/j1j2): ρ sin la pureza fuera del sector")
a = ap.parse_args()
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src_renyi.mps_ancilla.training import make_sampler
from src_renyi.mps_ancilla.tilted_loss import make_mix_partners
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.models import mps_overlap_aligned
from src_renyi.mps_ancilla.ladder import load_params, init_params
from src_renyi.mps_ancilla.anc_generic import compute_rho_ov

assert "/data/0" not in os.path.abspath(a.out)
tmap = jax.tree_util.tree_map


def group(N, Lx=None, Ly=None, flip=True):
    """Lista de (nombre, perm (N,), signo): σ(s) = signo * s[:, perm]."""
    els = []
    if Lx is None:
        for k in range(N):
            els.append((f"T{k}", np.roll(np.arange(N), -k), 1.0))
    else:
        x, y = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="xy"); x, y = x.ravel(), y.ravel()   # sitio = x + Lx y
        for dy in range(Ly):
            for dx in range(Lx):
                els.append((f"T{dx},{dy}", ((x + dx) % Lx) + Lx * ((y + dy) % Ly), 1.0))
    if flip:
        els += [(n + "P", p, -1.0) for n, p, _ in list(els)]
    return els


def build(fj):
    r = json.load(open(fj)); ar = r["args"]
    if "Lx" in ar:
        from models2d import build_model_2d
        from models_sbs_chain import build_model_sbs_chain
        from models_ttn import build_model_ttn, make_ttn_overlap
        Lx, Ly, chi, arch = ar["Lx"], ar["Ly"], ar["chi"], ar["arch"]; N = Lx * Ly
        if ":ttn" in arch:
            model = build_model_ttn(Lx, Ly, chi, arch, ar["d"], ar["heads"], ar["blocks"]); ov = make_ttn_overlap(Lx, Ly)
        else:
            model = (build_model_sbs_chain if ":sbs" in arch else build_model_2d)(Lx, Ly, chi, arch, ar["d"], ar["heads"], ar["blocks"])
            ov = lambda M1, M2: mps_overlap_aligned(M1, M2, chi)
        geo = (Lx, Ly)
    else:
        N, chi, arch = ar["N"], ar["chi"], ar["arch"]
        model = build_alt_model(N, chi, arch, ar.get("d", 16), ar.get("heads", 4), ar.get("blocks", 2))
        ov = lambda M1, M2: mps_overlap_aligned(M1, M2, chi); geo = (None, None)
        if a.sz_label:                               # la traslación y la inversión conservan |Sz|: compatible
            from src_renyi.mps_ancilla.sz_label import SzLabelModel, assert_sz_ok
            assert_sz_ok(ar.get("model", "j1j2"), None, True)
            base = model; model = SzLabelModel(base, N)
            pf_ = fj[:-5] + ("_params_ema.msgpack" if a.which == "ema" else "_params.msgpack")
            return r, N, geo, model, ov, load_params(pf_, base, N)
    pf = fj[:-5] + ("_params_ema.msgpack" if a.which == "ema" else "_params.msgpack")
    return r, N, geo, model, ov, load_params(pf, model, N)


def make_batch_fn(model, ov, N, perms, signs):
    sampler = make_sampler(model, N, a.M)
    mix = make_mix_partners(model, N, a.kappas, a.K)
    K = a.K
    rep = lambda M: tmap(lambda x: jnp.repeat(x, K, axis=0), M)

    @jax.jit
    def batch(p, key):
        k1, k2 = jax.random.split(key)
        s = sampler(p, k1)
        t, lq = mix(p, s, k2)
        lA_s, M_s = model.apply({"params": p}, s); lA_t, M_t = model.apply({"params": p}, t)
        n_s = jnp.real(ov(M_s, M_s)); n_t = jnp.real(ov(M_t, M_t))
        G_st = ov(rep(M_s), M_t) / jnp.sqrt(jnp.repeat(n_s, K) * n_t)
        lw = 2.0 * jnp.real(lA_t) - lq
        lA_sr = jnp.repeat(lA_s, K)

        def one(ps):
            perm, sgn = ps
            gs = sgn * s[:, perm]; gt = sgn * t[:, perm]
            lA_gs, M_gs = model.apply({"params": p}, gs); lA_gt, M_gt = model.apply({"params": p}, gt)
            n_gs = jnp.real(ov(M_gs, M_gs)); n_gt = jnp.real(ov(M_gt, M_gt))
            G_g = ov(M_gt, rep(M_gs)) / jnp.sqrt(n_gt * jnp.repeat(n_gs, K))
            h = jnp.exp(lw + (lA_gt - lA_t) + jnp.conj(jnp.repeat(lA_gs, K) - lA_sr)) * G_st * G_g
            return jnp.real(jnp.mean(h))
        return jax.lax.map(one, (perms, signs))
    return batch


def exact_X(model, ov, p, N, perms, signs):
    idx = np.arange(2 ** N); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
    rho = np.asarray(compute_rho_ov(model, p, jnp.asarray(all_s), ov, chunk=a.rho_chunk))
    out = []
    for perm, sg in zip(perms, signs):
        gs = sg * all_s[:, perm]; j = (((gs + 1) / 2).astype(np.int64) << np.arange(N)).sum(1)   # índice de σ(s)
        out.append(float(np.real(np.sum(rho * rho[np.ix_(j, j)].T))))                          # Σ ρ(s,t) ρ(σt,σs)
    return np.array(out), float(np.real(np.trace(rho)))


def jack(fn, Pb, Xb):
    """valor y error jackknife de fn(media de P, medias de X) sobre lotes."""
    n = len(Pb); v = fn(Pb.mean(), Xb.mean(0))
    vi = np.array([fn(np.delete(Pb, i).mean(), np.delete(Xb, i, 0).mean(0)) for i in range(n)])
    return float(v), float(np.sqrt((n - 1) / n * np.sum((vi - vi.mean()) ** 2)))


out = {}
runs = sorted(glob.glob(a.runs)) if not a.test_random else [a.test_random]
for fj in runs:
    t0 = time.time()
    if a.test_random:                                        # red aleatoria perturbada (solo para comprobar el estimador)
        kind, sz = a.test_random.split(":")
        if kind == "j1j2":
            N = int(sz); model = build_alt_model(N, 4, "vit:mlp", 16, 4, 2); ov = lambda M1, M2: mps_overlap_aligned(M1, M2, 4)
            geo = (None, None)
        else:
            from models_sbs_chain import build_model_sbs_chain
            Lx, Ly = map(int, sz.split("x")); N = Lx * Ly; geo = (Lx, Ly)
            model = build_model_sbs_chain(Lx, Ly, 4, "vit:sbs:r=1,win=plus,paths=r", 16, 4, 2)
            ov = lambda M1, M2: mps_overlap_aligned(M1, M2, 4)
        p = init_params(model, N, 3)
        lv, td = jax.tree_util.tree_flatten(p); ks = jax.random.split(jax.random.PRNGKey(5), len(lv))
        p = jax.tree_util.tree_unflatten(td, [x + 0.3 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(lv, ks)])
        r = {"args": {"T": 1.0, "label": a.test_random}, a.which: {"E": 0.0}, "ref": None}
    else:
        r, N, geo, model, ov, p = build(fj)
    els = group(N, *geo, flip=not a.no_flip)
    if a.refl and not geo[0]:                                            # T_k ∘ R (todas involuciones), R: i -> -i
        rf = (-np.arange(N)) % N
        els = els + [(f"T{k}R", np.asarray(els[k][1])[rf], 1.0) for k in range(N)]
    names = [e[0] for e in els]
    perms = jnp.asarray(np.stack([e[1] for e in els])); signs = jnp.asarray(np.array([e[2] for e in els]))
    batch = make_batch_fn(model, ov, N, perms, signs)
    key = jax.random.PRNGKey(a.seed); Xb = []
    for b in range(a.n_batch):
        key, k = jax.random.split(key); Xb.append(np.asarray(batch(p, k)))
    Xb = np.array(Xb); Pb = Xb[:, 0]                                    # elemento 0 = identidad
    T = r["args"]["T"]; E = r[a.which]["E"]; Fm = (r.get("ref") or {}).get("F")
    i_T1 = names.index("T1,0") if geo[0] else names.index("T1")
    i_P = names.index("T0,0P" if geo[0] else "T0P") if not a.no_flip else None
    ntr = (geo[0] * geo[1]) if geo[0] else N
    sets = {"T1": lambda P, X: -np.log((P + X[i_T1]) / 2), "trasl": lambda P, X: -np.log(X[:ntr].mean())}
    if i_P is not None:
        nP = 2 * ntr
        sets["P"] = lambda P, X: -np.log((P + X[i_P]) / 2); sets["trasl_P"] = lambda P, X: -np.log(X[:nP].mean())
    if a.refl and not geo[0]:
        i_R = names.index("T0R"); iR = [names.index(f"T{k}R") for k in range(N)]
        sets["R"] = lambda P, X: -np.log((P + X[i_R]) / 2)
        sets["trasl_R"] = lambda P, X: -np.log(np.concatenate([X[:ntr], X[iR]]).mean())
    S2, S2e = jack(lambda P, X: -np.log(P), Pb, Xb)
    res = dict(label=r["args"]["label"], T=T, N=N, E=E, F_mre=Fm, P=float(Pb.mean()), S2=S2, S2_err=S2e,
               r={n: float(x) for n, x in zip(names, Xb.mean(0) / Pb.mean())}, sym={})
    msg = f"{res['label']}: T={T} S2={S2:.4f}±{S2e:.4f}  r(T1)={res['r'][names[i_T1]]:.3f}"
    if i_P is not None:
        msg += f" r(P)={res['r'][names[i_P]]:.3f}"
    for nm, fn in sets.items():
        v, e = jack(fn, Pb, Xb); d, de = jack(lambda P, X: fn(P, X) + np.log(P), Pb, Xb)
        res["sym"][nm] = dict(S2=v, S2_err=e, dS2=d, dS2_err=de)
        if Fm is not None:
            res["sym"][nm]["dF_per_site"] = (E - T * v - Fm) / N
        msg += f" | {nm}: ΔS2={d:+.4f}±{de:.4f}" + (f" dF/N={(E - T * v - Fm) / N:+.5f}" if Fm is not None else "")
    if Fm is not None:
        res["dF_per_site"] = (E - T * S2 - Fm) / N; msg += f" | sin simetrizar dF/N={res['dF_per_site']:+.5f}"
    if a.exact and N <= 12:
        Xe, tr = exact_X(model, ov, p, N, np.asarray(perms), np.asarray(signs))
        res["exact"] = dict(P=float(Xe[0]), X=Xe.tolist(), trace=tr)
        dev = np.abs(Xb.mean(0) - Xe) / (Xb.std(0, ddof=1) / np.sqrt(a.n_batch))
        msg += f"\n   exacto: Tr ρ={tr:.6f} P={Xe[0]:.5f} (MC {Pb.mean():.5f})  max|X_MC − X_ex|/σ = {dev.max():.2f}  " \
               f"r_ex(T1)={Xe[i_T1] / Xe[0]:.4f}" + (f" r_ex(P)={Xe[i_P] / Xe[0]:.4f}" if i_P is not None else "")
    print(msg + f"  ({time.time() - t0:.0f} s)", flush=True)
    out[res["label"]] = res
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
