"""
Tests de la pipeline physical_ancilla con redes REALES y COMPLEJAS.

θ siempre real; log ψ puede ser complejo (ARSpinViT_Causal: módulo + fase).

Uso (siempre en CPU):
    JAX_PLATFORMS=cpu .venv/bin/python scripts/test_complex_physical_ancilla.py
        [--quick] [--skip-orig] [--skip-train] [--orig-dir DIR]

1. Referencia exacta (N=3 sistema + N_A=3 ancilla): ψ completo construido
   desde el modelo, E, S₂, S₃ y sus gradientes con jax.grad. Se comparan:
     - funciones exactas de entropy.py y FullSumState.expect_and_grad(H)
     - estimadores MC (energía, swap, swap-jacobiano, Wang n=2,3, DS n=2,3,
       Drut, Drut-direct, observable F con swap/wang/drut)
   para un modelo complejo (fase O(1)) y uno real (ARNNDense).
2. (si existe --orig-dir) mismo cálculo con la copia ORIGINAL del paquete,
   en un subproceso (evita conflictos del dispatch de NetKet):
     - modelo real: resultados nuevos == originales (mismas muestras)
     - modelo complejo: muestra el sesgo del código original.
3. Entrenamiento corto end-to-end con modelo complejo (N=2, N_A=2):
   free_energy_minimize_phases (swap → drut) y free_energy_minimize,
   comparado con el F exacto del estado MRE.
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl_"))
import matplotlib
matplotlib.use("Agg")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import netket as nk
import numpy as np
from jax.flatten_util import ravel_pytree
from netket.operator.spin import sigmax, sigmaz

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DEFAULT_ORIG = ("/home/ismael/.claude/tmp/claude-1000/"
                "-home-ismael-Documents-PhD-Projects-Thermal-Renyi-TFM/"
                "00cf2266-50d1-42da-8dad-0fabb018b434/scratchpad/"
                "physical_ancilla_orig")

GAMMA, V, T = 1.0, 1.0, 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Sistema, modelos, referencia exacta
# ══════════════════════════════════════════════════════════════════════════════

def build_system(N, N_A):
    hi = nk.hilbert.Spin(s=1 / 2, N=N + N_A)
    H = 0
    for i in range(N):
        H += -GAMMA * sigmax(hi, i)
        if N > 1 and (N > 2 or i == 0):
            H += -V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)
    return hi, H, list(range(N))


def build_model(pa, kind, hi, seed=0):
    """Modelo + parámetros float64. Para 'complex' se perturban las capas de
    salida para que |ψ| no sea uniforme y la fase sea O(1)."""
    if kind == "complex":
        model = pa.models.ARSpinViT_Causal(
            hilbert=hi, embedding_d=8, n_heads=2, n_blocks=1,
            n_ffn_layers=1, param_dtype=jnp.float64)
    else:
        model = nk.models.ARNNDense(hilbert=hi, layers=2, features=8,
                                    param_dtype=jnp.float64)
    vs = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model,
                        n_samples=16, seed=seed)
    params = jax.tree_util.tree_map(jnp.asarray, vs.parameters)
    if kind == "complex":
        params = dict(params)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed + 7), 3)
        fd = dict(params["final_dense"])
        fd["kernel"] = 0.7 * jax.random.normal(k1, fd["kernel"].shape)
        params["final_dense"] = fd
        ph = dict(params["phase_dense"])
        ph["kernel"] = 1.5 * jax.random.normal(k2, ph["kernel"].shape)
        ph["bias"] = 1.0 * jax.random.normal(k3, ph["bias"].shape)
        params["phase_dense"] = ph
    return model, params


def exact_quantities(model, params, hi, H, N):
    """E, S₂, S₃ exactos como funciones diferenciables de θ (real)."""
    configs = jnp.asarray(hi.all_states())
    hi_s = nk.hilbert.Spin(s=1 / 2, N=N)
    hi_a = nk.hilbert.Spin(s=1 / 2, N=hi.size - N)
    iA = np.asarray(hi_s.states_to_numbers(np.asarray(configs[:, :N])))
    iB = np.asarray(hi_a.states_to_numbers(np.asarray(configs[:, N:])))
    Hd = jnp.asarray(H.to_dense())

    def f(p):
        lp = model.apply({"params": p}, configs)
        psi = jnp.exp(lp.astype(jnp.complex128))
        psi = psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2))
        E = jnp.real(jnp.vdot(psi, Hd @ psi))
        M = jnp.zeros((2 ** N, 2 ** (hi.size - N)), complex).at[iA, iB].set(psi)
        rho = M @ M.conj().T
        S2 = -jnp.log(jnp.real(jnp.trace(rho @ rho)))
        S3 = -0.5 * jnp.log(jnp.real(jnp.trace(rho @ rho @ rho)))
        return E, S2, S3

    E, S2, S3 = f(params)
    gE = jax.grad(lambda p: f(p)[0])(params)
    gS2 = jax.grad(lambda p: f(p)[1])(params)
    gS3 = jax.grad(lambda p: f(p)[2])(params)
    lp = model.apply({"params": params}, configs)
    phase_std = float(jnp.std(jnp.imag(lp))) if jnp.iscomplexobj(lp) else 0.0
    return dict(E=float(E), S2=float(S2), S3=float(S3),
                gE=flat(gE), gS2=flat(gS2), gS3=flat(gS3),
                phase_std=phase_std)


def flat(tree):
    return np.asarray(ravel_pytree(tree)[0])


def make_vs(model, params, hi, n_samples, seed):
    vs = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model,
                        n_samples=n_samples, seed=seed, sampler_seed=seed + 1)
    vs.parameters = params
    return vs


# ══════════════════════════════════════════════════════════════════════════════
# Batería de estimadores (se ejecuta con el paquete nuevo o con el original)
# ══════════════════════════════════════════════════════════════════════════════

def run_estimators(pa, kind, quick=False):
    N, N_A = 3, 3
    hi, H, part = build_system(N, N_A)
    model, params = build_model(pa, kind, hi)
    ent, obs = pa.entropy, pa.observables
    NS = 2 ** 13 if quick else 2 ** 15
    out = {}

    def rec(name, fn):
        t0 = time.time()
        try:
            res = fn()
            out[name] = {k: (np.asarray(v).tolist() if not np.isscalar(v) else float(v))
                         for k, v in res.items()}
        except Exception as e:  # noqa
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"   [{kind}] {name:24s} {time.time() - t0:6.1f}s"
              + (f"  ERROR {out[name]['error'][:90]}" if "error" in out[name] else ""),
              flush=True)

    # ── exactas ───────────────────────────────────────────────────────────
    def _exact_entropy():
        vs = nk.vqs.FullSumState(hi, model); vs.parameters = params
        S2a, _ = ent.renyi2_entropy_exact(vs, part)
        S2b, g = ent.renyi2_entropy_and_grad_exact(vs, part, hi, isFullSum=True)
        S3, _ = ent.renyin_entropy_exact(vs, part, 3)
        return dict(S2=float(S2a), S2_g=float(S2b), gS2=flat(g), S3=float(S3))
    rec("exact_entropy", _exact_entropy)

    def _fullsum_E():
        vs = nk.vqs.FullSumState(hi, model); vs.parameters = params
        E, g = vs.expect_and_grad(H)
        return dict(E=float(E.mean.real), gE=np.real(flat(g)))
    rec("fullsum_expect_grad_H", _fullsum_E)

    # ── energía MC ────────────────────────────────────────────────────────
    for s in (0, 1):
        def _nk_E(s=s):
            vs = make_vs(model, params, hi, NS, 100 + s)
            E, g = vs.expect_and_grad(H)
            return dict(E=float(E.mean.real), gE=np.real(flat(g)))
        rec(f"netket_expect_grad_H_{s}", _nk_E)

        def _E_chunk(s=s):
            vs = make_vs(model, params, hi, NS, 200 + s)
            op = obs.FreeRenyiEnergyObservable(hi, H, part, T, chunk_size=1024)
            E, g = obs._energy_and_grad_chunked(vs, op, 1024)
            return dict(E=float(E.mean.real), gE=flat(g))
        rec(f"energy_chunked_{s}", _E_chunk)

        # ── swap ──────────────────────────────────────────────────────────
        def _swap(s=s):
            vs = make_vs(model, params, hi, NS, 300 + s)
            S2, g = ent.renyi2_entropy_and_grad_sampled(
                vs, part, NS, key=s, chunk_size=1024)
            vs2 = make_vs(model, params, hi, NS, 300 + s)
            S2f = ent.renyi2_entropy_sampled(vs2, part, NS, key=s, chunk_size=1024)
            return dict(S2=float(S2), S2_fwd=float(S2f), gS2=flat(g))
        rec(f"swap_sampled_{s}", _swap)

        def _swap2(s=s):
            vs = make_vs(model, params, hi, NS, 400 + s)
            S2, g = ent.renyi2_entropy_and_grad_sampled2(vs, part, NS // 4, key=s)
            return dict(S2=float(S2), gS2=flat(g))
        rec(f"swap_jacobian_{s}", _swap2)

        def _obs_swap(s=s):
            vs = make_vs(model, params, hi, 2 * NS, 500 + s)
            op = obs.FreeRenyiEnergyObservable(hi, H, part, T, chunk_size=1024,
                                               method="swap")
            F, g = vs.expect_and_grad(op)
            return dict(F=float(F.mean.real), gF=flat(g))
        rec(f"obs_swap_{s}", _obs_swap)

        # ── DS Rényi-n ────────────────────────────────────────────────────
        def _ds(s=s):
            vs = make_vs(model, params, hi, NS, 600 + s)
            S2, g2 = ent.renyin_entropy_and_grad_sampled(vs, part, 2, NS, key=s,
                                                         chunk_size=1024)
            S3, g3 = ent.renyin_entropy_and_grad_sampled(vs, part, 3, NS, key=s,
                                                         chunk_size=1024)
            return dict(S2=S2, gS2=flat(g2), S3=S3, gS3=flat(g3))
        rec(f"renyin_DS_{s}", _ds)

    # ── Wang (red reversa real, entrenada sobre |ψ|²) ─────────────────────
    model_R = nk.models.ARNNDense(hilbert=hi, layers=2, features=16,
                                  param_dtype=jnp.float64)
    vs_p = make_vs(model, params, hi, 1024, 700)
    vs_R = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model_R,
                          n_samples=1024, seed=701, sampler_seed=702)
    try:
        vs_R = ent.train_reverse_network(vs_p, vs_R, n_steps=150 if quick else 400,
                                         batch=1024, lr=1e-2, verbose=False)
        dkl = float(pa.training.compute_kl(vs_p, vs_R, 4096))
        out["wang_R_dkl"] = {"dkl": dkl}
    except Exception as e:  # noqa
        out["wang_R_dkl"] = {"error": f"{type(e).__name__}: {e}"}

    for s in (0, 1):
        def _wang(s=s):
            vs = make_vs(model, params, hi, 16, 800 + s)
            S2, g2 = ent.renyi2_wang_cs(vs, vs_R, part, NS, key=s, chunk_size=4096)
            S3, g3 = ent.renyin_wang_cs(vs, vs_R, part, 3, NS, key=s, chunk_size=4096)
            return dict(S2=S2, gS2=flat(g2), S3=S3, gS3=flat(g3))
        rec(f"wang_{s}", _wang)

        def _obs_wang(s=s):
            vs = make_vs(model, params, hi, NS, 900 + s)
            op = obs.FreeRenyiEnergyObservable(
                hi, H, part, T, chunk_size=4096, chunk_size_energy=1024,
                method="wang", vstate_R=vs_R,
                wang_kwargs={"n_samples": NS}, wang_seed=10 * s)
            F, g = vs.expect_and_grad(op)
            return dict(F=float(F.mean.real), gF=flat(g))
        rec(f"obs_wang_{s}", _obs_wang)

    # ── Drut ──────────────────────────────────────────────────────────────
    dk = dict(n_chains=512 if quick else 2048, n_lambda=8, n_sweeps_per_lam=10)
    for s in (0, 1):
        def _drut(s=s):
            vs = make_vs(model, params, hi, 16, 1000 + s)
            S2, g = ent.renyi2_drut_sampling(vs, part, key=s, **dk)
            vs = make_vs(model, params, hi, 16, 1000 + s)
            S2d, gd = ent.renyi2_drut_sampling_direct(vs, part, key=s, **dk)
            return dict(S2=S2, gS2=flat(g), S2_direct=S2d, gS2_direct=flat(gd))
        rec(f"drut_{s}", _drut)

    def _obs_drut():
        vs = make_vs(model, params, hi, NS, 1100)
        op = obs.FreeRenyiEnergyObservable(hi, H, part, T, chunk_size=1024,
                                           method="drut", drut_kwargs=dk)
        F, g = vs.expect_and_grad(op)
        return dict(F=float(F.mean.real), gF=flat(g))
    rec("obs_drut", _obs_drut)

    # ── estimadores sin soporte complejo (deben lanzar NotImplementedError) ──
    def _lam():
        vs = make_vs(model, params, hi, 16, 1200)
        S2, g = ent.renyi2_entropy_and_grad_lambda_integral(vs, part, NS // 2, n_lambda=6)
        return dict(S2=S2, gS2=flat(g))
    rec("lambda_integral", _lam)

    def _inc():
        vs = make_vs(model, params, hi, 16, 1300)
        S2, g, _, _ = ent.renyi2_increment_sampling(vs, part, 512, n_sweeps_per_site=10)
        return dict(S2=S2, gS2=flat(g))
    rec("increment", _inc)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Comparación con la referencia exacta
# ══════════════════════════════════════════════════════════════════════════════

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def compare(kind, res, ex, label="new"):
    gF_ex = ex["gE"] - T * ex["gS2"]
    F_ex = ex["E"] - T * ex["S2"]
    rows = []

    def grad_row(name, key, exkey, gex, valkey=None, valex=None):
        runs = [res.get(f"{name}_{s}", {}) for s in (0, 1)] \
            if f"{name}_0" in res else [res.get(name, {})]
        if any("error" in r for r in runs) or not runs[0]:
            err = next((r["error"] for r in runs if "error" in r), "missing")
            rows.append((name + ":" + key, None, None, None, None, None, err))
            return
        gs = [np.asarray(r[key]) for r in runs]
        g = np.mean(gs, axis=0)
        noise = (np.linalg.norm(gs[0] - gs[1]) / 2 / np.linalg.norm(gex)
                 if len(gs) == 2 else float("nan"))
        val = np.mean([r[valkey] for r in runs]) if valkey else None
        rows.append((name + ":" + key, cos(g, gex), relerr(g, gex), noise,
                     val, valex, ""))

    ok = True
    print(f"\n=== {label} — modelo {kind}  "
          f"(E={ex['E']:.5f}  S2={ex['S2']:.5f}  S3={ex['S3']:.5f}  "
          f"F={F_ex:.5f}  std(fase)={ex['phase_std']:.2f}) ===")

    # exactas
    r = res.get("exact_entropy", {})
    if "error" in r:
        print(f"  exact_entropy ERROR {r['error']}"); ok = False
    elif r:
        c = cos(np.asarray(r["gS2"]), ex["gS2"])
        e = relerr(np.asarray(r["gS2"]), ex["gS2"])
        print(f"  exact_entropy: S2={r['S2']:.8f} (ref {ex['S2']:.8f})  "
              f"S3={r['S3']:.8f} (ref {ex['S3']:.8f})  grad cos={c:.8f} rel={e:.2e}")
        ok &= abs(r["S2"] - ex["S2"]) < 1e-8 and e < 1e-6
    r = res.get("fullsum_expect_grad_H", {})
    if r and "error" not in r:
        e = relerr(np.asarray(r["gE"]), ex["gE"])
        print(f"  FullSumState.expect_and_grad(H): E={r['E']:.8f} (ref {ex['E']:.8f}) "
              f"grad rel={e:.2e}")
        ok &= e < 1e-6

    grad_row("netket_expect_grad_H", "gE", "gE", ex["gE"], "E", ex["E"])
    grad_row("energy_chunked", "gE", "gE", ex["gE"], "E", ex["E"])
    grad_row("swap_sampled", "gS2", "gS2", ex["gS2"], "S2", ex["S2"])
    grad_row("swap_jacobian", "gS2", "gS2", ex["gS2"], "S2", ex["S2"])
    grad_row("renyin_DS", "gS2", "gS2", ex["gS2"], "S2", ex["S2"])
    grad_row("renyin_DS", "gS3", "gS3", ex["gS3"], "S3", ex["S3"])
    grad_row("wang", "gS2", "gS2", ex["gS2"], "S2", ex["S2"])
    grad_row("wang", "gS3", "gS3", ex["gS3"], "S3", ex["S3"])
    grad_row("drut", "gS2", "gS2", ex["gS2"], "S2", ex["S2"])
    grad_row("drut", "gS2_direct", "gS2", ex["gS2"], "S2_direct", ex["S2"])
    grad_row("obs_swap", "gF", "gF", gF_ex, "F", F_ex)
    grad_row("obs_wang", "gF", "gF", gF_ex, "F", F_ex)
    grad_row("obs_drut", "gF", "gF", gF_ex, "F", F_ex)
    grad_row("lambda_integral", "gS2", "gS2", ex["gS2"], "S2", ex["S2"])
    grad_row("increment", "gS2", "gS2", ex["gS2"], "S2", ex["S2"])

    print(f"  {'estimador':32s} {'cos':>8s} {'relerr':>8s} {'ruido':>8s}"
          f" {'valor':>10s} {'exacto':>10s}")
    for name, c, e, nz, v, vex, err in rows:
        if err:
            print(f"  {name:32s} {err[:100]}")
            continue
        vs = f"{v:10.5f}" if v is not None else ""
        vx = f"{vex:10.5f}" if vex is not None else ""
        # aceptación: dentro de ~4σ del ruido (ruido ≈ σ del promedio) o cos alto
        good = (c > 0.99) or (np.isfinite(nz) and e < 4 * nz + 0.02)
        flag = "" if good else "  <-- FUERA"
        print(f"  {name:32s} {c:8.4f} {e:8.4f} {nz:8.4f} {vs} {vx}{flag}")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Entrenamiento corto end-to-end (modelo complejo)
# ══════════════════════════════════════════════════════════════════════════════

def mre_target(H, hi, N):
    from scipy.optimize import brentq
    # H sólo actúa en el sistema → espectro de H_sys (cada nivel ×2^N_A)
    hi_s = nk.hilbert.Spin(s=1 / 2, N=N)
    Hs = 0
    for i in range(N):
        Hs += -GAMMA * sigmax(hi_s, i)
        if N > 1 and (N > 2 or i == 0):
            Hs += -V * sigmaz(hi_s, i) @ sigmaz(hi_s, (i + 1) % N)
    Ev = np.linalg.eigvalsh(np.asarray(Hs.to_dense()))

    def sc(c):
        w = np.maximum(c - Ev, 0)
        return c - (w * Ev).sum() / w.sum() - 2 * T
    c = brentq(sc, Ev.min() + 1e-9, Ev.max() + 20 * T + 10, xtol=1e-12)
    w = np.maximum(c - Ev, 0); p = w / w.sum()
    E, S2 = (p * Ev).sum(), -np.log((p ** 2).sum())
    return E - T * S2, E, S2


def run_training(pa, quick=False):
    import optax
    N, N_A = 2, 2
    hi, H, part = build_system(N, N_A)
    model, params0 = build_model(pa, "complex", hi, seed=3)
    F_mre, E_mre, S2_mre = mre_target(H, hi, N)

    def exact_F(p):
        ex = exact_quantities(model, p, hi, H, N)
        return ex["E"] - T * ex["S2"], ex

    F0, ex0 = exact_F(params0)
    print(f"\n=== Entrenamiento end-to-end (complejo, N={N}, N_A={N_A}, T={T}) ===")
    print(f"  MRE exacto:  F={F_mre:.5f}  E={E_mre:.5f}  S2={S2_mre:.5f}")
    print(f"  inicial   :  F={F0:.5f}  E={ex0['E']:.5f}  S2={ex0['S2']:.5f}  "
          f"std(fase)={ex0['phase_std']:.2f}")
    results = {}

    n1 = 40 if quick else 80
    vs = make_vs(model, params0, hi, 4096, 11)
    phases = [
        {"method": "swap", "n_steps": n1, "optimizer": optax.adam(1e-2)},
        {"method": "drut", "n_steps": 5, "optimizer": optax.adam(3e-3),
         "drut_kwargs": dict(n_chains=256, n_lambda=6, n_sweeps_per_lam=5)},
    ]
    t0 = time.time()
    hist, best_F, E_b, S2_b = pa.training.free_energy_minimize_phases(
        vs, T, part, H, phases, verbose=False, plot=False, chunk_size=512)
    Ff, exf = exact_F(vs.parameters)
    print(f"  phases(swap {n1} + drut 5) [{time.time() - t0:.0f}s]: F_hist "
          f"{np.mean(hist[:5]):.4f} → {np.mean(hist[n1 - 5:n1]):.4f};  "
          f"F_exacto final={Ff:.5f} (E={exf['E']:.4f}, S2={exf['S2']:.4f}); "
          f"reportado F={best_F:.5f} (E={E_b:.4f}, S2={S2_b:.4f})")
    results["phases"] = dict(F0=F0, F_final=Ff, F_mre=F_mre, best_F=best_F)

    vs = make_vs(model, params0, hi, 4096, 12)
    t0 = time.time()
    hist2, best_F2, E_b2, S2_b2 = pa.training.free_energy_minimize(
        vs, T, part, H, n_steps=n1, fine_steps=3,
        fine_drut_kwargs=dict(n_chains=256, n_lambda=6, n_sweeps_per_lam=5),
        optimizer=optax.adam(1e-2), verbose=False, plot=False, chunk_size=512)
    Ff2, exf2 = exact_F(vs.parameters)
    print(f"  free_energy_minimize(swap {n1} + drut 3) [{time.time() - t0:.0f}s]: "
          f"F_exacto final={Ff2:.5f} (E={exf2['E']:.4f}, S2={exf2['S2']:.4f}); "
          f"reportado F={best_F2:.5f} (E={E_b2:.4f}, S2={S2_b2:.4f})")
    results["minimize"] = dict(F0=F0, F_final=Ff2, F_mre=F_mre, best_F=best_F2)
    return results


# ══════════════════════════════════════════════════════════════════════════════

def import_pkg(which, orig_dir):
    if which == "orig":
        sys.path.insert(0, os.path.dirname(orig_dir))
        return importlib.import_module(os.path.basename(orig_dir))
    return importlib.import_module("src_renyi.physical_ancilla")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-orig", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--only", choices=["complex", "real"], default=None)
    ap.add_argument("--orig-dir", default=DEFAULT_ORIG)
    ap.add_argument("--dump", default=None, help="(interno) ejecutar y volcar JSON")
    ap.add_argument("--pkg", default="new", choices=["new", "orig"])
    args = ap.parse_args()
    kinds = [args.only] if args.only else ["complex", "real"]

    if args.dump:
        pa = import_pkg(args.pkg, args.orig_dir)
        out = {k: run_estimators(pa, k, args.quick) for k in kinds}
        with open(args.dump, "w") as f:
            json.dump(out, f)
        return

    pa = import_pkg("new", args.orig_dir)

    # referencia exacta
    exact = {}
    for k in kinds:
        hi, H, _ = build_system(3, 3)
        model, params = build_model(pa, k, hi)
        exact[k] = exact_quantities(model, params, hi, H, 3)

    # resultados con el paquete nuevo
    print("── estimadores (paquete NUEVO) ──")
    new = {k: run_estimators(pa, k, args.quick) for k in kinds}

    # paquete original en subproceso
    orig = None
    if not args.skip_orig and os.path.isdir(args.orig_dir):
        print("── estimadores (paquete ORIGINAL, subproceso) ──", flush=True)
        dump = os.path.join(tempfile.mkdtemp(), "orig.json")
        cmd = [sys.executable, __file__, "--dump", dump, "--pkg", "orig",
               "--orig-dir", args.orig_dir] + (["--quick"] if args.quick else []) \
            + (["--only", args.only] if args.only else [])
        subprocess.run(cmd, check=True)
        orig = json.load(open(dump))

    for k in kinds:
        compare(k, new[k], exact[k], label="NUEVO")
        if "wang_R_dkl" in new[k]:
            print(f"  (D_KL red reversa = {new[k]['wang_R_dkl']})")
        if orig is not None:
            compare(k, orig[k], exact[k], label="ORIGINAL")

    # real: nuevo == original (mismas semillas ⇒ mismas muestras)
    if orig is not None and "real" in kinds:
        print("\n=== modelo REAL: nuevo vs original (mismas muestras) ===")
        worst = 0.0
        for name, r in new["real"].items():
            ro = orig["real"].get(name, {})
            if "error" in r or "error" in ro:
                print(f"  {name:26s} nuevo: {r.get('error', 'ok')[:50]} | "
                      f"orig: {ro.get('error', 'ok')[:50]}")
                continue
            diffs = []
            for key, v in r.items():
                a, b = np.asarray(v), np.asarray(ro[key])
                d = float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))
                diffs.append(f"{key}:{d:.1e}")
                worst = max(worst, d)
            print(f"  {name:26s} " + "  ".join(diffs))
        print(f"  máx diferencia relativa = {worst:.2e}")

    if not args.skip_train:
        run_training(pa, args.quick)


if __name__ == "__main__":
    main()
