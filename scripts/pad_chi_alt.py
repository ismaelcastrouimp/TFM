"""
pad_chi_alt.py
==============
Amplia chi -> chi2 un modelo models_alt con ancilla local compartida (anc = conv | convg | causal), p.ej. vit:conv:r=1.
Ahi M_k(s) = W(s) + eye/2 + s_k * c, con W = salida de la capa "anc_out" (2 chi^2) y c = "mps_copy" (2 chi^2).
Como el MPS se contrae con e0 en los dos bordes (Phi(a) = e0^T M^{a1}...M^{aK} e0), si los bloques fuera de la
diagonal (viejo<->nuevo) son 0 el bloque nuevo no contribuye: el estado es EXACTAMENTE el mismo.
  - anc_out/kernel: columnas nuevas = 0 + ruido N(0, noise) (solo ahi; para que haya gradiente hacia el bloque nuevo)
  - anc_out/bias:   entradas nuevas = 0
  - mps_copy:       bloque viejo copiado; resto como en un modelo nuevo (copia diagonal; fuera de diagonal = 0)
Con noise=0 F no cambia (lo comprueba --check).  Todo lo demas (ViT, capas ocultas de la ancilla) se copia.
Uso:
  python scripts/pad_chi_alt.py --params P.msgpack --arch vit:conv:r=1 --N 50 --chi 8 --chi2 16 --noise 1e-3 --out Q.msgpack
  python scripts/pad_chi_alt.py --selftest      # prueba exacta a N=8 (ED): F igual con noise=0 y entrenamiento corto
"""
import argparse, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import flax.serialization as serialization
from src_renyi.mps_ancilla.ladder import load_params, init_params
from src_renyi.mps_ancilla.models_alt import build_alt_model


def pad_chi_alt(p, p_new, chi, chi2, noise=0.0, seed=0):
    """p: params a chi; p_new: params recien inicializados a chi2 (dan formas y el mps_copy por defecto)."""
    rng = np.random.default_rng(seed)
    p = jax.tree_util.tree_map(np.asarray, p); out = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), p_new)
    old = np.zeros((2, chi2, chi2), bool); old[:, :chi, :chi] = True; old = old.reshape(-1)

    def put(a_old, a_def):                                        # ultima dim 2chi^2 -> 2chi2^2, bloque viejo arriba-izq
        lead = a_old.shape[:-1]; b = a_def.copy().reshape(lead + (2, chi2, chi2))
        b[..., :chi, :chi] = a_old.reshape(lead + (2, chi, chi))
        return b.reshape(lead + (-1,))

    for k in p:
        if k == "anc_out":
            ker = put(p[k]["kernel"], np.zeros_like(out[k]["kernel"]))
            ker[:, ~old] = rng.normal(0.0, noise, (ker.shape[0], (~old).sum())) if noise > 0 else 0.0
            out[k] = dict(kernel=ker, bias=put(p[k]["bias"], np.zeros_like(out[k]["bias"])))
        elif k == "mps_copy":
            out[k] = put(p[k], out[k])
        else:
            assert jax.tree_util.tree_structure(p[k]) == jax.tree_util.tree_structure(out[k]), k
            for a, b in zip(jax.tree_util.tree_leaves(p[k]), jax.tree_util.tree_leaves(out[k])):
                assert a.shape == b.shape, (k, a.shape, b.shape)
            out[k] = p[k]
    assert "anc_out" in p and "mps_copy" in p, "solo ancilla conv/convg/causal (capa anc_out + mps_copy)"
    return out


def check_M(model1, p1, model2, p2, N, chi, n=64, seed=0):
    """M2[..., :chi, :chi] == M1, bloques cruzados = 0, logA igual.  Devuelve max |diferencia|."""
    s = jnp.asarray(2.0 * jax.random.bernoulli(jax.random.PRNGKey(seed), 0.5, (n, N)) - 1)
    a1, M1 = model1.apply({"params": p1}, s); a2, M2 = model2.apply({"params": p2}, s)
    return dict(logA=float(jnp.max(jnp.abs(a1 - a2))), M_old=float(jnp.max(jnp.abs(M2[..., :chi, :chi] - M1))),
                M_cross=float(max(jnp.max(jnp.abs(M2[..., :chi, chi:])), jnp.max(jnp.abs(M2[..., chi:, :chi])))))


def selftest(N=8, chi=4, chi2=8, T=1.0, arch="vit:conv:r=1"):
    from src_renyi.mps_ancilla import training
    idx = np.arange(2 ** N); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
    zz = sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
    H = np.zeros((2 ** N,) * 2); H[idx, idx] = -1.0 * zz
    for i in range(N): H[idx, idx ^ (1 << i)] += -1.5
    m1, m2 = build_alt_model(N, chi, arch), build_alt_model(N, chi2, arch)
    l1, _ = training.make_loss_fns(m1, all_s, H, T); l2, _ = training.make_loss_fns(m2, all_s, H, T)
    p1 = training.train(l1, init_params(m1, N, 1), 400, 5e-3, T, freq=400, tag="[chi=4 base] ")["params_best"]
    F1 = float(l1(p1)[0])
    q0 = pad_chi_alt(p1, init_params(m2, N, 2), chi, chi2, 0.0)
    q3 = pad_chi_alt(p1, init_params(m2, N, 2), chi, chi2, 1e-3, seed=5)
    c = check_M(m1, p1, m2, q0, N, chi); F0, F3 = float(l2(q0)[0]), float(l2(q3)[0])
    print(f"selftest N={N} chi {chi}->{chi2}: |dlogA|={c['logA']:.1e} |dM_viejo|={c['M_old']:.1e} |M_cruzado|={c['M_cross']:.1e}")
    print(f"  F(chi={chi})={F1:.10f}  F(pad, ruido 0)={F0:.10f} (dif {F0 - F1:+.1e})  F(pad, ruido 1e-3)={F3:.10f} (dif {F3 - F1:+.1e})")
    assert c["logA"] < 1e-12 and c["M_old"] < 1e-12 and c["M_cross"] == 0 and abs(F0 - F1) < 1e-10 and abs(F3 - F1) < 1e-4
    g = jax.grad(lambda q: l2(q)[0])(q3)["anc_out"]["kernel"]
    old = np.zeros((2, chi2, chi2), bool); old[:, :chi, :chi] = True; old = old.reshape(-1)
    print(f"  |grad| columnas nuevas de anc_out (ruido 1e-3): {float(jnp.abs(g[:, ~old]).max()):.2e}")
    # entrenamiento corto: continuar chi=4 vs continuar desde el ampliado (mismos pasos/lr)
    Fa = float(training.train(l1, p1, 400, 2e-3, T, freq=400, tag="[chi=4 sigue] ")["F"])
    Fb = float(training.train(l2, q3, 400, 2e-3, T, freq=400, tag="[chi=8 pad] ")["F"])
    print(f"  tras 400 pasos mas: chi={chi} {Fa:.6f}   chi={chi2} (ampliado) {Fb:.6f}   (dif {Fb - Fa:+.2e})")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--params"); ap.add_argument("--arch", default="vit:conv:r=1"); ap.add_argument("--N", type=int)
    ap.add_argument("--chi", type=int); ap.add_argument("--chi2", type=int)
    ap.add_argument("--noise", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        print("selftest:", selftest()); sys.exit()
    assert "/data/0/" not in os.path.abspath(a.out), "no escribir en data/0"
    m1, m2 = build_alt_model(a.N, a.chi, a.arch), build_alt_model(a.N, a.chi2, a.arch)
    p1 = load_params(a.params, m1, a.N); p1 = p1.get("params", p1)
    q = pad_chi_alt(p1, init_params(m2, a.N, a.seed), a.chi, a.chi2, a.noise, a.seed)
    q0 = pad_chi_alt(p1, init_params(m2, a.N, a.seed), a.chi, a.chi2, 0.0)
    c = check_M(m1, p1, m2, q0, a.N, a.chi)
    print(f"comprobacion (ruido 0): |dlogA|={c['logA']:.1e} |dM_viejo|={c['M_old']:.1e} |M_cruzado|={c['M_cross']:.1e}")
    assert c["logA"] < 1e-12 and c["M_old"] < 1e-12 and c["M_cross"] == 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f: f.write(serialization.to_bytes(q))
    print("guardado", a.out)
