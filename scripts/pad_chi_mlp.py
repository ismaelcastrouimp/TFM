"""
pad_chi_mlp.py
==============
Amplía chi -> chi2 un modelo models_alt con ancilla GLOBAL (anc = mlp | ctx | mix), p.ej. vit:mlp.
Ahí la cabeza de la ancilla da los tensores de TODOS los sitios a la vez:
    W(s) = anc_out(MLP(s))  (última dim N * 2 chi^2, bloque del sitio k en [k nb, (k+1) nb), nb = 2 chi^2)
    M_k(s) = W_k(s) + eye/2 + (s @ mps_copy)_k,     mps_copy: (N, N nb)
(en "ctx" la capa de salida se llama mps_head).  Igual que pad_chi_alt.py: en cada sitio el bloque viejo (chi x chi)
va arriba a la izquierda; los bloques viejo<->nuevo valen 0 y el bloque nuevo queda desacoplado (el MPS se contrae
con e0 en los bordes), así que con noise=0 el estado es EXACTAMENTE el mismo.
  - capa de salida, kernel: entradas nuevas = 0 + ruido N(0, noise) (para que haya gradiente hacia el bloque nuevo)
  - capa de salida, bias:   entradas nuevas = 0
  - mps_copy:               bloque viejo copiado; resto como en un modelo nuevo (copia diagonal; fuera de diagonal 0)
Todo lo demás (ViT, capas ocultas de la ancilla: su anchura no depende de chi) se copia.
Uso:
  python scripts/pad_chi_mlp.py --params P.msgpack --arch vit:mlp --N 10 --chi 4 --chi2 8 --noise 1e-3 --out Q.msgpack
  python scripts/pad_chi_mlp.py --selftest      # exacto a N=8 (j1j2): F igual con noise=0; con ruido, cambio pequeño
"""
import argparse, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from src_renyi.mps_ancilla.ladder import load_params, init_params, save_params
from src_renyi.mps_ancilla.models_alt import build_alt_model

OUT_LAYERS = ("anc_out", "mps_head")


def pad_chi_mlp(p, p_new, N, chi, chi2, noise=0.0, seed=0):
    """p: params a chi; p_new: params recién inicializados a chi2 (dan las formas y el mps_copy por defecto)."""
    rng = np.random.default_rng(seed)
    p = jax.tree_util.tree_map(np.asarray, p)
    out = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), p_new)
    old = np.zeros((N, 2, chi2, chi2), bool); old[:, :, :chi, :chi] = True; old = old.reshape(-1)

    def put(a_old, a_def):                    # última dim N*2chi^2 -> N*2chi2^2, bloque viejo arriba-izq en cada sitio
        lead = a_old.shape[:-1]; b = a_def.copy().reshape(lead + (N, 2, chi2, chi2))
        b[..., :chi, :chi] = a_old.reshape(lead + (N, 2, chi, chi))
        return b.reshape(lead + (-1,))

    found = set()
    for k in p:
        if k in OUT_LAYERS:
            ker = put(p[k]["kernel"], np.zeros_like(out[k]["kernel"]))
            ker[:, ~old] = rng.normal(0.0, noise, (ker.shape[0], (~old).sum())) if noise > 0 else 0.0
            out[k] = dict(kernel=ker, bias=put(p[k]["bias"], np.zeros_like(out[k]["bias"])))
            found.add("out")
        elif k == "mps_copy":
            out[k] = put(p[k], out[k]); found.add("copy")
        else:
            assert jax.tree_util.tree_structure(p[k]) == jax.tree_util.tree_structure(out[k]), k
            for a, b in zip(jax.tree_util.tree_leaves(p[k]), jax.tree_util.tree_leaves(out[k])):
                assert a.shape == b.shape, (k, a.shape, b.shape)
            out[k] = p[k]
    assert found == {"out", "copy"}, "solo ancilla global mlp/ctx/mix (capa anc_out o mps_head + mps_copy (N, N nb))"
    return out


def pad_file(path, arch, N, chi, chi2, noise, seed=0, d=16):
    m1, m2 = build_alt_model(N, chi, arch, d), build_alt_model(N, chi2, arch, d)
    return pad_chi_mlp(load_params(path, m1, N), init_params(m2, N, seed), N, chi, chi2, noise, seed), m2


def selftest(N=8, chi=4, chi2=8, T=1.0, arch="vit:mlp"):
    from src_renyi.mps_ancilla import training
    D = 2 ** N; idx = np.arange(D); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
    Hd = np.zeros((D, D))
    for r, Jb in ((1, 1.0), (2, 0.5)):                          # j1j2 J2 = 0.5 (Paulis)
        for i in range(N):
            j = (i + r) % N; si, sj = all_s[:, i], all_s[:, j]
            Hd[idx, idx] += Jb * si * sj
            f = si != sj; Hd[idx[f], (idx ^ (1 << i) ^ (1 << j))[f]] += 2 * Jb
    m1, m2 = build_alt_model(N, chi, arch), build_alt_model(N, chi2, arch)
    L1, _ = training.make_loss_fns(m1, all_s, Hd, T, chunk=64)
    L2, _ = training.make_loss_fns(m2, all_s, Hd, T, chunk=64)
    out = training.train(L1, init_params(m1, N, 1), 300, 5e-3, T, freq=100, tag="[chi4] ")
    p1 = out["params_best"]; F1 = float(L1(p1)[0])
    ok = True
    for noise in (0.0, 1e-3):
        p2 = pad_chi_mlp(p1, init_params(m2, N, 7), N, chi, chi2, noise)
        F2 = float(L2(p2)[0])
        print(f"selftest N={N} {arch} chi {chi}->{chi2} noise={noise:g}: F(chi)={F1:.12f}  F(chi2)={F2:.12f}  "
              f"ΔF={F2 - F1:+.2e}", flush=True)
        ok &= abs(F2 - F1) < (1e-10 if noise == 0 else 1e-2)
    out2 = training.train(L2, pad_chi_mlp(p1, init_params(m2, N, 7), N, chi, chi2, 1e-3), 300, 2e-3, T, freq=100,
                          tag="[chi8 desde chi4] ")
    print(f"tras 300 pasos a chi2: F={out2['F']:.6f} (antes {F1:.6f})", flush=True)
    ok &= out2["F"] <= F1 + 1e-6
    print("SELFTEST OK" if ok else "SELFTEST FALLO", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params"); ap.add_argument("--arch", default="vit:mlp")
    ap.add_argument("--N", type=int); ap.add_argument("--chi", type=int); ap.add_argument("--chi2", type=int)
    ap.add_argument("--noise", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    assert "/data/0" not in os.path.abspath(a.out), "no escribir en data/0"
    p2, _ = pad_file(a.params, a.arch, a.N, a.chi, a.chi2, a.noise, a.seed, a.d)
    save_params(a.out, p2); print("guardado", a.out)
