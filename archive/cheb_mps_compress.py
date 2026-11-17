"""
cheb_mps_compress.py
====================
¿Se puede aplicar el filtro de Chebyshev de la ancilla como aritmetica MPS con truncacion (lo que haria falta a N=25)?
Para un modelo conv ya entrenado (red congelada) y los coeficientes g de un filtro a posteriori (cheb_post_N{N}.json):
    phi_s = P(H) Phi_s,   P(E) = sum_k g_k T_k(y(E)),  y = (E - Emid)/(Ehalf*1.0001)   (convencion de cheb_anc_exact.py)
  "exacto": phi_s denso en la base propia (debe reproducir el dF a posteriori del json -> autocomprobacion).
  "tt":     phi_s exacto comprimido a chi' por TT-SVD (mejor caso: error solo de representacion).
  "rec":    recurrencia de Chebyshev en MPS: v_{k+1} = 2 y(H) v_k - v_{k-1}, acc += g_k v_k, truncando v_k y acc a chi'
            en cada paso (H = MPO OBC de bond 3 + termino de frontera Z_{N-1} Z_0 como producto).  Ruta realista.
rho(s,t) = a_s a_t^* <phi_t|phi_s>, a_s = A_s/||Phi_s||;  F exacto a partir de rho (N <= 12).
Uso:
  python scripts/cheb_mps_compress.py --N 10 --Ks 16 32 --chis 4 8 16 32 64 --out data/gap_arch/filter/cheb_mps_N10.json
"""
import argparse, json, os, sys, time
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "..")); sys.path.insert(0, script_dir)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import load_params
from mre_teff import MRE

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--chi", type=int, default=4); ap.add_argument("--arch", default="vit:conv:r=1")
ap.add_argument("--Ks", type=int, nargs="+", default=[16, 32]); ap.add_argument("--chis", type=int, nargs="+", default=[4, 8, 16, 32])
ap.add_argument("--modes", nargs="+", default=["tt", "rec"]); ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--post_json", default=None, help="json a posteriori con g (por defecto cheb_post_N{N}.json)")
ap.add_argument("--g_l1", type=float, default=None, help="usar la fila con este g_l1")
ap.add_argument("--out", required=True)
a = ap.parse_args()
N, T = a.N, a.T; D = 2 ** N; J, hx = -1.0, -1.5
assert N <= 12 and "/data/0/" not in os.path.abspath(a.out)
post = json.load(open(a.post_json or f"data/gap_arch/filter/cheb_post_N{N}.json"))
params_path = post[0]["params"]

# ---- H exacto, MRE ----
idx = np.arange(D)
bits = (idx[:, None] >> np.arange(N)[None, :]) & 1
all_s = (2 * bits - 1).astype(np.float64)
diagH = J * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
H = np.zeros((D, D)); H[idx, idx] = diagH
for i in range(N):
    H[idx, idx ^ (1 << i)] += hx
ev, V = np.linalg.eigh(H); del H
Ex, Sx, Fx = MRE(ev)(T)
Emid, Ehalf = 0.5 * (ev[0] + ev[-1]), 0.5 * (ev[-1] - ev[0]) * 1.0001
y = (ev - Emid) / Ehalf
flips = np.stack([idx ^ (1 << i) for i in range(N)])

# ---- modelo ----
model = build_alt_model(N, a.chi, a.arch, 16, 4, 2)
net = load_params(params_path, model, N)
logA, M = model.apply({"params": net}, jnp.asarray(all_s))
logA, M = np.asarray(logA), np.asarray(M)                       # M (D, N, 2, chi, chi): M[s,k][b,i,j]
A = np.exp(logA - np.max(np.real(logA)))
mps0 = [np.transpose(M[:, k], (0, 2, 1, 3)) for k in range(N)]   # (B, Dl, 2, Dr)
mps0[0] = mps0[0][:, :1]; mps0[-1] = mps0[-1][..., :1]


def dense(mps):                                                  # indice = sum_k b_k 2^k (como ancilla_vectors)
    v = mps[0][:, 0]                                             # (B, 2, Dr)
    for k in range(1, N):
        v = np.einsum("bxi,biyj->byxj", v, mps[k]).reshape(v.shape[0], -1, mps[k].shape[-1])
    return v[..., 0]


Phi = dense(mps0)
n = np.sum(np.abs(Phi) ** 2, 1)
assert np.allclose(n, np.real(np.sum(np.abs(Phi) ** 2, 1)))
aw = A / np.sqrt(n)


def F_of(phi):                                                   # phi (D, D): fila s = vector ancilla filtrado
    G = phi.conj() @ phi.T                                       # G[t, s] = <phi_t|phi_s>
    rho = aw[:, None] * G.T * aw.conj()[None, :]                 # rho[s, t] = a_s a_t^* <phi_t|phi_s>
    Z = np.real(np.trace(rho))
    E = np.real(np.sum(diagH * np.diag(rho)) + hx * sum(np.sum(rho[flips[i], idx]) for i in range(N))) / Z
    S2 = -np.log(np.sum(np.abs(rho) ** 2) / Z ** 2)
    return E - T * S2, E, S2


def cheb_vals(g, yv):
    Tm, Tk = np.ones_like(yv), yv; P = g[0] * Tm + (g[1] * Tk if len(g) > 1 else 0)
    for k in range(2, len(g)):
        Tm, Tk = Tk, 2 * yv * Tk - Tm; P = P + g[k] * Tk
    return P


# ---- aritmetica MPS por lotes ----
def compress(mps, chi):
    """QR izquierda->derecha, SVD derecha->izquierda truncando a chi.  Devuelve MPS normal (norma en el primer tensor)."""
    mps = [t.copy() for t in mps]
    for k in range(N - 1):
        B, Dl, d, Dr = mps[k].shape
        q, r = np.linalg.qr(mps[k].reshape(B, Dl * d, Dr))
        mps[k] = q.reshape(B, Dl, d, -1)
        mps[k + 1] = np.einsum("bij,bjxk->bixk", r, mps[k + 1])
    for k in range(N - 1, 0, -1):
        B, Dl, d, Dr = mps[k].shape
        u, s, vh = np.linalg.svd(mps[k].reshape(B, Dl, d * Dr), full_matrices=False)
        c = min(chi, s.shape[1])
        mps[k] = vh[:, :c].reshape(B, c, d, Dr)
        mps[k - 1] = np.einsum("bxdj,bjc->bxdc", mps[k - 1], u[:, :, :c] * s[:, None, :c])
    return mps


def lincomb(terms):
    """sum_i c_i |mps_i> por suma directa (bond = suma de bonds)."""
    out = []
    for k in range(N):
        ts = [(c if k == 0 else 1.0) * m[k] for c, m in terms]
        B, d = ts[0].shape[0], ts[0].shape[2]
        if k == 0:
            out.append(np.concatenate(ts, axis=3)); continue
        if k == N - 1:
            out.append(np.concatenate(ts, axis=1)); continue
        Dl, Dr = sum(t.shape[1] for t in ts), sum(t.shape[3] for t in ts)
        blk = np.zeros((B, Dl, d, Dr), ts[0].dtype); i0 = j0 = 0
        for t in ts:
            blk[:, i0:i0 + t.shape[1], :, j0:j0 + t.shape[3]] = t; i0 += t.shape[1]; j0 += t.shape[3]
        out.append(blk)
    return out


Iop, Xop, Zop = np.eye(2), np.array([[0., 1.], [1., 0.]]), np.diag([-1., 1.])   # base del bit: b=0 -> s=-1
W = np.zeros((3, 3, 2, 2)); W[0, 0] = Iop; W[0, 1] = Zop; W[0, 2] = hx * Xop; W[1, 2] = J * Zop; W[2, 2] = Iop


def apply_H(mps):
    out = []
    for k in range(N):
        Wk = W[:1] if k == 0 else W
        Wk = Wk[:, 2:] if k == N - 1 else Wk
        t = np.einsum("uvxy,bjyl->bjuxlv", Wk, mps[k])          # (B, Dl, wl, 2, Dr, wr)
        B, Dl, wl, d, Dr, wr = t.shape
        out.append(t.reshape(B, Dl * wl, d, Dr * wr))
    bnd = [m.copy() for m in mps]
    bnd[0] = np.einsum("xy,bjyl->bjxl", Zop, bnd[0]) * J
    bnd[-1] = np.einsum("xy,bjyl->bjxl", Zop, bnd[-1])
    return lincomb([(1.0, out), (1.0, bnd)])


def apply_y(mps, chi):                                           # y(H) v = (H v - Emid v) / Ehalf, comprimido
    return compress(lincomb([(1.0 / Ehalf, apply_H(mps)), (-Emid / Ehalf, mps)]), chi)


# autocomprobacion de apply_H frente al H exacto en unas pocas muestras
_v = [t[:3] for t in mps0]
_Hv = dense(apply_H(_v)); _ref = dense(_v)
_Href = diagH[None] * _ref + hx * sum(_ref[:, flips[i]] for i in range(N))
assert np.allclose(_Hv, _Href), "apply_H no coincide con H exacto"

res = json.load(open(a.out)) if os.path.exists(a.out) else []
PhiV = Phi @ V
for K in a.Ks:
    row = [r for r in post if r.get("filter") == "cheb" and r["K"] == K and (a.g_l1 is None or r.get("g_l1") == a.g_l1)][-1]
    g, dF_json = np.asarray(row["g"]), row["dF"]
    phi_ex = (PhiV * cheb_vals(g, y)[None]) @ V.T
    F_ex = F_of(phi_ex)[0]
    print(f"K={K}: exacto dF={F_ex - Fx:+.6f} (json {dF_json:+.6f})", flush=True)
    assert abs(F_ex - Fx - dF_json) < 1e-6, "no reproduce el dF a posteriori"
    base = dict(N=N, K=K, T=T, params=params_path, dF_exact=F_ex - Fx, dF_none=post[0]["dF_none"], g_l1=row.get("g_l1", 0.0),
                sum_abs_g=float(np.abs(g).sum()))
    for chi in a.chis:
        for mode in a.modes:
            t0 = time.time()
            if mode == "tt":                                     # TT-SVD del vector exacto
                v = phi_ex.reshape(D, *([2] * N)[::-1])
                v = np.moveaxis(v, list(range(1, N + 1)), list(range(N, 0, -1)))   # -> (B, b_0, ..., b_{N-1})
                cores, rest, Dl = [], v.reshape(D, 1, -1), 1
                for k in range(N - 1):
                    u, s, vh = np.linalg.svd(rest.reshape(D, Dl * 2, -1), full_matrices=False)
                    c = min(chi, s.shape[1])
                    cores.append(u[:, :, :c].reshape(D, Dl, 2, c))
                    rest = s[:, :c, None] * vh[:, :c]; Dl = c
                cores.append(rest.reshape(D, Dl, 2, 1))
                phi = dense(cores)
            else:                                                # recurrencia truncada
                parts = []
                for c0 in range(0, D, a.chunk):                  # por lotes de muestras (memoria)
                    v0 = [t[c0:c0 + a.chunk] for t in mps0]
                    vm, vk = v0, apply_y(v0, chi)
                    acc = compress(lincomb([(g[0], vm), (g[1], vk)]), chi)
                    for k in range(2, K + 1):
                        vn = compress(lincomb([(2.0, apply_y(vk, chi)), (-1.0, vm)]), chi)
                        vm, vk = vk, vn
                        acc = compress(lincomb([(1.0, acc), (g[k], vk)]), chi)
                    parts.append(dense(acc))
                phi = np.concatenate(parts)
            Fc = F_of(phi)[0]
            err = np.linalg.norm(phi - phi_ex) / np.linalg.norm(phi_ex)
            r = dict(base, mode=mode, chi_p=chi, dF=Fc - Fx, rel_err=float(err), secs=time.time() - t0)
            print(f"  {mode:3s} chi'={chi:3d}: dF={Fc - Fx:+.6f}  |phi-phi_ex|/|phi_ex|={err:.2e}  ({r['secs']:.0f} s)", flush=True)
            res.append(r); json.dump(res, open(a.out, "w"), indent=1)
print("FIN", flush=True)
