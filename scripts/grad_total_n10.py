"""
grad_total_n10.py
=================
Ruido y sesgo del gradiente TOTAL de F2 = E − T S2 por lote (energía + pureza, como en gap_mc_hybrid) frente al exacto
(Ψ densa), N <= 10, en unos parámetros fijos.  Mismas s para todas las configuraciones (las M de la energía; la pureza usa
las Mp primeras).  Configuración = "nombre:eform:pest:psym:Mp:K:kappas"
  eform std|vmc;  pest tilted|u1|loc1|ham2;  psym 0|1.
Uso:
  python scripts/grad_total_n10.py --params X.msgpack --nb 100 --out X.json --configs base:std:tilted:0:256:5:0.05,0.1,0.2,0.4,1.0 ...
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--J2", type=float, default=0.5); ap.add_argument("--N", type=int, default=10); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--chi", type=int, default=16); ap.add_argument("--arch", default="vit:mlp")
ap.add_argument("--params", required=True); ap.add_argument("--nb", type=int, default=100); ap.add_argument("--M", type=int, default=1024)
ap.add_argument("--e_chunk", type=int, default=128); ap.add_argument("--pairs_chunk", type=int, default=2048)
ap.add_argument("--configs", nargs="+", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--sz_label", type=int, default=0, help="1: etiqueta de Sz en la ancilla (sz_label.py)")
ap.add_argument("--perturb", type=float, default=0.0, help="ruido gaussiano sumado a los parámetros (lejos del óptimo)")
a = ap.parse_args()
assert "/data/0" not in os.path.abspath(a.out)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from flax import serialization
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla.training import make_sampler
from src_renyi.mps_ancilla.tilted_loss import make_mix_partners, make_tilted_purity
from src_renyi.mps_ancilla.generic_eloc import build_hamiltonian, make_energy_terms
from src_renyi.mps_ancilla.vmc_energy import make_energy_terms_vmc
from src_renyi.mps_ancilla.purity_lowvar import make_u1_mix_partners, make_u1_lowvar_purity, make_u1_mask_purity
from src_renyi.mps_ancilla.sz_label import maybe_label
from src_renyi.mps_ancilla.training import compute_rho

N, T, chi = a.N, a.T, a.chi; D = 2 ** N; sg = jax.lax.stop_gradient
model = build_alt_model(N, chi, a.arch, 16)
p = serialization.from_bytes(init_params(model, N), open(a.params, "rb").read())
if a.perturb > 0:
    p = jax.tree_util.tree_map(lambda x: x + a.perturb * jax.random.normal(jax.random.PRNGKey(7), x.shape, x.dtype), p)
model = maybe_label(model, N, a.sz_label)
H = build_hamiltonian("j1j2", N, J2=a.J2)
sampler = jax.jit(make_sampler(model, N, a.M))
idx = np.arange(D); Sx = jnp.asarray((2 * ((idx[:, None] >> np.arange(N)) & 1) - 1).astype(np.float64))
sp, mels = H.to_jax_operator().get_conn_padded(Sx); sp, mels = np.asarray(sp), np.asarray(mels).real
col = (((sp + 1) / 2).astype(np.int64) << np.arange(N)).sum(-1)
Hd = np.zeros((D, D)); np.add.at(Hd, (np.repeat(idx, sp.shape[1]), col.ravel()), mels.ravel()); Hx = jnp.asarray(Hd)
flat = lambda g: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(g)])


def psi(q):
    lA, Mt = model.apply({"params": q}, Sx)
    v = Mt[:, 0, :, 0, :]
    for k in range(1, N):
        v = jnp.einsum("sbi,saij->sbaj", v, Mt[:, k]).reshape(D, -1, Mt.shape[-1])
    phi = v[..., 0]; phi = phi / jnp.linalg.norm(phi, axis=1, keepdims=True)
    P = jnp.exp(lA - sg(jnp.max(jnp.real(lA))))[:, None] * phi
    return P / jnp.sqrt(jnp.sum(jnp.abs(P) ** 2))


_mg = np.asarray(Sx).sum(1); SAME = jnp.asarray((_mg[:, None] == _mg[None, :]).astype(np.float64))


def rho_ex(q):                                     # ψ densa con los N sitios de la ancilla; la etiqueta = δ de sector en ρ
    P = psi(q); r = P @ P.conj().T
    return r * SAME if a.sz_label else r


def F_ex(q):
    r = rho_ex(q)
    return jnp.real(jnp.sum(Hx.T * r)) + T * jnp.log(jnp.sum(jnp.abs(r) ** 2))


gx = np.asarray(flat(jax.grad(F_ex)(p)))
_r = np.asarray(rho_ex(p)); _a2 = np.abs(_r) ** 2; _sm = _mg[:, None] == _mg[None, :]
_rc = np.asarray(compute_rho(model, p, Sx, 64)); assert np.abs(_rc - _r).max() < 1e-10, "ρ densa != ρ por solapes"
print(f"|∇F| exacto = {np.linalg.norm(gx):.4f}   P = {_a2.sum():.6g} (S2 = {-np.log(_a2.sum()):.4f}),  fracción de P fuera del sector = "
      f"{(_a2.sum() - _a2[_sm].sum()) / _a2.sum():.3e}   F = {F_ex(p):.5f}   sz_label={a.sz_label} perturb={a.perturb}", flush=True)
TE = {"std": make_energy_terms(model, chi, H), "vmc": make_energy_terms_vmc(model, chi, H)}
ec = min(a.e_chunk, a.M)


def make_cfg(eform, pest, psym, Mp, K, ks):
    if pest == "tilted":
        pur, mix = make_tilted_purity(model, N, chi, T), make_mix_partners(model, N, ks, K)
    elif pest == "u1":
        pur, mix = make_u1_lowvar_purity(model, N, chi, T, ks, K, shell=False, cv_t=False, sym=psym), make_u1_mix_partners(model, N, ks, K)
    else:
        pur, mix = make_u1_mask_purity(model, N, chi, T, ks, K, kind=pest, cv_t=False, sym=psym), make_u1_mix_partners(model, N, ks, K)
    nsh = getattr(pur, "n_eval", 0); mc = max(1, min(Mp, a.pairs_chunk // (K + nsh)))
    while Mp % mc:
        mc -= 1
    nc = Mp // mc; terms = TE[eform]

    @jax.jit
    def g(s, key):
        sc = s.reshape(-1, ec, N)
        Ebar = jnp.sum(jax.lax.map(lambda c: terms(p, c, 0.0, a.M)[1], sc))
        gE = jax.lax.map(lambda c: flat(jax.grad(lambda q: terms(q, c, Ebar, a.M)[0])(p)), sc).sum(0)
        s_ = s[:Mp]; t, lq = mix(p, s_, key)
        cs = (s_.reshape(nc, mc, N), t.reshape(nc, mc * K, N), lq.reshape(nc, mc * K))
        P = jnp.mean(jax.lax.map(lambda c: pur(p, c[0], c[1], c[2], K)[1][0], cs)); Pf = sg(P)
        gP = jax.lax.map(lambda c: flat(jax.grad(lambda q: pur(q, c[0], c[1], c[2], K, Pf, Pf)[0])(p)), cs).mean(0)
        return gE + gP
    return g, Mp * K, nsh


G = {}
for c in a.configs:
    nm, ef, pe, ps, Mp, K, ks = c.split(":")
    G[nm] = make_cfg(ef, pe, bool(int(ps)), int(Mp), int(K), [float(x) for x in ks.split(",")])
key = jax.random.PRNGKey(a.seed); acc = {}; tw = {nm: 0.0 for nm in G}; t0 = time.time()
for i in range(a.nb):
    key, k = jax.random.split(key); k1, k2 = jax.random.split(k); s = sampler(p, k1)
    for j, (nm, (g, _, _)) in enumerate(G.items()):
        tt = time.time(); v = np.asarray(g(s, jax.random.fold_in(k2, j)))
        if i > 0:
            tw[nm] += time.time() - tt
        s1, s2, n = acc.get(nm, (0.0, 0.0, 0)); acc[nm] = (s1 + v, s2 + v * v, n + 1)
    if i in (0, 9) or (i + 1) % 25 == 0:
        print(f"  lote {i + 1}/{a.nb} ({time.time() - t0:.0f} s)", flush=True)
res = dict(args=vars(a), exact_norm=float(np.linalg.norm(gx)))
for nm, (s1, s2, n) in acc.items():
    m = s1 / n; var = (s2 - n * m * m) / (n - 1); noise = float(np.sqrt(np.sum(np.maximum(var, 0)))); se = noise / np.sqrt(n)
    _, pairs, nsh = G[nm]
    res[nm] = dict(noise_one_batch=noise, se_mean=se, norm_bias=float(np.linalg.norm(m - gx)), pairs=pairs, n_shell=nsh,
                   sec_per_batch=tw[nm] / max(n - 1, 1))
    print(f"{nm:14s} parejas={pairs:6d} cascarón={nsh:3d}  ruido de ∇F por lote = {noise:6.3f}  |media − exacto| = "
          f"{res[nm]['norm_bias']:.3f} (error de la media {se:.3f})  t = {res[nm]['sec_per_batch']:.2f} s/lote", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
