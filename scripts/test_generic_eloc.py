"""
test_generic_eloc.py
====================
Comprueba src_renyi/mps_ancilla/generic_eloc.py:
  1) TFIM: pérdida de energía y su gradiente IGUALES a los de mc_loss.make_mc_terms (código a mano) en el mismo lote.
  2) H de NetKet (Paulis) = H densa de robust_ed.py (heis) para xxz y j1j2.
  3) j1j2 / xxz / tfim_hz: media MC de E_loc = Tr(ρH) exacta, y gradiente MC medio = gradiente exacto de E.
Uso: JAX_PLATFORMS=cpu python scripts/test_generic_eloc.py --N 8
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=8); ap.add_argument("--chi", type=int, default=4)
ap.add_argument("--M", type=int, default=1024); ap.add_argument("--R", type=int, default=64)
ap.add_argument("--perturb", type=float, default=0.3)
a = ap.parse_args()
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.ladder import init_params
from src_renyi.mps_ancilla.training import make_sampler, make_loss_fns
from src_renyi.mps_ancilla.mc_loss import make_mc_terms
from src_renyi.mps_ancilla.generic_eloc import build_hamiltonian, make_energy_terms

N, chi = a.N, a.chi
model = build_alt_model(N, chi, "vit:conv:r=1")
p0 = init_params(model, N, 3)
lv, td = jax.tree_util.tree_flatten(p0)
ks = jax.random.split(jax.random.PRNGKey(5), len(lv))
params = jax.tree_util.tree_unflatten(td, [x + a.perturb * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(lv, ks)])
sampler = make_sampler(model, N, a.M)
flat = lambda g: np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(g)])
D = 2 ** N; idx = np.arange(D); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
ok = True

# 1) TFIM frente al código a mano
H = build_hamiltonian("tfim", N, h=-1.5)
terms = make_energy_terms(model, chi, H)
fE = make_mc_terms(model, N, 3.0, -1.5, -1.0, None, 2, "iid")
s = sampler(params, jax.random.PRNGKey(0))
Eb_hand = float(fE(params, s, None, a.M)[2][0]); Eb_gen = float(terms(params, s, 0.0, a.M)[1])
g_hand = flat(jax.grad(lambda q: fE(q, s, (Eb_hand, jnp.zeros(()), jnp.ones(())), a.M)[0])(params))
g_gen = flat(jax.grad(lambda q: terms(q, s, Eb_hand, a.M)[0])(params))
rel = np.linalg.norm(g_hand - g_gen) / np.linalg.norm(g_hand)
print(f"1) TFIM: Ē a mano {Eb_hand:.10f}  genérico {Eb_gen:.10f}   |Δ∇|/|∇| = {rel:.2e}", flush=True)
ok &= abs(Eb_hand - Eb_gen) < 1e-9 and rel < 1e-9


# 2) operador de NetKet en la base all_s, frente a robust_ed
def dense_in_all_s(H):
    Hj = H.to_jax_operator()
    sp, mels = Hj.get_conn_padded(jnp.asarray(all_s))
    sp, mels = np.asarray(sp), np.asarray(mels)
    col = (((sp + 1) / 2).astype(np.int64) << np.arange(N)).sum(-1)
    Md = np.zeros((D, D))
    np.add.at(Md, (np.repeat(idx, sp.shape[1]), col.ravel()), mels.ravel().real)
    return Md


def heis(bonds, Jb, Delta=1.0):                     # copia de robust_ed.py
    Hb = np.zeros((D, D))
    for i, j in bonds:
        si, sj = all_s[:, i], all_s[:, j]
        Hb[idx, idx] += Jb * Delta * si * sj
        flip = si != sj
        Hb[idx[flip], (idx ^ (1 << i) ^ (1 << j))[flip]] += 2 * Jb
    return Hb


nn1 = [(i, (i + 1) % N) for i in range(N)]; nn2 = [(i, (i + 2) % N) for i in range(N)]
refs = {"xxz": heis(nn1, 1.0, 0.7), "j1j2": heis(nn1, 1.0) + heis(nn2, 0.5)}
zz = sum(all_s[:, i] * all_s[:, j] for i, j in nn1)
Htz = np.diag(-1.0 * zz - 0.5 * all_s.sum(1))
for i in range(N):
    Htz[idx, idx ^ (1 << i)] += -1.5
refs["tfim_hz"] = Htz
Hs = {"xxz": build_hamiltonian("xxz", N, Delta=0.7), "j1j2": build_hamiltonian("j1j2", N, J2=0.5),
      "tfim_hz": build_hamiltonian("tfim_hz", N, h=-1.5, hz=-0.5)}
for k in refs:
    err = np.abs(dense_in_all_s(Hs[k]) - refs[k]).max()
    print(f"2) {k}: max|H_netket − H_robust_ed| = {err:.1e}", flush=True)
    ok &= err < 1e-12

# 3) E y ∇E: MC frente a exacto
for k in ("j1j2", "xxz", "tfim_hz"):
    terms = make_energy_terms(model, chi, Hs[k])
    loss, _ = make_loss_fns(model, all_s, refs[k], 3.0, chunk=64)
    Ex = float(loss(params)[1][0])
    gx = flat(jax.grad(lambda q: loss(q)[1][0])(params))

    @jax.jit
    def one(key):
        s = sampler(params, key)
        Eb = terms(params, s, 0.0, a.M)[1]
        return Eb, jax.grad(lambda q: terms(q, s, Eb, a.M)[0])(params)
    Es, G = [], []
    key = jax.random.PRNGKey(11)
    for r in range(a.R):
        key, kk = jax.random.split(key)
        e, g = one(kk); Es.append(float(e)); G.append(flat(g))
    Es, G = np.array(Es), np.array(G); gm = G.mean(0)
    nm = np.sqrt(np.mean(np.sum((G - gm) ** 2, 1)) / a.R)
    zE = (Es.mean() - Ex) / (Es.std(ddof=1) / np.sqrt(a.R))
    zg = np.linalg.norm(gm - gx) / nm
    print(f"3) {k}: E MC = {Es.mean():.5f} ± {Es.std(ddof=1) / np.sqrt(a.R):.5f}   exacta {Ex:.5f}  (z = {zE:+.2f})   "
          f"∇E: cos = {gm @ gx / np.linalg.norm(gm) / np.linalg.norm(gx):+.4f}  ‖ḡ − exacto‖/ruido rms de la media = {zg:.2f}",
          flush=True)
    ok &= abs(zE) < 4 and zg < 2.0
print("TODO OK" if ok else "FALLO", flush=True)
