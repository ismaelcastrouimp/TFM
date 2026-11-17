"""
ff_diag_sampler.py
==================
Muestreo EXACTO de la diagonal en z del MRE (o Gibbs) del TFIM periodico, p(s) = <s|rho|s>, a N grande (fermiones libres).
rho = sum_n w_n |n><n| (estados de Fock BdG por sector de paridad, ff_local_obs.py).  Una configuracion z esta fijada,
salvo el flip global (p(s) = p(-s), [rho, prod X] = 0), por las paredes d_j = Z_j Z_{j+1} = i g_{2j+1} g_{2j+2}, j=0..N-2
(bilineales de Majorana DISJUNTOS que conmutan; el enlace frontera queda fijado por prod d = 1).  Para cada |n> gaussiano:
  medir d_j:  P(d_j = d) = (1 + d M_ab)/2,  M_pq = <i g_p g_q>,  (a,b) = (2j+1, 2j+2)
  actualizacion (Wick):  M'_kl = M_kl + d (M_kb M_la - M_ka M_lb) / (1 + d M_ab),   y se anulan filas/columnas a, b.
Muestreo: n ~ w_n, despues d_0, d_1, ... secuencialmente, s_0 = +-1 al azar, s_{j+1} = s_j d_j.
Validacion (--validate, N<=10): p(s) exacta (suma sobre todos los n) frente a diag(rho_MRE) por ED, y szz_r muestreados.
Uso: python scripts/ff_diag_sampler.py --validate --N 8
     python scripts/ff_diag_sampler.py --N 25 --T 3 --samples 200000 --out data/gap_arch/qdistill/mre_diag_N25_T3.npz
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ff_local_obs as ff


def M_of(sc, occ):
    C = sc.covariance(occ)                                # (B,2N,2N) <g_p g_q>
    M = np.real(1j * (C - np.swapaxes(C, 1, 2)) / 2)      # <i g_p g_q>, antisimetrica real
    return M


def walk(M, N, rng=None, d_fixed=None):
    """Recorre j=0..N-2.  rng -> muestrea d; d_fixed (B,N-1) -> probabilidad de esas d.  Devuelve (d, log prob)."""
    B = M.shape[0]; M = M.copy(); logp = np.zeros(B); D = np.zeros((B, N - 1))
    for j in range(N - 1):
        a, b = 2 * j + 1, 2 * j + 2
        mab = M[:, a, b]
        if d_fixed is None:
            d = np.where(rng.random(B) < (1 + mab) / 2, 1.0, -1.0)
        else:
            d = d_fixed[:, j]
        pr = (1 + d * mab) / 2
        logp += np.log(np.maximum(pr, 1e-300)); D[:, j] = d
        den = np.where(pr > 1e-300, 2 * pr, 1.0)[:, None, None]           # = 1 + d M_ab
        Ma, Mb = M[:, :, a], M[:, :, b]                                    # M_ka, M_kb
        M = M + d[:, None, None] * (Mb[:, :, None] * Ma[:, None, :] - Ma[:, :, None] * Mb[:, None, :]) / den   # M_kb M_la - M_ka M_lb
        M[:, a, :] = 0; M[:, :, a] = 0; M[:, b, :] = 0; M[:, :, b] = 0
        M[:, a, b] = d; M[:, b, a] = -d
    return D, logp


def s_from_d(D, s0):
    return np.concatenate([s0[:, None], s0[:, None] * np.cumprod(D, axis=1)], axis=1)


def setup(N, T, kind="mre"):
    secs, E, ids, sid = ff.build(N)
    w, info = ff.weights(kind, E, T=T)
    return secs, E, ids, sid, w / w.sum(), info


def sample(N, T, n, rng, kind="mre", chunk=4096, setup_=None):
    secs, E, ids, sid, p, _ = setup_ or setup(N, T, kind)
    pick = rng.choice(len(p), size=n, p=p)
    out = []
    for k, sc in enumerate(secs):
        pk = pick[sid[pick] == k]
        for st in range(0, len(pk), chunk):
            D, _ = walk(M_of(sc, ff.occ_of(ids[pk[st:st + chunk]], N)), N, rng=rng)
            out.append(s_from_d(D, rng.choice([-1.0, 1.0], size=len(D))))
    s = np.concatenate(out); return s[rng.permutation(len(s))]


def validate(N, T):
    secs, E, ids, sid, p, info = setup(N, T)
    idx = np.arange(2 ** N); s_all = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(float)
    D_all = s_all[:, :-1] * s_all[:, 1:]                                   # d_j = s_j s_{j+1}
    keep = s_all[:, 0] == 1; Dk = D_all[keep]                               # una de cada pareja (s, -s)
    pd = np.zeros(len(Dk))
    for k, sc in enumerate(secs):
        nz = np.nonzero((sid == k) & (p > 0))[0]
        for n in nz:
            M = M_of(sc, ff.occ_of(ids[n:n + 1], N))
            _, lp = walk(np.repeat(M, len(Dk), 0), N, d_fixed=Dk)
            pd += p[n] * np.exp(lp)
    p_ff = np.zeros(2 ** N); p_ff[keep] = pd / 2
    comp = (2 ** N - 1) ^ idx                                               # s -> -s = complemento de bits
    p_ff[~keep] = p_ff[comp[~keep]]
    # ED
    H = np.zeros((2 ** N, 2 ** N)); H[idx, idx] = -ff.J * sum(s_all[:, i] * s_all[:, (i + 1) % N] for i in range(N))
    for i in range(N):
        H[idx, idx ^ (1 << i)] -= ff.Hf
    ev, V = np.linalg.eigh(H)
    w, _ = ff.weights("mre", ev, T=T); w = w / w.sum()
    p_ed = (V ** 2) @ w
    print(f"N={N} T={T}: sum p_ff={p_ff.sum():.10f}  max|p_ff - p_ED|={np.abs(p_ff - p_ed).max():.2e}  "
          f"(max p_ED={p_ed.max():.3e})", flush=True)
    rng = np.random.default_rng(0)
    s = sample(N, T, 200000, rng)
    for r in (1, 2, 3):
        ex = p_ed @ (s_all[:, 0] * s_all[:, r]); mc = np.mean(s * np.roll(s, -r, 1))
        print(f"  szz{r}: ED {ex:+.5f}   muestreo {mc:+.5f} +- {np.std(s * np.roll(s, -r, 1)) / np.sqrt(len(s) * N) * 3:.5f}")
    return np.abs(p_ff - p_ed).max()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--T", type=float, default=3.0)
    ap.add_argument("--validate", action="store_true"); ap.add_argument("--samples", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.validate:
        validate(a.N, a.T); sys.exit()
    assert a.out and "/data/0/" not in os.path.abspath(a.out)
    t0 = time.time(); rng = np.random.default_rng(a.seed)
    s = sample(a.N, a.T, a.samples, rng)
    print(f"N={a.N} T={a.T}: {len(s)} muestras en {time.time() - t0:.0f} s; " +
          " ".join(f"szz{r}={np.mean(s * np.roll(s, -r, 1)):+.4f}" for r in (1, 2, 3)), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, s=s.astype(np.int8), N=a.N, T=a.T)
