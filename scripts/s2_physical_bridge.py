"""
s2_physical_bridge.py  (16-11-2026)
==========================================
¿Por qué el AR inclinado va mal con ancilla FÍSICA y cómo arreglarlo?  Experimento exacto (numpy, sin redes).

Estado: purificación canónica del MRE del TFIM 1D (PBC, J=-1, h=-1.5):  ψ(s,a) = <s|ρ^{1/2}|a>, orden x = (s_0..s_{N-1},
a_0..a_{N-1}) (el de ARNNDense en s2_noise).  Los condicionales AR EXACTOS salen de los marginales de |ψ|², así que
cualquier estimador AR se puede simular sin error de red: solo se mide la varianza intrínseca del estimador.

Como el orden es s primero, ψ(s,a) = A(s) φ(a|s) con φ normalizada para cada s, y
    P = Tr ρ_S² = Σ_{s1,s2} p(s1) p(s2) |O(s1,s2)|²,     O(s1,s2) = <φ_{s1}|φ_{s2}> = Σ_a φ(a|s1) φ(a|s2).

Estimadores (todos insesgados), coste en "pasos de condicional" (1 sitio de 1 red):
  exact_anc  s1 ~ p, s2 ~ q_tilt(s2|s1) (inclinado SOLO en s), O exacto                  [suelo: = caso ancilla MPS]
  tilt_full  x1 ~ |ψ|², x2 ~ q_tilt(x2|x1) en los 2N sitios (lo que hace renyi2_tilted)    [actual]
  tilt_sys   igual pero inclinando solo los sitios del sistema (tilt_sites = s)
  bridge     s1 ~ p, s2 ~ q_tilt(s2|s1) en s;  O estimado con n_a muestras de la ancilla del AR "puente"
             r(a_k|a_<k) ∝ sqrt(p(a_k|a_<k,s1) p(a_k|a_<k,s2)), peso φ(a|s1)φ(a|s2)/r(a); |O|² ≈ Ô·Ô' (dos independientes)
Métrica: ineficiencia η = Var(P̂)/P² · coste por unidad (independiente del presupuesto; menor = mejor).
Uso:  python scripts/s2_physical_bridge.py --N 4 5 6 --T 1 2 3 5
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, nargs="+", default=[4, 5, 6]); ap.add_argument("--T", type=float, nargs="+", default=[1, 2, 3, 5])
ap.add_argument("--M", type=int, default=2000, help="unidades (x1 o pares s1) por repetición")
ap.add_argument("--rep", type=int, default=40); ap.add_argument("--K", type=int, default=5)
ap.add_argument("--n_a", type=int, nargs="+", default=[1, 4])
ap.add_argument("--n_u", type=int, nargs="*", default=[], help="tamaños del U-estadistico del puente")
ap.add_argument("--gauge", nargs="+", default=["none"],
                help="unitaria en la ancilla (F no cambia): none | rot:θ (rotación real R(θ) en cada sitio) | haar:seed (O global aleatoria)")
ap.add_argument("--psi_file", default=None,
                help=".npy con Ψ (2^N x 2^N, filas s) entrenada: mide los estimadores en ese estado (un N, un T)")
ap.add_argument("--oracle", action="store_true", help="añade el puente con propuesta óptima exacta (techo)")
ap.add_argument("--state", default="mre", choices=["mre", "gibbs"])
ap.add_argument("--out", default="data/gap_arch/physical_tilted/bridge.json")
a = ap.parse_args()
KAPPAS = np.array([0.05, 0.1, 0.2, 0.4, 1.0])


def tfim(N, J=-1.0, h=-1.5):
    D = 2 ** N; idx = np.arange(D)
    s = 1 - 2 * ((idx[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1)              # bit más significativo = sitio 0
    H = np.diag(J * sum(s[:, i] * s[:, (i + 1) % N] for i in range(N)).astype(float))
    for i in range(N): H[idx, idx ^ (1 << (N - 1 - i))] += h
    return np.linalg.eigh(H)


def mre_p(ev, T):
    D = len(ev); best = (ev[0], 0.0, np.r_[1.0, np.zeros(D - 1)]); S1c, S2c = np.cumsum(ev), np.cumsum(ev ** 2)
    for k in range(2, D + 1):
        S1, S2_ = S1c[k - 1], S2c[k - 1]; disc = (S1 + T * k) ** 2 - k * (S2_ + 2 * T * S1)
        if disc < 0: continue
        hi = ev[k] if k < D else np.inf
        for c in ((S1 + T * k) + np.sqrt(disc)) / k, ((S1 + T * k) - np.sqrt(disc)) / k:
            if not (ev[k - 1] < c <= hi + 1e-12): continue
            w = np.maximum(c - ev, 0); p = w / w.sum(); E, S = (p * ev).sum(), -np.log((p ** 2).sum())
            if E - T * S < best[0] - T * best[1]: best = (E, S, p)
    return best[2]


class ARTable:
    """ψ sobre 2N bits (sitio 0 = bit más significativo) con condicionales AR exactos."""
    def __init__(self, psi, n):
        self.psi, self.n = psi, n; w = np.abs(psi) ** 2
        self.M = [w.reshape(2 ** k, -1).sum(1) for k in range(n + 1)]              # M[k][prefijo]

    def cond1(self, k, pref):                                                       # p(x_k = 1 | prefijo)
        return self.M[k + 1][2 * pref + 1] / np.maximum(self.M[k][pref], 1e-300)

    def sample(self, B, rng, k0=0, pref=None, k1=None, anchor=None, kap=None):
        """Muestrea sitios k0..k1-1 desde el prefijo; si anchor/kap: inclinado hacia anchor (bits) con kap por fila.
        Devuelve (prefijo final, log q_κ para todos los κ (B, len(KAPPAS)))."""
        k1 = self.n if k1 is None else k1; pref = np.zeros(B, np.int64) if pref is None else pref.copy()
        lq = np.zeros((B, len(KAPPAS)))
        for k in range(k0, k1):
            p1 = self.cond1(k, pref)
            if anchor is None:
                b = (rng.random(B) < p1).astype(np.int64); lq += np.log(np.where(b, p1, 1 - p1))[:, None]
            else:
                ak = (anchor >> (self.n - 1 - k)) & 1; pa = np.where(ak, p1, 1 - p1)       # prob del bit ancla
                qa = pa / (pa + kap * (1 - pa))                                            # prob inclinada de copiar
                same = rng.random(B) < qa; b = np.where(same, ak, 1 - ak)
                cK = pa[:, None] + KAPPAS[None] * (1 - pa[:, None])
                lq += np.where(same[:, None], np.log(pa)[:, None], np.log(1 - pa)[:, None] + np.log(KAPPAS)[None]) - np.log(cK)
            pref = 2 * pref + b
        return pref, lq


def gauge_matrix(g, N):
    if g in ("none", "chol", "eig"): return None              # chol / eig: gauges fijados a partir de rho (en run)
    kind, v = g.split(":")
    if kind == "rot":
        t = float(v); r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]); U = np.ones((1, 1))
        for _ in range(N): U = np.kron(U, r)
        return U
    Q, R = np.linalg.qr(np.random.default_rng(int(v)).normal(size=(2 ** N, 2 ** N)))
    return Q * np.sign(np.diag(R))


def run(N, T, rng, gauge="none"):
    ev, V = tfim(N); p = mre_p(ev, T) if a.state == "mre" else np.exp(-(ev - ev[0]) / T) / np.exp(-(ev - ev[0]) / T).sum()
    sq = (V * np.sqrt(p)) @ V.T                                                     # ρ^{1/2}
    U = gauge_matrix(gauge, N)
    if gauge == "chol":                    # Ψ = L (Cholesky, triangular inferior en el índice entero: ψ(s,a)=0 si a>s)
        sq = np.linalg.cholesky((V * p) @ V.T + 1e-12 * np.eye(len(p)))
    elif gauge == "eig":                   # Ψ = V sqrt(p): la ancilla etiqueta autoestados
        sq = V * np.sqrt(p)
    if U is not None: sq = sq @ U.T                                                 # ψ(s,a) -> Σ_a' U_{a a'} ψ(s,a'): ρ_S igual
    if a.psi_file:
        sq = np.load(a.psi_file); sq = sq / np.sqrt(np.sum(np.abs(sq) ** 2))
        if np.abs(sq.imag).max() < 1e-14: sq = sq.real
    psi = sq.reshape(-1); D = 2 ** N; tab = ARTable(psi, 2 * N)
    rho = sq @ sq.conj().T if a.psi_file else (V * p) @ V.T; P_ex = float((np.abs(rho) ** 2).sum())
    pS = tab.M[N]                                                                    # p(s)
    Phi = sq / np.sqrt(np.maximum(pS, 1e-300))[:, None]                              # φ(a|s) (filas s)
    O = Phi @ Phi.conj().T                                                                  # O(s1,s2) exacto
    Kn = len(KAPPAS); K = a.K; Nt = 2 * N
    lnk = lambda lq: np.log(np.mean(np.exp(lq), 1))                                  # log q_mix (mezcla uniforme de κ)
    def kap_rows(B): return np.tile(KAPPAS, B // Kn + 1)[:B]

    def est_exact_anc(M):
        s1, _ = tab.sample(M, rng, k1=N); s1r = np.repeat(s1, K); kap = kap_rows(M * K)
        s2, lq = tab_sample_s(s1r, kap)
        return np.mean(pS[s2] * np.abs(O[s1r, s2]) ** 2 / np.exp(lnk(lq))), N + K * N

    def tab_sample_s(anchor_s, kap):                                                 # inclinado solo en los N sitios de s
        tabS = ARTable.__new__(ARTable); tabS.n = N; tabS.M = tab.M[:N + 1]
        return tabS.sample(len(anchor_s), rng, anchor=anchor_s, kap=kap)

    def est_tilt(M, sys_only):
        x1, _ = tab.sample(M, rng); x1r = np.repeat(x1, K); kap = kap_rows(M * K)
        if sys_only:
            s2, lq = tab_sample_s(x1r >> N, kap)
            x2, lq2 = tab.sample(M * K, rng, k0=N, pref=s2)                          # ancilla de x2 sin inclinar
            lq = lq + lq2
        else:
            x2, lq = tab.sample(M * K, rng, anchor=x1r, kap=kap)
        s1, a1, s2, a2 = x1r >> N, x1r & (D - 1), x2 >> N, x2 & (D - 1)
        w = psi[x2] * sq[s2, a1] * sq[s1, a2] / psi[x1r]                             # |ψ(x2)|² R
        return np.mean(w / np.exp(lnk(lq))), Nt + K * (Nt + 5)

    def overlap_bridge(s1, s2, n_a, raw=False):
        """Ô(s1,s2) con n_a muestras del AR puente r ∝ sqrt(p(.|s1) p(.|s2)) en la ancilla."""
        B = len(s1); s1 = np.repeat(s1, n_a); s2 = np.repeat(s2, n_a)
        pa1, pa2 = s1.copy(), s2.copy(); lr = np.zeros(B * n_a)
        for k in range(N, 2 * N):
            q1, q2 = tab.cond1(k, pa1), tab.cond1(k, pa2)
            g1, g0 = np.sqrt(q1 * q2), np.sqrt((1 - q1) * (1 - q2)); Z = g1 + g0
            b = (rng.random(B * n_a) < g1 / np.maximum(Z, 1e-300)).astype(np.int64)
            lr += np.log(np.maximum(np.where(b, g1, g0) / np.maximum(Z, 1e-300), 1e-300))
            pa1, pa2 = 2 * pa1 + b, 2 * pa2 + b
        aa = pa1 & (D - 1)
        w = Phi[s1, aa] * np.conj(Phi[s2, aa]) / np.exp(lr)
        return w.reshape(B, n_a).mean(1) if not raw else w.reshape(B, n_a)

    def est_bridge(M, n_a):
        s1, _ = tab.sample(M, rng, k1=N); s1r = np.repeat(s1, K); kap = kap_rows(M * K)
        s2, lq = tab_sample_s(s1r, kap)
        f = np.real(overlap_bridge(s1r, s2, n_a) * np.conj(overlap_bridge(s1r, s2, n_a)))
        return np.mean(pS[s2] * f / np.exp(lnk(lq))), N + K * (N + 2 * 2 * N * n_a)

    def est_bridge_u(M, n):
        """U-estadistico: n muestras del puente, |O|^2 ~ sum_{i!=j} w_i w_j / (n(n-1)) (usa todos los pares)."""
        s1, _ = tab.sample(M, rng, k1=N); s1r = np.repeat(s1, K); kap = kap_rows(M * K)
        s2, lq = tab_sample_s(s1r, kap)
        w = overlap_bridge(s1r, s2, n, raw=True); f = (w.sum(1) ** 2 - (w ** 2).sum(1)) / (n * (n - 1))
        return np.mean(pS[s2] * f / np.exp(lnk(lq))), N + K * (N + 2 * N * n)

    def est_bridge_oracle(M, n):
        """Techo de una propuesta aprendida: r*(a) ∝ |φ(a|s1)φ(a|s2)| exacta (enumerando a), U-estadístico con n."""
        s1, _ = tab.sample(M, rng, k1=N); s1r = np.repeat(s1, K); kap = kap_rows(M * K)
        s2, lq = tab_sample_s(s1r, kap)
        prod = Phi[s1r] * Phi[s2]; Z = np.abs(prod).sum(1); pr = np.abs(prod) / Z[:, None]
        u = rng.random((len(s1r), n)); idx = (pr.cumsum(1)[:, None, :] < u[..., None]).sum(-1).clip(max=D - 1)
        w = Z[:, None] * np.sign(np.take_along_axis(prod, idx, 1))
        f = (w.sum(1) ** 2 - (w ** 2).sum(1)) / (n * (n - 1))
        return np.mean(pS[s2] * f / np.exp(lnk(lq))), N + K * (N + 2 * N * n)

    herm = float(np.real(np.sum(sq * sq.T)))                    # Re Σ ψ(s,a)ψ(a,s) = 1 sii ψ hermítica (gauge canónico ρ^{1/2})
    out = dict(N=N, T=T, P=P_ex, S2=-np.log(P_ex), gauge=gauge, herm=herm, trace=float(np.real(np.trace(sq))), trace_canon=float(np.sqrt(np.clip(np.linalg.eigvalsh(rho), 0, None)).sum()))
    meths = {"exact_anc": lambda: est_exact_anc(a.M), "tilt_full": lambda: est_tilt(a.M, False),
             "tilt_sys": lambda: est_tilt(a.M, True)}
    if a.psi_file and np.iscomplexobj(sq):                 # est_tilt supone ψ real
        del meths["tilt_full"], meths["tilt_sys"]
    for n_a in a.n_a: meths[f"bridge_na{n_a}"] = (lambda n_a=n_a: est_bridge(a.M, n_a))
    for n in a.n_u: meths[f"bridgeU_{n}"] = (lambda n=n: est_bridge_u(a.M, n))
    if a.oracle: meths["oracle_U2"] = lambda: est_bridge_oracle(a.M, 2)
    for name, f in meths.items():
        vals = []; cost = None
        for _ in range(a.rep):
            v, cost = f(); vals.append(v)
        vals = np.array(vals); cv2 = vals.var() * a.M / P_ex ** 2                  # Var por unidad / P²
        out[name] = dict(mean=float(vals.mean()), bias_sigma=float((vals.mean() - P_ex) / (vals.std() / np.sqrt(a.rep))),
                         eta=float(cv2 * cost), cost=cost)
    return out


rng = np.random.default_rng(0); res = []
for N in a.N:
  for T in a.T:
    for g in a.gauge:
        r = run(N, T, rng, g); res.append(r)
        print(f"N={N} T={T} gauge={g} S2={r['S2']:.3f} herm={r['herm']:.3f} | " + "  ".join(
            f"{k} η={v['eta']:.3g} ({v['bias_sigma']:+.1f}σ)" for k, v in r.items() if isinstance(v, dict)), flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
