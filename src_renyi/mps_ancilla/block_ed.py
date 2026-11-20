"""
src_renyi/mps_ancilla/block_ed.py
=================================
Espectro completo de hamiltonianos de Heisenberg (xxz, j1j2) con PBC por bloques de simetría:
magnetización Sz (número de espines arriba) y momento k = 2π q / N (traslaciones).  Solo autovalores: es lo que
necesita el MRE (p ∝ (c − E)_+ sobre todo el espectro).

Convención (la de generic_eloc.build_hamiltonian / robust_ed.py, Paulis):
    H = Σ_bonds J_b [ (X_i X_j + Y_i Y_j) + Δ Z_i Z_j ]
      -> diagonal J_b Δ s_i s_j  (s = ±1),  giro de un par antiparalelo con amplitud 2 J_b.

Base de momento (Sandvik, arXiv:1101.3281, §4.1): representantes a (mínimo entero de su órbita de traslaciones),
periodo R_a, compatibles con q si q·R_a ≡ 0 (mod N);
    ⟨b(k)|H|a(k)⟩ += h · e^{−i k l} · sqrt(R_a / R_b),
con b el representante del estado de llegada y l la traslación que lo lleva a b.

Atajos (opcionales, use_sym=True): el espectro del sector m es el del −m (inversión de espines) y el del momento q
el del N − q (reflexión).  Ambos valen para xxz y j1j2 con PBC; se comprueban en scripts/test_block_ed.py.

  bonds_for(name, N, J, J2, Delta)  -> lista (i, j, J_b, Δ_b)
  spectrum(name, N, ..., use_k=True, use_sym=True, verbose=False) -> autovalores (2^N,) ordenados
  mre(ev, T) -> dict(E, S2, F, c)
"""
import time
import numpy as np


def bonds_for(name, N, J=1.0, J2=0.5, Delta=1.0):
    nn1 = [(i, (i + 1) % N) for i in range(N)]
    nn2 = [(i, (i + 2) % N) for i in range(N)]
    if name == "xxz":
        return [(i, j, J, Delta) for i, j in nn1]
    if name == "j1j2":
        return [(i, j, J, 1.0) for i, j in nn1] + [(i, j, J2 * J, 1.0) for i, j in nn2]
    raise ValueError(f"{name}: solo xxz y j1j2 conservan Sz")


def _popcount(x):
    x = x.copy(); c = np.zeros_like(x)
    while np.any(x):
        c += x & 1; x >>= 1
    return c


def _rots(x, N):
    """(N, n): todas las traslaciones cíclicas (rotación de bits a la izquierda por l = 0..N-1)."""
    mask = (1 << N) - 1
    return np.stack([((x << l) | (x >> (N - l))) & mask if l else x for l in range(N)])


def _rep(x, N):
    r = _rots(x, N)
    return r.min(0), r.argmin(0), r


def _sector_elements(states, N, bonds, use_k):
    """Representantes del sector y elementos (fila, col, amplitud, l) de H en la base de representantes."""
    if use_k:
        rep, _, rots = _rep(states, N)
        reps = np.sort(states[rep == states])
        R = N // (_rots(reps, N) == reps[None, :]).sum(0)
    else:
        reps = np.sort(states); R = np.ones(len(reps), np.int64)
    n = len(reps)
    diag = np.zeros(n)
    rows, cols, amps, ls = [], [], [], []
    for i, j, Jb, Db in bonds:
        bi = (reps >> i) & 1; bj = (reps >> j) & 1
        diag += Jb * Db * (2 * bi - 1) * (2 * bj - 1)
        f = np.nonzero(bi != bj)[0]
        b = reps[f] ^ ((1 << i) | (1 << j))
        if use_k:
            rb, lb, _ = _rep(b, N)
        else:
            rb, lb = b, np.zeros(len(b), np.int64)
        rows.append(np.searchsorted(reps, rb)); cols.append(f); amps.append(np.full(len(f), 2.0 * Jb)); ls.append(lb)
    rows, cols, amps, ls = (np.concatenate(v) for v in (rows, cols, amps, ls))
    return reps, R, diag, rows, cols, amps, ls


def _block_eigs(N, q, R, diag, rows, cols, amps, ls, use_k):
    if not use_k:
        n = len(diag); H = np.zeros((n, n)); H[np.arange(n), np.arange(n)] = diag
        np.add.at(H, (rows, cols), amps)
        assert np.allclose(H, H.T)
        return np.linalg.eigvalsh(H)
    ok = (q * R) % N == 0
    if not ok.any():
        return np.zeros(0)
    idx = -np.ones(len(R), np.int64); idx[ok] = np.arange(ok.sum())
    m = ok[rows] & ok[cols]
    k = 2 * np.pi * q / N
    H = np.zeros((ok.sum(), ok.sum()), complex)
    H[np.arange(ok.sum()), np.arange(ok.sum())] = diag[ok]
    np.add.at(H, (idx[rows[m]], idx[cols[m]]), amps[m] * np.exp(-1j * k * ls[m]) * np.sqrt(R[cols[m]] / R[rows[m]]))
    herm = np.abs(H - H.conj().T).max()
    assert herm < 1e-10, f"bloque q={q} no hermítico ({herm:.1e})"
    if np.abs(H.imag).max() < 1e-12:
        return np.linalg.eigvalsh(H.real)
    return np.linalg.eigvalsh(H)


def spectrum(name, N, J=1.0, J2=0.5, Delta=1.0, use_k=True, use_sym=True, verbose=False):
    bonds = bonds_for(name, N, J, J2, Delta)
    allst = np.arange(2 ** N, dtype=np.int64)
    pc = _popcount(allst)
    out = []
    t0 = time.time()
    ups = range((N + 1) // 2, N + 1) if use_sym else range(N + 1)     # m >= 0 (n_up >= N/2)
    for nu in ups:
        mult_m = 2 if (use_sym and 2 * nu != N) else 1
        reps, R, diag, rows, cols, amps, ls = _sector_elements(allst[pc == nu], N, bonds, use_k)
        qs = range(N // 2 + 1) if (use_k and use_sym) else (range(N) if use_k else [0])
        for q in qs:
            mult_q = 2 if (use_k and use_sym and 0 < q and 2 * q != N) else 1
            ev = _block_eigs(N, q, R, diag, rows, cols, amps, ls, use_k)
            for _ in range(mult_m * mult_q):
                out.append(ev)
            if verbose and len(ev) > 2000:
                print(f"  n_up={nu} q={q}: dim {len(ev)}  ({time.time() - t0:.0f} s)", flush=True)
    ev = np.sort(np.concatenate(out))
    assert len(ev) == 2 ** N, (len(ev), 2 ** N)
    return ev


def mre(ev, T, method="minF"):
    """MRE a T: p ∝ (c − E)_+ con c − ⟨E⟩ = 2T.  La condición puede tener VARIAS raíces c (T(c) no es monótona cuando
    entra un multiplete grande, y a T < Δ/2 siempre está la raíz del fundamental); el MRE es la de MÍNIMA F = E − T S2.
    method="minF": enumera todas las raíces (k niveles en el soporte, cuadrática en c, como robust_ed.py);
    method="brentq": una raíz cualquiera (versión antigua; solo para comparar)."""
    ev = np.sort(np.asarray(ev, np.float64))
    if T <= 0:                                   # límite T -> 0+: mezcla uniforme del multiplete fundamental
        g = int(np.sum(np.abs(ev - ev[0]) < 1e-9 * max(1.0, abs(ev[0]))))
        return dict(E=float(ev[0]), S2=float(np.log(g)), F=float(ev[0]), c=float(ev[0]))
    if method == "brentq":
        from scipy.optimize import brentq
        cc = brentq(lambda c: c - (np.maximum(c - ev, 0) * ev).sum() / np.maximum(c - ev, 0).sum() - 2 * T,
                    ev.min() + 1e-9, ev.max() + 20 * T + 10, xtol=1e-12)
    else:
        e0 = ev[0]; x = ev - e0                                   # desplazar: mejor condicionado
        k = np.arange(1, len(x) + 1, dtype=np.float64)
        S1, S2c = np.cumsum(x), np.cumsum(x ** 2)
        b = S1 + T * k; disc = b ** 2 - k * (S2c + 2 * T * S1)
        hi = np.append(x[1:], np.inf)
        best = None
        for sgn in (1.0, -1.0):
            c = (b + sgn * np.sqrt(np.maximum(disc, 0))) / k
            ok = (disc >= 0) & (c > x) & (c <= hi + 1e-12)
            if not ok.any():
                continue
            kk, cc_, s1, s2 = k[ok], c[ok], S1[ok], S2c[ok]
            Z = kk * cc_ - s1
            E = (cc_ * s1 - s2) / Z
            P2 = (kk * cc_ ** 2 - 2 * cc_ * s1 + s2) / Z ** 2
            F = E + T * np.log(P2)
            i = int(np.argmin(F))
            if best is None or F[i] < best[0]:
                best = (F[i], cc_[i])
        cc = best[1] + e0
    pw = np.maximum(cc - ev, 0); pw /= pw.sum()
    E, S2 = float((pw * ev).sum()), float(-np.log((pw ** 2).sum()))
    return dict(E=E, S2=S2, F=E - T * S2, c=float(cc))


# ─── observables locales del MRE: ⟨n|O_r|n⟩ en cada autoestado (bloque a bloque) ─────────────────────────────────
# O_r = (1/N) Σ_i σ_i·σ_{i+r} (Paulis; conserva Sz y momento, y es invariante bajo inversión de espines y reflexión,
# así que valen los mismos atajos que para el espectro).  Dentro de un subespacio degenerado ⟨n|O|n⟩ depende de la base,
# pero el MRE da el mismo peso a niveles degenerados y solo importa la traza sobre el subespacio, que es invariante.

def _block_op(N, q, R, diag, rows, cols, amps, ls, ok, idx, dense):
    k = 2 * np.pi * q / N
    m = ok[rows] & ok[cols]
    vals = amps[m] * np.exp(-1j * k * ls[m]) * np.sqrt(R[cols[m]] / R[rows[m]])
    n = int(ok.sum())
    if dense:
        M = np.zeros((n, n), complex); M[np.arange(n), np.arange(n)] = diag[ok]
        np.add.at(M, (idx[rows[m]], idx[cols[m]]), vals)
        return M
    import scipy.sparse as sp
    M = sp.coo_matrix((vals, (idx[rows[m]], idx[cols[m]])), shape=(n, n)).tocsr()
    return M + sp.diags(diag[ok].astype(complex))


def spectrum_obs(name, N, J=1.0, J2=0.5, Delta=1.0, rs=None, use_sym=True, verbose=False, col_chunk=1024):
    """Autovalores (2^N,) ordenados y O (len(rs), 2^N) con ⟨n|O_r|n⟩ en el mismo orden.  Solo PBC con momento."""
    from scipy.linalg import eigh
    rs = list(range(1, N // 2 + 1)) if rs is None else list(rs)
    bonds = bonds_for(name, N, J, J2, Delta)
    obonds = {r: [(i, (i + r) % N, 1.0 / N, 1.0) for i in range(N)] for r in rs}
    allst = np.arange(2 ** N, dtype=np.int64)
    pc = _popcount(allst)
    evs, obs = [], []
    t0 = time.time()
    ups = range((N + 1) // 2, N + 1) if use_sym else range(N + 1)
    for nu in ups:
        mult_m = 2 if (use_sym and 2 * nu != N) else 1
        st = allst[pc == nu]
        hs = _sector_elements(st, N, bonds, True)
        R = hs[1]
        os_ = {r: _sector_elements(st, N, obonds[r], True) for r in rs}
        qs = range(N // 2 + 1) if use_sym else range(N)
        for q in qs:
            mult_q = 2 if (use_sym and 0 < q and 2 * q != N) else 1
            ok = (q * R) % N == 0
            if not ok.any():
                continue
            idx = -np.ones(len(R), np.int64); idx[ok] = np.arange(ok.sum())
            H = _block_op(N, q, *hs[1:], ok, idx, dense=True)
            assert np.abs(H - H.conj().T).max() < 1e-10
            real = np.abs(H.imag).max() < 1e-12
            w, V = eigh(H.real if real else H, overwrite_a=True, check_finite=False)
            del H
            ob = np.zeros((len(rs), len(w)))
            for ir, r in enumerate(rs):
                O = _block_op(N, q, *os_[r][1:], ok, idx, dense=False)
                if real:
                    O = O.real
                for c0 in range(0, len(w), col_chunk):
                    Vc = V[:, c0:c0 + col_chunk]
                    ob[ir, c0:c0 + col_chunk] = np.real(np.sum(Vc.conj() * (O @ Vc), axis=0))
            del V
            for _ in range(mult_m * mult_q):
                evs.append(w); obs.append(ob)
            if verbose and len(w) > 2000:
                print(f"  n_up={nu} q={q}: dim {len(w)}  ({time.time() - t0:.0f} s)", flush=True)
    ev = np.concatenate(evs); O = np.concatenate(obs, axis=1)
    assert len(ev) == 2 ** N
    order = np.argsort(ev, kind="stable")
    return ev[order], O[:, order], rs


def mre_obs(ev, O, T, rs=None):
    """MRE a T con E, S2, F y C(r) = Σ_n p_n ⟨n|O_r|n⟩."""
    m = mre(ev, T)
    p = np.maximum(m["c"] - np.asarray(ev), 0); p /= p.sum()
    m["C"] = [float(x) for x in O @ p]
    if rs is not None:
        m["rs"] = list(rs)
    return m
