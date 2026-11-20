"""
test_block_ed.py
================
Comprueba src_renyi/mps_ancilla/block_ed.py:
  1) N <= 12: espectro por bloques (Sz + k, con y sin atajos de simetría) = espectro de la H densa de NetKet
     (generic_eloc.build_hamiltonian), para j1j2 (J2 = 0.5) y xxz (Δ = 0.7).
  2) N = 14: bloques Sz + k frente a bloques solo Sz (independiente de la parte de momento).
  3) MRE N = 10 j1j2 J2 = 0.5 frente a los valores del log de garmonegro (T = 1, 3).
Uso: JAX_PLATFORMS=cpu python scripts/test_block_ed.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from src_renyi.mps_ancilla.block_ed import spectrum, mre

ok = True


def dense(name, N, **kw):
    import jax.numpy as jnp
    from src_renyi.mps_ancilla.generic_eloc import build_hamiltonian
    H = build_hamiltonian(name, N, **kw)
    D = 2 ** N; idx = np.arange(D); s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
    sp, mels = H.to_jax_operator().get_conn_padded(jnp.asarray(s))
    sp, mels = np.asarray(sp), np.asarray(mels).real
    col = (((sp + 1) / 2).astype(np.int64) << np.arange(N)).sum(-1)
    Hd = np.zeros((D, D)); np.add.at(Hd, (np.repeat(idx, sp.shape[1]), col.ravel()), mels.ravel())
    return np.linalg.eigvalsh(Hd)


cases = [("j1j2", dict(J2=0.5)), ("xxz", dict(Delta=0.7)), ("j1j2", dict(J2=0.3))]
for N in (6, 8, 9, 10, 12):
    for name, kw in cases:
        ed = dense(name, N, **kw)
        for uk, us in ((True, True), (True, False), (False, True)):
            ev = spectrum(name, N, use_k=uk, use_sym=us, **kw)
            err = np.abs(ev - ed).max()
            ok &= err < 1e-9
            print(f"1) N={N:2d} {name} {kw}  k={uk!s:5} sym={us!s:5}  max|Δλ| = {err:.1e}", flush=True)

t0 = time.time(); a = spectrum("j1j2", 14, J2=0.5); t1 = time.time()
b = spectrum("j1j2", 14, J2=0.5, use_k=False); t2 = time.time()
err = np.abs(a - b).max(); ok &= err < 1e-9
print(f"2) N=14 j1j2: Sz+k ({t1 - t0:.1f} s) frente a solo Sz ({t2 - t1:.1f} s): max|Δλ| = {err:.1e}", flush=True)

ev = spectrum("j1j2", 10, J2=0.5)
ref = {1.0: (-1.30969, 3.0822, -16.17916), 3.0: (-0.87230, 5.4254, -24.99925)}
for T, (eN, S2, F) in ref.items():
    r = mre(ev, T)
    d = max(abs(r["E"] / 10 - eN), abs(r["S2"] - S2), abs(r["F"] - F))
    ok &= d < 1e-4
    print(f"3) MRE N=10 T={T}: E/N={r['E'] / 10:.5f} S2={r['S2']:.4f} F={r['F']:.5f}   (garmonegro {eN} {S2} {F})", flush=True)
print(f"   E0/N = {ev[0] / 10:.6f}")
print("TODO OK" if ok else "FALLO", flush=True)
