"""
tail_split.py
=============
Separa EXACTAMENTE el deficit de S2 diagonal (forma de q(E)) en "cola por encima del corte" y "resto de la forma".

Sea p el MRE a la energia del modelo (T_E): p_n = (c - E_n)_+ / Z, Z = sum_n (c - E_n)_+.  Si sum q = sum p = 1 y
sum q E = sum p E, entonces sum_n q_n (c - E_n) = sum_n p_n (c - E_n) = Z sum p^2, y por tanto
    sum q p = sum p^2 + tail / Z,      tail = <(H - c)_+>_q = sum_{E_n > c} q_n (E_n - c) >= 0
    sum q^2 = sum p^2 + ||q - p||^2 + 2 tail / Z
La pureza diagonal de mas se reparte en  2 tail/Z  (peso del modelo por encima del corte, donde el MRE es 0) y
||q - p||^2 (cualquier otra diferencia de forma).  En S2:  deficit_d = log(sum q^2 / sum p^2), y se da cada parte como
fraccion del exceso de pureza.  q = diagonal de rho en la base propia de H (incluye la no uniformidad en capas, ~0).

Uso: python scripts/tail_split.py data/gap_arch/varh/split_*_qE.npz
"""
import sys
import numpy as np

print(f"{'label':<16}{'N':>3}{'T_E':>8}{'peso>c':>9}{'<(H-c)+>':>10}{'deficit_d':>11}{'excesoP':>10}"
      f"{'cola':>10}{'resto':>10}{'%cola':>7}{'  S2 si solo cola':>18}")
for path in sys.argv[1:]:
    d = np.load(path)
    ev, q, c, TE, N, lab = d["ev"], d["q"], float(d["c"]), float(d["T_E"]), int(d["N"]), str(d["label"])
    q = q / q.sum()
    w = np.maximum(c - ev, 0); Z = w.sum(); p = w / Z
    assert abs((q * ev).sum() - (p * ev).sum()) < 1e-6, "E del modelo != E del MRE(T_E)"
    tail = (q * np.maximum(ev - c, 0)).sum()
    Pq, Pp = (q ** 2).sum(), (p ** 2).sum()
    cola, resto = 2 * tail / Z, ((q - p) ** 2).sum()
    assert abs(Pq - Pp - cola - resto) < 1e-12 * max(1, Pq)
    print(f"{lab:<16}{N:>3}{TE:>8.4f}{q[ev > c].sum():>9.4f}{tail:>10.5f}{np.log(Pq / Pp):>11.5f}{Pq - Pp:>10.2e}"
          f"{cola:>10.2e}{resto:>10.2e}{cola / (Pq - Pp):>7.0%}{np.log((Pp + cola) / Pp):>18.5f}")
