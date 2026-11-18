#!/bin/bash
# Techo del gauge aprendido (N=6 T=3): U completa de la ancilla que minimiza el ruido del puente, y η real después.
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cpu
for L in free_psi herm_s0; do
  .venv/bin/python -u scripts/gauge_opt.py --psi data/gap_arch/phys_bridge/ex_N6_T3_${L}_psi.npy --label $L --steps 3000
  .venv/bin/python -u scripts/s2_physical_bridge.py --N 6 --T 3 --rep 20 --M 1000 --n_a 1 4 --psi_file data/gap_arch/phys_bridge/gauge_opt/${L}_psi.npy --out data/gap_arch/phys_bridge/gauge_opt/eta_$L.json
done
