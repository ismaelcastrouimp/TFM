#!/bin/bash
# Entrenamiento exacto N=6 T=3 con F + μ J (J = ruido del puente): ¿se llega a dF ~7e-4 con un gauge de poco ruido?
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cpu
C="--N 6 --T 3 --mode exact --arch encdec --phase_net mlp --copy_init 0 --lr 3e-3 --steps 4000"
for mu in 0.01 0.1; do
  .venv/bin/python -u scripts/train_phys_bridge.py $C --mu $mu --label ex_N6_T3_mu$mu
  .venv/bin/python -u scripts/s2_physical_bridge.py --N 6 --T 3 --rep 20 --M 1000 --n_a 1 4 --psi_file data/gap_arch/phys_bridge/ex_N6_T3_mu${mu}_psi.npy --out data/gap_arch/phys_bridge/eta_mu$mu.json
done
