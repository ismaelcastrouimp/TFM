#!/bin/bash
# Continuación: desde el estado libre (dF/N 7.1e-4), encender μ J (μ = 0.1, 1) con lr bajo; η real al final.
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cpu
C="--N 6 --T 3 --mode exact --arch encdec --phase_net mlp --copy_init 0 --steps 4000 --lr 1e-3 --init data/gap_arch/phys_bridge/ex_N6_T3_encdec_phmlp_c0_params.msgpack"
for mu in 0.1 1; do
  .venv/bin/python -u scripts/train_phys_bridge.py $C --mu $mu --label ex_N6_T3_cont_mu$mu
  .venv/bin/python -u scripts/s2_physical_bridge.py --N 6 --T 3 --rep 20 --M 1000 --n_a 1 4 --psi_file data/gap_arch/phys_bridge/ex_N6_T3_cont_mu${mu}_psi.npy --out data/gap_arch/phys_bridge/eta_cont_mu$mu.json
done
