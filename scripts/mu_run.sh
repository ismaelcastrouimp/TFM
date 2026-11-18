#!/bin/bash
# Un entrenamiento exacto N=6 T=3 encdec + fase MLP con F + μJ y la η real del puente al final.
# Uso: bash scripts/mu_run.sh <etiqueta> <flags extra de train_phys_bridge.py...>
cd "$(dirname "$0")/.."
L=$1; shift
O=data/gap_arch/phys_bridge
.venv/bin/python -u scripts/train_phys_bridge.py --N 6 --T 3 --mode exact --arch encdec --phase_net mlp --copy_init 0 \
  --lr 3e-3 --label $L "$@"
.venv/bin/python -u scripts/s2_physical_bridge.py --N 6 --T 3 --rep 20 --M 1000 --n_a 1 4 --psi_file $O/${L}_psi.npy \
  --out $O/eta_$L.json
