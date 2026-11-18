#!/bin/bash
# Prueba Ψ hermítica por construcción (N=6 T=3, exacto) frente a la red libre (λ=0) y η del puente en cada estado.
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cpu
P=".venv/bin/python -u"; C="--N 6 --T 3 --mode exact --arch encdec --phase_net mlp --copy_init 0 --lr 3e-3"
# Ψ del estado libre ya entrenado (λ=0, H≈0): recargar y guardar Ψ (2 pasos con lr ~0)
$P scripts/train_phys_bridge.py $C --steps 2 --warmup 1 --lr 1e-14 --init data/gap_arch/phys_bridge/ex_N6_T3_encdec_phmlp_c0_params.msgpack --label ex_N6_T3_free_psi
for s in 0 1; do
  $P scripts/train_phys_bridge.py $C --steps 4000 --herm --seed $s --label ex_N6_T3_herm_s$s
done
for L in free_psi herm_s0 herm_s1; do
  $P scripts/s2_physical_bridge.py --N 6 --T 3 --rep 20 --M 1000 --n_a 1 4 --psi_file data/gap_arch/phys_bridge/ex_N6_T3_${L}_psi.npy --out data/gap_arch/phys_bridge/eta_$L.json
done
