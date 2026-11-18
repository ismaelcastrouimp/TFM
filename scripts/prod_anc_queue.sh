#!/bin/bash
# Ancilla producto neuronal (RBM generalizada, sin gauge): β = 1, 2, 4 cúbits/sitio, chi=1, MLP de toda la red.
# 3x3 (h=1.5, 3.5; T=3) frente a SBS/conv2d ya medidos, y 1D N=10 (T=1, 3) frente a una MPS local chi=4.
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cpu
P=".venv/bin/python -u 2D/ed2d.py"
A="vit:sbs:win=all,paths=w,w=64 vit:sbs:win=all,paths=ww,w=64 vit:sbs:win=all,paths=wwww,w=64"
for h in 1.5 3.5; do
  $P --Lx 3 --Ly 3 --h_x -$h --T 3 --archs $A --chis 1 --steps 3000 --lr 5e-3 --out data/2D/prod_3x3_h$h.json
done
$P --Lx 10 --Ly 1 --T 1 3 --archs $A --chis 1 --steps 3000 --lr 5e-3 --out data/2D/prod_1d_N10.json
$P --Lx 10 --Ly 1 --T 1 3 --archs "vit:sbs:r=1,win=plus,paths=w" --chis 4 --steps 3000 --lr 5e-3 --out data/2D/prod_1d_N10_ref.json
