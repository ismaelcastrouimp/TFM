# 17-11: robustez movida al CESGA (cl_rob_*).  GPU local: 1) s2_noise con puente  2) s2_noise_large (150 pasos) con puente  3) semillas 2D h=3.5 T=3
cd "$(dirname "$0")/.."
D=data/gap_arch/physical_tilted
.venv/bin/python -u scripts/s2_noise.py > $D/s2_noise_bridge.log 2>&1;            echo "$(date) s2_noise (puente) exit $?"
.venv/bin/python -u scripts/s2_noise_large.py --train_steps 150 > $D/s2_noise_large_bridge.log 2>&1; echo "$(date) s2_noise_large (puente) exit $?"
.venv/bin/python -u 2D/ed2d.py --Lx 3 --Ly 3 --h_x -3.5 --T 3 --archs vit:conv2d:r=1,win=sq vit:conv:r=1 --chis 4 --seeds 2 3 \
  --steps 3000 --lr 5e-3 --out data/2D/ed_3x3_h3.5.json >> data/2D/ed_3x3_h3.5.log 2>&1; echo "$(date) semillas 2D exit $?"
echo "$(date) cola 36 terminada"
