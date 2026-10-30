"""ARViT + MPS ancilla (JAX/Flax-only, no NetKet)."""
from . import models, training
from .models import (
    CausalBlock,
    ARViT_MPSAncilla,
    mps_overlap_rect,
    mps_overlap_aligned,
    mps_overlap_matrix,
)
from .training import (
    # ED
    compute_rho,
    make_loss_fns,
    train,
    count_params,
    # MC
    make_sampler,
    make_mc_loss,
    train_mc,
)