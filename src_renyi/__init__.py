"""src_renyi — variational NQS for finite-temperature Rényi states.

Sub-packages
------------
physical_ancilla
    NetKet-based pipeline: ARNN/ViT on the extended Hilbert space with a
    physical ancilla. Requires NetKet + Flax + Optax.

    Example:
        from src_renyi.physical_ancilla.training import free_energy_minimize_phases
        from src_renyi.physical_ancilla.models import ARSpinViT_Causal_Interleaved

mps_ancilla
    JAX/Flax-only pipeline: ARViT for the system + conditional MPS ancilla
    traced analytically. Does NOT depend on NetKet.

    Example:
        from src_renyi.mps_ancilla.models import ARViT_MPSAncilla
        from src_renyi.mps_ancilla.training import train_mc, make_loss_fns, train

Import policy
-------------
This __init__ does *not* import either sub-package. They are loaded only
when first accessed, so:

  * `import src_renyi` does NOT load NetKet (safe for MPS-only scripts).
  * `import src_renyi` does NOT activate jax_enable_x64 (safe for
    NetKet-only scripts, which prefer float32).
"""