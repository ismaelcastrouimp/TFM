# Thermal states from entanglement entropy

This repository studies **thermal quantum states** obtained variationally by minimizing the free energy

$$
F = \langle H \rangle - T S_2,
$$

where $S_2$ is the **Rényi-2 entanglement entropy** between a physical system and an ancilla.

The goal is to reproduce thermal states using **entanglement-based variational principles**, analyze their statistical properties, and compare variational Monte Carlo results with exact and canonical approaches.

---

## Repository structure

```
.
├── scripts/
│   ├── diagnose.py
│   ├── training_single_T.py
│   ├── training_multiple_T.py
│   └── s2_noise.py
│
├── src_renyi/
│   ├── __init__.py
│   ├── entropy.py
│   ├── models.py
│   ├── observables.py
│   └── training.py
│
├── notebooks/
│   ├── Renyi_Ising_scipy.ipynb
│   ├── Renyi_Ising.ipynb
│   └── Renyi_vs_canonical.ipynb
│
├── data/
├── plots/
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Conceptual overview

We variationally construct thermal states by minimizing:

$$
F = E - T S_2
$$

where:

- $E = \langle H \rangle$ is the energy
- $S_2$ is the Rényi-2 entanglement entropy between system and ancilla
- $T$ is the temperature

The entropy can be computed using:

- swap trick (Monte Carlo)
- $\lambda$-integral method
- exact reduced density matrix (small systems)

The optimization is performed using neural-network quantum states implemented in NetKet, with explicit gradients of the free energy.

---

## Installation

The project is structured as an installable Python package using `pyproject.toml`.

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

The editable install (`-e`) allows importing the package:

```python
import src_renyi
```

## Typical workflow

The recommended workflow for running simulations is:

### Step 1 — Diagnose simulation parameters

Before training a system of size N, run:

```bash
python scripts/diagnose.py
```

This script estimates:

- optimal `chunk_size`
- expected training step time
- recommended `clip_norm`

This helps avoid inefficient configurations and memory issues.

### Step 2 — Train at a single temperature

```bash
python scripts/training_single_T.py
```

Used to:

- debug training
- study convergence
- analyze specific temperatures

### Step 3 — Train across multiple temperatures

```bash
python scripts/training_multiple_T.py
```

This script:

- trains states for an array of temperatures
- stores optimal parameters
- generates thermodynamic curves


## Core package: `src_renyi`

This directory contains the physics and optimization logic.

### `entropy.py`

Implements computation of the Rényi-2 entanglement entropy and its gradient.

Supported methods:
- Swap trick
- $\lambda$-integral method
- exact computation (small Hilbert spaces)



### `models.py`

Neural-network quantum state models with symmetry support.

Currently includes:

- RBM with $Z_2$ symmetry
- Autoregressive neural networks (ARNN) with $Z_2$ symmetry


### `observables.py`

Defines the main observable used in optimization:

**FreeRenyiEnergyObservable**

This class inherits from:

- `netket.experimental.observable.AbstractObservable`

It computes:

$$
F = E - T S_2
$$

and its gradient with respect to model parameters.


### `training.py`

Contains the training logic to minimize free energy, but also to maximize entanglement entropy.

This module is used internally by:

- `training_single_T.py`
- `training_multiple_T.py`

## Scripts

These are entry points for running experiments.

### `diagnose.py`

Utility script executed before training.

It estimates:

- optimal `chunk_size`
- expected step time
- recommended gradient clipping norm

This helps prevent:

- memory overflows
- unstable gradients
- inefficient sampling

### `training_single_T.py`

Trains a neural quantum state at a single temperature.

Used for:

- debugging
- convergence analysis
- controlled experiments

### `training_multiple_T.py`

Runs training across a temperature grid.

It:

- trains states sequentially
- stores optimal parameters
- generates thermodynamic datasets

This is the main production script.

### `s2_noise.py`

Analyzes statistical fluctuations of the entropy estimator.

Studies scaling with temperature (i.e. entropy value) and the number of samples

Useful for:

- error analysis
- method validation
- choosing sampling parameters

## Notebooks

These notebooks are primarily for validation and analysis.

### `Renyi_Ising_scipy.ipynb`

Small benchmark system.

**System:** 2 qubits (system) + 2 qubits (ancilla)

**Method:** optimization using `scipy.optimize`

**Purpose:**
- verify correctness of the implementation
- provide a baseline

### `Renyi_Ising.ipynb`

Variational Monte Carlo implementation.

It:
- minimizes free energy at different temperatures
- compares results with SciPy optimization

### `Renyi_vs_canonical.ipynb`

Post-training analysis.

It compares:
- variational thermal states
- canonical ensemble predictions

Typical observables:
- energy
- magnetization
- correlations

## Data directory

### `data/`

This directory stores training outputs.

Structure:
```
data/
    N{N}/
        params_T{T}.msgpack
        observables.json
```

The JSON file contains:
- `F(T)`
- `E(T)`
- `S2(T)`

## Plots directory

### `plots/`

Used to store generated figures.

## Dependencies

Main libraries:

- NetKet
- JAX
- Optax
- NumPy
- SciPy
- Matplotlib

Full dependency list: `requirements.txt`