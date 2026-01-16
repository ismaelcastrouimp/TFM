# Thermal states from entanglement entropy

This repository studies **thermal quantum states** obtained variationally by minimizing the free energy

$$
F = \langle H \rangle - T S_2,
$$

where \(S_2\) is the **Rényi-2 entanglement entropy** between a physical system and an ancilla.

The goal is to reproduce thermal states using entanglement-based variational principles, and to compare exact and variational approaches.

---

## Repository structure
```
2spin.nb
Ancilla_exact.ipynb
Ancilla_ising.ipynb
params_fullsum.msgpack
README.md
```

---

## Files overview

### 🔹 `2spin.nb` (Mathematica)

- **System:** 2 qubits  
- **Ancilla:** none  
- **Entropy:** von Neumann  

Used as a baseline for comparison.

---

### 🔹 `Ancilla_exact.ipynb`

- **System:** 2 qubits  
- **Ancilla:** 2 qubits  
- **Entropy:** Rényi-2 (system–ancilla)  

- Uses `FullSumState` from **NetKet** (no Monte Carlo).
- \(S_2\) computed exactly via reduced density matrix and partial trace.
- Free energy minimized using `scipy.optimize.minimize`.

Acts as an exact benchmark with ancilla.

---

### 🔹 `Ancilla_ising.ipynb`

- **System:** 2 qubits  
- **Ancilla:** 2 qubits  
- **Entropy:** Rényi-2 (system–ancilla)  

Variational Monte Carlo approach:

- \(S_2\) computed using:
  - The swap trick
  - NetKet's built-in Rényi entropy observable
- Explicit computation of the gradient of \(S_2\)
- Free energy minimization with:
  - Adam
  - gradient descent
  - SGD

This notebook is the most general and closest to applications to larger systems.

---


## Notes

- `params_fullsum.msgpack` stores parameters from `Ancilla_exact.ipynb`.

