"""
sn_noise.py
===========
Comparación de estimadores de S_n y su gradiente para varios órdenes n
a presupuesto igualado.

Métodos de extrapolación a S_vN incluidos:
  - Polinomios en n:      linear_n, quadratic_n
  - Polinomios en 1/n:    linear_inv_n, quadratic_inv_n
  - Padé racional:        pade_2_2 (principal), pade_auto (diagnóstico)
  - Chebyshev-BPA:        cheb_ncK (Wang & Davis 2020, Apéndice C)
  - Chebyshev drop_n0:    parche heurístico para rank alto

Secciones especiales:
  - 0  : S_vN y su gradiente vía Wang contra S_vN exacta (SVD).
         Compara Padé, cuadrática y Chebyshev.
  - 0.5: convergencia de S_vN con el número de órdenes k usados.
  - 1  : comparación swap vs Wang a presupuesto fijo.
  - Tras sección 1: figura de calidad del gradiente de S_n vs T.

Referencia exacta de S_vN: von_neumann_entropy_exact (via SVD de Ψ).

Gráficas en ../plots/Sn_noise/ en formato .pgf y .pdf.
"""

import os
import copy
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmax, sigmaz
import optax
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
})
import matplotlib.pyplot as plt

from src_renyi.physical_ancilla.entropy import (
    renyin_entropy_exact,
    renyin_entropy_and_grad_exact,
    renyin_entropy_and_grad_sampled,
    renyin_wang_cs,
    train_reverse_network,
)
from src_renyi.physical_ancilla.training import free_energy_minimize

# ── directorios ────────────────────────────────────────────────────────────────
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots/Sn_noise")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path + ".pgf", bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    print(f"  guardado: {path}.pdf")
    plt.close(fig)


def cosine_similarity(g1, g2):
    flat1, _ = jax.flatten_util.ravel_pytree(g1)
    flat2, _ = jax.flatten_util.ravel_pytree(g2)
    flat1 = jnp.array(flat1, float)
    flat2 = jnp.array(flat2, float)
    return float(
        jnp.dot(flat1, flat2) /
        (jnp.linalg.norm(flat1) * jnp.linalg.norm(flat2) + 1e-30)
    )


# ══════════════════════════════════════════════════════════════════════════════
# S_vN exacta via SVD
# ══════════════════════════════════════════════════════════════════════════════

def _build_index_maps(vstate, subsystem):
    """Precomputa los índices (a_idx, b_idx) para reordenar la base."""
    hi = vstate.hilbert
    N_total = hi.size
    complement = [i for i in range(N_total) if i not in subsystem]

    all_states = hi.all_states()
    bits = ((all_states + 1) // 2).astype(jnp.int32)

    a_idx = jnp.zeros(bits.shape[0], dtype=jnp.int32)
    for q in subsystem:
        a_idx = (a_idx << 1) | bits[:, q]

    b_idx = jnp.zeros(bits.shape[0], dtype=jnp.int32)
    for q in complement:
        b_idx = (b_idx << 1) | bits[:, q]

    return all_states, a_idx, b_idx


def von_neumann_entropy_exact(vstate, subsystem, return_rho_A=False,
                              rank_tol=1e-12):
    """
    S_vN = -Tr(ρ_A ln ρ_A) exacta via SVD de Ψ[a, b].
    """
    hi = vstate.hilbert
    N_total = hi.size
    nA = len(subsystem)
    nB = N_total - nA

    all_states, a_idx, b_idx = _build_index_maps(vstate, subsystem)

    log_psi = vstate.log_value(all_states)
    psi = jnp.exp(jnp.asarray(log_psi).reshape(-1))
    norm2 = jnp.sum(jnp.abs(psi) ** 2)
    psi = psi / jnp.sqrt(norm2)

    Psi = jnp.zeros((2**nA, 2**nB), dtype=psi.dtype)
    Psi = Psi.at[a_idx, b_idx].set(psi)

    S = jnp.linalg.svd(Psi, compute_uv=False)
    p = np.asarray((S ** 2) / jnp.sum(S ** 2))

    S_vN = -float(np.sum(np.where(p > 1e-30, p * np.log(p), 0.0)))
    rank_A = int(np.sum(p > rank_tol))

    if return_rho_A:
        return S_vN, rank_A, p
    return S_vN


def svn_and_grad_exact(vstate, subsystem):
    """
    S_vN exacta y su gradiente exacto por autodiff a través del SVD de Ψ.
    """
    hi = vstate.hilbert
    N_total = hi.size
    nA = len(subsystem)
    nB = N_total - nA

    all_states, a_idx, b_idx = _build_index_maps(vstate, subsystem)

    def S_vN_func(params):
        log_psi = vstate._apply_fun(
            {"params": params, **vstate.model_state}, all_states
        )
        psi = jnp.exp(log_psi.reshape(-1))
        norm2 = jnp.sum(jnp.abs(psi) ** 2)
        psi = psi / jnp.sqrt(norm2)

        Psi = jnp.zeros((2**nA, 2**nB), dtype=psi.dtype)
        Psi = Psi.at[a_idx, b_idx].set(psi)

        S = jnp.linalg.svd(Psi, compute_uv=False)
        p = S ** 2
        p = p / jnp.sum(p)
        return -jnp.sum(jnp.where(p > 1e-30, p * jnp.log(p), 0.0))

    S_vN, grad = jax.value_and_grad(S_vN_func)(vstate.parameters)
    return float(S_vN), grad


# ══════════════════════════════════════════════════════════════════════════════
# Padé
# ══════════════════════════════════════════════════════════════════════════════

def _pade_solve(t, S, L, M):
    t = np.asarray(t, dtype=float)
    S = np.asarray(S, dtype=float)
    k = L + M + 1
    if len(t) < k:
        raise ValueError(f"Padé [{L}/{M}] requiere ≥ {k} puntos, hay {len(t)}")

    tau = float(np.max(np.abs(t))) or 1.0
    u = t / tau

    A = np.zeros((len(u), k))
    for i_ in range(L + 1):
        A[:, i_] = u ** i_
    for j_ in range(1, M + 1):
        A[:, L + j_] = -S * (u ** j_)

    sol, *_ = np.linalg.lstsq(A, S, rcond=None)
    a = sol[: L + 1]
    b = sol[L + 1 :]
    return float(a[0]), a, b


def _pade_eval(a, b, t_eval):
    t_eval = np.asarray(t_eval, dtype=float)
    num = sum(a[i_] * t_eval ** i_ for i_ in range(len(a)))
    den = 1.0 + sum(b[j_] * t_eval ** j_ for j_ in range(len(b)))
    return num / den


def pade_extrapolate(n_orders, S_n_values, L=2, M=2):
    n_arr = np.asarray(n_orders, dtype=float)
    S_arr = np.asarray(S_n_values, dtype=float)
    S_vN, _, _ = _pade_solve(n_arr - 1.0, S_arr, L, M)
    return float(S_vN)


def pade_scan(n_orders, S_n_values, verbose=True):
    n_arr = np.asarray(n_orders, dtype=float)
    S_arr = np.asarray(S_n_values, dtype=float)
    t = n_arr - 1.0
    n_pts = len(t)

    if n_pts < 4:
        if verbose:
            print(f"  [pade_scan] solo {n_pts} puntos → Padé [1/1] forzado")
        return pade_extrapolate(n_orders, S_n_values, L=1, M=1), []

    orders = []
    for L in range(0, n_pts):
        for M in range(0, n_pts):
            if 2 <= L + M + 1 <= n_pts:
                orders.append((L, M))

    table = []
    for L, M in orders:
        try:
            _, a, b = _pade_solve(t[1:], S_arr[1:], L, M)
            pred_n2 = float(_pade_eval(a, b, t[0]))
            holdout_err = abs(pred_n2 - S_arr[0])
        except Exception:
            pred_n2, holdout_err = np.nan, np.nan

        try:
            S_vN, a, b = _pade_solve(t, S_arr, L, M)
        except Exception:
            S_vN, a, b = np.nan, None, None

        pole_penalty = 0.0
        if M > 0 and b is not None and np.all(np.isfinite(b)):
            q_coeffs = np.concatenate([[1.0], b[::-1]])
            try:
                poles = np.roots(q_coeffs)
                if len(poles) > 0:
                    d_pole = float(np.min(np.abs(poles)))
                    if d_pole < 1.0:
                        pole_penalty = 1e6 * (1.0 - d_pole)
            except Exception:
                pass

        score = (holdout_err if np.isfinite(holdout_err) else 1e9) + pole_penalty
        table.append({
            "L": L, "M": M, "S_vN": S_vN,
            "holdout_pred_S2": pred_n2, "holdout_err": holdout_err,
            "pole_penalty": pole_penalty, "score": score,
        })

    if verbose:
        print(f"  [pade_scan] {len(table)} órdenes probadas:")
        for row in sorted(table, key=lambda r: r["score"])[:8]:
            flag = "  ← pole" if row["pole_penalty"] > 0 else ""
            print(f"    [{row['L']}/{row['M']}]  "
                  f"S_vN={row['S_vN']:.5f}  "
                  f"holdout_err={row['holdout_err']:.2e}  "
                  f"score={row['score']:.2e}{flag}")

    valid = [r for r in table if np.isfinite(r["S_vN"]) and r["score"] < 1e8]
    if not valid:
        if verbose:
            print("  ⚠ ningún Padé válido → fallback cuadrática en n")
        c = np.polyfit(n_arr, S_arr, 2)
        return float(np.polyval(c, 1.0)), table

    valid.sort(key=lambda r: r["score"])
    best = valid[0]
    if verbose:
        print(f"  → mejor: [{best['L']}/{best['M']}]  S_vN = {best['S_vN']:.5f}")
    return float(best["S_vN"]), table


# ══════════════════════════════════════════════════════════════════════════════
# Chebyshev-BPA para S_vN (Wang & Davis 2020, Apéndice C)
# ══════════════════════════════════════════════════════════════════════════════
#
# Aproxima f(x) = -x ln x por un polinomio de Chebyshev truncado a orden n_c:
#     f(x) ≈ Σ_{k=0}^{n_c} a_k T̂_k(x),  T̂_k(x) = T_k(2x-1)
# y reescribe S_vN = Σ_i f(λ_i) ≈ Σ_{n=0}^{n_c} α_n Tr(ρ_A^n),
# con α_n los coeficientes en la base monomial x^n.
#
# AVISO: los α_n crecen como 2^{2n_c-1}/(n_c(n_c²-1)), y α_0 ≠ 0. En el
# régimen de rank alto (χ ~ 1000), α_0 · χ domina sobre S_vN y el método
# falla. Se incluye por completitud metodológica.

def chebyshev_coeffs_monomial(n_c):
    """
    Coeficientes α_n tales que Σ_k a_k T̂_k(x) = Σ_n α_n x^n.

    Devuelve un array (n_c + 1,) con α_0, α_1, ..., α_{n_c}.
    """
    a = np.zeros(n_c + 1)
    a[0] = np.log(2) - 0.5
    if n_c >= 1:
        a[1] = np.log(2) - 0.75
    for k in range(2, n_c + 1):
        a[k] = (-1) ** (k + 1) / (k * (k ** 2 - 1))

    # Cambio de variable y → 2x - 1 (polinomio en x: [-1, 2])
    lin = np.array([-1.0, 2.0])

    alpha = np.zeros(n_c + 1)
    for k in range(n_c + 1):
        Tk_y = np.zeros(k + 1); Tk_y[k] = 1.0
        Tk_mono_y = np.polynomial.chebyshev.cheb2poly(Tk_y)
        Tk_mono_x = np.zeros(1)
        for c in Tk_mono_y[::-1]:
            Tk_mono_x = np.polynomial.polynomial.polyadd(
                np.polynomial.polynomial.polymul(Tk_mono_x, lin),
                [c],
            )
        alpha[: len(Tk_mono_x)] += a[k] * Tk_mono_x
    return alpha


def chebyshev_extrapolate(n_orders, S_n_values, n_c=None,
                          rank_A=None, N_A=None, drop_n0=False):
    """
    S_vN ≈ Σ_{n=0}^{n_c} α_n Tr(ρ_A^n), con Tr(ρ_A^n) = exp((1-n) S_n).
    """
    n_orders = np.asarray(list(n_orders), dtype=int)
    S_n_values = np.asarray(list(S_n_values), dtype=float)

    if n_c is None:
        n_c = int(np.max(n_orders))

    alpha = chebyshev_coeffs_monomial(n_c)

    Tr = {}
    for n, s in zip(n_orders, S_n_values):
        Tr[int(n)] = float(np.exp((1.0 - int(n)) * s))

    Tr[1] = 1.0

    if drop_n0:
        Tr[0] = 0.0
    elif rank_A is not None:
        Tr[0] = float(rank_A)
    elif N_A is not None:
        Tr[0] = float(2 ** N_A)
    else:
        Tr[0] = 1.0

    if drop_n0:
        a_pos = alpha[1: n_c + 1]
        sgn = np.sign(np.sum(a_pos))
        scale = sgn if abs(np.sum(a_pos)) > 1e-30 else 1.0
        S_vN = 0.0
        for n in range(1, n_c + 1):
            if n in Tr:
                S_vN += alpha[n] * Tr[n] / scale
        return float(S_vN)

    S_vN = 0.0
    for n in range(n_c + 1):
        if n in Tr:
            S_vN += alpha[n] * Tr[n]
    return float(S_vN)


def test_chebyshev_diagonal(n_c=6, seed=0, verbose=True):
    """Test de sanidad sobre espectro diagonal de rank bajo."""
    rng = np.random.default_rng(seed)
    lam = np.exp(-rng.uniform(0, 3, size=10))
    lam = lam / lam.sum()

    S_vN_true = -np.sum(lam * np.log(lam))

    n_orders = list(range(2, n_c + 1))
    S_n_vals = []
    for n in n_orders:
        Tr = float(np.sum(lam ** n))
        S_n_vals.append(np.log(Tr) / (1 - n))

    S_vN_cheb = chebyshev_extrapolate(n_orders, S_n_vals, n_c=n_c,
                                      rank_A=len(lam))

    if verbose:
        print(f"  [cheb test] S_vN exacta   = {S_vN_true:.6f}")
        print(f"  [cheb test] S_vN cheb nc={n_c} = {S_vN_cheb:.6f}")
        print(f"  [cheb test] err           = "
              f"{abs(S_vN_cheb - S_vN_true):.2e}")
    return S_vN_true, S_vN_cheb


# ══════════════════════════════════════════════════════════════════════════════
# Extrapolación unificada
# ══════════════════════════════════════════════════════════════════════════════

def extrapolate_renyi_to_vN(n_orders, S_n_values, method="pade_2_2"):
    """
    Estima S_vN = S_1 a partir de S_n (n > 1).

    Métodos:
      - 'linear_n', 'quadratic_n', 'linear_inv_n', 'quadratic_inv_n'
      - 'pade_L_M'   : Padé explícito (ej. 'pade_2_2')
      - 'pade_auto'  : scan Padé con hold-out
      - 'cheb_ncK'   : Chebyshev-BPA con n_c = K (rank por defecto = 2)
    """
    n_arr = np.asarray(n_orders, dtype=float)
    S_arr = np.asarray(S_n_values, dtype=float)

    if method == "linear_n":
        c = np.polyfit(n_arr, S_arr, 1); return float(np.polyval(c, 1.0))
    if method == "quadratic_n":
        c = np.polyfit(n_arr, S_arr, 2); return float(np.polyval(c, 1.0))
    if method == "linear_inv_n":
        c = np.polyfit(1.0 / n_arr, S_arr, 1); return float(np.polyval(c, 1.0))
    if method == "quadratic_inv_n":
        c = np.polyfit(1.0 / n_arr, S_arr, 2); return float(np.polyval(c, 1.0))

    if method.startswith("pade_") and method != "pade_auto":
        parts = method.split("_")[1:]
        if len(parts) == 2:
            L, M = int(parts[0]), int(parts[1])
            return pade_extrapolate(n_orders, S_n_values, L=L, M=M)

    if method == "pade_auto":
        S_vN, _ = pade_scan(n_orders, S_n_values, verbose=False)
        return S_vN

    if method.startswith("cheb_nc"):
        n_c = int(method.replace("cheb_nc", ""))
        # Fallback: N_A=1 → rank=2. Para uso real, llamar a chebyshev_extrapolate.
        return chebyshev_extrapolate(n_orders, S_n_values, n_c=n_c, N_A=1)

    raise ValueError(f"método desconocido: {method}")


def svn_from_k_orders(S_n_values, k, method=None):
    """
    Estima S_vN usando los primeros k órdenes.

    Esquema progresivo:
      k=1: S_2 como proxy
      k=2: lineal en n
      k=3: cuadrática en n
      k=4: Padé [2/1]
      k>=5: Padé [2/2]
    """
    n_use = N_ORDERS[:k]
    S_use = np.asarray(S_n_values[:k], dtype=float)

    if method is not None:
        if k == 1:
            return float(S_use[0])
        return extrapolate_renyi_to_vN(n_use, S_use, method)

    if k == 1:
        return float(S_use[0])
    if k == 2:
        return extrapolate_renyi_to_vN(n_use, S_use, "linear_n")
    if k == 3:
        return extrapolate_renyi_to_vN(n_use, S_use, "quadratic_n")
    if k == 4:
        return extrapolate_renyi_to_vN(n_use, S_use, "pade_2_1")
    if k >= 5:
        return extrapolate_renyi_to_vN(n_use, S_use, "pade_2_2")
    raise ValueError(f"k={k} no soportado")


# ══════════════════════════════════════════════════════════════════════════════
# Gradiente de S_vN por chain rule sobre S_n estimados
# ══════════════════════════════════════════════════════════════════════════════

def svn_grad_from_Sn_estimators(S_n_values, grad_Sn_list,
                                method="pade_2_2", eps=1e-4):
    n_orders = list(range(2, 2 + len(S_n_values)))
    S_arr = np.asarray(S_n_values, dtype=float)

    coefs = []
    for i in range(len(S_arr)):
        S_plus  = S_arr.copy(); S_plus[i]  += eps
        S_minus = S_arr.copy(); S_minus[i] -= eps
        SvN_plus  = extrapolate_renyi_to_vN(n_orders, S_plus,  method)
        SvN_minus = extrapolate_renyi_to_vN(n_orders, S_minus, method)
        coefs.append((SvN_plus - SvN_minus) / (2.0 * eps))

    grad_total = None
    for c, g in zip(coefs, grad_Sn_list):
        if grad_total is None:
            grad_total = jax.tree_util.tree_map(lambda x: c * x, g)
        else:
            grad_total = jax.tree_util.tree_map(
                lambda a, b: a + c * b, grad_total, g
            )
    return grad_total, coefs


def test_pade_diagonal(L=2, M=2, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    lam = np.exp(-rng.uniform(0, 3, size=10))
    lam = lam / lam.sum()

    S_vN_true = -np.sum(lam * np.log(lam))

    n_orders = [2, 3, 4, 5, 6]
    S_n_vals = []
    for n in n_orders:
        Tr = float(np.sum(lam ** n))
        S_n = np.log(Tr) / (1 - n)
        S_n_vals.append(S_n)

    S_vN_pade = pade_extrapolate(n_orders, S_n_vals, L=L, M=M)

    if verbose:
        print(f"  [pade test] S_vN exacta = {S_vN_true:.6f}")
        print(f"  [pade test] S_vN [{L}/{M}] = {S_vN_pade:.6f}")
        print(f"  [pade test] err         = "
              f"{abs(S_vN_pade - S_vN_true):.2e}")
    return S_vN_true, S_vN_pade


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
N              = 10
GAMMA          = -1.5
V              = -1.0
TEMPS          = [2.0, 2.5, 3.0, 3.5, 4.0]
N_ORDERS       = [2, 3, 4, 5, 6]
n_rep          = 10

BUDGET_FIXED   = 2**20
BUDGET_LIST    = [2**14, 2**16, 2**18, 2**20, 2**22]

N_WANG_SVN     = 2**16
SVN_METHOD     = "pade_2_2"
SVN_METHOD_ALT = "quadratic_n"
CHEB_NC        = max(N_ORDERS)   # n_c para Chebyshev


def cost_swap(n):
    return 3 * n

def cost_wang(n, N_total):
    return n * N_total + 2 * n + 3


# ══════════════════════════════════════════════════════════════════════════════
# Tests de sanidad
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test Padé sobre ρ_A diagonal")
print("=" * 60)
test_pade_diagonal(L=2, M=2, seed=0)

print("=" * 60)
print("Test Chebyshev sobre ρ_A diagonal")
print("=" * 60)
test_chebyshev_diagonal(n_c=CHEB_NC, seed=0)


# ══════════════════════════════════════════════════════════════════════════════
# Hilbert / Hamiltoniano / Modelos
# ══════════════════════════════════════════════════════════════════════════════
subsystem = list(range(N))
partition = subsystem

hi = nk.hilbert.Spin(s=1/2, N=2 * N)
N_TOTAL = 2 * N

print(f"\nPresupuesto fijo: {BUDGET_FIXED} evaluaciones")
for n in N_ORDERS:
    print(f"  n={n}:  swap_n_samples = {BUDGET_FIXED // cost_swap(n):>8}   "
          f"wang_n_samples = {BUDGET_FIXED // cost_wang(n, N_TOTAL):>8}")

H_extended = 0
for i in range(N):
    H_extended += GAMMA * sigmax(hi, i)
    H_extended += V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N)

model = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                            activation=jax.nn.gelu)
sampler = nk.sampler.ARDirectSampler(hi)
vstate_ref = nk.vqs.MCState(sampler, model, n_samples=2**16)


# ══════════════════════════════════════════════════════════════════════════════
# Entrenamiento vstates + vstate_R
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Entrenando vstates y vstate_R")
print("=" * 60)

trained_vstates   = {}
trained_vstates_R = {}

for T in TEMPS:
    vstate = copy.deepcopy(vstate_ref)
    lr = optax.linear_schedule(0.05, 0.001, 300)
    print(f"\n  T={T:.2f}")
    free_energy_minimize(
        vstate, T, partition, H_extended, n_steps=300,
        optimizer=optax.sgd(lr),
        plot=False, verbose=False,
        chunk_size=vstate.n_samples // 2,
    )
    trained_vstates[T] = vstate

    S2_ex, _ = renyin_entropy_exact(vstate, subsystem, 2)
    print(f"         S₂ exacto = {float(S2_ex):.4f}")

    model_R = nk.models.ARNNDense(hilbert=hi, layers=1, features=16,
                                  activation=jax.nn.gelu)
    sampler_R = nk.sampler.ARDirectSampler(hi)
    vstate_R = nk.vqs.MCState(sampler_R, model_R, n_samples=2**16)

    train_reverse_network(
        vstate, vstate_R,
        n_steps=4000, batch=4096, lr=1e-3,
        verbose=True, freq=500,
    )
    trained_vstates_R[T] = vstate_R


# ══════════════════════════════════════════════════════════════════════════════
# Referencias exactas
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Referencias exactas (ED)")
print("=" * 60)

exact_Sn  = {n: {} for n in N_ORDERS}
exact_gSn = {n: {} for n in N_ORDERS}

for T, vstate in trained_vstates.items():
    for n in N_ORDERS:
        S_n, g = renyin_entropy_and_grad_exact(
            vstate, subsystem, n, hi_extended=hi, isFullSum=False
        )
        exact_Sn[n][T]  = float(S_n)
        exact_gSn[n][T] = g
        print(f"  T={T:.2f}  n={n}  S_n = {float(S_n):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# S_vN exacta + extrapolaciones con S_n exactos
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("S_vN exacta vs Rényi + extrapolación")
print("=" * 60)

exact_SvN           = {}
exact_rank_A        = {}
extrap_linear_n     = {}
extrap_quad_n       = {}
extrap_linear_inv_n = {}
extrap_quad_inv_n   = {}
extrap_pade_22      = {}
extrap_pade_auto    = {}
extrap_cheb         = {}

for T, vstate in trained_vstates.items():
    SvN, rank_A, _eigs = von_neumann_entropy_exact(
        vstate, subsystem, return_rho_A=True
    )
    exact_SvN[T]    = SvN
    exact_rank_A[T] = rank_A

    S_arr = np.array([exact_Sn[n][T] for n in N_ORDERS])

    extrap_linear_n[T]     = extrapolate_renyi_to_vN(N_ORDERS, S_arr, "linear_n")
    extrap_quad_n[T]       = extrapolate_renyi_to_vN(N_ORDERS, S_arr, "quadratic_n")
    extrap_linear_inv_n[T] = extrapolate_renyi_to_vN(N_ORDERS, S_arr, "linear_inv_n")
    extrap_quad_inv_n[T]   = extrapolate_renyi_to_vN(N_ORDERS, S_arr, "quadratic_inv_n")
    extrap_pade_22[T]      = extrapolate_renyi_to_vN(N_ORDERS, S_arr, "pade_2_2")
    extrap_pade_auto[T], _ = pade_scan(N_ORDERS, S_arr, verbose=False)
    extrap_cheb[T]         = chebyshev_extrapolate(
        N_ORDERS, S_arr, n_c=CHEB_NC, rank_A=rank_A
    )

    def rel_err(x):
        return abs(x - SvN) / (abs(SvN) + 1e-30)

    print(f"\nT={T:.2f}  S_vN={SvN:.5f}  rank(ρ_A)={rank_A}")
    for name, val in [
        ("lin(n)",      extrap_linear_n[T]),
        ("quad(n)",     extrap_quad_n[T]),
        ("lin(1/n)",    extrap_linear_inv_n[T]),
        ("quad(1/n)",   extrap_quad_inv_n[T]),
        ("pade[2/2]",   extrap_pade_22[T]),
        ("pade_auto",   extrap_pade_auto[T]),
        (f"cheb_nc{CHEB_NC}", extrap_cheb[T]),
    ]:
        print(f"   {name:>14s}: {val:.5f}   err={rel_err(val):.4%}")


# ══════════════════════════════════════════════════════════════════════════════
# Figura: S_vN y S_n vs T
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 4.2))
Ts = np.array(TEMPS)
cmap = plt.cm.viridis

for k, n in enumerate(N_ORDERS):
    Ss = np.array([exact_Sn[n][T] for T in TEMPS])
    ax.plot(Ts, Ss, "o-", color=cmap(k / max(1, len(N_ORDERS) - 1)),
            label=fr"$S_{n}$ (exacta)", alpha=0.85)

SvN_arr   = np.array([exact_SvN[T] for T in TEMPS])
quadn_arr = np.array([extrap_quad_n[T] for T in TEMPS])
pade_arr  = np.array([extrap_pade_22[T] for T in TEMPS])

ax.plot(Ts, SvN_arr,   "s-",  color="k",   lw=2.0, label=r"$S_{\rm vN}$ (exacta)")
ax.plot(Ts, quadn_arr, "^--", color="C3",           label=r"$S_1$ extrap. cuadrática $n$")
ax.plot(Ts, pade_arr,  "d--", color="C0",           label=r"$S_1$ Padé [2/2]")

ax.axhline(N * np.log(2), color="k", ls=":", alpha=0.4, label=r"$N_A \ln 2$")
ax.set_xlabel(r"$T$")
ax.set_ylabel(r"$S$")
ax.set_title(fr"$S_{{\rm vN}}$ y $S_n$ exactas ($N={N}$)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
save_fig(fig, "SvN_vs_Renyi_vs_T")


# ══════════════════════════════════════════════════════════════════════════════
# Figura: error relativo de cada método de extrapolación
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 4.5))

methods_plot = [
    ("lin(n)",    extrap_linear_n,     "C0", "o-"),
    ("quad(n)",   extrap_quad_n,       "C1", "s-"),
    ("lin(1/n)",  extrap_linear_inv_n, "C2", "^-"),
    ("quad(1/n)", extrap_quad_inv_n,   "C3", "v-"),
    ("pade[2/2]", extrap_pade_22,      "C4", "d-"),
]
for name, dic, color, marker in methods_plot:
    errs = np.array([abs(dic[T] - exact_SvN[T]) / (exact_SvN[T] + 1e-30)
                     for T in TEMPS])
    ax.semilogy(Ts, errs, marker, color=color, label=name, alpha=0.85)

# Chebyshev en su propia escala (fallará en rank alto)
errs_cheb = np.array([abs(extrap_cheb[T] - exact_SvN[T]) / (exact_SvN[T] + 1e-30)
                      for T in TEMPS])
ax.semilogy(Ts, errs_cheb, "x--", color="C2",
            label=f"cheb_nc{CHEB_NC} (rank alto)", alpha=0.85)

ax.set_xlabel(r"$T$")
ax.set_ylabel(r"$|S_1^{\rm est} - S_{\rm vN}| / S_{\rm vN}$")
ax.set_title(fr"Error relativo de la extrapolación ($N={N}$)")
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
save_fig(fig, "SvN_extrap_errors_vs_T")


# ══════════════════════════════════════════════════════════════════════════════
# Sección 0: S_vN via Wang (valor y gradiente) vs exacta
# Comparación de tres métodos: Padé, cuadrática, Chebyshev.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("0. S_vN via Wang: valor y gradiente (3 métodos)")
print("=" * 60)
print(f"    n_samples por S_n : {N_WANG_SVN}")
print(f"    métodos          : Padé [2/2] / cuadrática n / Chebyshev n_c={CHEB_NC}")

results_svn = {}

for T, vstate in trained_vstates.items():
    vstate_R = trained_vstates_R[T]

    SvN_ex, g_SvN_ex = svn_and_grad_exact(vstate, subsystem)
    rank_A = exact_rank_A[T]

    # ── Estimar S_n y ∇S_n via Wang ──
    S_n_wang, g_Sn_wang = [], []
    for n in N_ORDERS:
        S_n_est, g_n_est = renyin_wang_cs(
            vstate, vstate_R, subsystem, n,
            n_samples=N_WANG_SVN, key=n,
        )
        S_n_wang.append(float(S_n_est))
        g_Sn_wang.append(g_n_est)

    # ── Valores ──
    SvN_wang_pade = extrapolate_renyi_to_vN(
        N_ORDERS, np.array(S_n_wang), "pade_2_2"
    )
    SvN_wang_quad = extrapolate_renyi_to_vN(
        N_ORDERS, np.array(S_n_wang), "quadratic_n"
    )
    # Chebyshev estándar (con rank exacto)
    SvN_wang_cheb = chebyshev_extrapolate(
        N_ORDERS, S_n_wang, n_c=CHEB_NC, rank_A=rank_A
    )
    # Chebyshev sin el término α_0·rank (parche heurístico)
    SvN_wang_cheb_dn0 = chebyshev_extrapolate(
        N_ORDERS, S_n_wang, n_c=CHEB_NC, rank_A=rank_A, drop_n0=True
    )

    # ── Gradientes ──
    g_SvN_pade, coefs_pade = svn_grad_from_Sn_estimators(
        S_n_wang, g_Sn_wang, method="pade_2_2"
    )
    g_SvN_quad, coefs_quad = svn_grad_from_Sn_estimators(
        S_n_wang, g_Sn_wang, method="quadratic_n"
    )

    g_SvN_quad3, coefs_quad3 = svn_grad_from_Sn_estimators(
        S_n_wang[:3], g_Sn_wang[:3], method="quadratic_n"
    )
    cos_quad3 = cosine_similarity(g_SvN_quad3, g_SvN_ex)

    
    # Chebyshev vía chain rule (usa dispatch con rank=2 por defecto)
    g_SvN_cheb, coefs_cheb = svn_grad_from_Sn_estimators(
        S_n_wang, g_Sn_wang, method="cheb_nc" + str(CHEB_NC)
    )

    cos_pade = cosine_similarity(g_SvN_pade, g_SvN_ex)
    cos_quad = cosine_similarity(g_SvN_quad, g_SvN_ex)
    cos_cheb = cosine_similarity(g_SvN_cheb, g_SvN_ex)

    print(f"cuadrática 5 pts: cos={cos_quad:+.3f}  ‖c‖₁={sum(abs(c) for c in coefs_quad):.2f}")
    print(f"cuadrática 3 pts: cos={cos_quad3:+.3f}  ‖c‖₁={sum(abs(c) for c in coefs_quad3):.2f}")

    def rel(x):
        return abs(x - SvN_ex) / (abs(SvN_ex) + 1e-30)

    print(f"\n  T={T:.2f}  S_vN_ex={SvN_ex:.5f}  rank={rank_A}")
    print(f"    Padé [2/2]    : valor={SvN_wang_pade:.5f}  err={rel(SvN_wang_pade):>7.3%}  "
          f"cos={cos_pade:+.3f}  ‖c‖₁={sum(abs(c) for c in coefs_pade):6.2f}")
    print(f"    cuadrática n  : valor={SvN_wang_quad:.5f}  err={rel(SvN_wang_quad):>7.3%}  "
          f"cos={cos_quad:+.3f}  ‖c‖₁={sum(abs(c) for c in coefs_quad):6.2f}")
    print(f"    Cheb nc={CHEB_NC}    : valor={SvN_wang_cheb:.5f}  err={rel(SvN_wang_cheb):>7.3%}  "
          f"cos={cos_cheb:+.3f}  ‖c‖₁={sum(abs(c) for c in coefs_cheb):6.2f}")
    print(f"    Cheb nc={CHEB_NC} drop_n0: valor={SvN_wang_cheb_dn0:.5f}  err={rel(SvN_wang_cheb_dn0):>7.3%}")

    results_svn[T] = {
        "SvN_ex":        SvN_ex,
        "rank_A":        rank_A,
        "SvN_pade":      SvN_wang_pade,
        "SvN_quad":      SvN_wang_quad,
        "SvN_cheb":      SvN_wang_cheb,
        "SvN_cheb_dn0":  SvN_wang_cheb_dn0,
        "cos_pade":      cos_pade,
        "cos_quad":      cos_quad,
        "cos_cheb":      cos_cheb,
        "coefs_pade":    coefs_pade,
        "coefs_quad":    coefs_quad,
        "coefs_cheb":    coefs_cheb,
        "S_n_wang":      list(S_n_wang),
    }


# ── Figura: tres métodos (valor y gradiente) ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

Ts_arr           = np.array(TEMPS)
SvN_ex_arr       = np.array([results_svn[T]["SvN_ex"]       for T in TEMPS])
SvN_pade_arr     = np.array([results_svn[T]["SvN_pade"]     for T in TEMPS])
SvN_quad_arr     = np.array([results_svn[T]["SvN_quad"]     for T in TEMPS])
SvN_cheb_arr     = np.array([results_svn[T]["SvN_cheb"]     for T in TEMPS])
SvN_cheb_dn0_arr = np.array([results_svn[T]["SvN_cheb_dn0"] for T in TEMPS])
cos_pade_arr     = np.array([results_svn[T]["cos_pade"]     for T in TEMPS])
cos_quad_arr     = np.array([results_svn[T]["cos_quad"]     for T in TEMPS])
cos_cheb_arr     = np.array([results_svn[T]["cos_cheb"]     for T in TEMPS])

axes[0].plot(Ts_arr, SvN_ex_arr,       "s-",  color="k",
             label=r"$S_{\rm vN}$ exacta")
axes[0].plot(Ts_arr, SvN_pade_arr,     "d--", color="C0",
             label="Padé [2/2]")
axes[0].plot(Ts_arr, SvN_quad_arr,     "^--", color="C3",
             label="cuadrática $n$")
axes[0].plot(Ts_arr, SvN_cheb_arr,     "x--", color="C2",
             label=f"Chebyshev $n_c={CHEB_NC}$")
axes[0].plot(Ts_arr, SvN_cheb_dn0_arr, "o:",  color="C2",
             label=f"Cheb $n_c={CHEB_NC}$, drop $n{{=}}0$")
axes[0].set_xlabel(r"$T$")
axes[0].set_ylabel(r"$S_{\rm vN}$")
axes[0].set_title(r"$S_{\rm vN}$: exacta vs extrapolaciones")
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8)

axes[1].plot(Ts_arr, cos_pade_arr, "o-", color="C0", label="Padé [2/2]")
axes[1].plot(Ts_arr, cos_quad_arr, "s-", color="C3", label="cuadrática $n$")
axes[1].plot(Ts_arr, cos_cheb_arr, "x-", color="C2", label=f"Chebyshev $n_c={CHEB_NC}$")
axes[1].axhline(0.7, color="k", ls=":", alpha=0.4)
axes[1].axhline(0.3, color="k", ls=":", alpha=0.2)
axes[1].set_xlabel(r"$T$")
axes[1].set_ylabel(r"$\cos(\nabla S_{\rm vN}^{\rm est}, \nabla S_{\rm vN}^{\rm exact})$")
axes[1].set_title(r"Calidad del gradiente (3 métodos)")
axes[1].set_ylim(-0.05, 1.05)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=8)

fig.tight_layout()
save_fig(fig, "SvN_three_methods_vs_T")


# ── Figura: norma de coeficientes ──
fig, ax = plt.subplots(figsize=(7, 4))

n1_pade = np.array([sum(abs(c) for c in results_svn[T]["coefs_pade"]) for T in TEMPS])
n1_quad = np.array([sum(abs(c) for c in results_svn[T]["coefs_quad"]) for T in TEMPS])
n1_cheb = np.array([sum(abs(c) for c in results_svn[T]["coefs_cheb"]) for T in TEMPS])

ax.semilogy(Ts_arr, n1_pade, "o-", color="C0", label=r"$\|c\|_1$ Padé [2/2]")
ax.semilogy(Ts_arr, n1_quad, "s-", color="C3", label=r"$\|c\|_1$ cuadrática")
ax.semilogy(Ts_arr, n1_cheb, "x-", color="C2", label=r"$\|c\|_1$ Chebyshev")
ax.set_xlabel(r"$T$")
ax.set_ylabel(r"$\|c\|_1$")
ax.set_title(r"Amplificación del ruido: norma $\ell_1$ de los coeficientes")
ax.grid(True, alpha=0.3, which="both")
ax.legend(fontsize=9)

fig.tight_layout()
save_fig(fig, "SvN_coef_norm_vs_T")


# ══════════════════════════════════════════════════════════════════════════════
# Sección 0.5: convergencia de S_vN con el número de órdenes k usados
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("0.5. S_vN vs número de órdenes k usados")
print("=" * 60)
print("    Esquema progresivo:")
print("      k=1: S_2 como proxy")
print("      k=2: lineal en n")
print("      k=3: cuadrática en n")
print("      k=4: Padé [2/1]")
print("      k>=5: Padé [2/2]")

K_VALUES = list(range(1, len(N_ORDERS) + 1))

svn_vs_k_wang = {}
svn_vs_k_exact = {}

for T, vstate in trained_vstates.items():
    SvN_ex = exact_SvN[T]
    S_n_wang = results_svn[T]["S_n_wang"]

    svn_vs_k_wang[T]  = {}
    svn_vs_k_exact[T] = {}

    print(f"\n  T={T:.2f}  S_vN_ex={SvN_ex:.5f}")
    for k in K_VALUES:
        SvN_k_w = svn_from_k_orders(S_n_wang, k)
        err_w   = abs(SvN_k_w - SvN_ex) / SvN_ex
        svn_vs_k_wang[T][k] = {"SvN": SvN_k_w, "err": err_w}

        S_n_ex_list = [exact_Sn[n][T] for n in N_ORDERS]
        SvN_k_e = svn_from_k_orders(S_n_ex_list, k)
        err_e   = abs(SvN_k_e - SvN_ex) / SvN_ex
        svn_vs_k_exact[T][k] = {"SvN": SvN_k_e, "err": err_e}

        print(f"    k={k}: Wang={SvN_k_w:.5f} (err={err_w:>7.3%})   "
              f"exact={SvN_k_e:.5f} (err={err_e:>7.3%})")


# ── Figura: error relativo vs k ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
cmap_T = plt.cm.viridis

for T in TEMPS:
    errs_w = [svn_vs_k_wang[T][k]["err"]  for k in K_VALUES]
    errs_e = [svn_vs_k_exact[T][k]["err"] for k in K_VALUES]
    color  = cmap_T(TEMPS.index(T) / max(1, len(TEMPS) - 1))

    axes[0].semilogy(K_VALUES, errs_w, "o-", color=color, label=fr"$T={T}$")
    axes[1].semilogy(K_VALUES, errs_e, "o-", color=color, label=fr"$T={T}$")

for ax, title in zip(axes, [r"via Wang", r"via $S_n$ exactos"]):
    ax.set_xlabel(r"número de órdenes $k$")
    ax.set_xticks(K_VALUES)
    ax.set_ylabel(r"$|S_{\rm vN}^{\rm est} - S_{\rm vN}^{\rm ex}| / S_{\rm vN}^{\rm ex}$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)

fig.suptitle(r"Convergencia de $S_{\rm vN}$ con el número de órdenes usados", y=1.02)
fig.tight_layout()
save_fig(fig, "SvN_vs_num_orders")


# ══════════════════════════════════════════════════════════════════════════════
# Sección 1: comparación a PRESUPUESTO FIJO (estimadores muestreados)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"1. Comparación a presupuesto fijo = {BUDGET_FIXED} evals")
print("=" * 60)

results = {n: {"swap": {}, "wang": {}} for n in N_ORDERS}

for n in N_ORDERS:
    ns_swap = BUDGET_FIXED // cost_swap(n)
    ns_wang = BUDGET_FIXED // cost_wang(n, N_TOTAL)
    print(f"\n  ── n = {n} ──")
    print(f"     swap: n_samples = {ns_swap}   wang: n_samples = {ns_wang}")

    for T, vstate in trained_vstates.items():
        vstate_R = trained_vstates_R[T]
        S_n_ex   = exact_Sn[n][T]
        g_n_ex   = exact_gSn[n][T]

        s_sw, cos_sw     = [], []
        s_wang, cos_wang = [], []

        for rep in range(n_rep):
            S_est, g = renyin_entropy_and_grad_sampled(
                vstate, subsystem, n, n_samples=ns_swap, key=rep
            )
            s_sw.append(float(S_est))
            cos_sw.append(cosine_similarity(g, g_n_ex))

            S_est, g = renyin_wang_cs(
                vstate, vstate_R, subsystem, n,
                n_samples=ns_wang, key=rep
            )
            s_wang.append(float(S_est))
            cos_wang.append(cosine_similarity(g, g_n_ex))

        s_sw   = np.array(s_sw)
        s_wang = np.array(s_wang)
        err_sw   = np.abs(s_sw   - S_n_ex)
        err_wang = np.abs(s_wang - S_n_ex)

        results[n]["swap"][T] = {
            "S_n": S_n_ex,
            "err_mean": err_sw.mean(),
            "err_p25":  np.percentile(err_sw, 25),
            "err_p75":  np.percentile(err_sw, 75),
            "cos_mean": float(np.mean(cos_sw)),
            "cos_std":  float(np.std(cos_sw)),
        }
        results[n]["wang"][T] = {
            "S_n": S_n_ex,
            "err_mean": err_wang.mean(),
            "err_p25":  np.percentile(err_wang, 25),
            "err_p75":  np.percentile(err_wang, 75),
            "cos_mean": float(np.mean(cos_wang)),
            "cos_std":  float(np.std(cos_wang)),
        }

        print(f"     T={T:.2f}  S_n={S_n_ex:.4f}  "
              f"swap |Δ|/S={err_sw.mean()/S_n_ex:.4f} "
              f"cos={np.mean(cos_sw):.3f}  |  "
              f"wang |Δ|/S={err_wang.mean()/S_n_ex:.4f} "
              f"cos={np.mean(cos_wang):.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Figura: calidad del gradiente de S_n vs T, un panel por n
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(N_ORDERS),
                         figsize=(3.4 * len(N_ORDERS), 3.6),
                         sharey=True, squeeze=False)
axes = axes[0]

for ax, n in zip(axes, N_ORDERS):
    cos_sw   = np.array([results[n]["swap"][T]["cos_mean"] for T in TEMPS])
    cos_wang = np.array([results[n]["wang"][T]["cos_mean"] for T in TEMPS])
    std_sw   = np.array([results[n]["swap"][T]["cos_std"]  for T in TEMPS])
    std_wang = np.array([results[n]["wang"][T]["cos_std"]  for T in TEMPS])

    ax.errorbar(Ts, cos_sw,   yerr=std_sw,   fmt="o-", color="C0",
                label="swap (direct)", capsize=3)
    ax.errorbar(Ts, cos_wang, yerr=std_wang, fmt="d-", color="C3",
                label="Wang (conditional)", capsize=3)
    ax.axhline(0.7, color="k", ls=":", alpha=0.4)
    ax.axhline(0.3, color="k", ls=":", alpha=0.2)

    ax.set_xlabel(r"$T$")
    if ax is axes[0]:
        ax.set_ylabel(r"$\cos(\nabla S_n^{\rm est}, "
                      r"\nabla S_n^{\rm ex})$")
    ax.set_title(fr"$n = {n}$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

fig.suptitle(fr"Calidad del gradiente de $S_n$ vs $T$ "
             fr"(presupuesto $B = {BUDGET_FIXED}$)", y=1.02)
fig.tight_layout()
save_fig(fig, "Sn_grad_cos_vs_T")


print("\nDone. Gráficas guardadas en", os.path.abspath(PLOTS_DIR))