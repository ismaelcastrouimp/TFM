"""
src_renyi/extrapolation.py
==========================
Extrapolación Rényi → S_vN.

Self-contained: no importa de `src_renyi` ni de `scripts/`, para evitar
imports circulares cuando `observables.py` lo usa.

Métodos:
  - 'linear_n', 'quadratic_n', 'linear_inv_n', 'quadratic_inv_n'
  - 'pade_L_M' (ej. 'pade_2_2'), 'pade_auto'
  - 'cheb_ncK' (Chebyshev-BPA; solo válido para rank bajo)
"""

import numpy as np
import jax


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
# Chebyshev-BPA
# ══════════════════════════════════════════════════════════════════════════════

def chebyshev_coeffs_monomial(n_c):
    a = np.zeros(n_c + 1)
    a[0] = np.log(2) - 0.5
    if n_c >= 1:
        a[1] = np.log(2) - 0.75
    for k in range(2, n_c + 1):
        a[k] = (-1) ** (k + 1) / (k * (k ** 2 - 1))

    lin = np.array([-1.0, 2.0])
    alpha = np.zeros(n_c + 1)
    for k in range(n_c + 1):
        Tk_y = np.zeros(k + 1); Tk_y[k] = 1.0
        Tk_mono_y = np.polynomial.chebyshev.cheb2poly(Tk_y)
        Tk_mono_x = np.zeros(1)
        for c in Tk_mono_y[::-1]:
            Tk_mono_x = np.polynomial.polynomial.polyadd(
                np.polynomial.polynomial.polymul(Tk_mono_x, lin), [c],
            )
        alpha[: len(Tk_mono_x)] += a[k] * Tk_mono_x
    return alpha


def chebyshev_extrapolate(n_orders, S_n_values, n_c=None,
                          rank_A=None, N_A=None, drop_n0=False):
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


# ══════════════════════════════════════════════════════════════════════════════
# Extrapolación unificada
# ══════════════════════════════════════════════════════════════════════════════

def extrapolate_renyi_to_vN(n_orders, S_n_values, method="pade_2_2"):
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
        return chebyshev_extrapolate(n_orders, S_n_values, n_c=n_c, N_A=1)

    raise ValueError(f"método desconocido: {method}")


# ══════════════════════════════════════════════════════════════════════════════
# Gradiente de S_vN por chain rule
# ══════════════════════════════════════════════════════════════════════════════

def svn_grad_from_Sn_estimators(S_n_values, grad_Sn_list,
                                method="pade_2_2", eps=1e-4,
                                n_orders=None):
    """
    ∇_θ S_vN ≈ Σ_n (∂S_vN/∂S_n) · ∇_θ S_n

    Parámetros
    ----------
    S_n_values : list[float]  — estimaciones de S_n
    grad_Sn_list : list[pytree]  — ∇S_n correspondientes
    method : str  — método de extrapolación
    eps : float  — paso de diferencias finitas
    n_orders : list[int] or None
        Órdenes de Rényi correspondientes a S_n_values.
        Si None, se infiere como [2, 3, ..., len+1].
        IMPORTANTE: si usas órdenes no consecutivos (ej. [2, 4, 6]),
        pásalos explícitamente.

    Devuelve
    --------
    grad_total : pytree  — gradiente de S_vN
    coefs : list[float]  — coeficientes ∂S_vN/∂S_n
    """
    if n_orders is None:
        n_orders = list(range(2, 2 + len(S_n_values)))
    else:
        n_orders = list(n_orders)

    if len(n_orders) != len(S_n_values):
        raise ValueError(
            f"len(n_orders)={len(n_orders)} != len(S_n_values)={len(S_n_values)}"
        )

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