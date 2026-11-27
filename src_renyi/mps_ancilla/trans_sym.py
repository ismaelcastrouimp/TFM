"""
src_renyi/mps_ancilla/trans_sym.py
==================================
Pureza de la purificación simetrizada por un grupo G de simetrías de H (traslaciones, inversión global de espines,
reflexión espacial), compatible con las parejas en el sector (u1/u1c) y la etiqueta de Sz:
    ρ_G = (1/|G|) Σ_g U_g ρ U_g†,   E(ρ_G) = E(ρ),   P_G = (1/|G|) Σ_g X_g,   X_g = Tr(ρ U_g ρ U_g†) = Σ_{s,t} ρ(s,t) ρ(σt,σs)
  - X_g = X_{g⁻¹} (ciclicidad): basta un representante de cada par {g, g⁻¹} con multiplicidad 2 (1 si g = g⁻¹).
  - Forma "diff" (variable de control con las mismas parejas):  P_G = P + (1/|G|) Σ_{g≠e} m_g (X_g − P),
      término por pareja  [ρ(s,t) ρ(σt,σs) − ρ(s,t) ρ(t,s)] / (p(s) q(t|s)),  P con el estimador de sector (sym, etiqueta).
    Forma "direct": P_G = (1/|G|) [P + Σ_{g≠e} m_g X_g].
  - Peso (s ~ p, t ~ q(t|s), sin gradiente):  h_g = p(t)/q · (A(σt)/A(t)) (A(σs)/A(s))* · G(s,t) G(σt,σs);  identidad: w = p(t)/q |G(s,t)|².
  - Gradiente: ∇X_g = E[ Re( h_g ∇ log F_g ) ],  log F_g = lA(s) + lA(t)* + log G(s,t) + lA(σt) + lA(σs)* + log G(σt,σs)
    (4 log A y 2 log G; la distribución de muestreo no se deriva).  Identidad: ∇ log |ρ(s,t)|² = 2Re lA(s) + 2Re lA(t) + log|G|².
  Denominador: P_G fijo (media móvil en el entreno; aquí el valor que se pase).
"""
import numpy as np
import jax
import jax.numpy as jnp

from .models import mps_overlap_aligned

sg = jax.lax.stop_gradient


def group_elements(N, kind):
    """Lista de (perm, signo) de TODOS los elementos (el primero es la identidad) y su tamaño.
    kind: T1 ({1,T}, conjunto), T2 (desplazamientos pares), trasl (Z_N), trasl_P (× inversión global de espines),
          R ({1, reflexión}), trasl_R (diédrico: Z_N × reflexión i -> -i)."""
    ar = np.arange(N); tr = [(np.roll(ar, -k), 1.0) for k in range(N)]; refl = (-ar) % N
    els = {"T1": tr[:2], "T2": tr[::2], "trasl": tr, "trasl_P": tr + [(pm, -1.0) for pm, _ in tr],
           "R": [tr[0], (refl, 1.0)], "trasl_R": tr + [(pm[refl], 1.0) for pm, _ in tr]}[kind]
    return els


def representatives(els):
    """Elementos distintos de la identidad agrupados por {g, g⁻¹}: lista de (perm, signo, multiplicidad)."""
    key = lambda pm, sgn: (tuple(int(x) for x in pm), float(sgn))
    present = {key(pm, s) for pm, s in els}
    seen, reps = set(), []
    for pm, s in els[1:]:
        k = key(pm, s)
        if k in seen:
            continue
        inv = key(np.argsort(pm), s)
        seen.add(k)
        if inv != k and inv in present:
            seen.add(inv); reps.append((np.asarray(pm), s, 2))
        else:
            reps.append((np.asarray(pm), s, 1))
    assert sum(m for _, _, m in reps) == len(els) - 1
    return reps


def make_tsym_dX_one(model, chi, reps, nG, form="diff"):
    """Forma diff con UN elemento del grupo por pareja (coste fijo por pareja, independiente de |G|).
    Lista L = representantes repetidos según su multiplicidad (|L| = |G| − 1); la pareja k usa L[(k + off) % |L|] (off
    aleatorio por lote: estratificado).  Peso de la pareja: m_g / (|G| n_g), n_g = nº de parejas del lote con ese g →
    E[Σ_k peso_k D_g(k)] = (1/|G|) Σ_g m_g (X_g − P) exactamente.
    dX(params, s, t, lq, K, key) -> (L_X, val_X) como make_tsym_dX."""
    L = [(pm, sg_, m) for pm, sg_, m in reps for _ in range(m)]
    PERM = jnp.asarray(np.stack([pm for pm, _, _ in L])); SGN = jnp.asarray(np.array([s_ for _, s_, _ in L]))
    MREP = jnp.asarray(np.array([m for _, _, m in L], float))
    RID = np.array([ri for ri, (_, _, m) in enumerate(reps) for _ in range(m)]); RID = jnp.asarray(RID)
    nL = len(L); nR = len(reps)
    rep = lambda x, K: jnp.repeat(x, K, axis=0)

    def dX(params, s, t, lq, K, key):
        Pn = t.shape[0]
        off = jax.random.randint(key, (), 0, nL)
        li = (jnp.arange(Pn) + off) % nL
        perm = PERM[li]; sgn = SGN[li][:, None]; rid = RID[li]
        counts = jnp.zeros(nR).at[rid].add(1.0)
        mult = jnp.asarray(np.array([m for _, _, m in reps], float))
        wk = mult[rid] / (nG * counts[rid])                                            # m_g / (|G| n_g)
        lA_s, M_s = model.apply({"params": params}, s); lA_t, M_t = model.apply({"params": params}, t)
        n_s = jnp.real(mps_overlap_aligned(M_s, M_s, chi)); n_t = jnp.real(mps_overlap_aligned(M_t, M_t, chi))
        G_st = mps_overlap_aligned(rep(M_s, K), M_t, chi) / jnp.sqrt(rep(n_s, K) * n_t)
        lw = sg(2.0 * jnp.real(lA_t) - lq)
        S = rep(s, K); lA_sr = rep(lA_s, K)
        gs = sgn * jnp.take_along_axis(S, perm, axis=1); gt = sgn * jnp.take_along_axis(t, perm, axis=1)
        lA_gs, M_gs = model.apply({"params": params}, gs); lA_gt, M_gt = model.apply({"params": params}, gt)
        n_gs = jnp.real(mps_overlap_aligned(M_gs, M_gs, chi)); n_gt = jnp.real(mps_overlap_aligned(M_gt, M_gt, chi))
        G_g = mps_overlap_aligned(M_gt, M_gs, chi) / jnp.sqrt(n_gt * n_gs)
        w = sg(jnp.exp(lw) * jnp.abs(G_st) ** 2)
        h = sg(jnp.exp(lw + (lA_gt - lA_t) + jnp.conj(lA_gs - lA_sr)) * G_st * G_g)
        logF = lA_sr + jnp.conj(lA_t) + jnp.log(G_st) + lA_gt + jnp.conj(lA_gs) + jnp.log(G_g)
        logP = 2.0 * jnp.real(lA_sr) + 2.0 * jnp.real(lA_t) + jnp.log(jnp.abs(G_st) ** 2)
        term = jnp.real(h * logF); val = jnp.real(h)
        if form == "parts":                                    # piezas por pareja para make_tsym_dX_beta
            return wk, term, val, w, w * logP
        if form == "diff":                                     # direct: Σ m X_g / |G| (la P va aparte con peso 1/|G|)
            term = term - w * logP; val = val - w
        return jnp.sum(sg(wk) * term), sg(jnp.sum(wk * val))
    return dX


def make_tsym_dX_beta(model, chi, reps, nG):
    """Un g por pareja con variable de control de β óptimo:  P̂_G = c_β P̂ + Σ_k w_k (h_k − β w_k),  c_β = 1/|G| + β (|G|−1)/|G|
    (β = 1: diff1; β = 0: direct1; insesgado para todo β fijo).  β = cov(h, w)/var(w) en el lote, sin gradiente (sesgo O(1/M)).
    dX(params, s, t, lq, K, key) -> (L_X, val_X, β); el que llama pondera la pérdida de P con sg(c_β)."""
    base = make_tsym_dX_one(model, chi, reps, nG, form="parts")

    def dX(params, s, t, lq, K, key):
        wk, hterm, hval, w, wlogP = base(params, s, t, lq, K, key)
        x = hval; y = w
        beta = sg(jnp.mean((x - x.mean()) * (y - y.mean())) / jnp.maximum(jnp.var(y), 1e-300))
        L = jnp.sum(sg(wk) * (hterm - beta * wlogP))
        V = sg(jnp.sum(wk * (hval - beta * w)))
        return L, V, beta
    return dX


def make_tsym_dX(model, chi, reps, nG, form="diff"):
    """dX(params, s, t, lq, K) -> (L_X, val_X):  ∇L_X ≈ (1/|G|) Σ m_g ∇(X_g − P)  [diff]  o  (1/|G|) Σ m_g ∇X_g  [direct];
    val_X ≈ lo mismo sin gradiente.  Se suma a la pérdida de la pureza (forma diff) o a ella / |G| (direct)."""
    rep = lambda x, K: jnp.repeat(x, K, axis=0)

    def dX(params, s, t, lq, K):
        lA_s, M_s = model.apply({"params": params}, s); lA_t, M_t = model.apply({"params": params}, t)
        n_s = jnp.real(mps_overlap_aligned(M_s, M_s, chi)); n_t = jnp.real(mps_overlap_aligned(M_t, M_t, chi))
        G_st = mps_overlap_aligned(rep(M_s, K), M_t, chi) / jnp.sqrt(rep(n_s, K) * n_t)
        lw = sg(2.0 * jnp.real(lA_t) - lq)
        lA_sr = rep(lA_s, K)
        w = sg(jnp.exp(lw) * jnp.abs(G_st) ** 2)
        logP = 2.0 * jnp.real(lA_sr) + 2.0 * jnp.real(lA_t) + jnp.log(jnp.abs(G_st) ** 2)
        L, V = 0.0, 0.0
        for perm, sgn, m in reps:
            gs = sgn * s[:, perm]; gt = sgn * t[:, perm]
            lA_gs, M_gs = model.apply({"params": params}, gs); lA_gt, M_gt = model.apply({"params": params}, gt)
            n_gs = jnp.real(mps_overlap_aligned(M_gs, M_gs, chi)); n_gt = jnp.real(mps_overlap_aligned(M_gt, M_gt, chi))
            G_g = mps_overlap_aligned(M_gt, rep(M_gs, K), chi) / jnp.sqrt(n_gt * rep(n_gs, K))
            lA_gsr = rep(lA_gs, K)
            h = sg(jnp.exp(lw + (lA_gt - lA_t) + jnp.conj(lA_gsr - lA_sr)) * G_st * G_g)
            logF = lA_sr + jnp.conj(lA_t) + jnp.log(G_st) + lA_gt + jnp.conj(lA_gsr) + jnp.log(G_g)
            term = jnp.real(h * logF)
            val = jnp.real(h)
            if form == "diff":
                term = term - w * logP; val = val - w
            L = L + m * jnp.mean(term); V = V + m * jnp.mean(val)
        return L / nG, sg(V / nG)
    return dX
