"""Entrenamiento de ARViT+MPS: exacto (ED) y Monte Carlo (MC)."""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

from .models import mps_overlap_rect, mps_overlap_aligned, mps_overlap_matrix

sg = jax.lax.stop_gradient


# ════════════════════════════════════════════════════════════════════════════
# ED (exacto, enumera 2^N; solo N ≤ 10)
# ════════════════════════════════════════════════════════════════════════════

def compute_rho(model, params, all_s, chunk=None):
    """chunk: filas por trozo en la matriz de solapes (lax.map + checkpoint). Necesario para N>=10
    en GPU (la memoria del gradiente crece como K * S^2 * chi^2 sin trozos)."""
    logA, M = model.apply({"params": params}, all_s)
    A = jnp.exp(logA - jnp.max(jnp.real(logA)))
    A = A / jnp.linalg.norm(A)
    O = mps_overlap_rect(M, M, model.chi, chunk)
    nrm = jnp.sqrt(jnp.real(jnp.diagonal(O)) + 1e-300)
    G = O / (nrm[:, None] * nrm[None, :])                                   # solape normalizado
    rho = A[:, None] * jnp.conj(A)[None, :] * G
    return rho


def make_loss_fns(model, all_s, H_dense, T, chunk=None):
    all_s = jnp.asarray(all_s)
    H = jnp.asarray(H_dense)

    def loss(params):
        rho = compute_rho(model, params, all_s, chunk)
        E = jnp.real(jnp.sum(H.T * rho))                                    # Tr(H rho)
        purity = jnp.sum(jnp.abs(rho) ** 2)                                 # Tr rho^2
        S2 = -jnp.log(purity)
        return E - T * S2, (E, S2)

    return loss, jax.jit(lambda p: compute_rho(model, p, all_s, chunk))


def count_params(params):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def train(loss, params, n_steps, lr, T, *, freq=100, warmup=100, end_frac=1e-2,
          clip=10.0, diag_fn=None, verbose=True, tag=""):
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, n_steps, lr * end_frac)
    tx = optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched))
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state):
        (F, (E, S2)), g = jax.value_and_grad(loss, has_aux=True)(params)
        upd, opt_state = tx.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, F, E, S2

    best_F, best_params = np.inf, params
    t0 = time.time()
    for it in range(n_steps):
        cur = params
        params, opt_state, F, E, S2 = step(params, opt_state)
        if verbose and it % freq == 0:
            msg = f"{tag}Step {it:5d} | F={float(F):.6f} | E={float(E):.6f} | S2={float(S2):.6f}"
            if diag_fn is not None:
                msg += " | " + diag_fn(cur)
            print(msg, flush=True)
        if it % 10 == 0 or it == n_steps - 1:
            Fv = float(F)
            if np.isfinite(Fv) and Fv < best_F:
                best_F, best_params = Fv, cur
    jax.block_until_ready(params)
    dt = time.time() - t0
    _, (E_b, S2_b) = loss(best_params)
    E_b, S2_b = float(E_b), float(S2_b)
    return {"params_best": best_params, "F": E_b - T * S2_b, "E": E_b, "S2": S2_b,
            "time": dt, "s_per_step": dt / n_steps}

# ════════════════════════════════════════════════════════════════════════════
# MC (escala a N grande)
# ════════════════════════════════════════════════════════════════════════════

def make_sampler(model, N, M):
    """Devuelve sample(params, key) -> s (M, N) con valores +-1, muestreo autorregresivo exacto."""
    @jax.jit
    def sample(params, key):
        def body(i, carry):
            s, key = carry
            key, sub = jax.random.split(key)
            logits = model.apply({"params": params}, s, only_logits=True)       # (M, N, 2)
            li = jax.lax.dynamic_index_in_dim(logits, i, axis=1, keepdims=False)  # (M, 2)
            draw = jax.random.categorical(sub, li, axis=-1)                     # 0 -> -1, 1 -> +1
            s = s.at[:, i].set(2.0 * draw - 1.0)
            return s, key
        s0 = jnp.zeros((M, N), jnp.float64)
        s, _ = jax.lax.fori_loop(0, N, body, (s0, key))
        return s
    return sample


def make_mc_loss(model, N, T, hx, J, chunk=None, pair_block=None, terms="both"):
    chi = model.chi

    def mc_loss(params, s):
        s = sg(s)
        M = s.shape[0]
        logA, Mt = model.apply({"params": params}, s)
        n2 = jnp.real(mps_overlap_aligned(Mt, Mt, chi))                         # ||Phi_s||^2
        logp = 2.0 * jnp.real(logA)                                             # log p(s)

        # ---- energia local ----
        flip = 1.0 - 2.0 * jnp.eye(N)
        sf = (s[:, None, :] * flip[None]).reshape(M * N, N)                     # s con el espin i invertido
        logAf, Mf = model.apply({"params": params}, sf)
        Mrep = jnp.repeat(Mt, N, axis=0)
        Of = mps_overlap_aligned(Mf, Mrep, chi)                                 # <Phi_s | Phi_s'> (orden: s', s)
        n2f = jnp.real(mps_overlap_aligned(Mf, Mf, chi))
        Gf = Of / jnp.sqrt(n2f * jnp.repeat(n2, N))
        ratio = jnp.exp(logAf - jnp.repeat(logA, N))
        offdiag = jnp.sum((ratio * Gf).reshape(M, N), axis=1)
        diag = jnp.sum(s * jnp.roll(s, -1, axis=1), axis=1)
        ReE = jnp.real(hx * offdiag + J * diag)
        Ebar = jnp.mean(ReE)
        L_E = jnp.mean(sg(ReE - Ebar) * logp + ReE)

        # ---- pureza ----
        # pair_block=None: todos los pares i!=j (coste M^2).  pair_block=b: se parte el lote en M/b
        # bloques y se usan todos los pares DENTRO de cada bloque (coste M*b).  b=2 == M/2 pares disjuntos.
        if pair_block is None:
            O = mps_overlap_rect(Mt, Mt, chi, chunk)
            f = jnp.abs(O) ** 2 / (n2[:, None] * n2[None, :])
            mask = 1.0 - jnp.eye(M)
            npairs = M * (M - 1)
            lp = logp[:, None] + logp[None, :]
        else:
            b = pair_block
            assert M % b == 0, "M debe ser multiplo de pair_block"
            nb = M // b
            Mb = Mt.reshape(nb, b, *Mt.shape[1:])
            O = jax.vmap(lambda m: mps_overlap_rect(m, m, chi))(Mb)               # (nb, b, b)
            n2b = n2.reshape(nb, b)
            f = jnp.abs(O) ** 2 / (n2b[:, :, None] * n2b[:, None, :])
            mask = jnp.broadcast_to(1.0 - jnp.eye(b), (nb, b, b))
            npairs = nb * b * (b - 1)
            lpb = logp.reshape(nb, b)
            lp = lpb[:, :, None] + lpb[:, None, :]
        Pb = jnp.sum(f * mask) / npairs
        L_P = T * (Pb + jnp.sum(mask * sg(f - Pb) * lp) / npairs) / sg(Pb)

        S2 = -jnp.log(sg(Pb))
        total = {"both": L_E + L_P, "E": L_E, "P": L_P}[terms]
        return total, (Ebar, S2, Ebar - T * S2)

    return mc_loss


def train_mc(model, params, N, T, hx, J, *, M=512, n_steps=2000, lr=5e-3, clip=10.0, warmup=100,
             end_frac=1e-2, seed=0, chunk=None, pair_block=None, freq=50, exact_fn=None, ref=None,
             ema_decay=0.995, polish_steps=0, polish_M=None, polish_lr=1e-3, polish_pair_block=None,
             ckpt_fn=None, ckpt_every=250, tag=""):
    """
    Entrena con el gradiente MC en dos fases:
      1) n_steps con warmup + coseno, lotes de M muestras.
      2) (opcional) polish_steps con lr CONSTANTE polish_lr y lotes de polish_M muestras (mas grandes).
    Mantiene ademas un promedio exponencial (EMA, con correccion de sesgo) de los parametros.

    exact_fn(params) -> (F, E, S2) exactos por ED (opcional, solo N pequeno): se imprime para los
                        parametros actuales y para la EMA.
    ref = (F, E, S2) del MRE exacto: se imprime como baseline y se usa para los gaps finales.
    Devuelve dict con params (ultimos), params_ema, hist (E, S2, F de MC por paso) y s_per_step.
    """
    if ref is not None:
        print(f"{tag}BASELINE exacto (MRE): F={ref[0]:.5f}  E={ref[1]:.5f}  S2={ref[2]:.4f}", flush=True)

    def make_phase(Mv, pb, sched):
        sampler = make_sampler(model, N, Mv)
        mc_loss = make_mc_loss(model, N, T, hx, J, chunk, pb)
        tx = optax.chain(optax.clip_by_global_norm(clip), optax.adam(sched))

        @jax.jit
        def step(params, opt_state, ema, key):
            key, sub = jax.random.split(key)
            s = sampler(params, sub)
            (_, aux), g = jax.value_and_grad(mc_loss, has_aux=True)(params, s)
            upd, opt_state = tx.update(g, opt_state, params)
            params = optax.apply_updates(params, upd)
            ema = jax.tree_util.tree_map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, ema, params)
            return params, opt_state, ema, key, aux
        return tx, step

    phases = [("main", M, pair_block, n_steps,
               optax.warmup_cosine_decay_schedule(0.0, lr, min(warmup, max(n_steps // 5, 1)), n_steps,
                                                  lr * end_frac))]
    if polish_steps > 0:
        phases.append(("polish", polish_M or 4 * M, polish_pair_block or pair_block, polish_steps,
                       optax.constant_schedule(polish_lr)))

    def ema_hat(ema, t):
        return jax.tree_util.tree_map(lambda e: e / (1.0 - ema_decay ** t), ema)

    key = jax.random.PRNGKey(seed)
    ema = jax.tree_util.tree_map(jnp.zeros_like, params)
    hist, t_total, t0 = [], 0, time.time()
    for name, Mv, pb, n, sched in phases:
        tx, step = make_phase(Mv, pb, sched)
        opt_state = tx.init(params)
        for it in range(n):
            params, opt_state, ema, key, (E, S2, F) = step(params, opt_state, ema, key)
            t_total += 1
            hist.append((float(E), float(S2), float(F)))
            if ckpt_fn is not None and t_total % ckpt_every == 0:
                ckpt_fn(params, ema_hat(ema, t_total))
            if it % freq == 0:
                msg = f"{tag}[{name}] Step {it:5d} | MC: F={float(F):.5f} E={float(E):.5f} S2={float(S2):.4f}"
                if exact_fn is not None:
                    try:
                        Fx, Ex, Sx = exact_fn(params)
                        Fe, Ee, Se = exact_fn(ema_hat(ema, t_total))
                        msg += (f" | ED: F={float(Fx):.5f} E={float(Ex):.5f} S2={float(Sx):.4f}"
                                f" | ED(EMA): F={float(Fe):.5f}")
                    except Exception as e:                      # la monitorizacion no debe tirar el entrenamiento
                        msg += f" | (monitorizacion ED fallo: {type(e).__name__})"
                print(msg, flush=True)
    params_ema = ema_hat(ema, t_total)
    if exact_fn is not None:
        print(f"{tag}RESUMEN (evaluacion exacta por ED)")
        for lab, p in (("ultimos ", params), ("EMA     ", params_ema)):
            Fx, Ex, Sx = exact_fn(p)
            line = f"{tag}  {lab}: F={float(Fx):.5f}  E={float(Ex):.5f}  S2={float(Sx):.4f}"
            if ref is not None:
                line += f"   gap F = {float(Fx) - ref[0]:+.4f}   (dE={float(Ex) - ref[1]:+.3f}, dS2={float(Sx) - ref[2]:+.3f})"
            print(line, flush=True)
    return {"params": params, "params_ema": params_ema, "hist": np.array(hist),
            "s_per_step": (time.time() - t0) / max(t_total, 1)}