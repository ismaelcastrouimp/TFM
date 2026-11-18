"""
train_phys_bridge.py
=========================================
Minimiza F = E - T·S₂ del TFIM 1D (PBC, J_ZZ = -1, h_x = -1.5 por defecto) con ancilla FÍSICA:
ψ(s, a) sobre 2N espines, sistema = sitios 0..N-1, ancilla = N..2N-1 (src_renyi/physical_ancilla/models_encdec.py).
  --arch encdec : sistema ARViT causal + encoder bidireccional de s + decoder AR de a con atención cruzada.
  --arch prefix : la arquitectura previa (un transformer causal sobre [s, a]; = ARSpinViT_Causal).
ψ compleja (módulo + fase) con parámetros reales.

Modos:
  --mode exact : (N <= 8) F, E, S₂ y gradiente EXACTOS enumerando las 4^N configuraciones (aísla la expresividad).
  --mode mc    : E por MC (NetKet, muestreo AR directo) y S₂ con el estimador puente (bridge.renyi2_bridge);
                 opcional --lam: penalización de gauge λ(1 - H) (bridge.hermiticity), no cambia F.
Optimizador como mps_ancilla.training.train_mc: Adam + clip 10 + warmup/coseno + EMA de parámetros.
Referencia: MRE exacto (fermiones libres, mre_teff.spectrum).  Salida: <out_dir>/<label>.json (+ _params.msgpack).

Ejemplos:
  python scripts/train_phys_bridge.py --N 6 --T 3 --mode exact --arch encdec --steps 2000 --label ex_N6_encdec
  python scripts/train_phys_bridge.py --N 25 --T 3 --mode mc --arch encdec --M 1024 --steps 3000 --label mc_N25_encdec
"""
import argparse, json, os, sys, time
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cfg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..")); sys.path.insert(0, _here)

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True); ap.add_argument("--T", type=float, default=3.0)
ap.add_argument("--J", type=float, default=-1.0); ap.add_argument("--h", type=float, default=-1.5)
ap.add_argument("--arch", choices=["encdec", "prefix"], default="encdec")
ap.add_argument("--d", type=int, default=16); ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--sys_blocks", type=int, default=2); ap.add_argument("--enc_blocks", type=int, default=1)
ap.add_argument("--dec_blocks", type=int, default=2); ap.add_argument("--blocks", type=int, default=4,
                                                                  help="prefix: nº de bloques")
ap.add_argument("--phase_net", choices=["head", "mlp", "enc"], default="head")
ap.add_argument("--phase_after", type=int, default=0,
                help="K > 0: fase a 0 y congelada los K primeros pasos (solo módulo), luego se entrena todo")
ap.add_argument("--copy_init", type=float, default=4.0, help="0 = sin inicialización copia")
ap.add_argument("--mode", choices=["exact", "mc"], default="mc")
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--end_frac", type=float, default=1e-2)
ap.add_argument("--clip", type=float, default=10.0); ap.add_argument("--ema", type=float, default=0.995)
ap.add_argument("--n_E", type=int, default=1024, help="muestras para la energía (mc)")
ap.add_argument("--M", type=int, default=1024, help="configuraciones s1 del puente (mc)")
ap.add_argument("--K", type=int, default=5); ap.add_argument("--n_a", type=int, default=2)
ap.add_argument("--lam", type=float, default=0.0, help="penalización de gauge λ(1-H)")
ap.add_argument("--mu", type=float, default=0.0,
                help="exacto: + μ·J (segundo momento exacto del solapamiento del puente, gauge_opt.py): elige un gauge de poco ruido")
ap.add_argument("--mu_ramp", type=int, default=0, help="μ sube linealmente de 0 a μ en estos pasos")
ap.add_argument("--mu_end", type=float, default=1.0, help="tras la rampa, μ baja linealmente hasta μ·mu_end al final")
ap.add_argument("--logJ", action="store_true", help="penaliza μ·log J en vez de μ·J")
ap.add_argument("--herm", action="store_true",
                help="exacto: Ψ = Re(G + G^T)/2 (real simétrica, gauge hermítico por construcción; G = red); guarda Ψ final")
ap.add_argument("--e_chunk", type=int, default=256, help="trozo (muestras) de la energía")
ap.add_argument("--chunk", type=int, default=0, help="trozo (nº de s1) del puente; 0 = sin trocear")
ap.add_argument("--log_every", type=int, default=25); ap.add_argument("--ckpt_every", type=int, default=250)
ap.add_argument("--eval_n_E", type=int, default=16384); ap.add_argument("--eval_M", type=int, default=8192)
ap.add_argument("--max_hours", type=float, default=None, help="corta (guardando) tras estas horas")
ap.add_argument("--init", default=None, help="msgpack de parámetros para continuar")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--label", required=True)
ap.add_argument("--out_dir", default="data/gap_arch/phys_bridge"); ap.add_argument("--no_x64", action="store_true")
a = ap.parse_args()

import numpy as np
import jax
jax.config.update("jax_enable_x64", not a.no_x64)
import jax.numpy as jnp
import netket as nk
import optax
import flax.serialization as ser
from netket.operator.spin import sigmax, sigmaz
from src_renyi.physical_ancilla.models_encdec import ARPhysEncDec, ARPhysPrefix
from src_renyi.physical_ancilla.bridge import renyi2_bridge, hermiticity
from mre_teff import MRE, spectrum

assert "/data/0/" not in os.path.abspath(a.out_dir) + "/"
os.makedirs(a.out_dir, exist_ok=True)
base = os.path.join(a.out_dir, a.label)
N, T = a.N, a.T
dt = jnp.float32 if a.no_x64 else jnp.float64
hi = nk.hilbert.Spin(1 / 2, 2 * N)
sys_sites, anc_sites = list(range(N)), list(range(N, 2 * N))

# ── referencia MRE ───────────────────────────────────────────────────────────────────────────────────────────────
if (a.J, a.h) == (-1.0, -1.5):
    Ev = spectrum(N)                                          # fermiones libres (mismo espectro)
else:
    assert N <= 14, "MRE con J/h distintos solo por ED densa (N <= 14)"
    hs = nk.hilbert.Spin(1 / 2, N)
    Ev = np.linalg.eigvalsh(sum(a.J * sigmaz(hs, i) @ sigmaz(hs, (i + 1) % N) + a.h * sigmax(hs, i)
                                for i in range(N)).to_dense())
E_mre, S2_mre, F_mre = MRE(Ev)(T)

# ── modelo ───────────────────────────────────────────────────────────────────────────────────────────────────────
if a.arch == "encdec":
    model = ARPhysEncDec(hilbert=hi, embedding_d=a.d, n_heads=a.heads, sys_blocks=a.sys_blocks,
                         enc_blocks=a.enc_blocks, dec_blocks=a.dec_blocks, copy_init=a.copy_init, param_dtype=dt,
                         phase_net=a.phase_net)
else:
    model = ARPhysPrefix(hilbert=hi, embedding_d=a.d, n_heads=a.heads, n_blocks=a.blocks, copy_init=a.copy_init,
                         param_dtype=dt, phase_net=a.phase_net)
# sin chunk_size en NetKet: su expect_and_grad troceado usa jax.lax.pvary, eliminado en JAX 0.10
vs = nk.vqs.MCState(nk.sampler.ARDirectSampler(hi), model, n_samples=a.n_E, seed=a.seed, sampler_seed=a.seed + 1)
params = vs.parameters
if a.init:
    params = ser.from_bytes(params, open(a.init, "rb").read())
n_par = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def _is_phase(path):
    keys = [getattr(k, "key", str(k)) for k in path]
    return "phase" in keys or "theta" in keys


def _zero_phase_out(path, x):          # fase inicial 0: cabezas de fase y capa de salida de la red de fase a cero
    keys = [getattr(k, "key", str(k)) for k in path]
    return jnp.zeros_like(x) if ("phase" in keys or ("theta" in keys and "out" in keys)) else x


if a.phase_after > 0:
    params = jax.tree_util.tree_map_with_path(_zero_phase_out, params)
mask_phase = jax.jit(lambda g: jax.tree_util.tree_map_with_path(lambda pth, x: jnp.zeros_like(x) if _is_phase(pth) else x, g))
print(f"[{a.label}] N={N} T={T} arch={a.arch} mode={a.mode} n_par={n_par} copy_init={a.copy_init} lam={a.lam} "
      f"dispositivos={jax.devices()}", flush=True)
print(f"[{a.label}] MRE: E={E_mre:.5f} S2={S2_mre:.5f} F={F_mre:.5f}", flush=True)

# ── exacto (N pequeño) ───────────────────────────────────────────────────────────────────────────────────────────
exact_fn = None
if N <= 8:
    cfg = jnp.asarray(hi.all_states())                       # el sitio 0 es el bit más significativo
    Hs = jnp.asarray(sum(a.J * sigmaz(nk.hilbert.Spin(1 / 2, N), i) @ sigmaz(nk.hilbert.Spin(1 / 2, N), (i + 1) % N)
                         + a.h * sigmax(nk.hilbert.Spin(1 / 2, N), i) for i in range(N)).to_dense())

    def exact_quantities(p):
        psi = jnp.exp(model.apply({"params": p}, cfg).astype(jnp.result_type(1j)))
        Psi = psi.reshape(2 ** N, 2 ** N)
        if a.herm:                                           # ρ real -> basta Ψ real simétrica (ρ^{1/2} lo es)
            Psi = 0.5 * (Psi + Psi.T).real.astype(Psi.dtype)
        Psi = Psi / jnp.sqrt(jnp.sum(jnp.abs(Psi) ** 2))
        rho = Psi @ Psi.conj().T
        E = jnp.real(jnp.trace(Hs @ rho)); S2 = -jnp.log(jnp.sum(jnp.abs(rho) ** 2))
        return E - T * S2, E, S2, jnp.real(jnp.sum(Psi * Psi.T))
    exact_fn = jax.jit(exact_quantities)

    def bridge_J(p):                                         # = J de scripts/gauge_opt.py (n_a = 1)
        psi = jnp.exp(model.apply({"params": p}, cfg).astype(jnp.result_type(1j)))
        Psi = psi.reshape(2 ** N, 2 ** N) / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2))
        D = 2 ** N; w = jnp.abs(Psi) ** 2; ps = w.sum(1); ph2 = w / ps[:, None]
        P = jnp.sum(jnp.abs(Psi @ Psi.conj().T) ** 2)
        lr = jnp.zeros((D, D, D))
        for k in range(N):
            Mk = ph2.reshape(D, 2 ** k, -1).sum(-1); Mk1 = ph2.reshape(D, 2 ** (k + 1), -1).sum(-1)
            q1 = Mk1[:, 1::2] / jnp.maximum(Mk, 1e-300)
            g1 = jnp.sqrt(q1[:, None] * q1[None]); g0 = jnp.sqrt((1 - q1)[:, None] * (1 - q1)[None]); Z = g1 + g0
            pre = jnp.arange(D) >> (N - k); bit = (jnp.arange(D) >> (N - k - 1)) & 1
            g = jnp.where(bit[None, None] == 1, g1[:, :, pre], g0[:, :, pre]) / Z[:, :, pre]
            lr = lr + jnp.log(jnp.maximum(g, 1e-300))
        S = jnp.sum(ph2[:, None] * ph2[None] * jnp.exp(-lr), -1)
        return jnp.sum(ps[:, None] * ps[None] * S ** 2) / P - P
    bridge_J_fn = jax.jit(bridge_J)
elif a.mode == "exact" or a.herm:
    sys.exit("--mode exact solo para N <= 8")

# ── optimizador ──────────────────────────────────────────────────────────────────────────────────────────────────
sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, min(a.warmup, max(a.steps // 5, 1)), a.steps, a.lr * a.end_frac)
tx = optax.chain(optax.clip_by_global_norm(a.clip), optax.adam(sched))
opt_state = tx.init(params)
ema = jax.tree_util.tree_map(jnp.zeros_like, params)
upd_ema = jax.jit(lambda e, p: jax.tree_util.tree_map(lambda x, y: a.ema * x + (1 - a.ema) * y, e, p))
ema_hat = lambda e, t: jax.tree_util.tree_map(lambda x: x / (1 - a.ema ** t), e)
apply_upd = jax.jit(lambda g, s, p: (lambda u, s2: (optax.apply_updates(p, u), s2))(*tx.update(g, s, p)))
real = lambda g: jax.tree_util.tree_map(jnp.real, g)

if a.mode == "exact":
    _pen = (lambda p: jnp.log(bridge_J(p))) if a.logJ else bridge_J
    gfun = jax.jit(jax.value_and_grad(lambda p, mu: (lambda F, E, S2, Hh: F - a.lam * Hh)(*exact_quantities(p))
                                      + (mu * _pen(p) if a.mu else 0.0)))


def mu_at(t):
    """μ(t): rampa lineal 0 -> μ en mu_ramp pasos, luego lineal hasta μ·mu_end en el último paso."""
    if t <= a.mu_ramp:
        return a.mu * t / a.mu_ramp
    f = (t - a.mu_ramp) / max(a.steps - a.mu_ramp, 1)
    return a.mu * (1 + (a.mu_end - 1) * f)


# ── energía MC propia (troceada): E_loc(x) = J Σ s_i s_{i+1} + h Σ_i ψ(flip_i x)/ψ(x), solo sitios del sistema ────
flips = jnp.asarray(np.concatenate([np.zeros((1, 2 * N)), np.eye(2 * N)[:N]]), dtype=bool)   # (N+1, 2N): x y flips


def _lpsi(p, x):
    return model.apply({"params": p}, x).astype(jnp.result_type(1j))


def _eloc_chunk(p, x):
    xs = jnp.where(flips[None], -x[:, None, :], x[:, None, :]).reshape(-1, 2 * N)
    lp = _lpsi(p, xs).reshape(x.shape[0], N + 1)
    s = x[:, :N]
    return a.J * jnp.sum(s * jnp.roll(s, -1, axis=1), 1) + a.h * jnp.sum(jnp.exp(lp[:, 1:] - lp[:, :1]), 1)


@jax.jit
def energy_and_grad(p, x):
    nc = x.shape[0] // ec
    xc = x.reshape(nc, ec, -1)
    el = jax.lax.map(lambda c: _eloc_chunk(p, c), xc).reshape(-1)
    Em = jnp.mean(el)
    w = jax.lax.stop_gradient(jnp.conj(el - Em)).reshape(nc, ec)

    def loss(q):     # ∇E = 2 Re E[(E_loc - Ē) conj(∇ ln ψ)]  (parámetros reales)
        return jnp.sum(jax.lax.map(jax.checkpoint(lambda c: 2 * jnp.real(jnp.sum(jnp.conj(_lpsi(q, c[0])) * jnp.conj(c[1])))),
                                   (xc, w))) / x.shape[0]
    return el, jax.grad(loss)(p)


@jax.jit
def energy_only(p, x):
    return jax.lax.map(lambda c: _eloc_chunk(p, c), x.reshape(x.shape[0] // ec, ec, -1)).reshape(-1)


ec = max(1, min(a.e_chunk, a.n_E))
assert a.n_E % ec == 0 and a.eval_n_E % ec == 0, "e_chunk debe dividir a n_E y eval_n_E"


def mc_grad(p, t):
    vs.parameters = p
    vs.reset()
    x = jnp.asarray(vs.sample(n_samples=a.n_E)).reshape(-1, 2 * N)[: a.n_E]
    el, gE = energy_and_grad(p, x)
    el = np.real(np.asarray(el))
    Est = type("S", (), dict(mean=el.mean(), error_of_mean=el.std() / np.sqrt(len(el))))
    S2, gS, info = renyi2_bridge(vs, sys_sites, a.M, K=a.K, n_a=a.n_a, key=10 * t + a.seed * 7919,
                                 chunk_size=a.chunk or None, return_info=True, check_order=(t == 0))
    g = jax.tree_util.tree_map(lambda x, y: x - T * y, real(gE), real(gS))
    Hh = np.nan
    if a.lam > 0:
        Hh, gH = hermiticity(vs, sys_sites, anc_sites, a.n_E, key=10 * t + 3 + a.seed * 7919, chunk_size=a.chunk or None)
        g = jax.tree_util.tree_map(lambda x, y: x - a.lam * y, g, real(gH))
        Hh = float(Hh)
    E = float(Est.mean.real)
    return g, dict(E=E, E_err=float(Est.error_of_mean), S2=float(S2), S2_err=info["S2_err"], F=E - T * float(S2), H=Hh)


def save(p, e, t, hist, final=None):
    open(base + "_params.msgpack", "wb").write(ser.to_bytes(p))
    open(base + "_params_ema.msgpack", "wb").write(ser.to_bytes(ema_hat(e, max(t, 1))))
    json.dump(dict(args=vars(a), n_par=n_par, E_mre=E_mre, S2_mre=S2_mre, F_mre=F_mre, steps_done=t, hist=hist,
                   final=final), open(base + ".json", "w"), indent=1)


# ── bucle ────────────────────────────────────────────────────────────────────────────────────────────────────────
hist, t0, t_done = [], time.time(), 0
for t in range(1, a.steps + 1):
    if a.mode == "exact":
        _, g = gfun(params, mu_at(t))
        aux = {}
    else:
        g, aux = mc_grad(params, t)
    if t <= a.phase_after:
        g = mask_phase(g)
    params, opt_state = apply_upd(g, opt_state, params)
    ema = upd_ema(ema, params)
    t_done = t
    if t % a.log_every == 0 or t == 1 or t == a.steps:
        row = dict(step=t, s=time.time() - t0, **aux)
        msg = f"[{a.label}] {t:5d} {row['s'] / t:.2f}s/paso"
        if aux:
            msg += f" | MC F={aux['F']:.4f} E={aux['E']:.4f} S2={aux['S2']:.4f}±{aux['S2_err']:.4f} dF/N={(aux['F'] - F_mre) / N:+.5f}"
        if exact_fn is not None:
            F, E, S2, Hh = map(float, exact_fn(params))
            row.update(F_ex=F, E_ex=E, S2_ex=S2, H_ex=Hh)
            msg += f" | exacto F={F:.5f} dF/N={(F - F_mre) / N:+.6f} S2={S2:.4f} H={Hh:+.3f}"
            if a.mu or t == a.steps:
                row["J"] = float(bridge_J_fn(params)); msg += f" J={row['J']:.4g}"
        hist.append(row); print(msg, flush=True)
    if t % a.ckpt_every == 0:
        save(params, ema, t, hist)
    if a.max_hours and time.time() - t0 > 3600 * a.max_hours:
        print(f"[{a.label}] corte por tiempo en el paso {t}", flush=True)
        break

# ── evaluación final (últimos parámetros y EMA) ──────────────────────────────────────────────────────────────────
final = {}
for name, p in (("last", params), ("ema", ema_hat(ema, t_done))):
    r = {}
    if exact_fn is not None:
        F, E, S2, Hh = map(float, exact_fn(p))
        r.update(F_ex=F, E_ex=E, S2_ex=S2, H_ex=Hh, dF_ex=F - F_mre, dF_ex_per_site=(F - F_mre) / N)
    if a.mode == "mc" or exact_fn is None:
        vs.parameters = p
        vs.reset()
        x = jnp.asarray(vs.sample(n_samples=a.eval_n_E)).reshape(-1, 2 * N)[: a.eval_n_E]
        el = np.real(np.asarray(energy_only(p, x)))
        Est = type("S", (), dict(mean=el.mean(), error_of_mean=el.std() / np.sqrt(len(el))))
        cs = min(a.chunk or a.eval_M, a.eval_M)
        S2, _, info = renyi2_bridge(vs, sys_sites, a.eval_M, K=a.K, n_a=a.n_a, key=987654, compute_grad=False,
                                    chunk_size=cs, return_info=True)
        E = float(Est.mean.real); F = E - T * float(S2)
        r.update(E=E, E_err=float(Est.error_of_mean), S2=float(S2), S2_err=info["S2_err"], F=F, dF=F - F_mre,
                 dF_per_site=(F - F_mre) / N, F_err=float(np.hypot(float(Est.error_of_mean), T * info["S2_err"])))
    final[name] = r
    print(f"[{a.label}] FINAL {name}: " + "  ".join(f"{k}={v:+.5f}" if isinstance(v, float) else f"{k}={v}"
                                                   for k, v in r.items()), flush=True)
if exact_fn is not None:                                     # Ψ final (filas s, columnas a) para medir el puente
    _psi = jnp.exp(model.apply({"params": params}, cfg).astype(jnp.result_type(1j))).reshape(2 ** N, 2 ** N)
    if a.herm:
        _psi = 0.5 * (_psi + _psi.T).real
    np.save(base + "_psi.npy", np.asarray(_psi / jnp.sqrt(jnp.sum(jnp.abs(_psi) ** 2))))
save(params, ema, t_done, hist, final)
print(f"[{a.label}] guardado {base}.json  ({(time.time() - t0) / 60:.1f} min)", flush=True)
