"""
diagnose_mps.py
===============
Diagnostico PREVIO a un entrenamiento largo con el modelo ARViT (o variantes de models_alt) + MPS ancilla.
Equivalente de scripts/diagnose.py (arquitectura physical_ancilla / NetKet) para src_renyi/mps_ancilla.
Mismo estilo de CLI que scripts/gap_mc.py; al final imprime los flags recomendados para gap_mc.py.

Secciones (--sections 1,2,...; cada una protegida con try/except: si una falla, sigue el resto):
  0. (implicita) calentamiento opcional (--warm_steps): unos pasos de train_mc para no diagnosticar el estado
     copy_init (rho = I/2^N), donde la parte lejana de la pureza es 0 y el ruido no es representativo.
  1. Resumen del modelo: arquitectura, #params, bytes_per_sample, auto_chunks para M y eval_M.
  2. Barrido de memoria: menor n_chunks (troceado de muestras de make_mc_vg) que no da OOM, con la
     memory_analysis de XLA de cada candidato; compara con auto_chunks(--mem_gb) y sugiere --mem_gb.
  3. Tiempo por paso (sin compilacion) para cada --time_purities, tiempo de un lote de evaluacion y
     tiempo total previsto para --steps + --polish_steps + evaluacion.
  4. Normas del gradiente en K pasos (con Adam y el warmup de train_mc) -> mediana/p90/p95/max y clip sugerido.
  5. Ruido del estimador frente a M: S2 (=-log media P), std S2, std ||g||, cos entre repeticiones, cos y beta
     con un gradiente de referencia (ED exacto si N<=10; si no, media de gradientes a --ref_M);
     ruido_rel = sqrt(<|g|^2> - |g_verd|^2)/|g_verd|; ajustes std||g|| ~ M^-a y ruido_rel ~ M^-a;
     M sugerida = la que da ruido_rel = --rel_target.
     5b. Colas de la pureza (como s2_bias_check.py): fracciones de P en d=0,1,2,>=3, CV por lote de cada parte,
         S2 nbr1 (exacto d<=1) vs nbr2 (exacto d<=2) y S2 acumulada al anadir lotes (sesgo de colas).
  6. Cadenas swap (como swap_warm_mps.py): aceptacion por movimiento, beta / cos con la referencia en funcion
     de la longitud de las cadenas -> swap_warm sugerido.
  7. Barrido corto de lr (--lr_sweep 1e-3,3e-3,5e-3 --sweep_steps 200): train_mc desde el mismo estado, F por MC
     (y por ED si N<=10) de los ultimos parametros y de la EMA.
  8. N <= 10: verificacion con ED: rho exacta del modelo vs estimaciones MC de E y S2 (sesgo de los
     estimadores), cos/beta del gradiente MC con el exacto, y hueco con el MRE exacto.
  9. Resumen final + JSON (--out) + flags recomendados para gap_mc.py.

Uso (todo opcional salvo N, T, chi):
  # diagnostico completo por defecto a N=25 (GPU) desde los parametros de un entrenamiento
  python scripts/diagnose_mps.py --N 25 --T 3.0 --chi 4 --init data/0/mps_N25/params/params_0030_chi4.msgpack:4 \
      --M 1024 --pair_block 8 --purity nbr --steps 3000
  # una variante de arquitectura desde cero, solo memoria y tiempos
  python scripts/diagnose_mps.py --N 25 --T 3.0 --chi 4 --arch vit:conv:r=1 --sections 1,2,3
  # N pequeno con ED y barrido de lr
  JAX_PLATFORMS=cpu python scripts/diagnose_mps.py --N 8 --T 3.0 --chi 4 --M 256 --warm_steps 200 \
      --sections 1,3,4,5,7,8 --lr_sweep 1e-3,5e-3 --sweep_steps 100
"""
import argparse, contextlib, io, itertools, json, os, sys, time, traceback
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from src_renyi.mps_ancilla.training import train_mc, make_loss_fns, make_sampler
from src_renyi.mps_ancilla.mc_loss import make_mc_vg, auto_chunks, bytes_per_sample
from src_renyi.mps_ancilla.models_alt import build_alt_model
from src_renyi.mps_ancilla.models import count_params, mps_overlap_aligned, mps_overlap_rect
from src_renyi.mps_ancilla.ladder import build_model, init_params, pad_chi, load_params, make_evaluator
from src_renyi.mps_ancilla.swap_mc import make_swap_stepper, make_swap_chains, make_swap_loss, SWAP_TRAIN_MOVES, MOVES

# ══════════════════════════════════════════════════════════════════════════════
# argumentos
# ══════════════════════════════════════════════════════════════════════════════
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
# modelo / estado (como gap_mc.py)
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--T", type=float, required=True)
ap.add_argument("--h_x", type=float, default=-1.5)
ap.add_argument("--J_ZZ", type=float, default=-1.0)
ap.add_argument("--chi", type=int, required=True)
ap.add_argument("--d", type=int, default=16)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--arch", default=None, help="variante de models_alt (p.ej. vit:conv:r=1); por defecto el ARViT de siempre")
ap.add_argument("--init", default="scratch", help="scratch | fichero.msgpack:chi0 (como gap_mc.py; amplia con pad_chi si chi>chi0)")
ap.add_argument("--params", default=None, help="atajo de --init fichero:chi (parametros guardados con el mismo chi)")
ap.add_argument("--noise", type=float, default=0.02, help="ruido de pad_chi (como gap_mc.py)")
ap.add_argument("--seed", type=int, default=1)
# entrenamiento previsto (lo que se pasaria a gap_mc.py)
ap.add_argument("--M", type=int, default=1024)
ap.add_argument("--pair_block", type=int, default=8)
ap.add_argument("--purity", default="nbr", choices=["iid", "nbr", "swap"])
ap.add_argument("--mem_gb", type=float, default=1.5)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--warmup", type=int, default=30)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--polish_steps", type=int, default=0)
ap.add_argument("--polish_M", type=int, default=None)
ap.add_argument("--clip", type=float, default=10.0, help="clip de train_mc (por defecto 10; gap_mc.py no lo expone)")
ap.add_argument("--eval_M", type=int, default=4096)
ap.add_argument("--eval_batches", type=int, default=16)
# swap (valores por defecto de train_mc)
ap.add_argument("--swap_chains", type=int, default=256)
ap.add_argument("--swap_group", type=int, default=256)
ap.add_argument("--swap_stride", type=int, default=20)
ap.add_argument("--swap_warm", type=int, default=100)
# control del diagnostico
ap.add_argument("--sections", default=None,
                help="lista, p.ej. 1,2,3 o 'all'.  Por defecto 1,2,3,4,5,8 (+6 si purity=swap, +7 si --lr_sweep); 9 siempre")
ap.add_argument("--warm_steps", type=int, default=None,
                help="pasos de train_mc antes de las secciones 4-8 (por defecto 100 si --init scratch, 0 si se cargan params)")
ap.add_argument("--scan_max", type=int, default=6, help="sec 2: maximo de candidatos de n_chunks probados")
ap.add_argument("--mem_limit_gb", type=float, default=None,
                help="sec 2: limite de memoria para considerar OOM por la estimacion de XLA (por defecto 90%% del limite de la GPU; en CPU ninguno)")
ap.add_argument("--time_purities", default=None, help="sec 3: estimadores a cronometrar, p.ej. nbr,iid,swap (por defecto --purity)")
ap.add_argument("--n_time", type=int, default=5, help="sec 3: pasos cronometrados")
ap.add_argument("--grad_K", type=int, default=20, help="sec 4: pasos para las normas del gradiente")
ap.add_argument("--noise_Ms", default=None, help="sec 5: lista de M (por defecto M/4,M/2,M,2M)")
ap.add_argument("--noise_reps", type=int, default=8)
ap.add_argument("--ref_M", type=int, default=None, help="sec 5/6: M de la referencia sin ED (por defecto 4M)")
ap.add_argument("--ref_reps", type=int, default=4)
ap.add_argument("--eps_grad", type=float, default=0.1, help="sec 5: M sugerida tal que std||g|| = eps |g|")
ap.add_argument("--rel_target", type=float, default=1.0,
                help="sec 5: M sugerida tal que ruido_rel = sqrt(<|g|^2>-|g|^2)/|g| = rel_target (1 = SNR 1)")
ap.add_argument("--bias_M", type=int, default=None, help="sec 5b: muestras por lote (por defecto M)")
ap.add_argument("--bias_batches", type=int, default=32)
ap.add_argument("--cfg_chunk", type=int, default=8192, help="sec 5b: configuraciones por trozo (memoria)")
ap.add_argument("--swap_lengths", default="20,60,120,300", help="sec 6: checkpoints (multiplos de --swap_stride)")
ap.add_argument("--swap_reps", type=int, default=4)
ap.add_argument("--lr_sweep", default=None, help="sec 7: lista de lr, p.ej. 1e-3,3e-3,5e-3")
ap.add_argument("--sweep_steps", type=int, default=200)
ap.add_argument("--sweep_eval_batches", type=int, default=4)
ap.add_argument("--ed_M", type=int, default=None, help="sec 8: M de los gradientes MC comparados con el exacto (por defecto M)")
ap.add_argument("--out", default=None, help="JSON de salida (por defecto data/diagnose/diag_...json)")
args = ap.parse_args()

N, T, chi, hx, J, pb = args.N, args.T, args.chi, args.h_x, args.J_ZZ, args.pair_block
if args.params:
    args.init = f"{args.params}:{chi}"
arch_tag = (args.arch or "vit").replace(":", "-").replace("=", "").replace(",", "_")
out_file = args.out or os.path.join(script_dir, "..", "data", "diagnose", f"diag_N{N}_chi{chi}_{arch_tag}_T{T:g}.json")
data0 = os.path.abspath(os.path.join(script_dir, "..", "data", "0")) + os.sep
assert not os.path.abspath(out_file).startswith(data0), "no escribir en data/0"

if args.sections in (None, ""):
    SECTIONS = {1, 2, 3, 4, 5, 8}
    if args.purity == "swap":
        SECTIONS.add(6)
    if args.lr_sweep:
        SECTIONS.add(7)
elif args.sections == "all":
    SECTIONS = set(range(1, 9))
else:
    SECTIONS = {int(x) for x in args.sections.split(",") if x.strip()}
if N > 10:
    SECTIONS.discard(8)
if 7 in SECTIONS and not args.lr_sweep:
    print("(seccion 7 pedida sin --lr_sweep: se usa 0.5*lr, lr, 2*lr)")
    args.lr_sweep = f"{args.lr / 2},{args.lr},{2 * args.lr}"

dev = jax.devices()[0]
ON_GPU = dev.platform == "gpu"
print(f"N={N} T={T} chi={chi} arch={args.arch or 'ARViT (models.py)'} d={args.d} blocks={args.blocks} init={args.init}\n"
      f"M={args.M} pair_block={pb} purity={args.purity} mem_gb={args.mem_gb} lr={args.lr} steps={args.steps}\n"
      f"secciones={sorted(SECTIONS)}  dispositivos={jax.devices()}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# utilidades
# ══════════════════════════════════════════════════════════════════════════════
OOM_MARKERS = ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory", "OOM")
RES = {"args": vars(args), "device": str(dev), "sections": {}}


def is_oom(e):
    return any(m in str(e) for m in OOM_MARKERS)


class Section:
    """with Section(k, nombre): ... -> try/except; registra ok/error en RES['sections']."""
    def __init__(self, k, name):
        self.k, self.name = k, name

    def __enter__(self):
        self.t0 = time.time()
        print("\n" + "=" * 92 + f"\n{self.k}. {self.name}\n" + "=" * 92, flush=True)
        return self

    def __exit__(self, et, ev, tb):
        dt = time.time() - self.t0
        if et is None:
            RES["sections"][str(self.k)] = {"status": "ok", "time_s": dt}
            print(f"<<< fin seccion {self.k} ({dt:.0f} s)", flush=True)
            return False
        if et is KeyboardInterrupt:
            return False
        why = "OOM" if is_oom(ev) else f"{et.__name__}"
        print(f"!!! seccion {self.k} abortada por {why}: {str(ev)[:400]}", flush=True)
        print("    " + "".join(traceback.format_exception(et, ev, tb, limit=4)).replace("\n", "\n    ").rstrip())
        RES["sections"][str(self.k)] = {"status": f"error {why}: {str(ev)[:300]}", "time_s": dt}
        return True


def flat(g):
    return np.asarray(ravel_pytree(g)[0])


def cosm(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def mean_pair_cos(gs):
    c = [cosm(gs[a], gs[b]) for a, b in itertools.combinations(range(len(gs)), 2)]
    return (float(np.mean(c)), float(np.std(c))) if c else (float("nan"), float("nan"))


def mem_report(label=""):
    """Pico de memoria: GPU (memory_stats) y proceso (VmHWM de /proc si existe, si no ru_maxrss)."""
    out = {}
    try:
        st = jax.local_devices()[0].memory_stats()
        if st:
            out.update(gpu_peak_gb=st.get("peak_bytes_in_use", 0) / 2 ** 30,
                       gpu_in_use_gb=st.get("bytes_in_use", 0) / 2 ** 30,
                       gpu_limit_gb=st.get("bytes_limit", 0) / 2 ** 30)
    except Exception:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    out["proc_hwm_gb"] = int(line.split()[1]) / 2 ** 20
    except Exception:
        pass
    if "proc_hwm_gb" not in out:
        try:
            import resource
            out["proc_hwm_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20   # kB en Linux
        except Exception:
            pass
    msg = " | ".join(f"{k}={v:.2f}" for k, v in out.items())
    print(f"  memoria {label}: {msg or 'no disponible'}", flush=True)
    return out


def gpu_limit_gb():
    try:
        st = jax.local_devices()[0].memory_stats()
        return st.get("bytes_limit", 0) / 2 ** 30 if st else None
    except Exception:
        return None


def round_M(M):
    return max(pb, int(M) // pb * pb)


def mem_gb_for(M, n):
    """menor --mem_gb (2 decimales) con el que auto_chunks(M) da <= n trozos."""
    g = np.ceil(bytes_per_sample(N, chi, True) * (M / n) / 2 ** 30 * 100) / 100
    for _ in range(200):
        if auto_chunks(M, N, chi, pb, None, g, grad=True) <= n:
            return float(g)
        g = round(g + 0.01, 2)
    return float(g)


@contextlib.contextmanager
def quiet(prefix="    | ", drop=("cadenas swap arrancadas",)):
    """Captura la salida de train_mc y la reimprime filtrada (train_mc imprime 'cadenas swap arrancadas' en cada
    paso cuando purity != swap)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield
    for line in buf.getvalue().splitlines():
        if not any(d in line for d in drop):
            print(prefix + line)
    sys.stdout.flush()


# ── modelo y parametros (como gap_mc.py) ──────────────────────────────────────
def mk(c):
    return (build_alt_model(N, c, args.arch, args.d, args.heads, args.blocks) if args.arch
            else build_model(N, c, args.d, args.heads, args.blocks))


model = mk(chi)
if args.init == "scratch":
    params0 = init_params(model, N, args.seed)
else:
    path, chi0 = args.init.rsplit(":", 1)
    chi0 = int(chi0)
    assert args.arch is None or chi0 == chi, "pad_chi solo vale para el ARViT de siempre"
    p0 = load_params(path, mk(chi0), N)
    params0 = p0 if chi0 == chi else pad_chi(p0, chi0, chi, N, args.noise / np.sqrt(args.d), args.seed)
n_params = count_params(params0)
est_purity = "nbr" if args.purity == "swap" else args.purity        # estimador de valor / lote (swap -> nbr)

# ── caches de funciones compiladas ────────────────────────────────────────────
_samplers, _vgs = {}, {}


def get_sampler(M):
    if M not in _samplers:
        _samplers[M] = make_sampler(model, N, M)
    return _samplers[M]


def sample(params, key, M, gsz=None):
    """M muestras exactas; en lotes de gsz (el muestreador no trocea) si M es grande."""
    gsz = gsz or max(args.M, pb)
    if M <= gsz or M % gsz:
        return get_sampler(M)(params, key)
    smp = get_sampler(gsz)
    return jnp.concatenate([smp(params, jax.random.fold_in(key, k)) for k in range(M // gsz)], 0)


def get_vg(M, purity, terms="both", Tv=T, nch=None):
    if nch is None:
        nch = auto_chunks(M, N, chi, pb, None, MEM_GB, grad=True)
    k = (M, purity, terms, Tv, nch)
    if k not in _vgs:
        _vgs[k] = jax.jit(make_mc_vg(model, N, Tv, hx, J, None, pb, terms=terms, purity=purity, n_chunks=nch))
    return _vgs[k], nch


def make_train_step(M, purity, sched, nch):
    """Paso de entrenamiento como el de train_mc (sin EMA), devolviendo ademas ||g|| antes del clip."""
    sampler = get_sampler(M)
    swap = purity == "swap"
    vg = make_mc_vg(model, N, T, hx, J, None, pb, terms="E" if swap else "both",
                    purity="nbr" if swap else purity, n_chunks=nch)
    tx = optax.chain(optax.clip_by_global_norm(args.clip), optax.adam(sched))
    if swap:
        sw_init, sw_grad = make_swap_stepper(model, N, T, args.swap_chains, args.swap_group, args.swap_stride,
                                             args.swap_warm)
        chain_sampler = make_sampler(model, N, args.swap_group)

    @jax.jit
    def step(params, opt_state, key, chains):
        key, sub, sub2 = jax.random.split(key, 3)
        s = sampler(params, sub)
        (_, aux), g = vg(params, s)
        acc = {}
        if swap:
            gS, cs, ct, acc = sw_grad(params, sub2, chains[0], chains[1])
            g = jax.tree_util.tree_map(jnp.add, g, gS)
            chains = (cs, ct)
        gn = optax.global_norm(g)
        upd, opt_state = tx.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, key, chains, aux, gn, acc

    def init_chains(params, key):
        return sw_init(params, key, chain_sampler) if swap else None
    return step, tx, init_chains


def default_sched(n_steps):
    return optax.warmup_cosine_decay_schedule(0.0, args.lr, min(args.warmup, max(n_steps // 5, 1)), n_steps,
                                              args.lr * 1e-2)


# ── ED (N <= 10), perezoso ────────────────────────────────────────────────────
_ED = {}


def ed():
    """dict con ref MRE exacta y funciones exactas F,E,S2 y gradientes (o None si N > 10)."""
    if N > 10:
        return None
    if _ED:
        return _ED
    from scipy.optimize import brentq
    idx = np.arange(2 ** N); bits = (idx[:, None] >> np.arange(N)[None, :]) & 1
    all_s = (2 * bits - 1).astype(np.float64)
    H = np.zeros((2 ** N, 2 ** N))
    H[idx, idx] = J * sum(all_s[:, i] * all_s[:, (i + 1) % N] for i in range(N))
    for i in range(N):
        H[idx, idx ^ (1 << i)] += hx
    ev = np.linalg.eigvalsh(H)

    def sc(c):
        w = np.maximum(c - ev, 0); return c - (w * ev).sum() / w.sum() - 2 * T
    c = brentq(sc, ev.min() + 1e-9, ev.max() + 20 * T + 10, xtol=1e-12)
    w = np.maximum(c - ev, 0); p = w / w.sum()
    E_ex, S2_ex = (p * ev).sum(), -np.log((p ** 2).sum())
    loss, _ = make_loss_fns(model, all_s, H, T, chunk=min(64, 2 ** N))
    _ED.update(ref=(float(E_ex - T * S2_ex), float(E_ex), float(S2_ex)),
               loss=jax.jit(loss),
               gradF=jax.jit(jax.grad(lambda q: loss(q)[0])),                   # grad F
               gradlogP=jax.jit(jax.grad(lambda q: -loss(q)[1][1])))            # grad log P = -grad S2
    return _ED


def exact_fn(p):
    F, (E, S2) = ed()["loss"](p)
    return float(F), float(E), float(S2)


MEM_GB = args.mem_gb            # se actualiza con la recomendacion de la seccion 2
REC = {}                        # recomendaciones para el resumen

# ══════════════════════════════════════════════════════════════════════════════
# 1. resumen del modelo
# ══════════════════════════════════════════════════════════════════════════════
if 1 in SECTIONS:
    with Section(1, "Resumen del modelo"):
        bps_g, bps_f = bytes_per_sample(N, chi, True), bytes_per_sample(N, chi, False)
        nch_M = auto_chunks(args.M, N, chi, pb, None, args.mem_gb, grad=True)
        nch_E = auto_chunks(args.eval_M, N, chi, pb, None, args.mem_gb, grad=False)
        leaves = {jax.tree_util.keystr(k): int(v.size) for k, v in jax.tree_util.tree_flatten_with_path(params0)[0]}
        top = {}
        for k, v in leaves.items():
            top[k.split("]")[0] + "]"] = top.get(k.split("]")[0] + "]", 0) + v
        print(f"  arquitectura: {type(model).__name__} {args.arch or ''}  #params = {n_params:,}")
        print("  params por modulo: " + ", ".join(f"{k.strip('[]')}={v:,}" for k, v in sorted(top.items(), key=lambda x: -x[1])))
        print(f"  bytes_per_sample (estimacion mc_loss, ajustada para el ARViT): grad {bps_g / 2 ** 20:.2f} MiB | fwd {bps_f / 2 ** 20:.2f} MiB")
        print(f"  M={args.M}: memoria estimada sin trocear {args.M * bps_g / 2 ** 30:.2f} GiB -> auto_chunks(mem_gb={args.mem_gb}) = "
              f"{nch_M} trozos de {args.M // nch_M}")
        print(f"  eval_M={args.eval_M}: {args.eval_M * bps_f / 2 ** 30:.2f} GiB sin trocear -> {nch_E} trozos (fwd)")
        print(f"  evaluaciones del ansatz por paso: M*(1+N) = {args.M * (1 + N):,}"
              + (" (+ cadenas swap)" if args.purity == "swap" else ""))
        RES["model"] = dict(arch=args.arch, n_params=n_params, params_by_module=top, bytes_per_sample_grad=bps_g,
                            bytes_per_sample_fwd=bps_f, auto_chunks_M=nch_M, auto_chunks_eval=nch_E)
        if args.arch:
            print("  (aviso: bytes_per_sample se ajusto con el ARViT; para otras arquitecturas fiate mas de la seccion 2)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. barrido de memoria / trozos
# ══════════════════════════════════════════════════════════════════════════════
if 2 in SECTIONS:
    with Section(2, f"Memoria: menor n_chunks sin OOM (purity={est_purity}, M={args.M}, gradiente)"):
        M = round_M(args.M)
        u = M // pb
        divs = [n for n in range(1, u + 1) if u % n == 0]
        cands, last = [], 0
        for n in divs:                              # divisores, aproximadamente doblando
            if n >= 2 * last or n == divs[-1]:
                cands.append(n); last = n
        lim = args.mem_limit_gb if args.mem_limit_gb is not None else (0.9 * gpu_limit_gb() if gpu_limit_gb() else None)
        auto_n = auto_chunks(M, N, chi, pb, None, args.mem_gb, grad=True)
        print(f"  candidatos (divisores de M/pair_block={u}): {cands}   auto_chunks(mem_gb={args.mem_gb}) = {auto_n}")
        print(f"  limite para OOM estimado: {f'{lim:.2f} GiB' if lim else 'ninguno (CPU): solo OOM real'}")
        s = sample(params0, jax.random.PRNGKey(0), M)
        scan, best, tried = [], None, 0
        for n in cands:
            if tried >= args.scan_max:
                break
            tried += 1
            row = {"n_chunks": n, "chunk": M // n}
            try:
                vgf = make_mc_vg(model, N, T, hx, J, None, pb, terms="E" if args.purity == "swap" else "both",
                                 purity=est_purity, n_chunks=n)
                t0 = time.time()
                comp = jax.jit(vgf).lower(params0, s).compile()
                row["compile_s"] = time.time() - t0
                try:
                    ma = comp.memory_analysis()
                    row["xla_temp_gb"] = ma.temp_size_in_bytes / 2 ** 30
                    row["xla_total_gb"] = (ma.temp_size_in_bytes + ma.argument_size_in_bytes + ma.output_size_in_bytes) / 2 ** 30
                except Exception:
                    pass
                if lim and row.get("xla_total_gb", 0) > lim:
                    row["status"] = "OOM (estimado XLA)"
                else:
                    t0 = time.time()
                    jax.block_until_ready(comp(params0, s))
                    row["run_s"] = time.time() - t0
                    row["status"] = "ok"
            except Exception as e:
                row["status"] = "OOM" if is_oom(e) else f"error {type(e).__name__}: {str(e)[:120]}"
            scan.append(row)
            print(f"  n_chunks={n:4d} (trozo {M // n:5d}) | XLA temp={row.get('xla_temp_gb', float('nan')):7.3f} GiB "
                  f"total={row.get('xla_total_gb', float('nan')):7.3f} GiB | compila {row.get('compile_s', float('nan')):5.1f} s "
                  f"| ejecuta {row.get('run_s', float('nan')):6.2f} s | {row['status']}", flush=True)
            if row["status"] == "ok":
                best = row
                break
            if not row["status"].startswith("OOM"):
                break
        mem = mem_report("tras el barrido")
        RES["memory"] = {"scan": scan, "auto_chunks": auto_n, "mem": mem}
        if best:
            n = best["n_chunks"]
            # margen: si XLA dice que el trozo usa > 80% del limite, recomendar el siguiente candidato
            if lim and best.get("xla_total_gb", 0) > 0.8 * lim:
                nxt = [c for c in cands if c > n]
                n = nxt[0] if nxt else n
                print(f"  (el minimo usa >80% del limite: recomiendo {n} trozos por seguridad)")
            rec = mem_gb_for(M, n)
            print(f"  -> menor n_chunks sin OOM: {best['n_chunks']}; recomendado {n} trozos de {M // n} "
                  f"=> --mem_gb {rec} (auto_chunks con --mem_gb {args.mem_gb} daria {auto_n})")
            if ON_GPU or lim:
                REC["mem_gb"] = rec
                MEM_GB = rec            # el resto de secciones trocea con la recomendacion
            else:
                print(f"  (en CPU sin --mem_limit_gb no hay OOM real: no se cambia --mem_gb={args.mem_gb}; "
                      f"repite en GPU para un valor util)")
            RES["memory"].update(n_min=best["n_chunks"], n_rec=n, mem_gb_rec=rec)
        else:
            print("  -> ningun candidato cabe: reduce M o aumenta --scan_max")

# ══════════════════════════════════════════════════════════════════════════════
# 0. calentamiento (estado diagnosticado por 4-8)
# ══════════════════════════════════════════════════════════════════════════════
warm_steps = args.warm_steps if args.warm_steps is not None else (100 if args.init == "scratch" else 0)
params = params0
if warm_steps > 0 and SECTIONS & {4, 5, 6, 8}:
    with Section(0, f"Calentamiento: {warm_steps} pasos de train_mc (purity={est_purity}) antes de 4-8"):
        with quiet():
            o = train_mc(model, params0, N, T, hx, J, M=round_M(args.M), n_steps=warm_steps, lr=args.lr,
                         warmup=args.warmup, seed=args.seed + 17, pair_block=pb, freq=max(warm_steps // 4, 1),
                         ema_decay=0.995, purity=est_purity, mem_gb=MEM_GB)
        params = o["params"]
        h = o["hist"]
        print(f"  F MC (ultimos 10 pasos) = {h[-10:, 2].mean():.4f}   S2 = {h[-10:, 1].mean():.4f}")
        if ed():
            print(f"  F exacto del estado calentado = {exact_fn(params)[0]:.5f}  (MRE {ed()['ref'][0]:.5f})")

# ══════════════════════════════════════════════════════════════════════════════
# 3. tiempos
# ══════════════════════════════════════════════════════════════════════════════
if 3 in SECTIONS:
    with Section(3, "Tiempo por paso, evaluacion y tiempo total previsto"):
        M = round_M(args.M)
        purs = (args.time_purities or args.purity).split(",")
        tim = {}
        for pur in purs:
            nch = auto_chunks(M, N, chi, pb, None, MEM_GB, grad=True)
            step, tx, init_chains = make_train_step(M, pur, default_sched(args.steps), nch)
            opt = tx.init(params0)
            key = jax.random.PRNGKey(3)
            t0 = time.time()
            chains = init_chains(params0, jax.random.PRNGKey(4))
            jax.block_until_ready(chains)
            t_chains = time.time() - t0
            t0 = time.time()
            out = step(params0, opt, key, chains)
            jax.block_until_ready(out)
            t_first = time.time() - t0
            p, opt, key, chains = out[0], out[1], out[2], out[3]
            ts = []
            for _ in range(args.n_time):
                t0 = time.time()
                out = step(p, opt, key, chains)
                jax.block_until_ready(out)
                ts.append(time.time() - t0)
                p, opt, key, chains = out[0], out[1], out[2], out[3]
            t_step = float(np.median(ts))
            tim[pur] = dict(s_per_step=t_step, compile_s=t_first - t_step, chains_init_s=t_chains if pur == "swap" else 0.0,
                            n_chunks=nch)
            print(f"  {pur:>4s}: {1e3 * t_step:8.1f} ms/paso (mediana de {args.n_time}; min {1e3 * min(ts):.1f}) | "
                  f"compilacion ~{t_first - t_step:.1f} s | trozos {nch}"
                  + (f" | arranque+calentamiento cadenas {t_chains:.1f} s" if pur == "swap" else ""), flush=True)
        # evaluacion
        nchE = auto_chunks(args.eval_M, N, chi, pb, None, MEM_GB, grad=False)
        one = make_evaluator(model, N, T, hx, J, args.eval_M, pb, purity=est_purity, n_chunks=nchE)
        t0 = time.time(); jax.block_until_ready(one(params0, jax.random.PRNGKey(5))); t_e0 = time.time() - t0
        t0 = time.time(); jax.block_until_ready(one(params0, jax.random.PRNGKey(6))); t_e = time.time() - t0
        print(f"  evaluacion: {t_e:.2f} s/lote (eval_M={args.eval_M}, {nchE} trozos; compilacion ~{t_e0 - t_e:.1f} s) "
              f"-> {args.eval_batches} lotes = {args.eval_batches * t_e:.0f} s")
        tm = tim[args.purity] if args.purity in tim else list(tim.values())[0]
        pM = args.polish_M or args.M
        t_pol = tm["s_per_step"] * pM / M                 # estimacion lineal en M
        total = (args.steps * tm["s_per_step"] + args.polish_steps * t_pol + args.eval_batches * t_e
                 + tm["compile_s"] * (2 if args.polish_steps else 1) + t_e0 + tm["chains_init_s"])
        # train_mc anade un float() por paso (sincroniza) y la EMA: margen de ~5%
        print(f"  previsto: {args.steps} pasos x {tm['s_per_step']:.3f} s"
              + (f" + {args.polish_steps} polish x {t_pol:.3f} s (M={pM}, lineal en M)" if args.polish_steps else "")
              + f" + evaluacion -> ~{total / 60:.1f} min ({total / 3600:.2f} h, +5% por EMA/sincronizacion)")
        mem_report("tras los tiempos")
        RES["timing"] = dict(per_purity=tim, eval_s_per_batch=t_e, eval_compile_s=t_e0 - t_e, total_s=total)
        REC["total_min"] = total / 60

# ══════════════════════════════════════════════════════════════════════════════
# 4. normas del gradiente
# ══════════════════════════════════════════════════════════════════════════════
if 4 in SECTIONS:
    with Section(4, f"Normas del gradiente ({args.grad_K} pasos con Adam desde el estado diagnosticado, purity={args.purity})"):
        M = round_M(args.M)
        nch = auto_chunks(M, N, chi, pb, None, MEM_GB, grad=True)
        step, tx, init_chains = make_train_step(M, args.purity, default_sched(args.steps), nch)
        p, opt, key = params, tx.init(params), jax.random.PRNGKey(44)
        chains = init_chains(p, jax.random.PRNGKey(45))
        norms, Fs = [], []
        for _ in range(args.grad_K):
            p, opt, key, chains, aux, gn, acc = step(p, opt, key, chains)
            norms.append(float(gn)); Fs.append(float(aux[2]))
        g = np.array(norms)
        med, p90, p95, mx = np.median(g), np.percentile(g, 90), np.percentile(g, 95), g.max()
        spread = mx / (med + 1e-12)
        clip_s = p90 if spread > 5 else p95
        print(f"  ||g||: mediana {med:.4f} | p90 {p90:.4f} | p95 {p95:.4f} | max {mx:.4f} (max/mediana {spread:.1f})")
        print(f"  F MC durante los {args.grad_K} pasos: {Fs[0]:.4f} -> {Fs[-1]:.4f}")
        frac = float((g > args.clip).mean())
        print(f"  clip actual (train_mc) = {args.clip}: recorta el {100 * frac:.0f}% de los pasos")
        print(f"  -> clip sugerido = {clip_s:.4f} ({'p90, cola larga' if spread > 5 else 'p95, distribucion compacta'})"
              + ("  [gap_mc.py no expone --clip: pasar clip= a train_mc]" if abs(clip_s - args.clip) / args.clip > 0.5 else ""))
        RES["grad_norms"] = dict(norms=norms, median=med, p90=p90, p95=p95, max=mx, clip_suggested=clip_s,
                                 frac_clipped_current=frac, F=Fs)
        REC["clip"] = clip_s

# ══════════════════════════════════════════════════════════════════════════════
# 5. ruido del estimador frente a M
# ══════════════════════════════════════════════════════════════════════════════
if 5 in SECTIONS:
    with Section(5, f"Ruido del estimador frente a M (purity={est_purity}, gradiente total E - T S2)"):
        Ms = ([round_M(int(x)) for x in args.noise_Ms.split(",")] if args.noise_Ms
              else sorted({round_M(args.M // 4), round_M(args.M // 2), round_M(args.M), round_M(2 * args.M)}))
        R = args.noise_reps
        # referencia
        ref_ok = True
        if ed():
            ref = flat(ed()["gradF"](params)); ref_sq = float(ref @ ref); ref_lab = "ED exacto"
        else:
            Mr = round_M(args.ref_M or 4 * args.M)
            vgr, _ = get_vg(Mr, est_purity)
            refs = [flat(vgr(params, sample(params, jax.random.PRNGKey(9000 + r), Mr))[1]) for r in range(args.ref_reps)]
            ref = np.mean(refs, 0)
            ref_sq = float(np.mean([refs[a] @ refs[b] for a, b in itertools.combinations(range(len(refs)), 2)]))
            ref_lab = f"media de {args.ref_reps} gradientes MC a M={Mr}"
            print(f"  referencia: {ref_lab}; cos entre referencias = {mean_pair_cos(refs)[0]:.3f}; "
                  f"|g_verd|^2 (pares distintos) = {ref_sq:.4g} vs |ref|^2 = {ref @ ref:.4g}")
            if ref_sq <= 0.1 * float(ref @ ref):
                print("  (aviso: |g_verd|^2 << |ref|^2: la referencia es casi todo ruido; cos_ref/beta/ruido_rel no son fiables "
                      "y no se recomienda M -> sube --ref_M/--ref_reps)")
                ref_ok = False
                ref_sq = max(ref_sq, 1e-3 * float(ref @ ref))
        rows = []
        print(f"  {'M':>6s} {'trozos':>6s} | {'S2':>8s} {'std S2':>7s} {'E':>9s} {'std E':>6s} | {'<|g|>':>8s} {'std|g|':>8s} "
              f"{'ruido_rel':>9s} | {'cos_self':>8s} {'cos_ref':>7s} {'beta':>6s} | {'s/llamada':>9s}")
        for Mi in Ms:
            vg, nch = get_vg(Mi, est_purity)
            vg(params, sample(params, jax.random.PRNGKey(1), Mi))               # compilacion
            S2s, Es, gs, ts = [], [], [], []
            for r in range(R):
                s = sample(params, jax.random.PRNGKey(100 * Mi + r), Mi)
                t0 = time.time()
                (_, aux), g = vg(params, s)
                g = flat(g); ts.append(time.time() - t0)
                Es.append(float(aux[0])); S2s.append(float(aux[1])); gs.append(g)
            nrm = np.array([np.linalg.norm(g) for g in gs])
            S2agg = float(-np.log(np.mean(np.exp(-np.array(S2s)))))
            rel = float(np.sqrt(max(np.mean(nrm ** 2) - ref_sq, 0.0) / ref_sq))
            row = dict(M=Mi, n_chunks=nch, S2=S2agg, S2_mean_batch=float(np.mean(S2s)), S2_std=float(np.std(S2s, ddof=1)),
                       E=float(np.mean(Es)), E_std=float(np.std(Es, ddof=1)), gnorm_mean=float(nrm.mean()),
                       gnorm_std=float(nrm.std(ddof=1)), rel_noise=rel, cos_self=mean_pair_cos(gs)[0],
                       cos_ref=float(np.mean([cosm(g, ref) for g in gs])),
                       beta=float(np.mean([g @ ref for g in gs]) / ref_sq),
                       beta_err=float(np.std([g @ ref for g in gs], ddof=1) / ref_sq / np.sqrt(R)), time_s=float(np.mean(ts)))
            rows.append(row)
            print(f"  {Mi:6d} {nch:6d} | {row['S2']:8.4f} {row['S2_std']:7.4f} {row['E']:9.4f} {row['E_std']:6.3f} | "
                  f"{row['gnorm_mean']:8.4f} {row['gnorm_std']:8.4f} {rel:9.3f} | {row['cos_self']:8.3f} {row['cos_ref']:7.3f} "
                  f"{row['beta']:6.3f}+-{row['beta_err']:.2f} | {row['time_s']:9.3f}", flush=True)
        print(f"  referencia: {ref_lab}; |g_ref| = {np.sqrt(ref_sq):.4f}.  ruido_rel = sqrt(<|g|^2> - |g_verd|^2)/|g_verd| "
              f"(1 => SNR 1); beta = <g.ref>/|g_verd|^2 (1 = insesgado)")
        print("  S2 = -log(media de P entre repeticiones); si 'S2' baja al subir M, hay sesgo de colas (ver 5b)")
        lm = np.log([r["M"] for r in rows])
        fit = {}
        try:
            a, b = np.polyfit(lm, np.log([r["gnorm_std"] for r in rows]), 1)
            gref = np.sqrt(ref_sq)
            M_opt = float(np.exp((np.log(args.eps_grad * gref) - b) / a)) if a < 0 else float("inf")
            fit.update(std_gnorm_slope=a, std_gnorm_C=float(np.exp(b)), M_opt_std=M_opt)
            print(f"  ajuste std||g|| = {np.exp(b):.3g} M^{a:.3f}   (ideal -0.5) -> M con std||g|| = {args.eps_grad}|g|: ~{M_opt:,.0f}")
            M_snr = float("inf")
            if all(r["rel_noise"] > 0 for r in rows):
                a2, b2 = np.polyfit(lm, np.log([r["rel_noise"] for r in rows]), 1)
                M_snr = float(np.exp((np.log(args.rel_target) - b2) / a2)) if a2 < 0 else float("inf")
                fit.update(rel_noise_slope=a2, rel_noise_C=float(np.exp(b2)), M_rel_target=M_snr)
                print(f"  ajuste ruido_rel = {np.exp(b2):.3g} M^{a2:.3f}  (ideal -0.5) -> M con ruido_rel={args.rel_target}: ~{M_snr:,.0f}")
            # la recomendacion usa ruido_rel (mas robusto que std||g||, que es un estadistico ruidoso con pocas reps)
            M_use = M_snr if np.isfinite(M_snr) else M_opt
            if np.isfinite(M_use) and ref_ok:
                Mrec = int(2 ** np.ceil(np.log2(max(M_use, pb))))
                Mrec = int(min(max(Mrec, pb * 16), 16 * args.M))
                REC["M"] = round_M(Mrec)
                print(f"  -> M sugerida ~ {REC['M']} (por ruido_rel; potencia de 2 acotada a [16 pair_block, 16 M]).  "
                      f"Cerca del optimo |g| es pequeno y la M necesaria crece: es normal que salga grande en un estado entrenado")
        except Exception as e:
            print(f"  (ajuste fallido: {e})")
        RES["noise"] = dict(reference=ref_lab, reference_reliable=ref_ok, ref_norm=float(np.sqrt(ref_sq)), rows=rows, fit=fit)

    with Section("5b", "Colas de la pureza: fracciones por distancia, nbr1 vs nbr2, S2 acumulada"):
        Mb = round_M(args.bias_M or args.M)
        nb_ = args.bias_batches
        pairs = np.array([(i, j) for i in range(N) for j in range(i + 1, N)])
        F1 = 1.0 - 2.0 * np.eye(N)
        F2 = np.ones((len(pairs), N)); F2[np.arange(len(pairs)), pairs[:, 0]] = -1; F2[np.arange(len(pairs)), pairs[:, 1]] = -1
        FL = jnp.asarray(np.concatenate([F1, F2])); nF = FL.shape[0]
        sc = max([c for c in range(1, Mb + 1) if Mb % c == 0 and c * nF <= max(args.cfg_chunk, nF)])

        @jax.jit
        def tail_stats(params, s):
            def near(sc_):                                      # (sc, N): d=0, d=1, d=2 exactos
                logA, Mt = model.apply({"params": params}, sc_)
                n2 = jnp.real(mps_overlap_aligned(Mt, Mt, chi))
                sf = (sc_[:, None, :] * FL[None]).reshape(-1, N)
                logAf, Mf = model.apply({"params": params}, sf)
                O = mps_overlap_aligned(Mf, jnp.repeat(Mt, nF, 0), chi)
                n2f = jnp.real(mps_overlap_aligned(Mf, Mf, chi))
                w = (jnp.exp(2 * jnp.real(logAf)) * jnp.abs(O) ** 2 / (n2f * jnp.repeat(n2, nF))).reshape(-1, nF)
                return jnp.stack([jnp.exp(2 * jnp.real(logA)).sum(), w[:, :N].sum(), w[:, N:].sum()])
            nr = jax.lax.map(near, s.reshape(-1, sc, N)).sum(0) / s.shape[0]
            _, Mt = model.apply({"params": params}, s)          # lejana: parejas iid en bloques de pair_block
            n2 = jnp.real(mps_overlap_aligned(Mt, Mt, chi))
            nbk = s.shape[0] // pb
            Ob = jax.vmap(lambda m: mps_overlap_rect(m, m, chi))(Mt.reshape(nbk, pb, *Mt.shape[1:]))
            n2b = n2.reshape(nbk, pb)
            f = jnp.abs(Ob) ** 2 / (n2b[:, :, None] * n2b[:, None, :])
            sb = s.reshape(nbk, pb, N)
            dist = (N - jnp.einsum("nbk,nck->nbc", sb, sb)) / 2
            off = 1.0 - jnp.eye(pb)
            npairs = s.shape[0] * (pb - 1)
            return jnp.concatenate([nr, jnp.stack([jnp.sum(off * f * (dist >= 2)) / npairs,
                                                   jnp.sum(off * f * (dist >= 3)) / npairs])])
        t0 = time.time()
        Rw = np.array([np.asarray(tail_stats(params, sample(params, jax.random.PRNGKey(5000 + k), Mb))) for k in range(nb_)])
        P1 = Rw[:, 0] + Rw[:, 1] + Rw[:, 3]                 # nbr1: exacto d<=1 + iid d>=2  (el de entrenamiento)
        P2 = Rw[:, 0] + Rw[:, 1] + Rw[:, 2] + Rw[:, 4]      # nbr2: exacto d<=2 + iid d>=3
        Pi = None
        out5b = {"M": Mb, "batches": nb_, "time_s": time.time() - t0}
        for name, P in (("nbr1", P1), ("nbr2", P2)):
            S2 = -np.log(P.mean())
            err = P.std(ddof=1) / np.sqrt(len(P)) / P.mean()
            cum = [float(-np.log(P[:n].mean())) for n in (1, 2, 4, 8, 16, 32, 64, 128, 256) if n <= len(P)]
            out5b[name] = {"S2": float(S2), "S2_err": float(err), "S2_cum_1_2_4": cum, "cv_batch": float(P.std(ddof=1) / P.mean()),
                           "S2_per_batch_mean": float(np.mean(-np.log(P)))}
            print(f"  {name}: S2 = {S2:.4f} +- {err:.4f} | <-log P_lote> = {np.mean(-np.log(P)):.4f} (sesgo de un lote) | "
                  f"CV por lote {P.std(ddof=1) / P.mean():.2f}")
            print(f"        S2 acumulada (1,2,4,... lotes): {' '.join(f'{c:.3f}' for c in cum)}")
        Pm = Rw.mean(0); tot = Pm[0] + Pm[1] + Pm[2] + Pm[4]
        fr = {"d0": Pm[0] / tot, "d1": Pm[1] / tot, "d2": Pm[2] / tot, "d>=3": Pm[4] / tot}
        cv = {k: float(Rw[:, i].std(ddof=1) / max(Rw[:, i].mean(), 1e-300)) for i, k in enumerate(["d0", "d1", "d2", "far>=2", "far>=3"])}
        far = Rw[:, 3]
        tail = float(far.max() / max(far.mean(), 1e-300))
        dpair = P1 / P1.mean() - P2 / P2.mean()
        dS2 = out5b["nbr1"]["S2"] - out5b["nbr2"]["S2"]
        dS2_err = float(np.std(dpair, ddof=1) / np.sqrt(nb_))
        print("  fracciones de P (nbr2): " + "  ".join(f"{k}={v:.3f}" for k, v in fr.items())
              + f"   -> parte exacta del estimador de entrenamiento (d<=1) = {fr['d0'] + fr['d1']:.3f}")
        print("  CV por lote: " + "  ".join(f"{k}={v:.2f}" for k, v in cv.items())
              + f"   | lote lejano max/media = {tail:.1f} (colas pesadas si >> 3)")
        print(f"  dS2 (nbr1 - nbr2) = {dS2:+.4f} +- {dS2_err:.4f}  (>0 significativo => nbr1 sesga S2 hacia arriba)")
        cum1 = out5b["nbr1"]["S2_cum_1_2_4"]
        if len(cum1) >= 3 and cum1[0] - cum1[-1] > 2 * out5b["nbr1"]["S2_err"] * np.sqrt(len(P1)):
            print("  -> la S2 acumulada baja claramente al anadir lotes: sesgo de colas; S2 de un lote de M sobreestima.")
        if cv["far>=2"] > 1.0 and fr["d0"] + fr["d1"] < 0.5:
            print("  -> la cola iid (d>=2) domina P y es ruidosa: hace falta M/pair_block mayor o un estimador nbr2/swap")
        out5b.update(fractions=fr, cv=cv, far_max_over_mean=tail, dS2_nbr1_minus_nbr2=dS2, dS2_err=dS2_err)
        RES["tails"] = out5b

# ══════════════════════════════════════════════════════════════════════════════
# 6. cadenas swap
# ══════════════════════════════════════════════════════════════════════════════
if 6 in SECTIONS:
    with Section(6, "Cadenas swap: aceptacion, calentamiento y beta frente a la longitud"):
        stride = args.swap_stride
        lengths = sorted({int(x) for x in args.swap_lengths.split(",")})
        assert all(L % stride == 0 for L in lengths), "--swap_lengths deben ser multiplos de --swap_stride"
        C = args.swap_group
        moves = SWAP_TRAIN_MOVES
        run = jax.jit(make_swap_chains(model, N, stride, moves))
        gfun = jax.jit(jax.grad(make_swap_loss(model, N, 1.0)))         # grad log P
        if ed():
            refs = None
            ref = flat(ed()["gradlogP"](params)); ref_sq = float(ref @ ref); ref_lab = "ED exacto"
        else:
            Mr = round_M(args.ref_M or 4 * args.M)
            vgr, _ = get_vg(Mr, "nbr", terms="P", Tv=1.0)
            refs = [flat(vgr(params, sample(params, jax.random.PRNGKey(7000 + r), Mr))[1]) for r in range(args.ref_reps)]
            ref = np.mean(refs, 0)
            ref_sq = float(np.mean([refs[a] @ refs[b] for a, b in itertools.combinations(range(len(refs)), 2)]))
            if ref_sq <= 0:
                ref_sq = float(ref @ ref)
            ref_lab = f"nbr M={Mr} x{args.ref_reps} (cos entre refs {mean_pair_cos(refs)[0]:.3f})"
        print(f"  referencia de grad log P: {ref_lab}")
        G = {L: [] for L in lengths}
        accs = {}
        smp = get_sampler(C)
        t0 = time.time()
        for r in range(args.swap_reps):
            k = jax.random.PRNGKey(6000 + r)
            pool = smp(params, k)
            s = t = pool
            done, first = 0, True
            while done < max(lengths):
                s, t, acc = run(params, jax.random.fold_in(k, done + 1), pool, None if first else s, None if first else t)
                first = False
                done += stride
                if done in G:
                    G[done].append(flat(gfun(params, s, t)))
                    accs.setdefault(done, []).append({m: float(v) for m, v in acc.items()})
        t_ch = time.time() - t0
        Lmax = max(lengths)
        gmax = np.mean(G[Lmax], 0)
        rows = []
        print(f"  {C} cadenas x {args.swap_reps} reps, stride={stride}, movimientos={moves}  [{t_ch:.0f} s]")
        print(f"  {'L':>5s} | {'beta':>6s} {'cos_ref':>7s} {'cos_self':>8s} {'cos(L,Lmax)':>11s} | acc " + "/".join(moves))
        for L in lengths:
            gs = G[L]
            be = float(np.mean([g @ ref for g in gs]) / ref_sq)
            cr = float(np.mean([cosm(g, ref) for g in gs]))
            cs = mean_pair_cos(gs)[0]
            cl = cosm(np.mean(gs, 0), gmax)
            ac = {m: float(np.mean([a[m] for a in accs[L]])) for m in moves}
            rows.append(dict(L=L, beta=be, cos_ref=cr, cos_self=cs, cos_Lmax=cl, acc=ac))
            print(f"  {L:5d} | {be:6.3f} {cr:7.3f} {cs:8.3f} {cl:11.3f} | " + "/".join(f"{ac[m]:.2f}" for m in moves))
        ok = [r["L"] for r in rows if abs(r["beta"] - 1) < 0.15 and r["cos_ref"] >= 0.9 * rows[-1]["cos_ref"]]
        rec_w = ok[0] if ok else None
        print(f"  -> swap_warm sugerido: {rec_w if rec_w is not None else 'ninguno de los probados (beta lejos de 1: mezcla lenta)'}"
              f"   (train_mc usa swap_warm={args.swap_warm}, stride={stride}; gap_mc.py no los expone)")
        if rows and min(rows[-1]["acc"].values()) < 0.05:
            print("  -> algun movimiento acepta < 5%: la cadena apenas se mueve con ese movimiento")
        RES["swap"] = dict(reference=ref_lab, C=C, stride=stride, rows=rows, swap_warm_suggested=rec_w)

# ══════════════════════════════════════════════════════════════════════════════
# 7. barrido de lr
# ══════════════════════════════════════════════════════════════════════════════
if 7 in SECTIONS:
    with Section(7, f"Barrido de lr ({args.sweep_steps} pasos de train_mc por lr, desde el estado inicial)"):
        lrs = [float(x) for x in args.lr_sweep.split(",")]
        nchE = auto_chunks(args.eval_M, N, chi, pb, None, MEM_GB, grad=False)
        one = make_evaluator(model, N, T, hx, J, args.eval_M, pb, purity=est_purity, n_chunks=nchE)

        def ev(p, seed):
            keys = jax.random.split(jax.random.PRNGKey(seed), args.sweep_eval_batches)
            rr = np.array([[float(x) for x in one(p, k)[:2]] for k in keys])
            Fb = rr[:, 0] - T * rr[:, 1]
            S2 = -np.log(np.mean(np.exp(-rr[:, 1])))
            return float(rr[:, 0].mean() - T * S2), float(Fb.std(ddof=1) / np.sqrt(len(Fb))) if len(Fb) > 1 else float("nan")
        rows = []
        for lr in lrs:
            t0 = time.time()
            with quiet():
                o = train_mc(model, params0, N, T, hx, J, M=round_M(args.M), n_steps=args.sweep_steps, lr=lr,
                             warmup=args.warmup, seed=args.seed, pair_block=pb, freq=max(args.sweep_steps // 2, 1),
                             ema_decay=0.995, purity=args.purity, mem_gb=MEM_GB, clip=args.clip,
                             swap_chains=args.swap_chains, swap_group=args.swap_group, swap_stride=args.swap_stride,
                             swap_warm=args.swap_warm)
            row = {"lr": lr, "time_s": time.time() - t0, "F_mc_last10": float(o["hist"][-10:, 2].mean())}
            for lab, p in (("last", o["params"]), ("ema", o["params_ema"])):
                row[f"F_{lab}"], row[f"F_{lab}_err"] = ev(p, 77)
                if ed():
                    row[f"F_{lab}_exact"] = exact_fn(p)[0]
            rows.append(row)
            print(f"  lr={lr:.2e}: F(ultimos)={row['F_last']:.4f}+-{row['F_last_err']:.4f}  F(EMA)={row['F_ema']:.4f}+-{row['F_ema_err']:.4f}"
                  + (f" | ED: ultimos {row['F_last_exact']:.5f} EMA {row['F_ema_exact']:.5f}" if ed() else "")
                  + f"  [{row['time_s']:.0f} s]", flush=True)
        key_ = "F_ema_exact" if ed() else "F_ema"
        best = min(rows, key=lambda r: r[key_])
        REC["lr"] = best["lr"]
        print(f"  -> mejor lr (menor {key_}) = {best['lr']:.2e}"
              + ("" if ed() else "  (con pocos pasos y error MC: mira tambien las barras de error)"))
        RES["lr_sweep"] = dict(rows=rows, best=best["lr"], criterion=key_)

# ══════════════════════════════════════════════════════════════════════════════
# 8. ED
# ══════════════════════════════════════════════════════════════════════════════
if 8 in SECTIONS:
    with Section(8, "Verificacion con ED (N <= 10): estimadores MC vs exacto del estado y hueco con el MRE"):
        E_ = ed()
        Fm, Em, Sm = E_["ref"]
        Fx, Ex, Sx = exact_fn(params)
        print(f"  MRE exacto        : F={Fm:.5f}  E={Em:.5f}  S2={Sm:.4f}")
        print(f"  estado (exacto ED): F={Fx:.5f}  E={Ex:.5f}  S2={Sx:.4f}   hueco F={Fx - Fm:+.5f} "
              f"(dE={Ex - Em:+.4f}, dS2={Sx - Sm:+.4f})")
        out8 = dict(F_mre=Fm, E_mre=Em, S2_mre=Sm, F_state=Fx, E_state=Ex, S2_state=Sx, gap_F=Fx - Fm, estimators={})
        for pur in sorted({est_purity, "iid", "nbr"}):
            eM = round_M(min(args.eval_M, 4096))
            nchE = auto_chunks(eM, N, chi, pb, None, MEM_GB, grad=False)
            one = make_evaluator(model, N, T, hx, J, eM, pb, purity=pur, n_chunks=nchE)
            keys = jax.random.split(jax.random.PRNGKey(888), args.eval_batches)
            rr = np.array([[float(x) for x in one(params, k)[:2]] for k in keys])
            n = len(rr)
            E_mc, E_err = rr[:, 0].mean(), rr[:, 0].std(ddof=1) / np.sqrt(n)
            P = np.exp(-rr[:, 1])
            S2_mc = -np.log(P.mean()); S2_err = P.std(ddof=1) / np.sqrt(n) / P.mean()
            S2_1 = rr[:, 1].mean()
            F_mc = E_mc - T * S2_mc
            out8["estimators"][pur] = dict(eval_M=eM, batches=n, E=E_mc, E_err=E_err, S2=S2_mc, S2_err=S2_err,
                                           S2_mean_per_batch=S2_1, F=F_mc, zE=(E_mc - Ex) / E_err, zS2=(S2_mc - Sx) / S2_err)
            print(f"  MC {pur:>3s} (eval_M={eM} x{n}): E={E_mc:.5f}+-{E_err:.5f} (z={(E_mc - Ex) / E_err:+.1f})  "
                  f"S2={S2_mc:.4f}+-{S2_err:.4f} (z={(S2_mc - Sx) / S2_err:+.1f})  <S2_lote>={S2_1:.4f} "
                  f"(sesgo de un lote {S2_1 - Sx:+.4f})  F={F_mc:.5f}")
        print("  (z = (MC - exacto)/error con --eval_batches lotes; |z| < ~3 es normal, el propio error es ruidoso con pocos lotes)")
        # gradiente MC vs exacto
        gex = flat(E_["gradF"](params)); gsq = float(gex @ gex)
        Mg = round_M(args.ed_M or args.M)
        rows = []
        for pur in sorted({est_purity, "iid", "nbr"}):
            vg, _ = get_vg(Mg, pur)
            gs = [flat(vg(params, sample(params, jax.random.PRNGKey(4400 + r), Mg))[1]) for r in range(args.noise_reps)]
            gm = np.mean(gs, 0)
            row = dict(purity=pur, M=Mg, cos=float(np.mean([cosm(g, gex) for g in gs])),
                       beta=float(np.mean([g @ gex for g in gs]) / gsq),
                       beta_err=float(np.std([g @ gex for g in gs], ddof=1) / gsq / np.sqrt(len(gs))), cos_mean=cosm(gm, gex),
                       beta_mean=float(gm @ gex / gsq), norm_ratio=float(np.mean([np.linalg.norm(g) for g in gs]) / np.sqrt(gsq)))
            rows.append(row)
            print(f"  grad MC {pur:>3s} M={Mg} x{args.noise_reps}: cos con exacto {row['cos']:.3f} (media de reps {row['cos_mean']:.3f}) | "
                  f"beta {row['beta']:.3f}+-{row['beta_err']:.3f} | |g|/|g_ex| {row['norm_ratio']:.2f}")
        print(f"  |grad F exacto| = {np.sqrt(gsq):.4f}.  beta compatible con 1 => estimador insesgado del gradiente de F "
              f"(cerca del optimo |g_ex| es pequeno y el ruido domina: sube --noise_reps/--ed_M)")
        out8["grad"] = rows
        RES["ed"] = out8

# ══════════════════════════════════════════════════════════════════════════════
# 9. resumen + JSON
# ══════════════════════════════════════════════════════════════════════════════
with Section(9, "Resumen"):
    def fmt(x, f="{:.4g}"):
        return "—" if x is None else (f.format(x) if isinstance(x, (int, float, np.floating)) else str(x))
    st = {k: v["status"] for k, v in RES["sections"].items()}
    print("  secciones: " + "  ".join(f"{k}:{'ok' if v == 'ok' else 'FALLO'}" for k, v in st.items()))
    print(f"  modelo           : {type(model).__name__} {args.arch or ''} | #params {n_params:,} | N={N} chi={chi} T={T}")
    if "memory" in RES:
        m = RES["memory"]
        print(f"  memoria          : n_chunks minimo {fmt(m.get('n_min'))}, recomendado {fmt(m.get('n_rec'))} -> --mem_gb {fmt(m.get('mem_gb_rec'))}"
              f" | pico {', '.join(f'{k}={v:.2f}' for k, v in m['mem'].items())}")
    if "timing" in RES:
        tt = RES["timing"]
        print("  tiempo/paso      : " + ", ".join(f"{k} {1e3 * v['s_per_step']:.0f} ms" for k, v in tt["per_purity"].items())
              + f" | eval {tt['eval_s_per_batch']:.2f} s/lote | total previsto ~{tt['total_s'] / 60:.1f} min")
    if "grad_norms" in RES:
        g = RES["grad_norms"]
        print(f"  ||g||            : mediana {g['median']:.3g} p90 {g['p90']:.3g} max {g['max']:.3g} -> clip {g['clip_suggested']:.3g} (actual {args.clip})")
    if "noise" in RES:
        nz = RES["noise"]
        rM = [r for r in nz["rows"] if r["M"] == round_M(args.M)] or nz["rows"]
        r0 = rM[0]
        print(f"  ruido (M={r0['M']})   : std S2 {r0['S2_std']:.3g} | cos_self {r0['cos_self']:.3f} | cos_ref {r0['cos_ref']:.3f} | "
              f"beta {r0['beta']:.3f} | ruido_rel {r0['rel_noise']:.2f} | M sugerida {fmt(REC.get('M'))}")
    if "tails" in RES:
        tl = RES["tails"]
        print(f"  colas            : parte exacta d<=1 {tl['fractions']['d0'] + tl['fractions']['d1']:.3f} | CV far {tl['cv']['far>=2']:.2f} | "
              f"dS2(nbr1-nbr2) {tl['dS2_nbr1_minus_nbr2']:+.4f}+-{tl['dS2_err']:.4f}")
    if "swap" in RES:
        print(f"  swap             : swap_warm sugerido {fmt(RES['swap']['swap_warm_suggested'])}")
    if "lr_sweep" in RES:
        print(f"  lr               : mejor {RES['lr_sweep']['best']:.2e} ({RES['lr_sweep']['criterion']})")
    if "ed" in RES:
        e = RES["ed"]
        print(f"  ED               : F estado {e['F_state']:.5f} vs MRE {e['F_mre']:.5f} -> hueco {e['gap_F']:+.5f}; "
              + "; ".join(f"{k}: zE={v['zE']:+.1f} zS2={v['zS2']:+.1f}" for k, v in e["estimators"].items()))
    M_rec = REC.get("M", args.M)
    flags = (f"--N {N} --T {T} --chi {chi}" + (f" --arch {args.arch}" if args.arch else "")
             + (f" --d {args.d}" if args.d != 16 else "") + (f" --blocks {args.blocks}" if args.blocks != 2 else "")
             + f" --init {args.init} --M {M_rec} --pair_block {pb} --purity {args.purity}"
             + f" --mem_gb {REC.get('mem_gb', args.mem_gb)} --lr {REC.get('lr', args.lr):g} --steps {args.steps}"
             + f" --warmup {args.warmup}" + (f" --polish_steps {args.polish_steps}" if args.polish_steps else "")
             + f" --eval_M {args.eval_M} --eval_batches {args.eval_batches} --label <label> --out_dir <dir>")
    print("\n  flags recomendados para gap_mc.py:\n    python scripts/gap_mc.py " + flags)
    if "clip" in REC and abs(REC["clip"] - args.clip) / args.clip > 0.5:
        print(f"    (clip sugerido {REC['clip']:.3g} != {args.clip}: gap_mc.py no lo expone)")
    if "M" in REC and REC["M"] != args.M and "timing" in RES:
        print(f"    (con M={M_rec} el tiempo previsto escala ~ x{M_rec / args.M:.2f})")
    RES["recommended"] = dict(REC, gap_mc_flags=flags)

os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
with open(out_file, "w") as f:
    json.dump(RES, f, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o))
print(f"\nJSON: {os.path.abspath(out_file)}", flush=True)
