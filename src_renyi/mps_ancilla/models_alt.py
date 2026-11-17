# src_renyi/mps_ancilla/models_alt.py
"""
Variantes de arquitectura para las pruebas del hueco de F (notas_gap.md, Dia 2).
Misma interfaz que ARViT_MPSAncilla:  __call__(s, only_logits=False) -> (logA (B,), M (B,K,2,chi,chi))
o los logits (B,N,2); atributo .chi.  Asi se reutilizan sampler, estimadores, evaluacion y mps_overlap_*.

Sistema (p(s) y fase):
  sys="vit"   ARViT identico al de models.py (tokens BOS, s_0..s_{N-1}; posicion i predice s_i)
  sys="vitp"  ARViT con sesgo de posicion relativa periodico en la atencion (PeriodicBiasBlock): el ultimo sitio
              ve s_0 como vecino.  Para el enlace periodico (N-1,0) que el ARViT deja en ~0.
  sys="made"  ARNN densa autorregresiva tipo MADE (capas densas enmascaradas, sin NetKet).  Tiene N+1
              "posiciones" de f rasgos cada una: la posicion p ve s_<p, igual que el token p del ARViT;
              la posicion N ve todo s y hace de contexto para la ancilla (equivalente a x[:,N] del ARViT).

Ancilla (tensores M_k(s), k = 0..N-1; siempre + bias = I*0.5 + termino de copia, ver abajo):
  anc="ctx"     una Dense sobre el contexto (posicion N), como ahora                  [actual]
  anc="mlp"     MLP propio sobre s completo (no pasa por el transformer)               [prueba 1]
  anc="mix"     MLP sobre concat(s, contexto)                                          [prueba 1, opcional]
  anc="conv"    M_k depende solo de la ventana periodica s_{k-r..k+r}, MLP compartido  [prueba 2]
  anc="causal"  M_k sale del estado oculto de la posicion k+1 (ve s_<=k), MLP compartido [prueba 3b]

copy_init en todas: amp_head a cero (p uniforme), salida de las cabezas de la ancilla pequena y
M_k^a(s) = 0.5 I + 0.5 (2a-1) s_k I  (= delta(a,s_k) I: la ancilla copia al sistema, rho = I/2^N).
En "ctx"/"mlp"/"mix" el termino de copia es una matriz (N, n_out) entrenable, como mps_copy en models.py;
en "conv"/"causal" es un vector (2 chi^2) compartido por todos los sitios (invariante traslacional).
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from .models import CausalBlock

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


# ─── capas enmascaradas (MADE) ────────────────────────────────────────────────

def _masked_init(mask):
    """lecun-normal con el fan_in efectivo de cada columna (numero de entradas no enmascaradas)."""
    fan = np.maximum(mask.sum(0), 1).astype(np.float64)

    def init(key, shape, dtype=F64):
        return jax.random.normal(key, shape, dtype) / jnp.sqrt(fan)[None, :] * mask
    return init


class MaskedDense(nn.Module):
    mask: np.ndarray            # (n_in, n_out) bool
    kernel_init: object = None
    name_bias_init: object = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        m = jnp.asarray(self.mask, F64)
        k = self.param("kernel", self.kernel_init or _masked_init(self.mask), self.mask.shape, F64)
        b = self.param("bias", self.name_bias_init, (self.mask.shape[1],), F64)
        return x @ (k * m) + b


def made_masks(N, f, n_layers):
    """Mascaras de la MADE con N+1 posiciones de f rasgos.  Capa 1 (exclusiva): posicion p ve s_j, j<p.
    Capas ocultas siguientes (inclusivas): posicion p ve posiciones q<=p.  Devuelve (lista de mascaras ocultas,
    posicion de cada rasgo oculto)."""
    pos = np.repeat(np.arange(N + 1), f)                       # posicion de cada neurona oculta
    m1 = np.arange(N)[:, None] < pos[None, :]                  # (N, (N+1) f)
    mh = pos[:, None] <= pos[None, :]                          # ((N+1) f, (N+1) f)
    return [m1] + [mh] * (n_layers - 1), pos


# ─── bloque causal con sesgo de posicion relativa periodico ───────────────────

def periodic_dist_table(N):
    """(N+1, N+1) indices de distancia para el sesgo: query en posicion p (predice s_p, p<N; p=N es contexto),
    key en posicion q (token s_{q-1}; q=0 es BOS).  Indice = distancia periodica min(dd, N-dd) entre el sitio
    p y el sitio q-1, en 0..N//2; BOS -> N//2+1.  Para p=N-1 la key s_0 esta a distancia 1 (vecino por PBC)."""
    t = np.zeros((N + 1, N + 1), dtype=np.int32)
    for p in range(N + 1):
        for q in range(N + 1):
            if q == 0:
                t[p, q] = N // 2 + 1
            else:
                dd = (p - (q - 1)) % N
                t[p, q] = min(dd, N - dd)
    return t


class PeriodicBiasBlock(nn.Module):
    """Igual que CausalBlock (atencion multi-cabeza + FFN, post-LN) pero con un sesgo aprendido b_h(dist periodica)
    sumado a los logits de atencion (inicializado a cero).  Analogo para transformer de pasar el estado del primer
    sitio al ultimo (Hibat-Allah): para el ultimo sitio, s_0 es un vecino mas."""
    n_heads: int
    embedding_d: int
    N: int

    @nn.compact
    def __call__(self, x, mask):
        B, L, d = x.shape
        H = self.n_heads; dh = d // H
        q = nn.Dense(d, param_dtype=F64, name="q")(x).reshape(B, L, H, dh)
        k = nn.Dense(d, param_dtype=F64, name="k")(x).reshape(B, L, H, dh)
        v = nn.Dense(d, param_dtype=F64, name="v")(x).reshape(B, L, H, dh)
        logits = jnp.einsum("bphd,bqhd->bhpq", q, k) / np.sqrt(dh)
        table = periodic_dist_table(self.N)
        bias = self.param("rel_bias", nn.initializers.zeros, (H, self.N // 2 + 2), F64)
        logits = logits + bias[:, table][None]
        logits = jnp.where(mask, logits, -1e30)
        w = jax.nn.softmax(logits, -1)
        a = jnp.einsum("bhpq,bqhd->bphd", w, v).reshape(B, L, d)
        a = nn.Dense(d, param_dtype=F64, name="o")(a)
        x = nn.LayerNorm(param_dtype=F64)(x + a)
        f = nn.gelu(nn.Dense(2 * d, param_dtype=F64)(x))
        f = nn.Dense(d, param_dtype=F64)(f)
        return nn.LayerNorm(param_dtype=F64)(x + f)


# ─── modelo ───────────────────────────────────────────────────────────────────

class AltModel(nn.Module):
    N_sys: int
    chi: int = 4
    sys: str = "vit"            # "vit" | "made"
    anc: str = "ctx"            # "ctx" | "mlp" | "mix" | "conv" | "convg" | "causal"
    # ARViT
    embedding_d: int = 16
    n_heads: int = 4
    n_blocks: int = 2
    # MADE
    made_f: int = 16            # rasgos por posicion
    made_layers: int = 2        # capas ocultas enmascaradas
    # ancilla
    anc_width: int = 0          # anchura de las capas ocultas de la cabeza de la ancilla (0 -> por defecto)
    anc_layers: int = 2         # capas ocultas del MLP de la ancilla
    conv_r: int = 1             # radio de la ventana (anc="conv")
    phase_init: float = 0.05
    mps_init: float = 0.05      # escala de la salida de las cabezas de la ancilla (como build_model)

    # ---- sistema ----
    def _system(self, s):
        """Devuelve (logits (B,N,2), phase (B,N,2), h (B,N+1,D)): h[:,p] solo depende de s_<p."""
        N = self.N_sys
        B = s.shape[0]
        if self.sys in ("vit", "vitp"):
            d = self.embedding_d
            tok = jnp.concatenate([jnp.zeros((B, 1), F64), s], axis=1)
            x = nn.Dense(d, param_dtype=F64, name="embed")(tok[..., None])
            x = x + self.param("pos", nn.initializers.normal(0.01), (N + 1, d), F64)
            mask = jnp.tril(jnp.ones((N + 1, N + 1), dtype=bool))[None, None]
            for i in range(self.n_blocks):
                if self.sys == "vit":
                    x = CausalBlock(self.n_heads, d, name=f"blk{i}")(x, mask)
                else:
                    x = PeriodicBiasBlock(self.n_heads, d, N, name=f"blk{i}")(x, mask)
            h = nn.LayerNorm(param_dtype=F64, name="ln_out")(x)                 # (B, N+1, d)
            hs = h[:, :N]
            logits = nn.Dense(2, param_dtype=F64, name="amp_head", kernel_init=nn.initializers.zeros)(hs)
            phase = nn.Dense(2, param_dtype=F64, name="phase_head",
                             kernel_init=nn.initializers.normal(self.phase_init))(hs)
            return logits, phase, h
        elif self.sys == "made":
            f = self.made_f
            masks, pos = made_masks(N, f, self.made_layers)
            x = s
            for li, m in enumerate(masks):
                x = nn.selu(MaskedDense(m, name=f"made{li}")(x))
            h = x.reshape(B, N + 1, f)                                            # (B, N+1, f)
            # salida del sitio i desde las posiciones q <= i (inclusiva) -> ve s_<i
            mo = pos[:, None] <= np.repeat(np.arange(N), 2)[None, :]             # ((N+1) f, 2N)
            logits = MaskedDense(mo, kernel_init=nn.initializers.zeros, name="amp_head")(x).reshape(B, N, 2)
            ph_init = lambda k, sh, dt=F64: jax.random.normal(k, sh, dt) * self.phase_init * mo
            phase = MaskedDense(mo, kernel_init=ph_init, name="phase_head")(x).reshape(B, N, 2)
            return logits, phase, h
        raise ValueError(self.sys)

    # ---- ancilla ----
    def _mlp(self, z, width, n_out, name):
        for li in range(self.anc_layers):
            z = nn.gelu(nn.Dense(width, param_dtype=F64, name=f"{name}_h{li}")(z))
        return nn.Dense(n_out, param_dtype=F64, name=f"{name}_out",
                        kernel_init=nn.initializers.normal(self.mps_init / np.sqrt(z.shape[-1])),
                        bias_init=nn.initializers.zeros)(z)

    def _ancilla(self, s, h):
        N, chi = self.N_sys, self.chi
        B = s.shape[0]
        K = N
        nb = 2 * chi * chi
        eye = np.broadcast_to(np.eye(chi) * 0.5, (2, chi, chi)).reshape(-1)          # (2 chi^2,)
        cpy = np.zeros((2, chi, chi))
        for a in (0, 1):
            cpy[a] = 0.5 * (2 * a - 1) * np.eye(chi)
        cpy = cpy.reshape(-1)

        if self.anc in ("ctx", "mlp", "mix"):
            n_out = K * nb
            if self.anc == "ctx":
                W = nn.Dense(n_out, param_dtype=F64, name="mps_head",
                             kernel_init=nn.initializers.normal(self.mps_init / np.sqrt(h.shape[-1])),
                             bias_init=nn.initializers.zeros)(h[:, N])
            else:
                z = s if self.anc == "mlp" else jnp.concatenate([s, h[:, N]], -1)
                W = self._mlp(z, self.anc_width or 4 * N, n_out, "anc")
            kern = np.zeros((N, n_out))
            for k in range(K):
                kern[k, k * nb:(k + 1) * nb] = cpy
            W = W + jnp.tile(jnp.asarray(eye), K)[None] + s @ self.param("mps_copy", lambda *_: jnp.asarray(kern),
                                                                        (N, n_out), F64)
            return W.reshape(B, K, 2, chi, chi)

        if self.anc == "conv":
            r = self.conv_r
            idx = (np.arange(N)[:, None] + np.arange(-r, r + 1)[None, :]) % N           # (N, 2r+1) periodica
            z = s[:, idx]                                                               # (B, N, 2r+1)
            W = self._mlp(z, self.anc_width or 32, nb, "anc")                          # (B, N, nb)
        elif self.anc == "convg":
            # ventana local periodica + resumen global h[:, N] (ultimo token: depende de todo s) en cada sitio
            r = self.conv_r
            idx = (np.arange(N)[:, None] + np.arange(-r, r + 1)[None, :]) % N
            g = jnp.broadcast_to(h[:, N][:, None, :], (B, N, h.shape[-1]))
            z = jnp.concatenate([s[:, idx], g], -1)                                     # (B, N, 2r+1+d)
            W = self._mlp(z, self.anc_width or 32, nb, "anc")
        elif self.anc == "causal":
            z = h[:, 1:N + 1]                                                           # posicion k+1 ve s_<=k
            W = self._mlp(z, self.anc_width or 4 * z.shape[-1], nb, "anc")
        else:
            raise ValueError(self.anc)
        c = self.param("mps_copy", lambda *_: jnp.asarray(cpy), (nb,), F64)
        W = W + jnp.asarray(eye) + s[..., None] * c                                     # (B, N, nb)
        return W.reshape(B, K, 2, chi, chi)

    @nn.compact
    def __call__(self, s, only_logits=False):
        s = s.astype(F64)
        logits, phase, h = self._system(s)
        if only_logits:
            return logits
        idx = ((s + 1) / 2).astype(jnp.int32)[..., None]
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits, -1), idx, -1)[..., 0]
        ph = jnp.take_along_axis(phase, idx, -1)[..., 0]
        logA = jax.lax.complex(0.5 * jnp.sum(logp, 1), jnp.sum(ph, 1))
        M = self._ancilla(s, h).astype(jnp.complex128)
        return logA, M


# ─── constructor desde una cadena "arch" ───────────────────────────────────────

def build_alt_model(N, chi, arch, d=16, heads=4, blocks=2):
    """arch = "sys:anc[:clave=valor,...]", p.ej. "vit:mlp", "vit:conv:r=2", "made:ctx:f=16,L=2",
    "vit:mlp:w=64,al=3".  Claves: r (conv_r), f (made_f), L (made_layers), w (anc_width), al (anc_layers)."""
    parts = arch.split(":")
    sys_, anc = parts[0], parts[1]
    if anc == "convmlp":                       # hibrido conv + residuo MLP global (models_hybrid.py)
        from .models_hybrid import build_hybrid_model
        return build_hybrid_model(N, chi, arch, d, heads, blocks)
    if anc in ("convlin", "convloc", "convdelta"):          # ablaciones conv vs mlp (models_hybrid.py)
        from .models_hybrid import build_ablation_model
        return build_ablation_model(N, chi, arch, d, heads, blocks)
    kw = {}
    keymap = {"r": "conv_r", "f": "made_f", "L": "made_layers", "w": "anc_width", "al": "anc_layers"}
    if len(parts) > 2 and parts[2]:
        for kv in parts[2].split(","):
            k, v = kv.split("=")
            kw[keymap[k]] = int(v)
    return AltModel(N_sys=N, chi=chi, sys=sys_, anc=anc, embedding_d=d, n_heads=heads, n_blocks=blocks, **kw)
