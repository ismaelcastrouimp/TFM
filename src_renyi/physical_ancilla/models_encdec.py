"""
models_encdec.py  (17-11-2026)
=====================================
Ancilla FÍSICA con el sistema ENTERO antes que la ancilla en el orden AR (compatible con bridge.renyi2_bridge):
    ψ(s, a) = A(s) · φ(a | s),   sitios 0..N-1 = sistema s,  N..2N-1 = ancilla a  (hilbert = Spin(1/2, 2N)).
Salida de conditionals_log_psi: (B, 2N, 2) complejo = 0.5·log p + i·fase (como ARSpinViT_Causal: ψ compleja con
parámetros reales; la fase del sitio i depende de x_i y de los sitios anteriores).

  ARPhysEncDec  encoder-decoder:
      sistema   : transformer CAUSAL sobre s (desplazado)                  -> A(s)
      encoder   : transformer BIDIRECCIONAL sobre s completo (sin máscara) -> memoria (B, N, d)
      decoder   : transformer causal sobre a (desplazado) + atención CRUZADA a la memoria -> φ(a|s)
  ARPhysPrefix  la arquitectura previa (= ARSpinViT_Causal sobre [s, a]): un único transformer causal; la ancilla solo
                ve representaciones causales de s y comparte pesos con el sistema.

phase_net != "head": fase extra θ(s, a) de una red aparte sobre la configuración completa (la fase lineal por sitio
sobre una red real no aprende la estructura de signos de ρ_MRE; ver progreso 17-11).
copy_init > 0: sesgo aprendible g·[a_k == s_k] en los logits de la ancilla (g inicial = copy_init).  Con la cabeza
final a cero, al empezar φ(a|s) ≈ Π_k δ(a_k = s_k) y p(s) uniforme  =>  ρ_S ≈ I/2^N (T = ∞), ya en el gauge
hermítico (ψ ∝ identidad).  g = 4 -> P(a_k = s_k) = 0.98.

CACHÉ (muestreo rápido, exacto): `conditional(inputs, i)` guarda en la colección "cache" de NetKet las claves/valores
de la auto-atención de cada bloque, así cada sitio cuesta UN token en vez de una pasada completa (como los FastARNN;
lo usan ARDirectSampler, bridge._bridge_sample y entropy._tilted_sample sin cambios).  Se puede empezar a muestrear
en i = 0 o en i = N (el puente muestrea solo la ancilla): en i = N se recalcula lo que depende de s completo
(memoria del encoder y sus K/V cruzadas; en prefix, las K/V de los N primeros tokens).  Hay que llamar a los sitios
en orden creciente desde 0 o desde N.
"""
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk


class _MHA(nn.Module):
    n_heads: int
    d: int
    param_dtype: Any = jnp.float64

    def setup(self):
        dh = self.d // self.n_heads
        D = lambda name: nn.DenseGeneral((self.n_heads, dh), param_dtype=self.param_dtype, name=name)
        self.q, self.k, self.v = D("q"), D("k"), D("v")
        self.o = nn.DenseGeneral(self.d, axis=(-2, -1), param_dtype=self.param_dtype, name="o")

    def kv(self, x):
        return self.k(x), self.v(x)

    def attend(self, xq, K, V, mask=None):
        q = self.q(xq)
        s = jnp.einsum("bqhd,bkhd->bhqk", q, K) / jnp.sqrt(jnp.asarray(q.shape[-1], q.dtype))
        if mask is not None:
            s = jnp.where(mask, s, jnp.finfo(s.dtype).min)
        return self.o(jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(s, -1), V))

    def __call__(self, xq, xkv, mask=None):
        K, V = self.kv(xkv)
        return self.attend(xq, K, V, mask)


class _Block(nn.Module):
    """Bloque post-LN: auto-atención (máscara opcional) [+ atención cruzada] + FFN."""
    n_heads: int
    d: int
    n_ffn_layers: int = 1
    cross: bool = False
    param_dtype: Any = jnp.float64

    def setup(self):
        dt = self.param_dtype
        self.sa = _MHA(self.n_heads, self.d, dt)
        self.ln1 = nn.LayerNorm(param_dtype=dt)
        if self.cross:
            self.ca = _MHA(self.n_heads, self.d, dt)
            self.lnc = nn.LayerNorm(param_dtype=dt)
        self.ff = [nn.Dense(2 * self.d, param_dtype=dt) for _ in range(self.n_ffn_layers)]
        self.ff_out = nn.Dense(self.d, param_dtype=dt)
        self.ln2 = nn.LayerNorm(param_dtype=dt)

    def _ffn(self, x):
        f = x
        for l in self.ff:
            f = nn.gelu(l(f))
        return self.ln2(x + self.ff_out(f))

    def __call__(self, x, mask=None, memory=None):
        """Pasada completa.  Devuelve también K, V de la auto-atención (para rellenar la caché)."""
        K, V = self.sa.kv(x)
        x = self.ln1(x + self.sa.attend(x, K, V, mask))
        if self.cross:
            x = self.lnc(x + self.ca(x, memory))
        return self._ffn(x), K, V

    def step(self, xt, K, V, t, cross_kv=None):
        """Un token (B, 1, d) en la posición t con la caché K, V (B, L, h, dh); atiende a las posiciones <= t."""
        kt, vt = self.sa.kv(xt)
        K = jax.lax.dynamic_update_slice_in_dim(K, kt, t, axis=1)
        V = jax.lax.dynamic_update_slice_in_dim(V, vt, t, axis=1)
        mask = (jnp.arange(K.shape[1]) <= t)[None, None, None, :]
        x = self.ln1(xt + self.sa.attend(xt, K, V, mask))
        if self.cross:
            x = self.lnc(x + self.ca.attend(x, *cross_kv))
        return self._ffn(x), K, V

    def cross_kv(self, memory):
        return self.ca.kv(memory)


class _Embed(nn.Module):
    """Embedding lineal del valor del espín + posicional aprendido."""
    n: int
    d: int
    param_dtype: Any = jnp.float64

    def setup(self):
        self.lin = nn.Dense(self.d, param_dtype=self.param_dtype)
        self.pos = self.param("pos", nn.initializers.normal(0.01), (self.n, self.d), self.param_dtype)

    def __call__(self, x):                     # x (B, L) con L = n
        return self.lin(x.astype(self.param_dtype)[..., None]) + self.pos[None, : x.shape[1]]

    def token(self, v, t):                     # v (B,) valor del token, t posición -> (B, 1, d)
        return self.lin(v.astype(self.param_dtype)[:, None, None]) + self.pos[t][None, None]


class _Head(nn.Module):
    """Cabeza de salida: 0.5·log_softmax(logits + sesgo) + i·fase."""
    param_dtype: Any = jnp.float64

    def setup(self):
        dt = self.param_dtype
        self.ln = nn.LayerNorm(param_dtype=dt)
        self.logits = nn.Dense(2, kernel_init=nn.initializers.zeros, param_dtype=dt)
        self.phase = nn.Dense(2, kernel_init=nn.initializers.normal(0.05), param_dtype=dt)

    def __call__(self, x, bias=0.0):
        x = self.ln(x)
        return jax.lax.complex(0.5 * jax.nn.log_softmax(self.logits(x) + bias, axis=-1), self.phase(x))

    def probs(self, x, bias=0.0):
        x = self.ln(x)
        self.phase(x)                          # (no se usa; así init por esta vía crea los mismos parámetros)
        return jax.nn.softmax(self.logits(x) + bias, axis=-1)


class _Store(nn.Module):
    """Lee/escribe un array de la colección "cache" (creado a ceros la primera vez)."""
    @nn.compact
    def __call__(self, name, shape, dtype, value=None):
        v = self.variable("cache", name, jnp.zeros, shape, dtype)
        if value is not None:
            v.value = value
        return v.value


def _shift(x):
    return jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)


class _PhaseNet(nn.Module):
    """Fase θ(x) sobre la configuración COMPLETA x = (s, a) (no autorregresiva: no afecta a la normalización ni al
    muestreo).  kind="mlp": MLP sobre los 2N espines;  kind="enc": transformer bidireccional + media sobre sitios."""
    kind: str
    n: int
    d: int = 16
    n_heads: int = 4
    width: int = 64
    param_dtype: Any = jnp.float64

    def setup(self):
        dt = self.param_dtype
        if self.kind == "mlp":
            self.layers = [nn.Dense(self.width, param_dtype=dt) for _ in range(2)]
        else:
            self.emb = _Embed(self.n, self.d, dt)
            self.blocks = [_Block(self.n_heads, self.d, param_dtype=dt) for _ in range(2)]
        self.out = nn.Dense(1, kernel_init=nn.initializers.normal(0.01), param_dtype=dt)

    def __call__(self, x):
        if self.kind == "mlp":
            h = x.astype(self.param_dtype)
            for l in self.layers:
                h = jnp.tanh(l(h))
        else:
            h = self.emb(x)
            for b in self.blocks:
                h = b(h)[0]
            h = jnp.mean(h, axis=1)
        return self.out(h)[:, 0]


def _onehot_up(s, dtype):
    return jax.nn.one_hot((s > 0).astype(jnp.int32), 2, dtype=dtype)


class ARPhysEncDec(nk.models.AbstractARNN):
    embedding_d: int = 16
    n_heads: int = 4
    sys_blocks: int = 2
    enc_blocks: int = 1
    dec_blocks: int = 2
    n_ffn_layers: int = 1
    phase_net: str = "head"     # "head" (fase lineal por sitio) | "mlp" | "enc" (red de fase aparte, sobre x completo)
    copy_init: float = 0.0
    machine_pow: int = 2
    param_dtype: Any = jnp.float64

    def setup(self):
        N, d, h, dt = self.hilbert.size // 2, self.embedding_d, self.n_heads, self.param_dtype
        self.sys_embed = _Embed(N, d, dt)
        self.sys = [_Block(h, d, self.n_ffn_layers, param_dtype=dt) for _ in range(self.sys_blocks)]
        self.sys_head = _Head(dt)
        self.enc_embed = _Embed(N, d, dt)
        self.enc = [_Block(h, d, self.n_ffn_layers, param_dtype=dt) for _ in range(self.enc_blocks)]
        self.dec_embed = _Embed(N, d, dt)
        self.dec = [_Block(h, d, self.n_ffn_layers, cross=True, param_dtype=dt) for _ in range(self.dec_blocks)]
        self.anc_head = _Head(dt)
        if self.copy_init > 0:
            self.copy_gain = self.param("copy_gain", nn.initializers.constant(self.copy_init), (), dt)
        if self.phase_net != "head":
            self.theta = _PhaseNet(self.phase_net, self.hilbert.size, self.embedding_d, self.n_heads,
                                   param_dtype=self.param_dtype)
        self.store = _Store()

    def __call__(self, inputs):
        lp = nk.models.AbstractARNN.__call__(self, inputs)
        if self.phase_net != "head":
            x = inputs if inputs.ndim == 2 else inputs[None]
            lp = lp + 1j * self.theta(x)
        return lp

    def _bias(self, s):
        return self.copy_gain * _onehot_up(s, self.param_dtype) if self.copy_init > 0 else 0.0

    def _encode(self, s):
        m = self.enc_embed(s)
        for blk in self.enc:
            m = blk(m)[0]
        return m

    def conditionals_log_psi(self, inputs):
        B, N2 = inputs.shape
        N = N2 // 2
        s, a = inputs[:, :N], inputs[:, N:]
        causal = nn.make_causal_mask(jnp.empty((B, N)))
        h = self.sys_embed(_shift(s))
        for blk in self.sys:
            h = blk(h, causal)[0]
        m = self._encode(s)
        g = self.dec_embed(_shift(a))
        for blk in self.dec:
            g = blk(g, causal, m)[0]
        return jnp.concatenate([self.sys_head(h), self.anc_head(g, self._bias(s))], axis=1)

    def conditional(self, inputs, index):
        B, N2 = inputs.shape
        N, dt = N2 // 2, self.param_dtype
        dh = self.embedding_d // self.n_heads
        shp = (B, N, self.n_heads, dh)
        i = jnp.asarray(index)
        # ── sistema (posición ts; se calcula siempre, solo se usa si i < N) ──
        ts = jnp.minimum(i, N - 1)
        x = self.sys_embed.token(jnp.where(ts > 0, jnp.take(inputs, jnp.maximum(ts - 1, 0), axis=1), 0), ts)
        for b, blk in enumerate(self.sys):
            x, K, V = blk.step(x, self.store(f"sK{b}", shp, dt), self.store(f"sV{b}", shp, dt), ts)
            self.store(f"sK{b}", shp, dt, K); self.store(f"sV{b}", shp, dt, V)
        p_s = self.sys_head.probs(x[:, 0])
        # ── ancilla (posición k); en i == N se recalculan la memoria del encoder y sus K/V cruzadas ──
        k = jnp.maximum(i - N, 0)
        old = [(self.store(f"cK{b}", shp, dt), self.store(f"cV{b}", shp, dt)) for b in range(self.dec_blocks)]

        def fill(mdl, s, old):
            m = mdl._encode(s)
            return [blk.cross_kv(m) for blk in mdl.dec]

        s_full = inputs[:, :N]
        if self.is_initializing():
            ckv = fill(self, s_full, old)
        else:
            ckv = nn.cond(i == N, fill, lambda mdl, s, old: old, self, s_full, old)
        for b, (Kc, Vc) in enumerate(ckv):
            self.store(f"cK{b}", shp, dt, Kc); self.store(f"cV{b}", shp, dt, Vc)
        y = self.dec_embed.token(jnp.where(k > 0, jnp.take(inputs, N + jnp.maximum(k - 1, 0), axis=1), 0), k)
        for b, blk in enumerate(self.dec):
            y, K, V = blk.step(y, self.store(f"dK{b}", shp, dt), self.store(f"dV{b}", shp, dt), k, ckv[b])
            self.store(f"dK{b}", shp, dt, K); self.store(f"dV{b}", shp, dt, V)
        p_a = self.anc_head.probs(y[:, 0], self._bias(jnp.take(inputs, k, axis=1)))
        return jnp.where(i < N, p_s, p_a)


class ARPhysPrefix(nk.models.AbstractARNN):
    """Arquitectura previa (ARSpinViT_Causal sobre [s, a]) con copy_init opcional y caché."""
    embedding_d: int = 16
    n_heads: int = 4
    n_blocks: int = 2
    n_ffn_layers: int = 1
    phase_net: str = "head"     # "head" (fase lineal por sitio) | "mlp" | "enc" (red de fase aparte, sobre x completo)
    copy_init: float = 0.0
    machine_pow: int = 2
    param_dtype: Any = jnp.float64

    def setup(self):
        N2, d, dt = self.hilbert.size, self.embedding_d, self.param_dtype
        self.embed = _Embed(N2, d, dt)
        self.blocks = [_Block(self.n_heads, d, self.n_ffn_layers, param_dtype=dt) for _ in range(self.n_blocks)]
        self.head = _Head(dt)
        if self.copy_init > 0:
            self.copy_gain = self.param("copy_gain", nn.initializers.constant(self.copy_init), (), dt)
        if self.phase_net != "head":
            self.theta = _PhaseNet(self.phase_net, self.hilbert.size, self.embedding_d, self.n_heads,
                                   param_dtype=self.param_dtype)
        self.store = _Store()

    def __call__(self, inputs):
        lp = nk.models.AbstractARNN.__call__(self, inputs)
        if self.phase_net != "head":
            x = inputs if inputs.ndim == 2 else inputs[None]
            lp = lp + 1j * self.theta(x)
        return lp

    def _bias(self, s):
        return self.copy_gain * _onehot_up(s, self.param_dtype) if self.copy_init > 0 else 0.0

    def conditionals_log_psi(self, inputs):
        B, N2 = inputs.shape
        N = N2 // 2
        h = self.embed(_shift(inputs))
        causal = nn.make_causal_mask(jnp.empty((B, N2)))
        for blk in self.blocks:
            h = blk(h, causal)[0]
        bias = self._bias(inputs[:, :N])
        bias = jnp.concatenate([jnp.zeros((B, N, 2), self.param_dtype),
                                bias * jnp.ones((B, N, 2), self.param_dtype)], axis=1)
        return self.head(h, bias)

    def conditional(self, inputs, index):
        B, N2 = inputs.shape
        N, dt = N2 // 2, self.param_dtype
        shp = (B, N2, self.n_heads, self.embedding_d // self.n_heads)
        i = jnp.asarray(index)
        caches = [(self.store(f"K{b}", shp, dt), self.store(f"V{b}", shp, dt)) for b in range(self.n_blocks)]

        def fill(mdl, x, caches):              # K/V de los N primeros tokens con una pasada (s completo)
            h = mdl.embed(_shift(x))
            causal = nn.make_causal_mask(jnp.empty(x.shape))
            out = []
            for blk, (K, V) in zip(mdl.blocks, caches):
                h, Kn, Vn = blk(h, causal)
                out.append((K.at[:, :N].set(Kn), V.at[:, :N].set(Vn)))
            return out

        s_full = inputs[:, :N]
        if self.is_initializing():
            caches = fill(self, s_full, caches)
        else:
            caches = nn.cond(i == N, fill, lambda mdl, x, c: c, self, s_full, caches)
        x = self.embed.token(jnp.where(i > 0, jnp.take(inputs, jnp.maximum(i - 1, 0), axis=1), 0), i)
        for b, blk in enumerate(self.blocks):
            x, K, V = blk.step(x, *caches[b], i)
            self.store(f"K{b}", shp, dt, K); self.store(f"V{b}", shp, dt, V)
        bias = jnp.where(i >= N, 1.0, 0.0) * self._bias(jnp.take(inputs, jnp.maximum(i - N, 0), axis=1)) \
            if self.copy_init > 0 else 0.0
        return self.head.probs(x[:, 0], bias)
