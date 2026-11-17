# src_renyi/mps_ancilla/models_hybrid.py
"""
Ancilla hibrida "convmlp": M_k(s) = conv_k(ventana local de s)  +  residuo_k(MLP sobre s COMPLETO)  (+ bias + copia).
  - la parte conv es IDENTICA a anc="conv" de models_alt (mismos nombres de parametros anc_h*, anc_out, mps_copy), asi que
    un modelo vit:conv:r=1 entrenado se carga tal cual y el residuo arranca a CERO (salida res_out con kernel 0):
    el hibrido empieza exactamente en el estado conv y solo puede bajar F.
  - el residuo es la cabeza de anc="mlp" (global, pesos por sitio, sin invariancia traslacional), que a N=10 chi=16 sin
    filtro bajo de +0.0073 (conv) a +0.0044 (cola 19 CPU).
arch: "vit:convmlp:r=1[,w=..,rw=..]"  (rw = anchura del residuo; por defecto 4N)
"""
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from .models_alt import AltModel, F64


class HybridModel(AltModel):
    res_width: int = 0

    def _ancilla(self, s, h):
        N, chi = self.N_sys, self.chi
        B = s.shape[0]
        nb = 2 * chi * chi
        eye = np.broadcast_to(np.eye(chi) * 0.5, (2, chi, chi)).reshape(-1)
        cpy = np.zeros((2, chi, chi))
        for a in (0, 1):
            cpy[a] = 0.5 * (2 * a - 1) * np.eye(chi)
        cpy = cpy.reshape(-1)
        r = self.conv_r
        idx = (np.arange(N)[:, None] + np.arange(-r, r + 1)[None, :]) % N
        W = self._mlp(s[:, idx], self.anc_width or 32, nb, "anc")                        # (B, N, nb), como "conv"
        z = s
        for li in range(self.anc_layers):
            z = nn.gelu(nn.Dense(self.res_width or 4 * N, param_dtype=F64, name=f"res_h{li}")(z))
        R = nn.Dense(N * nb, param_dtype=F64, name="res_out", kernel_init=nn.initializers.zeros,
                     bias_init=nn.initializers.zeros)(z).reshape(B, N, nb)
        c = self.param("mps_copy", lambda *_: jnp.asarray(cpy), (nb,), F64)
        W = W + R + jnp.asarray(eye) + s[..., None] * c
        return W.reshape(B, N, 2, chi, chi)


def build_hybrid_model(N, chi, arch, d=16, heads=4, blocks=2):
    parts = arch.split(":")
    assert parts[1] == "convmlp", arch
    kw = {}
    keymap = {"r": "conv_r", "w": "anc_width", "al": "anc_layers", "rw": "res_width"}
    if len(parts) > 2 and parts[2]:
        for kv in parts[2].split(","):
            k, v = kv.split("=")
            kw[keymap[k]] = int(v)
    return HybridModel(N_sys=N, chi=chi, sys=parts[0], anc="conv", embedding_d=d, n_heads=heads, n_blocks=blocks, **kw)


class AblationModel(AltModel):
    """Ablaciones para entender por que vit:mlp gana a vit:conv a N<=10 (exacto):
      "convlin": conv r + termino de copia LINEAL GLOBAL completo s @ C, C (N, N*nb) (como en mlp; init = copia diagonal)
      "convloc": ventana local r, pero MLP con pesos PROPIOS por sitio (sin compartir; mismas cuentas que conv).
      "convdelta": pesos por sitio = compartidos (nombres de conv) + eps*D_k, D_k=0 al inicio; eps = e/1000 (arch ...:e=100)."""
    variant: str = "convlin"
    delta_milli: int = 100      # convdelta: eps = delta_milli / 1000

    def _ancilla(self, s, h):
        N, chi = self.N_sys, self.chi
        B = s.shape[0]
        nb = 2 * chi * chi
        eye = np.broadcast_to(np.eye(chi) * 0.5, (2, chi, chi)).reshape(-1)
        cpy = np.zeros((2, chi, chi))
        for a in (0, 1):
            cpy[a] = 0.5 * (2 * a - 1) * np.eye(chi)
        cpy = cpy.reshape(-1)
        r = self.conv_r
        idx = (np.arange(N)[:, None] + np.arange(-r, r + 1)[None, :]) % N
        z = s[:, idx]                                                                    # (B, N, 2r+1)
        if self.variant == "convlin":
            W = self._mlp(z, self.anc_width or 32, nb, "anc")
            kern = np.zeros((N, N * nb))
            for k in range(N):
                kern[k, k * nb:(k + 1) * nb] = cpy
            C = self.param("mps_copy", lambda *_: jnp.asarray(kern), (N, N * nb), F64)
            W = W + jnp.asarray(eye) + (s @ C).reshape(B, N, nb)
        elif self.variant == "convloc":
            w = self.anc_width or 32
            for li in range(self.anc_layers):
                fan = z.shape[-1]
                Wk = self.param(f"loc_h{li}", nn.initializers.normal(1.0 / np.sqrt(fan)), (N, fan, w), F64)
                bk = self.param(f"loc_b{li}", nn.initializers.zeros, (N, w), F64)
                z = nn.gelu(jnp.einsum("bnf,nfw->bnw", z, Wk) + bk)
            Wo = self.param("loc_out", nn.initializers.normal(self.mps_init / np.sqrt(w)), (N, w, nb), F64)
            bo = self.param("loc_bout", nn.initializers.zeros, (N, nb), F64)
            c = self.param("mps_copy", lambda *_: jnp.asarray(np.tile(cpy, (N, 1))), (N, nb), F64)
            W = jnp.einsum("bnw,nwo->bno", z, Wo) + bo + jnp.asarray(eye) + s[..., None] * c
        elif self.variant == "convdelta":
            # pesos por sitio = compartidos (mismos nombres que conv: carga un conv tal cual) + eps * D_k (D_k = 0 al inicio).
            # Con Adam el paso de D_k es ~lr, asi que el de eps*D_k es ~eps*lr: lr efectivo menor para la parte por sitio.
            eps = self.delta_milli / 1000.0
            w = self.anc_width or 32
            for li in range(self.anc_layers):
                fan = z.shape[-1]
                sh = nn.Dense(w, param_dtype=F64, name=f"anc_h{li}")(z)
                Dk = self.param(f"del_h{li}", nn.initializers.zeros, (N, fan, w), F64)
                db = self.param(f"del_b{li}", nn.initializers.zeros, (N, w), F64)
                z = nn.gelu(sh + eps * (jnp.einsum("bnf,nfw->bnw", z, Dk) + db))
            sh = nn.Dense(nb, param_dtype=F64, name="anc_out",
                          kernel_init=nn.initializers.normal(self.mps_init / np.sqrt(w)), bias_init=nn.initializers.zeros)(z)
            Do = self.param("del_out", nn.initializers.zeros, (N, w, nb), F64)
            dbo = self.param("del_bout", nn.initializers.zeros, (N, nb), F64)
            c = self.param("mps_copy", lambda *_: jnp.asarray(cpy), (nb,), F64)
            W = sh + eps * (jnp.einsum("bnw,nwo->bno", z, Do) + dbo) + jnp.asarray(eye) + s[..., None] * c
        else:
            raise ValueError(self.variant)
        return W.reshape(B, N, 2, chi, chi)


def build_ablation_model(N, chi, arch, d=16, heads=4, blocks=2):
    parts = arch.split(":")
    kw = {}
    keymap = {"r": "conv_r", "w": "anc_width", "al": "anc_layers", "e": "delta_milli"}
    if len(parts) > 2 and parts[2]:
        for kv in parts[2].split(","):
            k, v = kv.split("=")
            kw[keymap[k]] = int(v)
    return AblationModel(N_sys=N, chi=chi, sys=parts[0], anc="conv", variant=parts[1], embedding_d=d, n_heads=heads,
                         n_blocks=blocks, **kw)
