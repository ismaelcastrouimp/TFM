"""
2D/models2d.py
==============
ARViT + ancilla MPS para el TFIM en una red Lx x Ly (PBC).  Sitio k = x + Lx*y (orden de barrido por filas), que es a
la vez el orden autorregresivo del ARViT y el orden de la cadena MPS de la ancilla.

Reutiliza AltModel (src_renyi/mps_ancilla/models_alt.py) sin tocarlo:
  - sistema: el ARViT "vit" tal cual (tabla de posiciones aprendida (N+1, d): puede aprender la geometria 2D).
  - ancilla "conv2d": como "conv", pero la ventana de M_k(s) son los vecinos 2D del sitio k en la red:
        win="plus": k y sus vecinos a distancia Manhattan <= r   (r=1: 5 sitios)
        win="sq"  : cuadrado (2r+1)x(2r+1)                         (r=1: 9 sitios)
    MLP compartido (invariante traslacional), mismo copy-init (rho = 1/2^N al empezar).
Con Ly=1 y win=plus, conv2d es EXACTAMENTE el conv 1D (mismos nombres de parametros y mismo orden de ventana):
lo comprueba test_1d_equivalence().
arch: "vit:conv2d:r=1,win=plus" (claves r, w, al como en models_alt; win plus|sq).  "vit:conv:r=1" = conv 1D en el
orden de barrido (control: no ve a los vecinos verticales).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import jax
import jax.numpy as jnp
from src_renyi.mps_ancilla.models_alt import AltModel, build_alt_model

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


def window_idx(Lx, Ly, r=1, win="plus"):
    """(N, n_win) indices de la ventana de cada sitio; orden: offsets (dy, dx) crecientes (dy=0 si Ly=1)."""
    offs = [(dy, dx) for dy in (range(-r, r + 1) if Ly > 1 else [0]) for dx in range(-r, r + 1)
            if win == "sq" or abs(dx) + abs(dy) <= r]
    idx = np.zeros((Lx * Ly, len(offs)), int)
    for y in range(Ly):
        for x in range(Lx):
            idx[x + Lx * y] = [((x + dx) % Lx) + Lx * ((y + dy) % Ly) for dy, dx in offs]
    return idx


class Model2D(AltModel):
    Lx: int = 3
    Ly: int = 3
    win: str = "plus"

    def _ancilla(self, s, h):
        if self.anc != "conv2d":
            return super()._ancilla(s, h)
        N, chi = self.N_sys, self.chi
        B, nb = s.shape[0], 2 * chi * chi
        eye = np.broadcast_to(np.eye(chi) * 0.5, (2, chi, chi)).reshape(-1)
        cpy = np.zeros((2, chi, chi))
        for a in (0, 1):
            cpy[a] = 0.5 * (2 * a - 1) * np.eye(chi)
        cpy = cpy.reshape(-1)
        z = s[:, window_idx(self.Lx, self.Ly, self.conv_r, self.win)]                 # (B, N, n_win)
        W = self._mlp(z, self.anc_width or 32, nb, "anc")
        c = self.param("mps_copy", lambda *_: jnp.asarray(cpy), (nb,), F64)
        W = W + jnp.asarray(eye) + s[..., None] * c
        return W.reshape(B, N, 2, chi, chi)


def build_model_2d(Lx, Ly, chi, arch, d=16, heads=4, blocks=2):
    parts = arch.split(":")
    if parts[1] != "conv2d":
        return build_alt_model(Lx * Ly, chi, arch, d, heads, blocks)
    kw, win = {}, "plus"
    keymap = {"r": "conv_r", "w": "anc_width", "al": "anc_layers"}
    if len(parts) > 2 and parts[2]:
        for kv in parts[2].split(","):
            k, v = kv.split("=")
            if k == "win": win = v
            else: kw[keymap[k]] = int(v)
    return Model2D(N_sys=Lx * Ly, chi=chi, sys=parts[0], anc="conv2d", embedding_d=d, n_heads=heads, n_blocks=blocks,
                   Lx=Lx, Ly=Ly, win=win, **kw)


def test_1d_equivalence(N=7, chi=4, seed=3):
    """Ly=1, win=plus  ==  vit:conv (1D) con los mismos parametros."""
    m2 = build_model_2d(N, 1, chi, "vit:conv2d:r=1,win=plus"); m1 = build_alt_model(N, chi, "vit:conv:r=1")
    p = m1.init(jax.random.PRNGKey(seed), jnp.ones((1, N)))["params"]
    p = jax.tree_util.tree_map(lambda x: x + 0.1 * jax.random.normal(jax.random.PRNGKey(1), x.shape), p)  # no trivial
    s = jnp.asarray(2.0 * jax.random.bernoulli(jax.random.PRNGKey(2), 0.5, (16, N)) - 1)
    a1, M1 = m1.apply({"params": p}, s); a2, M2 = m2.apply({"params": p}, s)
    assert jnp.allclose(a1, a2) and jnp.allclose(M1, M2), "conv2d(Ly=1) != conv 1D"
    assert sorted(window_idx(4, 4, 1, "plus")[5]) == sorted([5, 4, 6, 1, 9]), "ventana 2D mal"
    return True


if __name__ == "__main__":
    print("test_1d_equivalence:", test_1d_equivalence())
