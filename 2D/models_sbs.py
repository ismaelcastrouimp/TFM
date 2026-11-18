"""
2D/models_sbs.py
================
Ancilla "string-bond" (idea de Lu, Giudice, Cirac, PRB 111, 075102 (2025), SnakeSBS) para el ARViT + ancilla MPS en 2D:
varias MPS de ancilla ("strings"), cada una recorriendo la red con un camino distinto (serpentina por filas, por
columnas, ...).  La matriz densidad es el producto de Hadamard

    rho(s, t) = A(s) A*(t) * prod_i G_i(s, t),    G_i(s, t) = <phi_i(t)|phi_i(s)> / (||phi_i(s)|| ||phi_i(t)||),

equivalente a una purificación con n copias de ancilla (|phi_1(s)> ⊗ ... ⊗ |phi_n(s)>): hermítica y PSD por construcción,
y la ancilla se traza exactamente (no se muestrea).  Ningún corte de la red tiene que pasar todas las correlaciones por
el enlace de una sola MPS.

Tensores de la string i en la posición k de su camino: MLP (propio de cada string) de la ventana 2D alrededor del sitio
path_i[k] (como "conv2d"), + copy-init (cada G_i = delta al empezar -> rho = 1/2^N).
arch: "vit:sbs:r=1,win=plus,paths=rc"  (paths: r = serpentina por filas, c = por columnas, R/C = las mismas invertidas,
w = barrido por filas sin serpentina; p. ej. "r" = 1 string, "rc" = 2, "rcRC" = 4).  Claves r, w, al como en conv2d.
Ancilla producto neuronal (RBM generalizada): chi=1, win=all, paths="w"*β -> β cúbits de ancilla por sitio, cada uno
con estado φ_k(s) dado por un MLP de TODA la configuración; rho(s,t) = A(s)A*(t) prod_k <φ_k(t)|φ_k(s)> (sin gauge).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
import jax.numpy as jnp
from src_renyi.mps_ancilla.models_alt import AltModel
from src_renyi.mps_ancilla.models import mps_overlap_rect
from models2d import window_idx

jax.config.update("jax_enable_x64", True)
F64 = jnp.float64


def snake_path(Lx, Ly, kind):
    """Orden de sitios (k = x + Lx*y) del camino de una string."""
    if kind in "rR":
        p = [x + Lx * y for y in range(Ly) for x in (range(Lx) if y % 2 == 0 else range(Lx - 1, -1, -1))]
    elif kind in "cC":
        p = [x + Lx * y for x in range(Lx) for y in (range(Ly) if x % 2 == 0 else range(Ly - 1, -1, -1))]
    elif kind == "w":
        p = list(range(Lx * Ly))
    else:
        raise ValueError(kind)
    return np.asarray(p[::-1] if kind in "RC" else p)


class ModelSBS(AltModel):
    Lx: int = 3
    Ly: int = 3
    win: str = "plus"
    paths: str = "rc"

    def _ancilla(self, s, h):
        """Devuelve (B, n*N, 2, chi, chi): las n strings concatenadas (string i = filas i*N..(i+1)*N-1)."""
        N, chi = self.N_sys, self.chi
        B, nb = s.shape[0], 2 * chi * chi
        eye = np.broadcast_to(np.eye(chi) * 0.5, (2, chi, chi)).reshape(-1)
        cpy = np.zeros((2, chi, chi))
        for a in (0, 1):
            cpy[a] = 0.5 * (2 * a - 1) * np.eye(chi)
        cpy = cpy.reshape(-1)
        if self.win == "all":       # ventana = toda la red, desplazada al sitio (MLP compartido, ve toda s)
            win = (np.arange(N)[:, None] + np.arange(N)[None, :]) % N
        else:
            win = window_idx(self.Lx, self.Ly, self.conv_r, self.win)                 # (N, n_win)
        out = []
        for i, kind in enumerate(self.paths):
            path = snake_path(self.Lx, self.Ly, kind)
            z = s[:, win[path]]                                                       # (B, N, n_win) en orden del camino
            W = self._mlp(z, self.anc_width or 32, nb, f"anc{i}")
            c = self.param(f"mps_copy{i}", lambda *_: jnp.asarray(cpy), (nb,), F64)
            out.append((W + jnp.asarray(eye) + s[:, path][..., None] * c).reshape(B, N, 2, chi, chi))
        return jnp.concatenate(out, axis=1)


def build_model_sbs(Lx, Ly, chi, arch, d=16, heads=4, blocks=2):
    parts = arch.split(":")
    assert parts[1] == "sbs"
    kw, win, paths = {}, "plus", "rc"
    keymap = {"r": "conv_r", "w": "anc_width", "al": "anc_layers"}
    if len(parts) > 2 and parts[2]:
        for kv in parts[2].split(","):
            k, v = kv.split("=")
            if k == "win": win = v
            elif k == "paths": paths = v
            else: kw[keymap[k]] = int(v)
    return ModelSBS(N_sys=Lx * Ly, chi=chi, sys=parts[0], anc="sbs", embedding_d=d, n_heads=heads, n_blocks=blocks,
                    Lx=Lx, Ly=Ly, win=win, paths=paths, **kw)


def compute_rho_sbs(model, params, all_s, chunk=None):
    logA, M = model.apply({"params": params}, all_s)
    A = jnp.exp(logA - jnp.max(jnp.real(logA)))
    A = A / jnp.linalg.norm(A)
    N, n = model.N_sys, len(model.paths)
    G = 1.0
    for i in range(n):
        O = mps_overlap_rect(M[:, i * N:(i + 1) * N], M[:, i * N:(i + 1) * N], model.chi, chunk)
        nrm = jnp.sqrt(jnp.real(jnp.diagonal(O)) + 1e-300)
        G = G * (O / (nrm[:, None] * nrm[None, :]))
    return A[:, None] * jnp.conj(A)[None, :] * G


def make_loss_fns_sbs(model, all_s, H_dense, T, chunk=None):
    all_s = jnp.asarray(all_s)
    H = jnp.asarray(H_dense)

    def loss(params):
        rho = compute_rho_sbs(model, params, all_s, chunk)
        E = jnp.real(jnp.sum(H.T * rho))
        S2 = -jnp.log(jnp.sum(jnp.abs(rho) ** 2))
        return E - T * S2, (E, S2)

    return loss, jax.jit(lambda p: compute_rho_sbs(model, p, all_s, chunk))


def _tests():
    # caminos: permutaciones y vecinos consecutivos (serpentina)
    for k in "rcRCw":
        p = snake_path(3, 4, k); assert sorted(p) == list(range(12))
    for k in "rc":
        p = snake_path(4, 3, k)
        d = [abs(a % 4 - b % 4) + abs(a // 4 - b // 4) for a, b in zip(p[:-1], p[1:])]
        assert all(x == 1 for x in d), (k, d)
    # rho válida (hermítica, traza 1, PSD) y copy-init -> rho = I/2^N
    from src_renyi.mps_ancilla.ladder import init_params
    Lx, Ly = 3, 3; N = 9
    idx = np.arange(2 ** N); all_s = (2 * ((idx[:, None] >> np.arange(N)[None, :]) & 1) - 1).astype(np.float64)
    m = build_model_sbs(Lx, Ly, 2, "vit:sbs:r=1,win=plus,paths=rc")
    p0 = init_params(m, N, 1)
    rho0 = np.asarray(compute_rho_sbs(m, p0, all_s))
    p = jax.tree_util.tree_map(lambda x: x + 0.2 * jax.random.normal(jax.random.PRNGKey(x.size), x.shape), p0)
    rho = np.asarray(compute_rho_sbs(m, p, all_s))
    ev = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    print("copy-init: |rho - I/2^N|max =", np.abs(rho0 - np.eye(2 ** N) / 2 ** N).max())
    print("perturbado: herm", np.abs(rho - rho.conj().T).max(), " traza", np.trace(rho).real, " min autovalor", ev.min(),
          " S2", -np.log(np.sum(np.abs(rho) ** 2)))
    return True


if __name__ == "__main__":
    print("tests:", _tests())
