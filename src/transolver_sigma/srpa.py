"""Slice-Residual Physics-Attention, eqs 2-4."""

from __future__ import annotations

import jax
import jax.numpy as jnp

EPS = 1e-5
TAU_MIN = 0.1
TAU_MAX = 5.0
TAU_INIT = 0.5


def _lecun(key, shape, in_dim):
    return jax.random.normal(key, shape) * jnp.sqrt(1.0 / in_dim)


def init_srpa(key, c_phy: int, n_heads: int, n_slices: int) -> dict:
    """Routing, QKV, output projection, per-head tau, and ReZero gamma."""
    if c_phy % n_heads != 0:
        raise ValueError("Cphy must be divisible by nh")
    dh = c_phy // n_heads
    k_ws, k_q, k_k, k_v, k_o, k_r, k_f, k_dw_r, k_dw_f = jax.random.split(key, 9)
    return {
        "Ws": _lecun(k_ws, (dh, n_slices), dh),
        "bs": jnp.zeros((n_slices,)),
        "Wq": _lecun(k_q, (dh, dh), dh),
        "Wk": _lecun(k_k, (dh, dh), dh),
        "Wv": _lecun(k_v, (dh, dh), dh),
        "Wo": _lecun(k_o, (c_phy, c_phy), c_phy),
        "bo": jnp.zeros((c_phy,)),
        "Wr": _lecun(k_r, (c_phy, c_phy), c_phy),
        "br": jnp.zeros((c_phy,)),
        "Wf": _lecun(k_f, (c_phy, c_phy), c_phy),
        "bf": jnp.zeros((c_phy,)),
        "dw_r": jax.random.normal(k_dw_r, (5, 5, 1, c_phy)) * 0.02,
        "dw_f": jax.random.normal(k_dw_f, (5, 5, 1, c_phy)) * 0.02,
        "tau_tilde": jnp.full((n_heads,), TAU_INIT),
        "gamma": jnp.zeros(()),
    }


def clip_tau(tau_tilde):
    """tau_r = clip(tilde_tau_r, 0.1, 5)."""
    return jnp.clip(tau_tilde, TAU_MIN, TAU_MAX)


def srpa_core(R, F, Ws, bs, Wq, Wk, Wv, tau, gamma, eps: float = EPS):
    """Per-head SRPA algebra on already-split features.

    R, F: (..., N, dh)
    Ws: (dh, M), bs: (M,)
    Wq, Wk, Wv: (dh, dh), bias-free (eq 3)
    tau: positive scalar (already clipped)
    gamma: unconstrained residual scale, init 0 (ReZero)

    Returns P, D_diag, Z, A, DeltaZ, Z_plus, O.
    """
    # eq 2: P = softmax_M((R Ws + 1_N b_s^T) / tau)
    scores = (R @ Ws + bs) / tau
    P = jax.nn.softmax(scores, axis=-1)
    # eq 2: D = diag(P^T 1_N + eps 1_M)
    d_diag = jnp.sum(P, axis=-2) + eps
    # eq 2: Z = D^{-1} P^T F
    Z = jnp.einsum("...nm,...nd->...md", P, F) / d_diag[..., None]
    # eq 3: A = softmax_M((Z Wq)(Z Wk)^T / sqrt(dh)), DeltaZ = A Z Wv
    dh = R.shape[-1]
    Q = Z @ Wq
    K = Z @ Wk
    V = Z @ Wv
    attn_logits = (Q @ jnp.moveaxis(K, -1, -2)) / jnp.sqrt(jnp.asarray(dh, dtype=R.dtype))
    A = jax.nn.softmax(attn_logits, axis=-1)
    DeltaZ = A @ V
    # eq 3: Z+ = Z + gamma_ell DeltaZ
    Z_plus = Z + gamma * DeltaZ
    # eq 4: O^{(r)} = P^{(r)} Z+^{(r)}
    O = P @ Z_plus
    return {
        "P": P,
        "D_diag": d_diag,
        "Z": Z,
        "A": A,
        "DeltaZ": DeltaZ,
        "Z_plus": Z_plus,
        "O": O,
    }


def depthwise_conv2d(x, kernel, padding: int):
    """x: (B, H, W, C), kernel: (k, k, 1, C), feature_group_count = C."""
    c = x.shape[-1]
    return jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1, 1),
        padding=((padding, padding), (padding, padding)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=c,
    )


def project_routing_feature_linear(Y, W, b):
    """Pointwise fallback matching eq 2 on already-flat (..., N, Cphy) features."""
    return Y @ W + b


def project_routing_feature_conv(Y, dw_kernel, pw_w, pw_b):
    """Paper path: depthwise k=5 pad=2, then pointwise 1x1, on (B, H, W, Cphy)."""
    x = depthwise_conv2d(Y, dw_kernel, padding=2)
    return jnp.einsum("bhwc,co->bhwo", x, pw_w) + pw_b


def _split_heads(x, n_heads: int):
    """(B, N, Cphy) -> (B, nh, N, dh)."""
    b, n, c = x.shape
    dh = c // n_heads
    return x.reshape(b, n, n_heads, dh).transpose(0, 2, 1, 3)


def _merge_heads(x):
    """(B, nh, N, dh) -> (B, N, Cphy)."""
    b, nh, n, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, n, nh * dh)


def apply_srpa(params, Y, *, use_conv: bool = True, return_aux: bool = False):
    """Phi_phy(Y_phy). Grid path is (B, H, W, Cphy); linear path is (B, N, Cphy)."""
    tau = clip_tau(params["tau_tilde"])
    gamma = params["gamma"]
    if use_conv:
        if Y.ndim != 4:
            raise ValueError("grid SRPA expects (B, H, W, Cphy)")
        b, ht, wt, c_phy = Y.shape
        n = ht * wt
        R_grid = project_routing_feature_conv(Y, params["dw_r"], params["Wr"], params["br"])
        F_grid = project_routing_feature_conv(Y, params["dw_f"], params["Wf"], params["bf"])
        R = R_grid.reshape(b, n, c_phy)
        F = F_grid.reshape(b, n, c_phy)
        spatial = (ht, wt)
    else:
        if Y.ndim == 2:
            Y = Y[None, ...]
        if Y.ndim != 3:
            raise ValueError("linear SRPA expects (N, Cphy) or (B, N, Cphy)")
        b, n, c_phy = Y.shape
        R = project_routing_feature_linear(Y, params["Wr"], params["br"])
        F = project_routing_feature_linear(Y, params["Wf"], params["bf"])
        spatial = None
    dh = params["Ws"].shape[0]
    n_heads = c_phy // dh
    R_h = _split_heads(R, n_heads)
    F_h = _split_heads(F, n_heads)

    def _one_head(r, f, tau_h):
        return srpa_core(
            r, f, params["Ws"], params["bs"], params["Wq"], params["Wk"], params["Wv"], tau_h, gamma
        )

    aux = jax.vmap(_one_head, in_axes=(1, 1, 0), out_axes=1)(R_h, F_h, tau)
    O = _merge_heads(aux["O"])
    # eq 4: Phi_phy = Projo(Concat_r O^{(r)}), output projection has bias
    phi = O @ params["Wo"] + params["bo"]
    if use_conv:
        phi = phi.reshape(b, spatial[0], spatial[1], c_phy)
    if return_aux:
        return phi, aux
    return phi
