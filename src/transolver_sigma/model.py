"""Lift, L dual-subspace blocks, LN + linear readout."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from transolver_sigma.block import apply_block, init_block, layer_norm


def coordinate_grid(height: int, width: int, dtype):
    y = jnp.linspace(0.0, 1.0, height, dtype=dtype)
    x = jnp.linspace(0.0, 1.0, width, dtype=dtype)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    return jnp.stack([xx, yy], axis=-1)


def init_model(
    key,
    cin: int,
    cout: int,
    c: int = 16,
    n_blocks: int = 2,
    n_heads: int = 2,
    n_slices: int = 4,
    n_modes: int = 2,
    rho: float = 2.0,
) -> dict:
    if c % (2 * n_heads) != 0:
        raise ValueError("C must be divisible by 2*nh")
    k_l1, k_l2, k_blocks, k_read = jax.random.split(key, 4)
    in_dim = cin + 2
    lift_W1 = jax.random.normal(k_l1, (in_dim, c)) * jnp.sqrt(1.0 / in_dim)
    lift_W2 = jax.random.normal(k_l2, (c, c)) * jnp.sqrt(1.0 / c)
    block_keys = jax.random.split(k_blocks, n_blocks)
    blocks = tuple(
        init_block(block_keys[i], c, n_heads, n_slices, n_modes, rho) for i in range(n_blocks)
    )
    readout_W = jax.random.normal(k_read, (c, cout)) * jnp.sqrt(1.0 / c)
    return {
        "lift_W1": lift_W1,
        "lift_b1": jnp.zeros((c,)),
        "lift_W2": lift_W2,
        "lift_b2": jnp.zeros((c,)),
        "blocks": blocks,
        "ln_scale": jnp.ones((c,)),
        "ln_bias": jnp.zeros((c,)),
        "readout_W": readout_W,
        "readout_b": jnp.zeros((cout,)),
    }


def apply_model(params, field):
    """field: (B, H, W, Cin) -> (B, H, W, Cout)."""
    batch, height, width, _ = field.shape
    coords = coordinate_grid(height, width, field.dtype)
    coords = jnp.broadcast_to(coords[None, ...], (batch, height, width, 2))
    x = jnp.concatenate([field, coords], axis=-1)
    h = jax.nn.gelu(x @ params["lift_W1"] + params["lift_b1"])
    h = h @ params["lift_W2"] + params["lift_b2"]
    for blk in params["blocks"]:
        h = apply_block(blk, h)
    h = layer_norm(h, params["ln_scale"], params["ln_bias"])
    return h @ params["readout_W"] + params["readout_b"]


def relative_l2(pred, target, eps: float = 1e-8):
    num = jnp.sqrt(jnp.sum((pred - target) ** 2))
    den = jnp.sqrt(jnp.sum(target ** 2))
    return num / (den + eps)


def adam_init(params):
    return {
        "m": jtu.tree_map(jnp.zeros_like, params),
        "v": jtu.tree_map(jnp.zeros_like, params),
        "t": jnp.array(0, dtype=jnp.int32),
    }


def adam_step(params, grads, opt_state, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
    t = opt_state["t"] + 1
    m = jtu.tree_map(lambda mi, g: b1 * mi + (1.0 - b1) * g, opt_state["m"], grads)
    v = jtu.tree_map(lambda vi, g: b2 * vi + (1.0 - b2) * (g * g), opt_state["v"], grads)
    mhat = jtu.tree_map(lambda mi: mi / (1.0 - b1 ** t), m)
    vhat = jtu.tree_map(lambda vi: vi / (1.0 - b2 ** t), v)
    params = jtu.tree_map(lambda p, mi, vi: p - lr * mi / (jnp.sqrt(vi) + eps), params, mhat, vhat)
    return params, {"m": m, "v": v, "t": t}
