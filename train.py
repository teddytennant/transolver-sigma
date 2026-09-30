"""Tiny CPU smoke: 2-block Transolver-σ on random 8x8 fields."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from transolver_sigma.model import adam_init, adam_step, apply_model, init_model, relative_l2

C = 16
L = 2
M = 4
K = 2
STEPS = 30
BATCH = 16
HEIGHT = 8
WIDTH = 8
LR = 3e-3
SEED = 0


def smooth_field(key, batch, height, width, cutoff=3):
    noise = jax.random.normal(key, (batch, height, width, 1))
    spec = jnp.fft.rfft2(noise[..., 0], axes=(1, 2), norm="ortho")
    ky = jnp.fft.fftfreq(height)[:, None]
    kx = jnp.fft.rfftfreq(width)[None, :]
    mask = (ky**2 + kx**2) <= (cutoff / max(height, width)) ** 2
    spec = spec * mask[None, ...]
    out = jnp.fft.irfft2(spec, s=(height, width), axes=(1, 2), norm="ortho")
    return out[..., None]


def poisson_like(field):
    spec = jnp.fft.rfft2(field[..., 0], axes=(1, 2), norm="ortho")
    ky = jnp.fft.fftfreq(field.shape[1])[:, None]
    kx = jnp.fft.rfftfreq(field.shape[2])[None, :]
    k2 = ky**2 + kx**2
    filt = 1.0 / (1.0 + k2 * (field.shape[1] ** 2))
    out = jnp.fft.irfft2(spec * filt[None, ...], s=field.shape[1:3], axes=(1, 2), norm="ortho")
    return out[..., None]


def main() -> None:
    key = jax.random.PRNGKey(SEED)
    k_data, k_model = jax.random.split(key)
    x = smooth_field(k_data, BATCH, HEIGHT, WIDTH, cutoff=3)
    y = poisson_like(x)
    params = init_model(
        k_model, cin=1, cout=1, c=C, n_blocks=L, n_heads=2, n_slices=M, n_modes=K
    )
    opt = adam_init(params)

    def loss_fn(p):
        pred = apply_model(p, x)
        return relative_l2(pred, y)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    start = float(loss_fn(params))
    for _ in range(STEPS):
        _, grads = loss_and_grad(params)
        params, opt = adam_step(params, grads, opt, lr=LR)
    end = float(loss_fn(params))
    results = {
        "train_rel_l2_start": start,
        "train_rel_l2_end": end,
        "steps": STEPS,
        "C": C,
        "L": L,
        "M": M,
        "K": K,
    }
    out = Path("results.json")
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
