"""Forward shape, residual identity, and a tiny Adam fit."""

import jax
import jax.numpy as jnp
import numpy as np

from transolver_sigma.block import apply_block
from transolver_sigma.model import adam_init, adam_step, apply_model, init_model, relative_l2


def _smooth_field(key, batch, height, width, cutoff=3):
    noise = jax.random.normal(key, (batch, height, width, 1))
    spec = jnp.fft.rfft2(noise[..., 0], axes=(1, 2), norm="ortho")
    ky = jnp.fft.fftfreq(height)[:, None]
    kx = jnp.fft.rfftfreq(width)[None, :]
    mask = (ky**2 + kx**2) <= (cutoff / max(height, width)) ** 2
    spec = spec * mask[None, ...]
    out = jnp.fft.irfft2(spec, s=(height, width), axes=(1, 2), norm="ortho")
    return out[..., None]


def _poisson_like(field):
    spec = jnp.fft.rfft2(field[..., 0], axes=(1, 2), norm="ortho")
    ky = jnp.fft.fftfreq(field.shape[1])[:, None]
    kx = jnp.fft.rfftfreq(field.shape[2])[None, :]
    k2 = ky**2 + kx**2
    filt = 1.0 / (1.0 + k2 * (field.shape[1] ** 2))
    out = jnp.fft.irfft2(spec * filt[None, ...], s=field.shape[1:3], axes=(1, 2), norm="ortho")
    return out[..., None]


def test_forward_shape():
    key = jax.random.PRNGKey(0)
    params = init_model(key, cin=1, cout=1, c=16, n_blocks=2, n_heads=2, n_slices=4, n_modes=2)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 1))
    y = apply_model(params, x)
    assert y.shape == (2, 8, 8, 1)


def test_zero_spectral_weights_identity_on_spec_branch():
    key = jax.random.PRNGKey(2)
    params = init_model(key, cin=1, cout=1, c=16, n_blocks=1, n_heads=2, n_slices=4, n_modes=2)
    blk = params["blocks"][0]
    blk = dict(blk)
    blk["spec"] = {
        "W_real": jnp.zeros_like(blk["spec"]["W_real"]),
        "W_imag": jnp.zeros_like(blk["spec"]["W_imag"]),
    }
    h = jax.random.normal(jax.random.PRNGKey(3), (2, 8, 8, 16))
    _, aux = apply_block(blk, h, return_aux=True)
    np.testing.assert_allclose(
        np.asarray(aux["htilde_spec"]), np.asarray(aux["h_spec"]), atol=1e-5
    )
    np.testing.assert_allclose(float(blk["phy"]["gamma"]), 0.0, atol=0)


def test_adam_decreases_train_rel_l2():
    key = jax.random.PRNGKey(4)
    k_data, k_model = jax.random.split(key)
    x = _smooth_field(k_data, batch=16, height=8, width=8, cutoff=3)
    y = _poisson_like(x)
    params = init_model(k_model, cin=1, cout=1, c=16, n_blocks=2, n_heads=2, n_slices=4, n_modes=2)
    opt = adam_init(params)

    def loss_fn(p):
        pred = apply_model(p, x)
        return relative_l2(pred, y)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    start, _ = loss_and_grad(params)
    for _ in range(20):
        loss, grads = loss_and_grad(params)
        params, opt = adam_step(params, grads, opt, lr=3e-3)
    end = loss_fn(params)
    assert float(end) < float(start)
