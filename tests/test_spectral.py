"""Checks of axis-factorized Fourier mixing, eqs 40-43."""

import jax
import jax.numpy as jnp
import numpy as np

from transolver_sigma.spectral import (
    apply_axis,
    axis_factorized_fourier,
    init_spectral,
    mix_rfft,
    rfft_axis,
    rfft_irfft_roundtrip,
)


def test_rfft_irfft_roundtrip_identity():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 8, 8, 4)).astype(np.float32)
    y = rfft_irfft_roundtrip(jnp.asarray(x), axis=1)
    np.testing.assert_allclose(np.asarray(y), x, atol=1e-5)
    y2 = rfft_irfft_roundtrip(jnp.asarray(x), axis=2)
    np.testing.assert_allclose(np.asarray(y2), x, atol=1e-5)


def test_modes_above_k_zeroed_before_irfft():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(1, 8, 6, 3)).astype(np.float32)
    k_keep = 2
    w_real = rng.normal(size=(k_keep, 3, 3)).astype(np.float32)
    w_imag = rng.normal(size=(k_keep, 3, 3)).astype(np.float32)
    vhat = rfft_axis(jnp.asarray(v), axis=1)
    rhat = mix_rfft(vhat, jnp.asarray(w_real), jnp.asarray(w_imag), freq_axis=1)
    rhat_np = np.asarray(rhat)
    np.testing.assert_allclose(rhat_np[:, k_keep:, :, :], 0.0, atol=1e-7)
    assert rhat.shape == vhat.shape


def test_two_axis_sum_matches_numpy_fft():
    rng = np.random.default_rng(2)
    n1, n2, c = 4, 6, 2
    v = rng.normal(size=(1, n1, n2, c)).astype(np.float32)
    k_keep = 2
    wx_r = rng.normal(size=(k_keep, c, c)).astype(np.float32)
    wx_i = rng.normal(size=(k_keep, c, c)).astype(np.float32)
    wy_r = rng.normal(size=(k_keep, c, c)).astype(np.float32)
    wy_i = rng.normal(size=(k_keep, c, c)).astype(np.float32)
    params = {
        "W_real": jnp.stack([jnp.asarray(wx_r), jnp.asarray(wy_r)], axis=0),
        "W_imag": jnp.stack([jnp.asarray(wx_i), jnp.asarray(wy_i)], axis=0),
    }
    phi, axes = axis_factorized_fourier(params, jnp.asarray(v), return_axes=True)
    rx, ry = axes[0][0], axes[1][0]
    np.testing.assert_allclose(np.asarray(phi), np.asarray(rx + ry), atol=1e-5)

    def numpy_axis(field, w_r, w_i, axis):
        vhat = np.fft.rfft(field, axis=axis, norm="ortho")
        w = w_r + 1j * w_i
        v_m = np.moveaxis(vhat, axis, -2)
        f_len = v_m.shape[-2]
        k_eff = min(w.shape[0], f_len)
        mixed = np.zeros_like(v_m)
        for k in range(k_eff):
            mixed[..., k, :] = v_m[..., k, :] @ w[k]
        rhat = np.moveaxis(mixed, -2, axis)
        return np.fft.irfft(rhat, n=field.shape[axis], axis=axis, norm="ortho")

    field = v[0]
    rx_ref = numpy_axis(field, wx_r, wx_i, axis=0)
    ry_ref = numpy_axis(field, wy_r, wy_i, axis=1)
    np.testing.assert_allclose(np.asarray(phi[0]), rx_ref + ry_ref, atol=1e-5)


def test_same_w_at_every_remaining_location():
    rng = np.random.default_rng(3)
    n1, n2, c = 8, 5, 3
    v = rng.normal(size=(1, n1, n2, c)).astype(np.float32)
    k_keep = 3
    w_real = rng.normal(size=(k_keep, c, c)).astype(np.float32)
    w_imag = rng.normal(size=(k_keep, c, c)).astype(np.float32)
    vhat = np.fft.rfft(v, axis=1, norm="ortho")
    rhat = mix_rfft(jnp.asarray(vhat), jnp.asarray(w_real), jnp.asarray(w_imag), freq_axis=1)
    w = w_real + 1j * w_imag
    rhat_np = np.asarray(rhat)
    for loc in range(n2):
        for k in range(k_keep):
            expected = vhat[0, k, loc, :] @ w[k]
            np.testing.assert_allclose(rhat_np[0, k, loc, :], expected, atol=1e-5)
    loc0 = rhat_np[0, 1, 0, :]
    loc1 = rhat_np[0, 1, 3, :]
    recon0 = vhat[0, 1, 0, :] @ w[1]
    recon1 = vhat[0, 1, 3, :] @ w[1]
    np.testing.assert_allclose(loc0, recon0, atol=1e-5)
    np.testing.assert_allclose(loc1, recon1, atol=1e-5)


def test_apply_axis_restores_length():
    key = jax.random.PRNGKey(0)
    params = init_spectral(key, c_spec=4, n_modes=2, n_axes=2)
    y = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 8, 4))
    r0, _ = apply_axis(y, params["W_real"][0], params["W_imag"][0], axis=1)
    r1, _ = apply_axis(y, params["W_real"][1], params["W_imag"][1], axis=2)
    assert r0.shape == y.shape
    assert r1.shape == y.shape
