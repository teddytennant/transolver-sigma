"""Axis-factorized Fourier operator, eqs 40-43."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def init_spectral(key, c_spec: int, n_modes: int, n_axes: int = 2) -> dict:
    k1, k2 = jax.random.split(key)
    scale = 1.0 / c_spec
    return {
        "W_real": jax.random.normal(k1, (n_axes, n_modes, c_spec, c_spec)) * scale,
        "W_imag": jax.random.normal(k2, (n_axes, n_modes, c_spec, c_spec)) * scale,
    }


def rfft_axis(v, axis: int):
    """eq 40: Vhat_a = rFFT_a(V) with orthonormal normalization."""
    return jnp.fft.rfft(v, axis=axis, norm="ortho")


def irfft_axis(vhat, axis: int, n: int):
    """eq 42: R_a = irFFT_a(Rhat_a), restore length na, orthonormal."""
    return jnp.fft.irfft(vhat, n=n, axis=axis, norm="ortho")


def rfft_irfft_roundtrip(v, axis: int):
    """Unfiltered rFFT/irFFT roundtrip along one axis."""
    n = v.shape[axis]
    return irfft_axis(rfft_axis(v, axis), axis, n)


def mix_rfft(vhat, w_real, w_imag, freq_axis: int):
    """eq 41: Rhat_a(k) = Vhat_a(k) W_a(k) for 0 <= k < K_a, else 0.

    W_a(k) in C^{Cspec x Cspec} is stored as real/imag float arrays.
    The same W_a(k) is applied at every location of the remaining axes.
    K_a is capped at floor(na/2)+1, which is the rFFT length along freq_axis.
    """
    w = w_real.astype(vhat.real.dtype) + 1j * w_imag.astype(vhat.real.dtype)
    v = jnp.moveaxis(vhat, freq_axis, -2)
    freq_len = v.shape[-2]
    channels = v.shape[-1]
    k_keep = min(int(w.shape[0]), int(freq_len))
    mixed_low = jnp.einsum("...fc,fco->...fo", v[..., :k_keep, :], w[:k_keep])
    high = jnp.zeros(v.shape[:-2] + (freq_len - k_keep, channels), dtype=v.dtype)
    mixed = jnp.concatenate([mixed_low, high], axis=-2)
    return jnp.moveaxis(mixed, -2, freq_axis)


def apply_axis(v, w_real, w_imag, axis: int):
    """One axis of eqs 40-42: rFFT, mix/truncate, irFFT back to length na."""
    n_a = v.shape[axis]
    vhat = rfft_axis(v, axis)
    rhat = mix_rfft(vhat, w_real, w_imag, freq_axis=axis)
    return irfft_axis(rhat, axis, n_a), rhat


def axis_factorized_fourier(params, y, return_axes: bool = False):
    """eqs 40-43 on 1D (B, n1, Cspec) or 2D (B, n1, n2, Cspec) grids.

    Y_spec is already LN(H_spec) in grid shape. Phi_spec = sum_a R_a.
    """
    spatial_ndim = y.ndim - 2
    acc = jnp.zeros_like(y)
    axis_out = []
    for a in range(spatial_ndim):
        axis = a + 1
        r_a, rhat_a = apply_axis(y, params["W_real"][a], params["W_imag"][a], axis)
        acc = acc + r_a
        axis_out.append((r_a, rhat_a))
    if return_axes:
        return acc, axis_out
    return acc
