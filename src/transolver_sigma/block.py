"""Dual-subspace block, eqs 1a-1c, plus SwiGLU eq 39 and CPE."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from transolver_sigma.spectral import axis_factorized_fourier, init_spectral
from transolver_sigma.srpa import apply_srpa, depthwise_conv2d, init_srpa


def layer_norm(x, scale, bias, eps: float = 1e-5):
    """LN over the channel axis."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return scale * (x - mean) * jax.lax.rsqrt(var + eps) + bias


def swiglu_hidden_width(c: int, rho: float = 2.0) -> int:
    """Cf = floor(2 * rho * C / 3), default rho=2 so Cf = floor(4C/3)."""
    return int(np.floor(2.0 * rho * c / 3.0))


def init_swiglu(key, c: int, rho: float = 2.0) -> dict:
    cf = swiglu_hidden_width(c, rho)
    kg, ku, kd = jax.random.split(key, 3)
    scale_in = jnp.sqrt(1.0 / c)
    scale_out = jnp.sqrt(1.0 / cf)
    return {
        "Wg": jax.random.normal(kg, (c, cf)) * scale_in,
        "bg": jnp.zeros((cf,)),
        "Wu": jax.random.normal(ku, (c, cf)) * scale_in,
        "bu": jnp.zeros((cf,)),
        "Wd": jax.random.normal(kd, (cf, c)) * scale_out,
        "bd": jnp.zeros((c,)),
    }


def swiglu(params, h):
    """eq 39: SwiGLU(H) = [SiLU(H Wg + bg) odot (H Wu + bu)] Wd + bd."""
    gate = jax.nn.silu(h @ params["Wg"] + params["bg"])
    up = h @ params["Wu"] + params["bu"]
    return (gate * up) @ params["Wd"] + params["bd"]


def cpe(h, kernel):
    """Appendix B.3: depthwise conv kernel size 3, padding 1, one kernel per channel."""
    return depthwise_conv2d(h, kernel, padding=1)


def split_subspaces(h):
    """eq 1a: [H_spec, H_phy] = Split_{C/2}(H'), spectral channels first."""
    c = h.shape[-1]
    if c % 2 != 0:
        raise ValueError("C must be even")
    half = c // 2
    return h[..., :half], h[..., half:]


def init_ln(c: int) -> dict:
    return {"scale": jnp.ones((c,)), "bias": jnp.zeros((c,))}


def init_block(key, c: int, n_heads: int, n_slices: int, n_modes: int, rho: float = 2.0) -> dict:
    if c % (2 * n_heads) != 0:
        raise ValueError("C must be divisible by 2*nh")
    c_half = c // 2
    k_cpe, k_spec, k_phy, k_sw = jax.random.split(key, 4)
    return {
        "cpe_kernel": jax.random.normal(k_cpe, (3, 3, 1, c)) * 0.02,
        "ln_spec": init_ln(c_half),
        "ln_phy": init_ln(c_half),
        "ln_cat": init_ln(c),
        "spec": init_spectral(k_spec, c_half, n_modes, n_axes=2),
        "phy": init_srpa(k_phy, c_half, n_heads, n_slices),
        "swiglu": init_swiglu(k_sw, c, rho),
    }


def apply_block(params, h, return_aux: bool = False):
    """One dual-subspace block, eqs 1a-1c. h: (B, H, W, C)."""
    # eq 1a: H' = H + CPE(H)
    h_prime = h + cpe(h, params["cpe_kernel"])
    h_spec, h_phy = split_subspaces(h_prime)
    y_spec = layer_norm(h_spec, params["ln_spec"]["scale"], params["ln_spec"]["bias"])
    y_phy = layer_norm(h_phy, params["ln_phy"]["scale"], params["ln_phy"]["bias"])
    phi_spec = axis_factorized_fourier(params["spec"], y_spec)
    phi_phy = apply_srpa(params["phy"], y_phy, use_conv=True)
    # eq 1b: Htilde_* = H_* + Phi_*(LN_*(H_*))
    htilde_spec = h_spec + phi_spec
    htilde_phy = h_phy + phi_phy
    # eq 1c: H_cat = Concat(Htilde_spec, Htilde_phy)
    h_cat = jnp.concatenate([htilde_spec, htilde_phy], axis=-1)
    y_cat = layer_norm(h_cat, params["ln_cat"]["scale"], params["ln_cat"]["bias"])
    sw = swiglu(params["swiglu"], y_cat)
    h_next = h_cat + sw
    if return_aux:
        return h_next, {
            "h_prime": h_prime,
            "h_spec": h_spec,
            "h_phy": h_phy,
            "y_spec": y_spec,
            "y_phy": y_phy,
            "phi_spec": phi_spec,
            "phi_phy": phi_phy,
            "htilde_spec": htilde_spec,
            "htilde_phy": htilde_phy,
            "h_cat": h_cat,
            "y_cat": y_cat,
            "swiglu": sw,
        }
    return h_next
