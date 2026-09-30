"""Checks of the dual-subspace block, eqs 1a-1c and SwiGLU eq 39."""

import jax
import jax.numpy as jnp
import numpy as np

from transolver_sigma.block import (
    apply_block,
    init_block,
    layer_norm,
    split_subspaces,
    swiglu,
    swiglu_hidden_width,
)
from transolver_sigma.spectral import axis_factorized_fourier
from transolver_sigma.srpa import apply_srpa


def test_split_even_spectral_first():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(2, 4, 4, 16)).astype(np.float32)
    h_spec, h_phy = split_subspaces(jnp.asarray(h))
    c = h.shape[-1]
    assert h_spec.shape[-1] == c // 2
    assert h_phy.shape[-1] == c // 2
    np.testing.assert_allclose(np.asarray(h_spec), h[..., : c // 2], atol=0)
    np.testing.assert_allclose(np.asarray(h_phy), h[..., c // 2 :], atol=0)


def test_cf_floor_formula():
    assert swiglu_hidden_width(16, rho=2.0) == int(np.floor(4 * 16 / 3))
    assert swiglu_hidden_width(16, rho=2.0) == 21
    assert swiglu_hidden_width(15, rho=2.0) == int(np.floor(2 * 2 * 15 / 3))
    key = jax.random.PRNGKey(0)
    params = init_block(key, c=16, n_heads=2, n_slices=4, n_modes=2, rho=2.0)
    cf = swiglu_hidden_width(16, 2.0)
    assert params["swiglu"]["Wg"].shape == (16, cf)
    assert params["swiglu"]["Wu"].shape == (16, cf)
    assert params["swiglu"]["Wd"].shape == (cf, 16)


def test_swiglu_matches_explicit_matmuls():
    key = jax.random.PRNGKey(1)
    params = init_block(key, c=16, n_heads=2, n_slices=4, n_modes=2)
    h = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 4, 16))
    out = swiglu(params["swiglu"], h)
    sg = params["swiglu"]
    gate = jax.nn.silu(h @ sg["Wg"] + sg["bg"])
    up = h @ sg["Wu"] + sg["bu"]
    ref = (gate * up) @ sg["Wd"] + sg["bd"]
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-5)


def test_residual_subspaces_match_phi_of_ln():
    key = jax.random.PRNGKey(3)
    params = init_block(key, c=16, n_heads=2, n_slices=4, n_modes=2)
    h = jax.random.normal(jax.random.PRNGKey(4), (2, 8, 8, 16))
    _, aux = apply_block(params, h, return_aux=True)
    phi_spec = axis_factorized_fourier(params["spec"], aux["y_spec"])
    phi_phy = apply_srpa(params["phy"], aux["y_phy"], use_conv=True)
    np.testing.assert_allclose(
        np.asarray(aux["htilde_spec"] - aux["h_spec"]), np.asarray(phi_spec), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(aux["htilde_phy"] - aux["h_phy"]), np.asarray(phi_phy), atol=1e-5
    )
    y_spec = layer_norm(aux["h_spec"], params["ln_spec"]["scale"], params["ln_spec"]["bias"])
    y_phy = layer_norm(aux["h_phy"], params["ln_phy"]["scale"], params["ln_phy"]["bias"])
    np.testing.assert_allclose(np.asarray(aux["y_spec"]), np.asarray(y_spec), atol=1e-5)
    np.testing.assert_allclose(np.asarray(aux["y_phy"]), np.asarray(y_phy), atol=1e-5)


def test_concat_swiglu_residual_eq_1c():
    key = jax.random.PRNGKey(5)
    params = init_block(key, c=16, n_heads=2, n_slices=4, n_modes=2)
    h = jax.random.normal(jax.random.PRNGKey(6), (2, 6, 6, 16))
    h_next, aux = apply_block(params, h, return_aux=True)
    h_cat = jnp.concatenate([aux["htilde_spec"], aux["htilde_phy"]], axis=-1)
    np.testing.assert_allclose(np.asarray(aux["h_cat"]), np.asarray(h_cat), atol=1e-5)
    y_cat = layer_norm(h_cat, params["ln_cat"]["scale"], params["ln_cat"]["bias"])
    sw = swiglu(params["swiglu"], y_cat)
    ref = h_cat + sw
    np.testing.assert_allclose(np.asarray(h_next), np.asarray(ref), atol=1e-5)
