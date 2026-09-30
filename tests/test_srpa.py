"""Algebraic checks of Slice-Residual Physics-Attention, eqs 2-4."""

import jax
import jax.numpy as jnp
import numpy as np

from transolver_sigma.srpa import EPS, apply_srpa, init_srpa, srpa_core


def _rand_core(seed=0, n=12, dh=4, m=5):
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(n, dh)).astype(np.float32)
    F = rng.normal(size=(n, dh)).astype(np.float32)
    Ws = rng.normal(size=(dh, m)).astype(np.float32) * 0.3
    bs = rng.normal(size=(m,)).astype(np.float32) * 0.1
    Wq = rng.normal(size=(dh, dh)).astype(np.float32) * 0.3
    Wk = rng.normal(size=(dh, dh)).astype(np.float32) * 0.3
    Wv = rng.normal(size=(dh, dh)).astype(np.float32) * 0.3
    return tuple(jnp.asarray(x) for x in (R, F, Ws, bs, Wq, Wk, Wv))


def test_gamma_zero_identity_in_slice_space():
    R, F, Ws, bs, Wq, Wk, Wv = _rand_core()
    out = srpa_core(R, F, Ws, bs, Wq, Wk, Wv, tau=0.5, gamma=0.0)
    np.testing.assert_allclose(np.asarray(out["Z_plus"]), np.asarray(out["Z"]), atol=1e-6)


def test_p_rows_sum_to_one():
    R, F, Ws, bs, Wq, Wk, Wv = _rand_core()
    out = srpa_core(R, F, Ws, bs, Wq, Wk, Wv, tau=0.7, gamma=0.2)
    row_sum = np.asarray(out["P"].sum(axis=-1))
    np.testing.assert_allclose(row_sum, np.ones_like(row_sum), atol=1e-5)


def test_d_ii_is_column_mass_plus_eps():
    R, F, Ws, bs, Wq, Wk, Wv = _rand_core()
    out = srpa_core(R, F, Ws, bs, Wq, Wk, Wv, tau=0.5, gamma=0.0, eps=EPS)
    p = np.asarray(out["P"])
    d = np.asarray(out["D_diag"])
    np.testing.assert_allclose(d, p.sum(axis=0) + EPS, atol=1e-6)


def test_z_matches_numpy_reference():
    R, F, Ws, bs, Wq, Wk, Wv = _rand_core(seed=1)
    out = srpa_core(R, F, Ws, bs, Wq, Wk, Wv, tau=0.4, gamma=0.0)
    p = np.asarray(out["P"])
    f = np.asarray(F)
    mass = p.sum(axis=0) + EPS
    z_ref = (p.T @ f) / mass[:, None]
    np.testing.assert_allclose(np.asarray(out["Z"]), z_ref, atol=1e-5)


def test_a_is_row_stochastic():
    R, F, Ws, bs, Wq, Wk, Wv = _rand_core()
    out = srpa_core(R, F, Ws, bs, Wq, Wk, Wv, tau=0.5, gamma=1.0)
    row_sum = np.asarray(out["A"].sum(axis=-1))
    np.testing.assert_allclose(row_sum, np.ones_like(row_sum), atol=1e-5)


def test_deltaz_is_a_times_z_wv():
    R, F, Ws, bs, Wq, Wk, Wv = _rand_core()
    out = srpa_core(R, F, Ws, bs, Wq, Wk, Wv, tau=0.5, gamma=0.3)
    z = np.asarray(out["Z"])
    a = np.asarray(out["A"])
    wv = np.asarray(Wv)
    np.testing.assert_allclose(np.asarray(out["DeltaZ"]), a @ (z @ wv), atol=1e-5)


def test_deslice_o_equals_p_zplus():
    R, F, Ws, bs, Wq, Wk, Wv = _rand_core()
    out = srpa_core(R, F, Ws, bs, Wq, Wk, Wv, tau=0.5, gamma=0.4)
    p = np.asarray(out["P"])
    zplus = np.asarray(out["Z_plus"])
    np.testing.assert_allclose(np.asarray(out["O"]), p @ zplus, atol=1e-5)


def test_gamma_scales_deltaz_linearly():
    R, F, Ws, bs, Wq, Wk, Wv = _rand_core()
    g = 0.37
    out = srpa_core(R, F, Ws, bs, Wq, Wk, Wv, tau=0.5, gamma=g)
    z = np.asarray(out["Z"])
    zplus = np.asarray(out["Z_plus"])
    dz = np.asarray(out["DeltaZ"])
    np.testing.assert_allclose(zplus - z, g * dz, atol=1e-5)
    out0 = srpa_core(R, F, Ws, bs, Wq, Wk, Wv, tau=0.5, gamma=0.0)
    np.testing.assert_allclose(np.asarray(out0["DeltaZ"]), dz, atol=1e-5)


def test_linear_fallback_shape_and_identity_gamma():
    key = jax.random.PRNGKey(0)
    c_phy, nh, m, n = 8, 2, 4, 10
    params = init_srpa(key, c_phy, nh, m)
    y = jax.random.normal(jax.random.PRNGKey(1), (2, n, c_phy))
    phi, aux = apply_srpa(params, y, use_conv=False, return_aux=True)
    assert phi.shape == (2, n, c_phy)
    np.testing.assert_allclose(np.asarray(aux["Z_plus"]), np.asarray(aux["Z"]), atol=1e-6)
