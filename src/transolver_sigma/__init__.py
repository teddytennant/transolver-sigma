"""Transolver-σ JAX kernel (arXiv:2609.37279)."""

from transolver_sigma.block import apply_block, init_block, split_subspaces, swiglu, swiglu_hidden_width
from transolver_sigma.model import adam_init, adam_step, apply_model, init_model, relative_l2
from transolver_sigma.spectral import axis_factorized_fourier, mix_rfft, rfft_irfft_roundtrip
from transolver_sigma.srpa import apply_srpa, srpa_core

__all__ = [
    "adam_init",
    "adam_step",
    "apply_block",
    "apply_model",
    "apply_srpa",
    "axis_factorized_fourier",
    "init_block",
    "init_model",
    "mix_rfft",
    "relative_l2",
    "rfft_irfft_roundtrip",
    "split_subspaces",
    "srpa_core",
    "swiglu",
    "swiglu_hidden_width",
]
