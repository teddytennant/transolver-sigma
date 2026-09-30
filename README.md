Faithful JAX kernel of Transolver-σ (arXiv:2609.37279): dual-subspace blocks with Slice-Residual Physics-Attention and axis-factorized Fourier mixing.

## Install

```
python3 -m venv .venv && .venv/bin/pip install -e '.[tests]'
```

## Run

```
.venv/bin/pytest -q
.venv/bin/python train.py
```

## What does not match the paper

No paper-scale Darcy/NS2D/KF2D tables, no 8-layer C=256, no RealPDEBench, no 3D, CPE/SRPA convs may be 2D-only.
