# Original Variational Filtering Experiment

This directory preserves the original 2020-era TensorFlow experiment and scratch
files for historical comparison.

The important historical scripts are:

- `linear_regression.py`: exact scalar Kalman filtering for the dynamic
  regression example.
- `learning_linear_regression.py`: a TensorFlow Probability experiment that
  learns a variational posterior over `[z_t, z_{t-1}]` with an LSTM encoder.

The modern implementation lives under `src/vbf` with command-line entry points in
`scripts`. The original learned posterior already used the key two-state form
over `[z_t, z_{t-1}]`; the modern code makes that edge posterior explicit,
tested, and reusable.

`fib.py` and `issue.py` are retained as scratch files from the original
repository snapshot. `cool-vb.pdf` is retained as source research material.
