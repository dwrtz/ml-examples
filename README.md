# Learned Variational Bayesian Filtering Experiments

This repository contains experiments for learning Bayesian filtering updates using
edge-local variational inference.

The core filtering update maintains an explicit two-state variational posterior:

```text
q^E_t(z_t, z_{t-1}) = q^F_t(z_t) q^B_t(z_{t-1} | z_t)
```

The linear-Gaussian benchmark compares learned filters against exact Kalman
filtering and exact two-state edge posteriors.

## Motivation

The original experiment asked whether a neural network can learn to do Kalman
filtering in a dynamic scalar regression model. The modernization keeps the
important probabilistic structure explicit: each update reasons about the local
edge `(z_t, z_{t-1})`, then carries forward only the filtering marginal
`q^F_t(z_t)`.

This gives the project a clean path from exact linear-Gaussian debugging to
learned variational filters, posterior predictive heads, recurrent context
models, and nonlinear observation models.

## Repository Layout

```text
archive/      Original 2020-era TensorFlow experiment and scratch files
docs/         Modernization plans and implementation notes
experiments/  YAML experiment configurations
scripts/      Training, evaluation, and plotting entry points
src/vbf/      Modern variational Bayesian filtering package
tests/        Unit and smoke tests
outputs/      Local experiment outputs, ignored by git except .gitkeep
```

## Quickstart

Install `uv`, then create or update the project environment:

```bash
uv sync --dev
make test
```

## Planned Commands

Run the linear-Gaussian oracle check:

```bash
make evaluate-linear
```

Train the first strict MLP filter:

```bash
make train-linear LINEAR_CONFIG=experiments/linear_gaussian/01_supervised_edge_mlp.yaml
```

Generate evaluation plots:

```bash
make plot-linear RUN_DIR=outputs/<run-name>
```

These entry points are scaffolded as part of the repository reorganization. The
implementation plan in `docs/vbf-modernization/` defines the order in which the
underlying data generator, Kalman oracle, posterior distributions, losses, and
models should be filled in.

## Historical Experiment

The original TensorFlow and TensorFlow Probability scripts are preserved in
`archive/2020_variational_filtering_original/`. They remain useful for comparing
the modern implementation against the 2020-era dynamic regression experiment.
