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

The current strict MLP filter should be interpreted precisely: it is a
Kalman-structured marginal update with learned corrections and a learned
backward conditional, not a generic neural filter learned entirely from
scratch. Less-structured direct MLP ablations are included for claims about
learning the filtering recursion itself.

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

Install `uv`, then create or update the project environment. The default
experiments use JAX and are sized to run on a laptop CPU.

```bash
uv sync --dev
make test
```

## Common Commands

Run the linear-Gaussian oracle check:

```bash
make evaluate-linear
```

Train the first strict MLP filter:

```bash
make train-linear
```

Train the ELBO baseline:

```bash
make train-linear-elbo
```

Run the current seed sweeps:

```bash
make sweep-objective-budget
make sweep-self-fed-supervised
make sweep-self-fed-variance
make sweep-predictive-head
```

Generate evaluation plots:

```bash
make plot-linear RUN_DIR=outputs/<run-name>
```

## Current Status

The scalar linear-Gaussian benchmark is implemented with:

- exact Kalman filtering and exact two-state edge posterior oracles;
- a Kalman-structured strict MLP update cell;
- teacher-forced supervised edge distillation;
- self-fed supervised edge distillation;
- edge-local ELBO training;
- closed-form scalar Gaussian ELBO training;
- closed-form scalar Gaussian ELBO diagnostics;
- posterior predictive evaluation with analytic-residual and direct learned
  one-step predictive heads;
- direct, less-structured MLP filter ablations;
- calibration metrics, coverage, variance ratios, saved configs, and saved
  trained parameters.

The current strongest learned linear-Gaussian baseline is the self-fed supervised
edge objective. The ELBO objective remains the main unsupervised benchmark. The
implementation plan and progress notes in `docs/vbf-modernization/` track the
research interpretation and next experiments.

## Historical Experiment

The original TensorFlow and TensorFlow Probability scripts are preserved in
`archive/2020_variational_filtering_original/`. They remain useful for comparing
the modern implementation against the 2020-era dynamic regression experiment.
