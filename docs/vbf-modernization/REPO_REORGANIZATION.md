# Suggested Repository Reorganization

Prepared: 2026-04-24

## 1. Current repo state

The current repo appears to be a compact historical experiment with a few top-level scripts:

```text
README.md
LICENSE
.gitattributes
.vscode/settings.json
fib.py
issue.py
linear_regression.py
learning_linear_regression.py
```

The two important historical files are:

| File | Role |
|---|---|
| `linear_regression.py` | Exact scalar Kalman filter for the dynamic regression example |
| `learning_linear_regression.py` | Learned variational posterior over `[z_t, z_{t-1}]` using TensorFlow Probability and an LSTM |

`fib.py` and `issue.py` look like small TensorFlow scratch examples. The `.vscode/settings.json` points at a local Windows Python environment and should not remain as a shared repo default.

## 2. Recommended target layout

```text
ml-examples/
  README.md
  LICENSE
  pyproject.toml
  .gitignore
  .gitattributes

  docs/
    vbf-modernization/
      VBF_IMPLEMENTATION_PLAN.md
      REPO_REORGANIZATION.md
      ENGINEER_HANDOFF_CHECKLIST.md

  archive/
    2020_variational_filtering_original/
      README.md
      linear_regression.py
      learning_linear_regression.py
      issue.py
      fib.py
      cool-vb.pdf

  src/
    vbf/
      __init__.py
      data.py
      kalman.py
      distributions.py
      losses.py
      metrics.py
      predictive.py
      train.py
      plotting.py

      models/
        __init__.py
        cells.py
        mamba_context.py
        heads.py

  scripts/
    train_linear_gaussian.py
    evaluate_linear_gaussian.py
    plot_linear_gaussian.py
    train_nonlinear.py
    evaluate_nonlinear.py

  experiments/
    linear_gaussian/
      00_oracle_check.yaml
      01_supervised_edge_mlp.yaml
      02_elbo_edge_mlp.yaml
      03_elbo_edge_gru.yaml
      04_elbo_edge_lstm.yaml
      05_elbo_edge_mamba.yaml
      06_predictive_head.yaml
      07_random_qr_generalization.yaml
      08_weak_observability.yaml

    nonlinear/
      01_sine_observation.yaml
      02_student_t_observation.yaml

  tests/
    test_kalman_scalar.py
    test_edge_oracle.py
    test_gaussian_edge_posterior.py
    test_elbo_shapes.py
    test_positive_scales.py
    test_predictive_no_leakage.py
    test_reproducibility.py

  outputs/
    .gitkeep
```

## 3. Archive rather than delete the original experiment

Move the original experiment files into:

```text
archive/2020_variational_filtering_original/
```

Include a short `archive/2020_variational_filtering_original/README.md` explaining:

- these are the original 2020-era scripts;
- they are preserved for historical comparison;
- the modern implementation lives under `src/vbf` and `scripts`;
- the original learned posterior already used the important two-state form over `[z_t, z_{t-1}]`.

This keeps the old work available without forcing the new implementation to inherit old TensorFlow-specific choices.

## 4. Top-level README update

Replace the current minimal README with a project overview:

```text
# Learned Variational Bayesian Filtering Experiments

This repository contains experiments for learning Bayesian filtering updates using edge-local variational inference.

The core filtering update maintains an explicit two-state variational posterior:

q^E_t(z_t, z_{t-1}) = q^F_t(z_t) q^B_t(z_{t-1} | z_t)

The linear-Gaussian benchmark compares learned filters against exact Kalman filtering and exact two-state edge posteriors.
```

The README should include:

- research motivation;
- quickstart instructions;
- how to run the oracle check;
- how to train the first MLP filter;
- how to run evaluation and plots;
- where to find the original experiment.

## 5. Dependency management

Add `pyproject.toml`.

Recommended baseline dependencies:

```text
python >= 3.10
numpy
scipy
matplotlib
pandas
pyyaml
tqdm
pytest
ruff
```

Then choose one primary ML stack:

### Option A: PyTorch-first

Recommended if Mamba/Mamba-2 experiments are a near-term goal.

```text
torch
```

Optional later:

```text
mamba-ssm
wandb or tensorboard
```

### Option B: JAX-first

Recommended if the priority is functional scans, vectorization, and concise research code.

```text
jax
flax or equinox
optax
distrax or tensorflow-probability[jax]
```

### Practical recommendation

Use PyTorch if the engineer expects to try Mamba quickly. Use JAX if the first phase is mostly mathematical filtering, vectorized simulation, and clean scans. Either stack is fine as long as the posterior API and tests are clean.

## 6. Coding conventions

Use these conventions consistently:

- Use `z_tm1` for `z_{t-1}` in code.
- Use `q_filter` or `q_f` for `q^F_t(z_t)`.
- Use `q_backward` or `q_b` for `q^B_t(z_{t-1} | z_t)`.
- Use `q_edge` or `q_e` for `q^E_t(z_t, z_{t-1})`.
- Use `pred` for pre-assimilation quantities.
- Use `filter` for post-assimilation quantities.
- Keep tensors batch-major: `[batch, time, dim]`.
- Keep distribution parameters separate from sampled values.
- Avoid silent broadcasting in probability calculations; assert shapes.

## 7. Suggested package APIs

### `vbf.data`

```text
make_linear_gaussian_batch(config, seed) -> EpisodeBatch
```

### `vbf.kalman`

```text
kalman_filter_scalar(batch, params) -> KalmanOutputs
kalman_edge_posterior_scalar(batch, params) -> EdgeOracleOutputs
measurement_predictive_scalar(batch, params) -> PredictiveOutputs
```

### `vbf.distributions`

```text
GaussianBelief
ConditionalGaussianBackward
GaussianEdgePosterior
```

### `vbf.models.cells`

```text
StructuredMLPCell
GRUContextCell
LSTMContextCell
```

### `vbf.models.mamba_context`

```text
MambaContextCell
```

### `vbf.losses`

```text
edge_elbo_loss
supervised_filter_kl_loss
supervised_edge_kl_loss
predictive_nll_loss
hybrid_loss
```

### `vbf.metrics`

```text
gaussian_kl_1d
gaussian_kl_full
coverage
state_rmse
state_nll
predictive_nll
edge_covariance_error
```

## 8. Experiment organization

Each experiment should have a YAML config and produce an output directory:

```text
outputs/{timestamp}_{experiment_name}/
  config.yaml
  metrics.jsonl
  checkpoints/
  plots/
  evaluation_summary.md
```

Avoid committing generated outputs except `.gitkeep` or selected small reference plots.

## 9. Tests before research runs

Before running larger experiments, the following should pass:

1. scalar Kalman recursion test;
2. edge posterior marginal consistency test;
3. Gaussian edge posterior mean/covariance test;
4. ELBO shape and sign smoke test;
5. positive variance test;
6. predictive no-leakage test;
7. reproducibility test.

## 10. Cleanup recommendations

### Move or remove `.vscode/settings.json`

The current file points to a user-specific Windows path:

```text
C:\Users\david\env\tf-nightly\Scripts\python.exe
```

Either delete it or replace it with a generic workspace recommendation that does not hardcode a local environment.

### Move `fib.py` and `issue.py`

These are not part of the filtering experiment. Options:

- move them to `archive/2020_variational_filtering_original/`; or
- move them to `scratch/` and add `scratch/` to `.gitignore`; or
- delete them if they are not useful.

Archiving is safest.

### Keep `LICENSE`

The existing MIT license is fine.

### Add `.gitignore`

Suggested entries:

```text
__pycache__/
*.pyc
.venv/
.env
.ipynb_checkpoints/
outputs/*
!outputs/.gitkeep
wandb/
runs/
.DS_Store
.pytest_cache/
.ruff_cache/
```

## 11. Migration steps

1. Create the new directories.
2. Move original files into `archive/2020_variational_filtering_original/`.
3. Add handoff docs under `docs/vbf-modernization/`.
4. Add `pyproject.toml`, `.gitignore`, and a revised `README.md`.
5. Implement `src/vbf/data.py` and `src/vbf/kalman.py` first.
6. Add unit tests for the oracle.
7. Implement the Gaussian posterior family.
8. Implement supervised edge-posterior distillation.
9. Implement the Monte Carlo edge ELBO.
10. Add predictive head and evaluation reports.
11. Add GRU/LSTM/Mamba only after the strict MLP baseline is stable.

## 12. Suggested milestone branches

```text
milestone/00-repo-reorg
milestone/01-kalman-edge-oracle
milestone/02-gaussian-edge-posterior
milestone/03-supervised-edge-distillation
milestone/04-elbo-recovery
milestone/05-predictive-head
milestone/06-context-architectures
milestone/07-nonlinear-models
```

## 13. Deliverable expectations for engineer/researcher

The first implementation handoff should produce:

1. passing unit tests;
2. one working supervised edge distillation run;
3. one working unsupervised ELBO run;
4. an evaluation report comparing learned posterior vs exact Kalman posterior;
5. plots for at least one held-out sequence;
6. a short note describing any deviations from this plan.
