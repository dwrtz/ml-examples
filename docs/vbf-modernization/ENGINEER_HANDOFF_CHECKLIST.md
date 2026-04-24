# Engineer / Researcher Handoff Checklist

Prepared: 2026-04-24

## 1. Research objective

Build a modern learned variational Bayesian filtering benchmark that answers:

1. Can a learned update recover exact Kalman filtering in the linear-Gaussian case?
2. Can the edge-local variational objective recover the Kalman posterior without supervised posterior targets?
3. Can the same framework support learned posterior predictive measurement distributions?
4. Do modern sequence architectures such as Mamba/Mamba-2 help, once the probabilistic structure is correct?

The implementation must preserve the two-state edge posterior:

```text
q^E_t(z_t, z_{t-1})
```

and should use a marginal-preserving factorization:

```text
q^E_t(z_t, z_{t-1}) = q^F_t(z_t) q^B_t(z_{t-1} | z_t)
```

## 2. First two weeks: minimum useful implementation

### Task 1: repo setup

Deliverables:

- new `src/vbf` package;
- original scripts archived;
- `pyproject.toml` added;
- tests directory added;
- updated README.

Done when:

- `pytest` runs;
- package imports work;
- old experiment is preserved in `archive/`.

### Task 2: scalar linear-Gaussian data generator

Deliverables:

- `src/vbf/data.py`;
- support for sinusoidal `x_t`;
- support for random seeds;
- support for batched episodes;
- true latent `z_t` returned.

Done when:

- generated shapes are correct;
- same seed reproduces the same batch;
- plots resemble the original dynamic regression setup.

### Task 3: Kalman and edge oracle

Deliverables:

- `src/vbf/kalman.py`;
- standard filtering posterior;
- exact two-state edge posterior;
- exact measurement predictive distribution.

Done when:

- edge posterior marginal over `z_t` exactly matches the standard Kalman filter output;
- measurement predictive variance is `x_t^2 (P_{t-1} + Q) + R`;
- tests pass in float64.

### Task 4: Gaussian edge posterior distribution

Deliverables:

- `GaussianBelief`;
- `ConditionalGaussianBackward`;
- `GaussianEdgePosterior`.

Done when:

- can sample `(z_t, z_{t-1})`;
- can compute `log q^F_t(z_t)`, `log q^B_t(z_{t-1}|z_t)`, and `log q^E_t(z_t,z_{t-1})`;
- can return the filtering marginal `q^F_t`;
- joint mean/covariance matches analytic formulas.

## 3. Next milestone: supervised edge distillation

### Task 5: strict MLP update cell

Inputs:

```text
features(q^F_{t-1}), x_t, y_t, optional Q/R features
```

Outputs:

```text
mu_t, logvar_t, a_t, b_t, logtau_t^2
```

Done when:

- all output scales are positive after transformation;
- sequence scan works for variable `T`;
- no fixed batch size is hardcoded.

### Task 6: supervised edge KL objective

Train learned `q^E_t` against exact edge posterior from the oracle.

Done when:

- training loss decreases;
- validation `edge_kl` is low;
- validation `filter_kl` is low;
- plots show learned posterior intervals matching Kalman intervals.

Recommended report:

```text
outputs/.../evaluation_summary.md
outputs/.../plots/example_sequence.png
outputs/.../plots/edge_covariance.png
outputs/.../plots/filter_kl_over_time.png
```

## 4. Next milestone: unsupervised edge ELBO

### Task 7: Monte Carlo edge ELBO

Implement:

```text
E_{q^E_t}[
    log p(y_t | z_t, x_t)
  + log p(z_t | z_{t-1})
  + log q^F_{t-1}(z_{t-1})
  - log q^F_t(z_t)
  - log q^B_t(z_{t-1} | z_t)
]
```

Done when:

- ELBO has correct sign convention;
- gradients flow through reparameterized samples;
- no NaNs under default config;
- ELBO-trained model improves over naive baseline;
- evaluation still reports oracle KL, even though oracle is not used for training.

### Task 8: compare supervised and unsupervised results

Produce a short comparison table:

| Model | Objective | filter KL | edge KL | state RMSE | state NLL | predictive NLL |
|---|---|---:|---:|---:|---:|---:|
| MLP | supervised edge KL | | | | | |
| MLP | edge ELBO | | | | | |
| exact Kalman | oracle | 0 | 0 | | | |

Done when:

- failure modes are documented;
- if ELBO underperforms supervised distillation, determine whether the problem is architecture, objective variance, or implementation.

## 5. Next milestone: posterior predictive head

### Task 9: one-step predictive distribution

Implement:

```text
r_omega(y_t | q^F_{t-1}, x_t)
```

This head predicts `y_t` before assimilation.

Done when:

- code path cannot access `y_t` except as the target in the loss;
- exact Kalman predictive NLL is available for comparison;
- predictive intervals are plotted and calibrated.

### Task 10: model-consistent predictive distillation

Use Monte Carlo samples from:

```text
z_{t-1} ~ q^F_{t-1}
z_t     ~ p(z_t | z_{t-1})
y_t     ~ p(y_t | z_t, x_t)
```

to train or validate the predictive head.

Done when:

- learned predictive mean/variance match exact predictive in the scalar linear-Gaussian case;
- direct-data training and model-consistent distillation are compared.

## 6. Architecture milestone

### Task 11: GRU/LSTM comparison

Purpose:

- compare to the original LSTM-style experiment;
- determine whether recurrent context improves amortized updates.

Done when:

- GRU/LSTM models use the same posterior family and objective as the MLP;
- metrics are directly comparable.

### Task 12: Mamba/Mamba-2 context experiment

Purpose:

- test whether selective state-space sequence context improves long-horizon or out-of-distribution filtering.

Required guardrails:

- exported belief remains `q^F_t`;
- edge posterior remains `q^F_t q^B_t`;
- report `q^F_t`-only predictive results;
- report context-assisted predictive results separately;
- include context ablation.

Done when:

- Mamba result is compared to strict MLP, GRU, and LSTM;
- improvement is not solely due to hidden context bypassing the filtering belief.

## 7. Generalization milestone

Run the trained models on:

| Test suite | Expected behavior |
|---|---|
| Same distribution | Should match Kalman closely |
| Longer sequences | Should remain stable |
| Randomized `Q/R` | Should degrade gracefully or condition correctly on noise parameters |
| Weak observability | Variance should grow when `x_t ≈ 0` |
| Different `x_t` pattern | Should not depend only on sinusoid memorization |
| Missing `y_t` | Should fall back to prediction when observation is unavailable |

Done when:

- each suite has an evaluation summary;
- at least one failure case is analyzed in detail.

## 8. Nonlinear milestone

Only start this after the linear-Gaussian benchmark is stable.

Candidate models:

```text
y_t = sin(z_t) + v_t
```

```text
y_t = h_theta(z_t, x_t) + v_t
```

```text
v_t ~ StudentT(df, scale)
```

Done when:

- the same edge ELBO code path works;
- Kalman oracle is replaced by approximate diagnostics or particle/Monte Carlo references;
- filtering marginal remains explicit.

## 9. Open design choices to resolve early

### ML stack

Choose one:

- PyTorch-first, especially if Mamba/Mamba-2 is important soon.
- JAX-first, especially if clean scans/vectorization are more important.

### Config system

Choose one:

- simple YAML with a small loader;
- Hydra/OmegaConf if experiment management will grow.

### Logging

Choose one:

- JSONL + local plots;
- TensorBoard;
- Weights & Biases.

Local JSONL and plots are enough for the first milestone.

## 10. Red flags during implementation

Stop and debug if any of these happen:

- posterior variance becomes negative or NaN;
- ELBO improves while coverage collapses;
- learned `q^F_t` matches mean but not variance;
- `filter_kl` is good but `edge_kl` is bad;
- predictive head performs well only when it accidentally sees `y_t`;
- Mamba model performs well but `q^F_t`-only prediction performs poorly;
- edge posterior marginal does not match Kalman in the oracle implementation.

## 11. Final deliverable for first research review

The first review packet should contain:

1. code for data, oracle, posterior family, losses, metrics;
2. passing tests;
3. one supervised edge distillation run;
4. one unsupervised edge ELBO run;
5. one predictive-head run;
6. comparison table;
7. plots for representative sequences;
8. notes on failure modes;
9. recommendation on whether to proceed to Mamba or fix the baseline first.
