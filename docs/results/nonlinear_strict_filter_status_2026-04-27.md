# Nonlinear Strict-Filter Status Note, 2026-04-27

This note summarizes the recent nonlinear filtering work after the scalar
linear-Gaussian benchmark. It is intended as a research review handoff: what was
implemented, what we learned empirically, and what decision points remain.

## Context

The nonlinear benchmark uses the scalar random-walk latent state

```text
z_t = z_{t-1} + w_t
y_t = x_t sin(z_t) + v_t
```

with explicit stress patterns for `x_t`: weak sinusoidal, intermittent, zero,
random normal, and ordinary sinusoidal. The reference filter is a deterministic
1D grid filter. The learned filter is the existing strict Gaussian edge family:
a carried filtering marginal `q^F_t(z_t)` plus a backward conditional
`q^B_t(z_{t-1} | z_t)`. The nonlinear learned filter is EKF-residualized around
the sine observation model.

The important framing is that reference-calibrated rows are not unsupervised
baselines. They use grid-reference filtering moments as targets and should be
reported as oracle/reference-calibrated diagnostics.

## Recent Implemented Work

### Reference Caching

Grid-reference computations are now cached under:

```text
outputs/cache/nonlinear_reference/
```

The cache key includes data config, state-space parameters, seed, reference grid
config, and cache version. Evaluation and training scripts support:

```text
--cache-dir
--no-cache
```

This made repeated calibrated sweeps practical. In the later sweeps, repeated
train/eval references consistently hit cache after the first computation.

### Learned Nonlinear Suites

The nonlinear learned-filter sweep now supports:

- vanilla structured nonlinear MC ELBO
- direct nonlinear MC ELBO
- reference global variance calibration
- reference time-local variance calibration
- reference low-observation calibration
- reference log-variance calibration
- resampled-batch vanilla ELBO
- reference moment distillation

The most important code paths are:

- `scripts/train_nonlinear.py`
- `scripts/sweep_nonlinear_learned.py`
- `scripts/plot_nonlinear.py`
- `scripts/plot_nonlinear_sweep.py`
- `scripts/diagnose_nonlinear_mixture_projection.py`
- `src/vbf/nonlinear.py`
- `src/vbf/nonlinear_cache.py`

### Diagnostics

The plotting path now writes per-run diagnostics:

- posterior example plots
- predictive example plots
- variance-over-time plots
- `time_metrics.csv`
- `time_calibration.png`
- `reference_shape_metrics.csv`
- `reference_shape.png`

Additional reference-only diagnostics include:

- full grid posterior mass extraction
- posterior entropy
- significant local peak counts
- 90% credible width
- projection comparison: grid reference vs moment-matched Gaussian vs simple
  mixture projections vs learned Gaussian

## Main Empirical Results

### Zero Observation Is Solved By Reference Calibration

Seed replication over seeds `321,322,323` showed the zero-observation case is
stable under reference-calibrated time/global variance targets:

| case | model | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|
| zero | baseline | 15.471 +/- 1.579 | 0.264 +/- 0.034 | 0.050 +/- 0.001 |
| zero | time w1 | 2.732 +/- 0.058 | 0.910 +/- 0.024 | 1.004 +/- 0.004 |

Interpretation: when the observation carries no state information, the reference
variance target is enough to make the learned Gaussian uncertainty match the
grid reference.

### Weak And Intermittent Improve But Remain Under-Dispersed

Best replicated settings:

| case | best setting | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|
| weak | global variance w3 | 11.909 +/- 3.148 | 0.449 +/- 0.064 | 0.133 +/- 0.062 |
| intermittent | global variance w1 | 19.633 +/- 5.414 | 0.443 +/- 0.047 | 0.104 +/- 0.034 |

These are improvements over vanilla ELBO, but still far from the grid reference
coverage around `0.91` and far below variance ratio `1.0`.

### More Optimization Budget Did Not Fix The Gap

Focused 1000-step runs did not materially improve weak/intermittent calibration:

| case | model | steps | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|---:|
| weak | baseline | 250 | 17.735 | 0.403 | 0.065 |
| weak | baseline | 1000 | 17.748 | 0.403 | 0.065 |
| weak | global calibrated | 1000 | 8.393 | 0.513 | 0.089 |
| intermittent | baseline | 250 | 38.423 | 0.353 | 0.028 |
| intermittent | baseline | 1000 | 38.599 | 0.354 | 0.030 |
| intermittent | global calibrated | 1000 | 19.673 | 0.473 | 0.080 |

Interpretation: the remaining problem is probably not simple training budget.

### Naive Resampled-Batch Training Hurt

Resampled vanilla ELBO was added to test whether fixed-batch overfitting was the
problem. At 250 steps, it made weak/intermittent worse:

| case | model | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|
| weak | fixed baseline | 17.735 | 0.403 | 0.065 |
| weak | resampled baseline | 24.371 | 0.354 | 0.045 |
| intermittent | fixed baseline | 38.423 | 0.353 | 0.028 |
| intermittent | resampled baseline | 43.820 | 0.302 | 0.019 |

Interpretation: naive data resampling is not the immediate fix. If resampling is
revisited, it probably needs a different learning rate/budget schedule.

### Log-Variance Calibration Was A Useful Negative Result

An elementwise reference log-variance calibration objective was added:

```text
structured_elbo_ref_logvar_calibrated
```

Weights `1,3,10` did not beat the earlier global variance settings:

| case | log-var best-ish result | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|
| weak | log-var w3 | 13.241 | 0.456 | 0.086 |
| intermittent | log-var w1 | 20.450 | 0.451 | 0.080 |

Interpretation: simply making the variance target more local is not enough.

### Time-Local Diagnostics Shifted The Failure Story

The time-local plots showed the calibration failure is not only in low-`x`
windows. In both weak and intermittent cases, high-observation windows often
have lower variance ratio, lower coverage, and higher learned NLL.

Example low/high observation split:

| run | window | var ratio | learned cov 90 | learned NLL |
|---|---|---:|---:|---:|
| weak baseline | low x2 | 0.111 | 0.416 | 15.59 |
| weak baseline | high x2 | 0.058 | 0.396 | 18.86 |
| weak global w3 | low x2 | 0.149 | 0.536 | 6.96 |
| weak global w3 | high x2 | 0.097 | 0.503 | 9.10 |
| intermittent baseline | low x2 | 0.040 | 0.359 | 32.57 |
| intermittent baseline | high x2 | 0.015 | 0.335 | 57.00 |
| intermittent global w1 | low x2 | 0.109 | 0.487 | 15.42 |
| intermittent global w1 | high x2 | 0.042 | 0.429 | 33.56 |

Interpretation: focusing only on missing/weak observations is probably the wrong
next move. Informative nonlinear observations are also where the learned update
becomes confidently wrong.

### Reference Shape Diagnostics

Reference posterior peak count correlates strongly with learned NLL:

| run | corr peak count vs learned NLL |
|---|---:|
| weak baseline | 0.921 |
| weak global w3 | 0.914 |
| intermittent baseline | 0.788 |
| intermittent global w1 | 0.687 |

This initially suggested a posterior-family issue: the sine likelihood creates
multiple plausible modes, while the learned filter is a single Gaussian.

### Mixture Projection Diagnostic Revised That Interpretation

A reference-only projection diagnostic compared:

- full grid reference NLL
- moment-matched Gaussian NLL
- learned Gaussian NLL
- simple 2/3-component mixture projections

The key result is that the moment-matched Gaussian reference is already close to
the full grid reference, while the learned Gaussian is far worse:

| case | grid reference NLL | moment Gaussian NLL | learned Gaussian NLL |
|---|---:|---:|---:|
| weak global w3 run | 2.406 | 2.738 | 8.367 |
| intermittent global w1 run | 2.221 | 2.732 | 19.766 |

Interpretation: the single-Gaussian family is not the dominant bottleneck yet.
The current learned filter is not matching the best Gaussian moments of the grid
posterior. A mixture filter may be useful later, but it is premature as the next
main model change.

### Moment Distillation Diagnostic

To separate objective failure from architecture/capacity failure, a pure
reference moment-distillation mode was added:

```text
structured_moment_distilled
```

It uses:

- ELBO weight `0`
- mean loss `(learned_mean - reference_mean)^2 / reference_var`
- log-variance loss `(log learned_var - log reference_var)^2`

Results:

| case | steps | state NLL | coverage 90 | variance ratio |
|---|---:|---:|---:|---:|
| weak distilled | 250 | 25.525 | 0.331 | 0.061 |
| weak distilled | 1000 | 24.608 | 0.336 | 0.063 |
| intermittent distilled | 250 | 25.896 | 0.435 | 0.068 |
| intermittent distilled | 1000 | 25.486 | 0.393 | 0.070 |

Interpretation: direct moment supervision still does not make this
EKF-residualized update match reference moments. That points toward architecture
or optimization constraints in the current update parameterization, rather than
just the ELBO objective.

## Current Interpretation

The nonlinear benchmark is now telling a sharper story:

1. The grid-reference and diagnostics infrastructure is in good shape.
2. The zero-observation stress case is solved by reference variance calibration.
3. Weak/intermittent nonlinear cases remain badly under-dispersed.
4. More steps, naive resampling, log-variance calibration, and direct moment
   distillation do not close the gap.
5. The moment-matched Gaussian reference is good enough that a mixture posterior
   is not the immediate missing piece.
6. The likely bottleneck is the current EKF-residualized nonlinear update
   parameterization or its optimization dynamics.

In short: the model is not yet learning the reference Gaussian moments, even
when those moments are directly supervised.

## Suggested Next Decisions

### Decision 1: Investigate Architecture Versus Optimization

The next diagnostic should distinguish whether the EKF-residualized update is
too constrained or whether the training setup is failing to optimize it.

Possible checks:

- Increase hidden dimension for moment distillation only.
- Train moment distillation with a larger learning rate sweep.
- Add gradient/parameter diagnostics for the variance-scale output.
- Compare current EKF-residualized update against a direct mean/log-variance
  update head with the same inputs.

### Decision 2: Add A Less-Structured Nonlinear Filter Head

The most informative next model ablation would be a less-structured Gaussian
filter head that predicts:

```text
filter_mean
log_filter_var
backward_a
backward_b
log_backward_var
```

directly from `(prev_mean, prev_var, x_t, y_t, q, r)`, without the EKF gain/base
variance formula. Train it first with moment distillation, not ELBO.

If this direct head matches reference moments, the EKF-residualized update is the
bottleneck. If it also fails, the issue is likely optimization/features or the
stepwise Markov Gaussian parameterization more broadly.

### Decision 3: Defer Mamba/Mixture Until This Is Resolved

The current evidence does not yet justify Mamba or a learned mixture filter as
the next primary branch. The reference moment Gaussian is already strong, and a
directly supervised current model still fails to match it. The next best work is
therefore a controlled Gaussian update ablation, not a bigger sequence model.

## Suggested Immediate Next Experiment

Implement and run:

```text
structured_direct_moment_distilled
```

where the update is less EKF-residualized but still produces a Gaussian strict
filter. Run weak/intermittent at 250 steps and 1000 steps.

Interpretation:

- If direct moment distillation succeeds, keep Gaussian posterior family and
  redesign the nonlinear update architecture.
- If direct moment distillation fails, inspect optimization/features before
  adding richer posterior families.

## Relevant Commands

Recent useful commands:

```bash
uv run python scripts/sweep_nonlinear_learned.py \
  --configs experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  --models structured_elbo,structured_elbo_ref_calibrated \
  --steps 250 \
  --output-dir outputs/nonlinear_calibration_cached_250

uv run python scripts/sweep_nonlinear_learned.py \
  --configs experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  --models structured_moment_distilled \
  --steps 1000 \
  --output-dir outputs/nonlinear_moment_distillation_1000

make plot-nonlinear \
  RUN_DIR=outputs/nonlinear_calibration_weight_sweep_250/w3/nonlinear_weak_sine_observation_structured_elbo_ref_calibrated

uv run python scripts/diagnose_nonlinear_mixture_projection.py \
  --run-dir outputs/nonlinear_calibration_weight_sweep_250/w3/nonlinear_weak_sine_observation_structured_elbo_ref_calibrated
```

## Review Questions For The Research Director

1. Do you agree that the next branch should be a less-structured Gaussian update
   head rather than Mamba or mixtures?
2. Should moment distillation remain the first diagnostic objective for new
   nonlinear heads?
3. Is the current nonlinear claim framed correctly as: "the EKF-residualized
   strict Gaussian filter is insufficient for weak/intermittent sine
   observations," rather than "single Gaussian filtering is insufficient"?
4. Are there additional reference diagnostics you want before changing the
   update architecture?
