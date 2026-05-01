# Nonlinear Strict-Filter Final Result Summary, 2026-04-28

## Executive Summary

The nonlinear scalar benchmark now has a clear outcome. For the sine-observation
stress tests, the strict Gaussian filter family is not the main bottleneck:
a Gaussian matched to the grid-reference moments is close to the full grid
reference. The main failures were in the learned update parameterization and in
self-fed rollout stability.

The best current nonlinear learned filters are:

- `direct_moment_distilled`: best state NLL, especially on intermittent
  observations.
- `structured_moment_rollout_h4`: better calibrated strict Gaussian comparison,
  with near-reference variance on weak observations.

The direct head should be the default nonlinear candidate when state NLL is the
primary metric. The structured horizon-4 rollout head should be retained as the
better calibrated comparison. Simple variance calibration penalties for the
direct head should be considered exhausted.

## Benchmark And Reference

The benchmark uses:

```text
z_t = z_{t-1} + w_t
y_t = x_t sin(z_t) + v_t
```

with stress patterns for `x_t`: weak sinusoidal, intermittent, zero, random
normal, and ordinary sinusoidal. The reference is a deterministic 1D grid filter.
Reference computations are cached under:

```text
outputs/cache/nonlinear_reference/
```

The cache made repeated seed and calibration sweeps practical.

## Main Empirical Findings

### 1. Reference Calibration Solves The Zero-Observation Case

For zero observations, reference variance calibration recovers the grid
reference uncertainty:

| case | model | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|
| zero | baseline | 15.471 +/- 1.579 | 0.264 +/- 0.034 | 0.050 +/- 0.001 |
| zero | time w1 | 2.732 +/- 0.058 | 0.910 +/- 0.024 | 1.004 +/- 0.004 |

This confirmed that the reference machinery and calibration targets behave as
expected in a clean unobservable regime.

### 2. The Original Structured ELBO Head Remains Under-Dispersed

Weak/intermittent observations remained badly under-dispersed under the original
structured EKF-residualized learned head:

| case | setting | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|
| weak | global variance w3 | 11.909 +/- 3.148 | 0.449 +/- 0.064 | 0.133 +/- 0.062 |
| intermittent | global variance w1 | 19.633 +/- 5.414 | 0.443 +/- 0.047 | 0.104 +/- 0.034 |

Extra training budget, naive resampled batches, and log-variance calibration did
not close this gap.

### 3. Mixture Projections Were Not The Immediate Missing Piece

Reference-shape diagnostics showed multimodality, but the projection diagnostic
changed the interpretation:

| case | grid reference NLL | moment Gaussian NLL | learned Gaussian NLL |
|---|---:|---:|---:|
| weak global w3 run | 2.406 | 2.738 | 8.367 |
| intermittent global w1 run | 2.221 | 2.732 | 19.766 |

The moment-matched Gaussian reference is already strong. The current learned
filter was not matching the best Gaussian moments, so moving immediately to
mixtures or Mamba would be premature.

### 4. Direct Moment Distillation Is The Strongest State-NLL Head

Direct moment distillation uses a less-structured MLP update trained against
grid-reference moments. It is stable under self-fed rollout:

| case | model | steps | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|---:|
| weak | direct moment distilled | 1000 | 2.806 | 0.832 | 0.665 |
| intermittent | direct moment distilled | 1000 | 2.806 | 0.832 | 0.656 |

This established that the strict Gaussian filter family can perform well when
the update head is less constrained.

### 5. Structured One-Step Capacity Exists, But Rollout Was The Problem

Teacher-forced structured moment distillation showed that the structured
EKF-residualized head can learn the one-step reference map:

| case | head | eval mode | state NLL | coverage 90 | variance ratio |
|---|---|---|---:|---:|---:|
| weak | structured | teacher-forced | 2.790 | 0.888 | 1.112 |
| intermittent | structured | teacher-forced | 2.761 | 0.901 | 1.034 |

But self-fed structured rollout was unstable. The failure was rollout
stability/compounding error, not one-step representational capacity.

### 6. Horizon-4 Rollout Distillation Stabilizes The Structured Head

Short-horizon rollout distillation starts from reference beliefs, rolls the
structured head forward for a short horizon, and penalizes reference moment
error. Horizon 4 was the useful setting:

| case | rollout horizon | steps | state NLL | coverage 90 | variance ratio |
|---|---:|---:|---:|---:|---:|
| weak | 4 | 1000 | 2.808 | 0.881 | 0.993 |
| intermittent | 4 | 1000 | 3.340 | 0.861 | 1.124 |

This recovered the structured branch and made it a viable calibrated comparison,
though it still trails the direct head on intermittent state NLL.

### 7. Matched 3-Seed Head Comparison

The decisive matched comparison used seeds `321,322,323`, 1000 training steps,
and weak/intermittent observations:

| case | head | seeds | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|---:|
| weak | direct | 3 | 2.774 +/- 0.074 | 0.838 +/- 0.026 | 0.669 +/- 0.008 |
| weak | structured rollout h4 | 3 | 2.774 +/- 0.061 | 0.885 +/- 0.018 | 0.994 +/- 0.014 |
| intermittent | direct | 3 | 2.774 +/- 0.074 | 0.838 +/- 0.026 | 0.667 +/- 0.016 |
| intermittent | structured rollout h4 | 3 | 3.275 +/- 0.106 | 0.860 +/- 0.020 | 1.437 +/- 0.445 |

Interpretation:

- Weak observations: state NLL is effectively tied; structured rollout is much
  better calibrated.
- Intermittent observations: direct head has the better state NLL; structured
  rollout has higher coverage and less under-dispersion.

## Direct-Head Calibration Attempts

The direct head is under-covered, so three direct-head variance calibration
families were tested:

- global variance-ratio calibration
- time-local variance-ratio calibration
- low-observation-weighted variance-ratio calibration

All three were negative. Representative aggregate results:

| case | direct variant | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|
| weak | base | 2.774 +/- 0.074 | 0.838 +/- 0.026 | 0.669 +/- 0.008 |
| weak | global w3 | 2.774 +/- 0.074 | 0.839 +/- 0.025 | 0.673 +/- 0.007 |
| weak | time w3 | 2.775 +/- 0.074 | 0.838 +/- 0.025 | 0.670 +/- 0.007 |
| weak | low-obs w3 | 2.783 +/- 0.065 | 0.834 +/- 0.020 | 0.667 +/- 0.007 |
| intermittent | base | 2.774 +/- 0.074 | 0.838 +/- 0.026 | 0.667 +/- 0.016 |
| intermittent | global w3 | 2.773 +/- 0.074 | 0.839 +/- 0.025 | 0.671 +/- 0.016 |
| intermittent | time w3 | 2.775 +/- 0.074 | 0.838 +/- 0.025 | 0.668 +/- 0.016 |
| intermittent | low-obs w3 | 2.775 +/- 0.074 | 0.838 +/- 0.025 | 0.667 +/- 0.016 |

Simple variance penalties do not materially fix direct-head under-coverage.
Higher weights begin to hurt NLL and coverage. This branch should stop unless a
new calibration objective is introduced.

## Current Recommendation

Use the direct head as the default nonlinear strict Gaussian filter when the
primary target is state NLL:

```text
direct_moment_distilled
```

Keep the structured horizon-4 rollout head as the calibrated comparison:

```text
structured_moment_rollout_h4
```

Report the result as a real tradeoff:

- direct head: lower intermittent state NLL and simpler architecture
- structured rollout h4: better calibrated, especially on weak observations

Do not prioritize Mamba or learned mixture filters yet. The reference
moment-matched Gaussian is strong, and the learned update quality is still the
more immediate axis.

## Key Artifacts

Primary status note:

```text
docs/results/nonlinear_strict_filter_status_2026-04-27.md
```

Most useful plots:

```text
outputs/nonlinear_head_seed_sweep_1000/plots/sweep_comparison.png
outputs/nonlinear_direct_calibration_1000/plots/sweep_comparison.png
outputs/nonlinear_direct_time_calibration_1000/plots/sweep_comparison.png
outputs/nonlinear_direct_low_obs_calibration_1000/plots/sweep_comparison.png
outputs/nonlinear_rollout_distillation_h4_1000/plots/sweep_comparison.png
```

Most useful metrics:

```text
outputs/nonlinear_head_seed_sweep_1000/metrics.csv
outputs/nonlinear_direct_calibration_1000/w3/metrics.csv
outputs/nonlinear_direct_time_calibration_1000/w3/metrics.csv
outputs/nonlinear_direct_low_obs_calibration_1000/w3/metrics.csv
outputs/nonlinear_rollout_distillation_h4_1000/metrics.csv
```

## Next Step

The most pragmatic next step is not another calibration sweep. Either:

1. freeze this as the nonlinear strict-filter result and write it into the
   broader report, or
2. run `direct_moment_distilled` and `structured_moment_rollout_h4` on any
   additional nonlinear stressor the research director wants before finalizing
   the claim.

