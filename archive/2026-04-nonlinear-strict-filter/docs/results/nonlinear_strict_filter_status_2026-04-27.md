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

Interpretation at this stage: self-fed direct moment supervision did not make
the EKF-residualized update match reference moments. That pointed toward either
update parameterization or rollout/compounding-error constraints.

### Direct And Teacher-Forced Moment Distillation

The follow-up diagnostic added a direct nonlinear MLP moment-distillation mode
and teacher-forced variants for both structured and direct heads.

The direct self-fed head is much stronger than the structured self-fed head:

| case | model | steps | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|---:|
| weak | direct moment distilled | 250 | 3.260 | 0.664 | 0.317 |
| weak | direct moment distilled | 1000 | 2.806 | 0.832 | 0.665 |
| intermittent | direct moment distilled | 250 | 3.258 | 0.664 | 0.313 |
| intermittent | direct moment distilled | 1000 | 2.806 | 0.832 | 0.656 |

Teacher-forced moment distillation then separated one-step capacity from rollout
stability:

| case | head | eval mode | state NLL | coverage 90 | variance ratio |
|---|---|---|---:|---:|---:|
| weak | structured | rollout | 32.420 | 0.333 | 0.510 |
| weak | structured | teacher-forced | 2.790 | 0.888 | 1.112 |
| intermittent | structured | rollout | 3.219 | 0.767 | 4540.661 |
| intermittent | structured | teacher-forced | 2.761 | 0.901 | 1.034 |
| weak | direct | rollout | 2.809 | 0.830 | 0.657 |
| weak | direct | teacher-forced | 2.809 | 0.828 | 0.658 |
| intermittent | direct | rollout | 2.809 | 0.831 | 0.648 |
| intermittent | direct | teacher-forced | 2.808 | 0.828 | 0.648 |

This changes the interpretation. The structured EKF-residualized head can learn
the one-step reference moment map when fed reference previous beliefs. Its
failure is self-fed rollout stability, not one-step representational capacity.
The direct head is stable under both teacher-forced and self-fed evaluation.

### Short-Horizon Rollout Distillation

The next diagnostic trained the structured head with short self-fed rollouts
initialized from reference beliefs:

```text
structured_moment_rollout_h{2,4,8}
```

The 250-step weak/intermittent results were:

| case | rollout horizon | state NLL | coverage 90 | variance ratio |
|---|---:|---:|---:|---:|
| weak | 2 | 3.257 | 0.775 | 1.061 |
| weak | 4 | 2.933 | 0.834 | 1.127 |
| weak | 8 | 18.662 | 0.396 | 0.891 |
| intermittent | 2 | 7.653 | 0.783 | 1.215 |
| intermittent | 4 | 7.974 | 0.767 | 0.961 |
| intermittent | 8 | 10.311 | 0.698 | 0.924 |

Horizon 4 materially improves the weak-observation structured head, bringing
coverage near the direct head while keeping variance slightly above the grid
reference. The intermittent case remains poor: short-horizon rollout
distillation fixes variance scale but not state NLL.

The comparison plot is at:

```text
outputs/nonlinear_rollout_distillation_250/plots/sweep_comparison.png
```

The best short-horizon setting, horizon 4, was then run for 1000 steps:

| case | rollout horizon | steps | state NLL | coverage 90 | variance ratio |
|---|---:|---:|---:|---:|---:|
| weak | 4 | 1000 | 2.808 | 0.881 | 0.993 |
| intermittent | 4 | 1000 | 3.340 | 0.861 | 1.124 |

This is a real recovery for the structured branch. It still trails the direct
head on intermittent NLL, but no longer has the catastrophic rollout failure
seen in teacher-forced one-step training.

The expanded comparison plot is at:

```text
outputs/nonlinear_rollout_distillation_h4_1000/plots/sweep_comparison.png
```

### Matched Direct-Versus-Structured Seed Sweep

The direct head and the horizon-4 rollout-distilled structured head were then
run with matched seeds `321,322,323`, weak/intermittent observations, and 1000
training steps.

Aggregate results:

| case | head | seeds | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|---:|
| weak | direct | 3 | 2.774 +/- 0.074 | 0.838 +/- 0.026 | 0.669 +/- 0.008 |
| weak | structured rollout h4 | 3 | 2.774 +/- 0.061 | 0.885 +/- 0.018 | 0.994 +/- 0.014 |
| intermittent | direct | 3 | 2.774 +/- 0.074 | 0.838 +/- 0.026 | 0.667 +/- 0.016 |
| intermittent | structured rollout h4 | 3 | 3.275 +/- 0.106 | 0.860 +/- 0.020 | 1.437 +/- 0.445 |

The weak case is effectively tied on state NLL, but structured rollout is much
better calibrated. The intermittent case favors the direct head on state NLL by
about 0.50, while structured rollout has higher coverage and less
under-dispersion but a noisier variance ratio.

The seed-averaged comparison plot is at:

```text
outputs/nonlinear_head_seed_sweep_1000/plots/sweep_comparison.png
```

### Direct-Head Global Variance Calibration

A calibrated direct-head variant was added:

```text
direct_moment_calibrated
```

It keeps direct moment distillation and adds a global reference
variance-ratio penalty. We swept weights `0.1,1,3,10,30` with matched seeds
`321,322,323`.

Aggregate direct-head results:

| case | variance weight | state NLL | coverage 90 | variance ratio |
|---|---:|---:|---:|---:|
| weak | 0 | 2.774 +/- 0.074 | 0.838 +/- 0.026 | 0.669 +/- 0.008 |
| weak | 0.1 | 2.774 +/- 0.074 | 0.838 +/- 0.025 | 0.670 +/- 0.008 |
| weak | 1 | 2.773 +/- 0.074 | 0.838 +/- 0.025 | 0.672 +/- 0.008 |
| weak | 3 | 2.774 +/- 0.074 | 0.839 +/- 0.025 | 0.673 +/- 0.007 |
| weak | 10 | 2.779 +/- 0.069 | 0.837 +/- 0.022 | 0.673 +/- 0.006 |
| weak | 30 | 2.798 +/- 0.060 | 0.832 +/- 0.018 | 0.672 +/- 0.005 |
| intermittent | 0 | 2.774 +/- 0.074 | 0.838 +/- 0.026 | 0.667 +/- 0.016 |
| intermittent | 0.1 | 2.774 +/- 0.074 | 0.838 +/- 0.025 | 0.667 +/- 0.016 |
| intermittent | 1 | 2.773 +/- 0.074 | 0.839 +/- 0.025 | 0.670 +/- 0.016 |
| intermittent | 3 | 2.773 +/- 0.074 | 0.839 +/- 0.025 | 0.671 +/- 0.016 |
| intermittent | 10 | 2.779 +/- 0.069 | 0.837 +/- 0.022 | 0.671 +/- 0.014 |
| intermittent | 30 | 2.797 +/- 0.060 | 0.833 +/- 0.018 | 0.671 +/- 0.013 |

The global variance-ratio penalty is not an effective direct-head calibration
fix. It barely changes variance ratio or coverage through weight 10, and by
weight 30 it begins to hurt NLL without improving calibration.

The comparison plot is at:

```text
outputs/nonlinear_direct_calibration_1000/plots/sweep_comparison.png
```

### Direct-Head Time-Local Variance Calibration

A time-local calibrated direct-head variant was then added:

```text
direct_moment_time_calibrated
```

It keeps direct moment distillation and adds a per-time reference variance-ratio
penalty. We swept weights `1,3,10` with matched seeds `321,322,323`.

Aggregate direct-head results:

| case | time weight | state NLL | coverage 90 | variance ratio |
|---|---:|---:|---:|---:|
| weak | 0 | 2.774 +/- 0.074 | 0.838 +/- 0.026 | 0.669 +/- 0.008 |
| weak | 1 | 2.774 +/- 0.074 | 0.838 +/- 0.025 | 0.670 +/- 0.008 |
| weak | 3 | 2.775 +/- 0.074 | 0.838 +/- 0.025 | 0.670 +/- 0.007 |
| weak | 10 | 2.782 +/- 0.069 | 0.835 +/- 0.021 | 0.669 +/- 0.006 |
| intermittent | 0 | 2.774 +/- 0.074 | 0.838 +/- 0.026 | 0.667 +/- 0.016 |
| intermittent | 1 | 2.774 +/- 0.074 | 0.838 +/- 0.025 | 0.668 +/- 0.016 |
| intermittent | 3 | 2.775 +/- 0.074 | 0.838 +/- 0.025 | 0.668 +/- 0.016 |
| intermittent | 10 | 2.782 +/- 0.069 | 0.835 +/- 0.022 | 0.667 +/- 0.014 |

Time-local calibration also fails to recover the direct head's under-coverage.
This suggests the direct head's variance scale is not easily corrected by
simple aggregate variance penalties once self-fed moment distillation has
converged.

The comparison plot is at:

```text
outputs/nonlinear_direct_time_calibration_1000/plots/sweep_comparison.png
```

### Direct-Head Low-Observation Variance Calibration

A low-observation calibrated direct-head variant was added:

```text
direct_moment_low_obs_calibrated
```

It keeps direct moment distillation and weights the per-time reference
variance-ratio penalty toward low-observation timesteps. We swept weights
`1,3,10` with matched seeds `321,322,323`.

Aggregate direct-head results:

| case | low-obs weight | state NLL | coverage 90 | variance ratio |
|---|---:|---:|---:|---:|
| weak | 0 | 2.774 +/- 0.074 | 0.838 +/- 0.026 | 0.669 +/- 0.008 |
| weak | 1 | 2.775 +/- 0.074 | 0.837 +/- 0.025 | 0.668 +/- 0.008 |
| weak | 3 | 2.783 +/- 0.065 | 0.834 +/- 0.020 | 0.667 +/- 0.007 |
| weak | 10 | 2.783 +/- 0.070 | 0.834 +/- 0.022 | 0.666 +/- 0.006 |
| intermittent | 0 | 2.774 +/- 0.074 | 0.838 +/- 0.026 | 0.667 +/- 0.016 |
| intermittent | 1 | 2.774 +/- 0.074 | 0.838 +/- 0.025 | 0.667 +/- 0.016 |
| intermittent | 3 | 2.775 +/- 0.074 | 0.838 +/- 0.025 | 0.667 +/- 0.016 |
| intermittent | 10 | 2.782 +/- 0.069 | 0.834 +/- 0.021 | 0.666 +/- 0.014 |

Low-observation calibration is also negative. It does not recover coverage, and
larger weights slightly hurt NLL and coverage. This closes the simple
direct-head variance-penalty branch.

The comparison plot is at:

```text
outputs/nonlinear_direct_low_obs_calibration_1000/plots/sweep_comparison.png
```

## Current Interpretation

The nonlinear benchmark is now telling a sharper story:

1. The grid-reference and diagnostics infrastructure is in good shape.
2. The zero-observation stress case is solved by reference variance calibration.
3. Weak/intermittent nonlinear cases remain badly under-dispersed.
4. More steps, naive resampling, log-variance calibration, and direct moment
   distillation for the structured self-fed head do not close the gap.
5. The moment-matched Gaussian reference is good enough that a mixture posterior
   is not the immediate missing piece.
6. The structured EKF-residualized head can learn the one-step map under teacher
   forcing, but its self-fed rollout is unstable.
7. Short-horizon rollout distillation stabilizes the structured head at 1000
   steps, especially with horizon 4.
8. Across a 3-seed matched sweep, direct and structured rollout are tied on weak
   observations, but direct remains better on intermittent state NLL.
9. Global, time-local, and low-observation variance-ratio calibration do not
   materially fix direct-head under-coverage.
10. The direct head is a strong positive control and remains stable in rollout.

In short: the current structured nonlinear update is not primarily blocked by
one-step capacity. It is blocked by rollout stability/compounding error, and
rollout distillation is a viable training fix. The direct head remains the
cleaner architecture for intermittent observations when state NLL is the
dominant criterion. If calibration is the dominant criterion, the structured
horizon-4 rollout head remains the better calibrated option. Simple variance
penalties should no longer be the next branch for the direct head.

## Suggested Next Decisions

### Decision 1: Investigate Rollout Stability

The next diagnostic should target the gap between teacher-forced one-step
performance and self-fed rollout behavior.

Possible checks:

- add rollout stability plots for variance ratio and NLL by horizon
- inspect variance-path saturation diagnostics for the structured head during
  intermittent rollouts
- compare direct and structured rollout heads on additional nonlinear stressors
  before introducing richer posterior families

### Decision 2: Promote Direct Head To Main Nonlinear Candidate

The direct head should become the default nonlinear candidate. The structured
head is viable with rollout distillation and better calibrated on weak
observations, but the direct head has the better intermittent NLL and is a
simpler architecture.

### Decision 3: Defer Mamba/Mixture Until This Is Resolved

The current evidence still does not justify Mamba or a learned mixture filter as
the next primary branch. The reference moment Gaussian is already strong, and
the direct Gaussian head can learn useful moments. The next best work is
therefore rollout-stability training for strict Gaussian filters.

## Suggested Immediate Next Experiment

Stop simple direct-head calibration and make the next experiment a reporting or
robustness step:

```text
direct_moment_distilled
structured_moment_rollout_h4
```

Run these two heads on any additional nonlinear stressors the research director
wants in the final claim. Otherwise, report the current result as a
state-NLL-versus-calibration tradeoff.

Interpretation:

- Use the direct head as the state-NLL-focused nonlinear default.
- Use structured horizon-4 rollout as the better calibrated strict Gaussian
  comparison.

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
  --models structured_moment_distilled,direct_moment_distilled \
  --steps 1000 \
  --output-dir outputs/nonlinear_moment_distillation_1000

uv run python scripts/sweep_nonlinear_learned.py \
  --configs experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  --models structured_moment_teacher_forced,direct_moment_teacher_forced \
  --steps 1000 \
  --output-dir outputs/nonlinear_teacher_forced_moment_1000

uv run python scripts/sweep_nonlinear_learned.py \
  --configs experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  --models structured_moment_rollout_h2,structured_moment_rollout_h4,structured_moment_rollout_h8 \
  --steps 250 \
  --output-dir outputs/nonlinear_rollout_distillation_250

uv run python scripts/sweep_nonlinear_learned.py \
  --configs experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  --models structured_moment_rollout_h4 \
  --steps 1000 \
  --output-dir outputs/nonlinear_rollout_distillation_h4_1000

make sweep-nonlinear-learned \
  NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  NONLINEAR_LEARNED_MODELS=direct_moment_distilled,structured_moment_rollout_h4 \
  NONLINEAR_LEARNED_SEEDS=321,322,323 \
  NONLINEAR_LEARNED_STEPS=1000 \
  NONLINEAR_LEARNED_DIR=outputs/nonlinear_head_seed_sweep_1000

for w in 0.1 1 3 10 30; do \
  make sweep-nonlinear-learned \
    NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
    NONLINEAR_LEARNED_MODELS=direct_moment_calibrated \
    NONLINEAR_LEARNED_SEEDS=321,322,323 \
    NONLINEAR_REFERENCE_VARIANCE_RATIO_WEIGHT=$$w \
    NONLINEAR_LEARNED_STEPS=1000 \
    NONLINEAR_LEARNED_DIR=outputs/nonlinear_direct_calibration_1000/w$$w; \
done

make plot-nonlinear-sweep \
  NONLINEAR_SWEEP_METRICS=outputs/nonlinear_moment_distillation_1000/metrics.csv,outputs/nonlinear_teacher_forced_moment_1000/metrics.csv,outputs/nonlinear_rollout_distillation_250/metrics.csv \
  NONLINEAR_SWEEP_BASELINE_METRICS=outputs/nonlinear_moment_distillation_1000/metrics.csv \
  NONLINEAR_SWEEP_WEIGHTS=self-fed,teacher,rollout \
  NONLINEAR_SWEEP_PATTERNS=weak_sinusoidal,intermittent_sinusoidal \
  NONLINEAR_SWEEP_PLOT_DIR=outputs/nonlinear_rollout_distillation_250/plots

make plot-nonlinear-sweep \
  NONLINEAR_SWEEP_METRICS=outputs/nonlinear_moment_distillation_1000/metrics.csv,outputs/nonlinear_teacher_forced_moment_1000/metrics.csv,outputs/nonlinear_rollout_distillation_250/metrics.csv,outputs/nonlinear_rollout_distillation_h4_1000/metrics.csv \
  NONLINEAR_SWEEP_BASELINE_METRICS=outputs/nonlinear_moment_distillation_1000/metrics.csv \
  NONLINEAR_SWEEP_WEIGHTS=self-fed,teacher,rollout250,rollout1000 \
  NONLINEAR_SWEEP_PATTERNS=weak_sinusoidal,intermittent_sinusoidal \
  NONLINEAR_SWEEP_PLOT_DIR=outputs/nonlinear_rollout_distillation_h4_1000/plots

make plot-nonlinear-sweep \
  NONLINEAR_SWEEP_METRICS=outputs/nonlinear_head_seed_sweep_1000/metrics.csv \
  NONLINEAR_SWEEP_BASELINE_METRICS=outputs/nonlinear_head_seed_sweep_1000/metrics.csv \
  NONLINEAR_SWEEP_WEIGHTS=seed-sweep \
  NONLINEAR_SWEEP_PATTERNS=weak_sinusoidal,intermittent_sinusoidal \
  NONLINEAR_SWEEP_PLOT_DIR=outputs/nonlinear_head_seed_sweep_1000/plots

uv run python scripts/plot_nonlinear_sweep.py \
  --metrics outputs/nonlinear_head_seed_sweep_1000/metrics.csv,outputs/nonlinear_direct_calibration_1000/w0.1/metrics.csv,outputs/nonlinear_direct_calibration_1000/w1/metrics.csv,outputs/nonlinear_direct_calibration_1000/w3/metrics.csv,outputs/nonlinear_direct_calibration_1000/w10/metrics.csv,outputs/nonlinear_direct_calibration_1000/w30/metrics.csv \
  --weights base,0.1,1,3,10,30 \
  --patterns weak_sinusoidal,intermittent_sinusoidal \
  --output-dir outputs/nonlinear_direct_calibration_1000/plots

for w in 1 3 10; do \
  make sweep-nonlinear-learned \
    NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
    NONLINEAR_LEARNED_MODELS=direct_moment_time_calibrated \
    NONLINEAR_LEARNED_SEEDS=321,322,323 \
    NONLINEAR_REFERENCE_TIME_VARIANCE_RATIO_WEIGHT=$$w \
    NONLINEAR_LEARNED_STEPS=1000 \
    NONLINEAR_LEARNED_DIR=outputs/nonlinear_direct_time_calibration_1000/w$$w; \
done

uv run python scripts/plot_nonlinear_sweep.py \
  --metrics outputs/nonlinear_head_seed_sweep_1000/metrics.csv,outputs/nonlinear_direct_time_calibration_1000/w1/metrics.csv,outputs/nonlinear_direct_time_calibration_1000/w3/metrics.csv,outputs/nonlinear_direct_time_calibration_1000/w10/metrics.csv \
  --weights base,1,3,10 \
  --patterns weak_sinusoidal,intermittent_sinusoidal \
  --output-dir outputs/nonlinear_direct_time_calibration_1000/plots

for w in 1 3 10; do \
  make sweep-nonlinear-learned \
    NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
    NONLINEAR_LEARNED_MODELS=direct_moment_low_obs_calibrated \
    NONLINEAR_LEARNED_SEEDS=321,322,323 \
    NONLINEAR_REFERENCE_LOW_OBSERVATION_VARIANCE_RATIO_WEIGHT=$$w \
    NONLINEAR_LEARNED_STEPS=1000 \
    NONLINEAR_LEARNED_DIR=outputs/nonlinear_direct_low_obs_calibration_1000/w$$w; \
done

uv run python scripts/plot_nonlinear_sweep.py \
  --metrics outputs/nonlinear_head_seed_sweep_1000/metrics.csv,outputs/nonlinear_direct_low_obs_calibration_1000/w1/metrics.csv,outputs/nonlinear_direct_low_obs_calibration_1000/w3/metrics.csv,outputs/nonlinear_direct_low_obs_calibration_1000/w10/metrics.csv \
  --weights base,1,3,10 \
  --patterns weak_sinusoidal,intermittent_sinusoidal \
  --output-dir outputs/nonlinear_direct_low_obs_calibration_1000/plots

make plot-nonlinear \
  RUN_DIR=outputs/nonlinear_calibration_weight_sweep_250/w3/nonlinear_weak_sine_observation_structured_elbo_ref_calibrated

uv run python scripts/diagnose_nonlinear_mixture_projection.py \
  --run-dir outputs/nonlinear_calibration_weight_sweep_250/w3/nonlinear_weak_sine_observation_structured_elbo_ref_calibrated
```

## Review Questions For The Research Director

1. Should the nonlinear branch default to the simpler direct head given the
   intermittent NLL advantage?
2. Is the direct-head under-coverage acceptable for the current research claim,
   given that simple calibration penalties failed?
3. Is the current nonlinear claim framed correctly as: "the structured
   EKF-residualized head learns the one-step map and can be stabilized with
   short-horizon rollout distillation"?
4. Are there additional rollout-stability diagnostics you want before changing
   the update architecture?
