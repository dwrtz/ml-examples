# Nonlinear Unsupervised ELBO Recovery: Experiment Task Pack

Prepared: 2026-04-28

This task pack converts the research plan into implementation and experiment tickets. It assumes the current repository already has nonlinear grid references, strict nonlinear learned filters, nonlinear sweep scripts, and reports for the prior reference-distilled milestone.

## Task pack overview

| ID | Task | Type | Priority | Main files |
|---|---|---|---|---|
| T0 | Freeze current nonlinear baselines | Reporting | P0 | `docs/results/`, `scripts/aggregate_*` |
| T1 | Add nonlinear pre-assimilation predictive log probability | Implementation | P0 | `src/vbf/predictive.py`, `src/vbf/nonlinear.py` |
| T2 | Add causal predictive-y auxiliary loss | Implementation | P0 | `src/vbf/losses.py`, `scripts/train_nonlinear.py` |
| T3 | Add masked-y update semantics | Implementation | P0 | `src/vbf/nonlinear.py`, `scripts/train_nonlinear.py` |
| T4 | Add masked-y span training and diagnostics | Implementation | P1 | `src/vbf/nonlinear.py`, `scripts/plot_nonlinear.py` |
| T5 | Implement windowed joint edge ELBO | Implementation | P0 | `src/vbf/losses.py`, `scripts/train_nonlinear.py` |
| T6 | Implement random-prefix joint edge ELBO | Implementation | P1 | `src/vbf/losses.py`, `scripts/train_nonlinear.py` |
| T7 | Add model specs and sweep keys | Implementation | P0 | `scripts/sweep_nonlinear_learned.py` |
| T8 | Linear-Gaussian sanity run | Experiment | P0 | `scripts/sweep_weak_observability.py` |
| T9 | Nonlinear weak/intermittent pilot | Experiment | P0 | `scripts/sweep_nonlinear_learned.py` |
| T10 | Combined objective pilot | Experiment | P1 | `scripts/sweep_nonlinear_learned.py` |
| T11 | Robustness suite | Experiment | P2 | `scripts/sweep_nonlinear_learned.py` |
| T12 | Final aggregation report | Reporting | P1 | `scripts/aggregate_nonlinear_unsupervised_objective_report.py` |

## Shared definitions

### Fully unsupervised rows

Rows in this task pack are considered fully unsupervised only if training uses:

```text
x_t, y_t, p(z_t | z_{t-1}), p(y_t | z_t, x_t), p(z_0)
```

and does not use:

```text
grid posterior moments
true latent z_t
reference variance targets
oracle edge posteriors
```

### Fair comparison rows

Use these as the primary baselines:

```text
structured_elbo
direct_elbo
```

Use these only as reference-assisted upper-bound diagnostics:

```text
direct_moment_distilled
structured_moment_rollout_h4
```

### Primary nonlinear configs

Use these first:

```text
experiments/nonlinear/03_weak_sine_observation.yaml
experiments/nonlinear/04_intermittent_sine_observation.yaml
```

Use these for final robustness only after the pilot succeeds:

```text
experiments/nonlinear/01_sine_observation.yaml
experiments/nonlinear/03_weak_sine_observation.yaml
experiments/nonlinear/04_intermittent_sine_observation.yaml
experiments/nonlinear/05_zero_sine_observation.yaml
experiments/nonlinear/06_random_normal_sine_observation.yaml
```

## T0 — Freeze current nonlinear baselines

### Goal

Make the current nonlinear result a stable reference point before adding new unsupervised objective branches.

### Rationale

The current positive nonlinear learned rows use reference information. They are useful upper-bound diagnostics but should not be conflated with fully unsupervised ELBO recovery.

### Required actions

- Preserve the current interpretation in `docs/results/nonlinear_strict_filter_final_summary_2026-04-28.md`.
- Make sure future reports distinguish fully unsupervised rows from reference-distilled rows.
- Add a short note in any new report that `direct_moment_distilled` and `structured_moment_rollout_h4` are not unsupervised training results.

### Acceptance criteria

- A clean checkout still has a committed summary of the prior nonlinear result.
- New experiment reports have a `training_signal` or equivalent column with values like:
  - `unsupervised`
  - `reference_distilled`
  - `oracle_calibrated`

## T1 — Add nonlinear pre-assimilation predictive log probability

### Goal

Add a model-consistent log probability for `y_t` under the pre-assimilation predictive belief.

### Target file

```text
src/vbf/predictive.py
src/vbf/nonlinear.py
```

### Required behavior

Given:

```text
q^F_{t-1}(z_{t-1}) = Normal(mu_{t-1}, var_{t-1})
z_t | z_{t-1} ~ Normal(z_{t-1}, Q)
y_t | z_t, x_t ~ Normal(x_t sin(z_t), R)
```

compute:

```text
log p(y_t | D_{<t}, x_t)
=
log ∫ Normal(y_t ; x_t sin(z_t), R) Normal(z_t ; mu_{t-1}, var_{t-1}+Q) dz_t
```

Use a stable estimator:

- preferred: Gauss-Hermite quadrature;
- acceptable: fixed reparameterized Monte Carlo with log-sum-exp;
- not preferred as the primary loss: Gaussian moment approximation to `y_t`.

### Guardrails

- The function may receive `y_t` only as the scored target.
- It must not receive post-assimilation `q^F_t`.
- It must not depend on the grid reference.

### Acceptance criteria

- Works for batched `[batch, time]` arrays.
- Returns finite values in weak, intermittent, zero, and random-normal nonlinear configs.
- In the linear-Gaussian analog, predictive NLL matches the analytic predictive within tolerance if implemented for the linear case too.

## T2 — Add causal predictive-y auxiliary loss

### Goal

Add a self-supervised predictive auxiliary to nonlinear training without changing the filter rollout.

### Target file

```text
scripts/train_nonlinear.py
src/vbf/losses.py
```

### New training config keys

```text
predictive_y_weight: 0.0
predictive_y_num_samples: 32
predictive_y_estimator: quadrature
```

### New model keys

```text
structured_elbo_predictive_y
direct_elbo_predictive_y
```

### Objective

During normal filtering:

1. use `q^F_{t-1}` to form the transition predictive belief `q^-_t`;
2. score `y_t` under `p(y_t | q^-_t, x_t)`;
3. assimilate `y_t` normally using the learned update;
4. add the predictive log likelihood to the training objective.

### Acceptance criteria

- With `predictive_y_weight = 0`, training reproduces the old nonlinear ELBO path.
- With positive `predictive_y_weight`, `metrics.json` reports the training weight and predictive loss value.
- No reference targets are used.

## T3 — Add masked-y update semantics

### Goal

Support missing measurements during training and evaluation by masking `y_t` from the learned update path.

### Target file

```text
src/vbf/nonlinear.py
scripts/train_nonlinear.py
```

### New training config keys

```text
mask_y_probability: 0.0
mask_y_span_probability: 0.0
mask_y_span_length: 1
mask_y_seed_offset: 70000
```

### Required behavior

When `y_t` is observed:

```text
q^F_t = update(q^F_{t-1}, x_t, y_t)
```

When `y_t` is masked:

```text
q^F_t = transition_predict(q^F_{t-1})
```

The masked `y_t` should still be scored by the pre-assimilation predictive loss when `predictive_y_weight > 0`.

### Guardrails

- Do not pass placeholder zeros for masked `y_t` into the learned update unless the model is explicitly given a mask indicator and the update is defined to ignore it.
- The first implementation should skip the learned update entirely on masked timesteps.
- Do not mask `x_t` in this task.

### Acceptance criteria

- Evaluation diagnostics include the mask used for each sequence.
- During a masked span, learned variance should grow by transition propagation rather than collapse.
- With `mask_y_probability = 0`, the old rollout path is unchanged.

## T4 — Add masked-y span diagnostics

### Goal

Measure whether the filter recovers after hidden measurement spans.

### Target file

```text
scripts/plot_nonlinear.py
scripts/aggregate_nonlinear_unsupervised_objective_report.py
```

### New diagnostics

Add time-since-observation summaries:

| Metric | Meaning |
|---|---|
| `gap_age` | Number of consecutive masked steps since last observed `y` |
| `gap_state_nll` | State NLL grouped by `gap_age` |
| `gap_coverage_90` | Coverage grouped by `gap_age` |
| `gap_variance_ratio` | Variance ratio grouped by `gap_age` |
| `gap_predictive_nll` | Held-out predictive NLL grouped by `gap_age` |

### Acceptance criteria

- Plotting produces a CSV and figure for gap recovery.
- Diagnostics work even when no mask is present, with a clear no-op or all-zero gap age.

## T5 — Implement windowed joint edge ELBO

### Goal

Add trajectory-consistent ELBO terms over short windows assembled from `q^F` and `q^B`.

### Target file

```text
src/vbf/losses.py
scripts/train_nonlinear.py
```

### New training config keys

```text
joint_elbo_weight: 0.0
joint_elbo_horizon: 1
joint_elbo_num_samples: 16
joint_elbo_num_windows: 8
joint_elbo_window_seed_offset: 80000
```

### New model keys

```text
structured_joint_elbo_h1
structured_joint_elbo_h2
structured_joint_elbo_h4
structured_joint_elbo_h8
direct_joint_elbo_h4
```

### Required behavior

For each sampled window `[s, s + H]`:

1. run the model forward normally to obtain all `q^F_t` and `q^B_t`;
2. sample a latent trajectory backward from `q^F_{s+H}` through the learned backward conditionals;
3. score the trajectory with the carried initial belief, transitions, and observations;
4. subtract the trajectory posterior log probability;
5. average over windows and samples.

### Identity check

When `joint_elbo_horizon = 1`, the joint/windowed ELBO should match the existing local edge ELBO up to Monte Carlo noise and implementation convention.

### Acceptance criteria

- `structured_joint_elbo_h1` is numerically close to `structured_elbo` on a fixed batch and seed.
- Training with `joint_elbo_weight = 0` reproduces old behavior.
- No reference targets are used.

## T6 — Implement random-prefix joint edge ELBO

### Goal

Give each intermediate filtering marginal direct terminal-posterior pressure by sampling random prefix endpoints.

### Target file

```text
src/vbf/losses.py
scripts/train_nonlinear.py
```

### New training config keys

```text
prefix_joint_elbo_weight: 0.0
prefix_joint_elbo_num_samples: 8
prefix_joint_elbo_num_prefixes: 4
prefix_joint_elbo_seed_offset: 90000
```

### New model keys

```text
structured_prefix_joint_elbo
structured_joint_elbo_h4_prefix
```

### Acceptance criteria

- Prefix endpoint sampling covers early, middle, and late times over multiple steps.
- Metrics improve or at least do not regress on the scalar linear-Gaussian sanity suite before nonlinear pilots.
- Prefix objective is reported separately from windowed objective in `metrics.json`.

## T7 — Add model specs and sweep keys

### Goal

Expose all new objective variants in the existing nonlinear sweep path.

### Target file

```text
scripts/sweep_nonlinear_learned.py
```

### Required model keys

```text
structured_elbo_predictive_y
direct_elbo_predictive_y
structured_elbo_masked_y
structured_elbo_masked_y_spans_h2
structured_elbo_masked_y_spans_h4
structured_elbo_masked_y_spans_h8
structured_joint_elbo_h1
structured_joint_elbo_h2
structured_joint_elbo_h4
structured_joint_elbo_h8
direct_joint_elbo_h4
structured_prefix_joint_elbo
structured_joint_elbo_h4_predictive_y
structured_joint_elbo_h4_masked_y_spans_h4
```

### Acceptance criteria

- Each key generates a config with all objective weights recorded.
- The sweep summary includes:
  - objective weights;
  - mask settings;
  - joint horizon;
  - predictive estimator;
  - training signal classification.

## T8 — Linear-Gaussian sanity run

### Goal

Before trusting nonlinear results, verify that the new objective improves or preserves scalar linear-Gaussian weak-observability behavior.

### Command template

Run after adding linear-compatible variants or a linear wrapper for the new losses:

```bash
make sweep-weak-observability \
  WEAK_OBSERVABILITY_MODELS=elbo,elbo_joint_h1,elbo_joint_h4,elbo_predictive_y,elbo_joint_h4_predictive_y \
  WEAK_OBSERVABILITY_STEPS=1000 \
  WEAK_OBSERVABILITY_DIR=outputs/linear_gaussian_unsupervised_objective_sanity_1000
```

### Success criteria

| Metric | Gate |
|---|---|
| `coverage_90` | improves over vanilla MC ELBO |
| `variance_ratio` | moves toward 1.0 |
| `state_nll` | does not materially degrade |
| `predictive_nll` | stable or improved |

### Stop condition

If the new objective cannot improve the scalar weak-observability ELBO baseline, do not proceed to the full nonlinear sweep until the implementation is checked.

## T9 — Nonlinear weak/intermittent pilot

### Goal

Test whether each objective branch improves the true unsupervised nonlinear rows.

### Command

Run after implementing the model keys:

```bash
make sweep-nonlinear-learned \
  NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  NONLINEAR_LEARNED_MODELS=structured_elbo,structured_joint_elbo_h2,structured_joint_elbo_h4,structured_joint_elbo_h8,structured_elbo_predictive_y,structured_elbo_masked_y_spans_h4,direct_elbo,direct_elbo_predictive_y \
  NONLINEAR_LEARNED_SEEDS=321,322,323 \
  NONLINEAR_LEARNED_STEPS=1000 \
  NONLINEAR_LEARNED_DIR=outputs/nonlinear_unsupervised_objective_pilot_1000
```

### Expected useful signal

| Metric | Useful result |
|---|---|
| weak coverage 90 | above vanilla structured ELBO; ideally `> 0.70` |
| intermittent coverage 90 | above vanilla structured ELBO; ideally `> 0.70` |
| variance ratio | above `0.50` in at least one hard case |
| state NLL | materially below old structured ELBO failure |
| predictive NLL | improved without pure variance inflation |

### Interpretation

- If `structured_joint_elbo_h4` improves coverage and variance ratio, locality/trajectory inconsistency was a major failure source.
- If `structured_elbo_predictive_y` improves predictive NLL and coverage, observation-space calibration was a major missing signal.
- If neither helps, the issue is likely the divergence/objective family or posterior family.

## T10 — Combined objective pilot

### Goal

Test the strongest combination of trajectory consistency and self-supervised predictive calibration.

### Command

Run after T9 identifies nonnegative individual branches:

```bash
make sweep-nonlinear-learned \
  NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  NONLINEAR_LEARNED_MODELS=structured_elbo,structured_joint_elbo_h4,structured_elbo_predictive_y,structured_joint_elbo_h4_predictive_y,structured_joint_elbo_h4_masked_y_spans_h4 \
  NONLINEAR_LEARNED_SEEDS=321,322,323 \
  NONLINEAR_LEARNED_STEPS=1000 \
  NONLINEAR_LEARNED_DIR=outputs/nonlinear_unsupervised_combined_objective_1000
```

### Weight grid

Use only a compact grid initially:

```text
predictive_y_weight: 0.1, 0.3, 1.0
joint_elbo_weight: 0.3, 1.0
mask_y_probability: 0.15, 0.30
mask_y_span_length: 4
```

### Acceptance criteria

- At least one combined row beats both individual branches on the state-NLL/calibration tradeoff.
- No row is promoted if it improves coverage only by inflating variance while worsening state NLL.

## T11 — Robustness suite

### Goal

Check that the best fully unsupervised objective generalizes beyond weak/intermittent.

### Command

Run only after T9/T10 produce a promising row:

```bash
make sweep-nonlinear-learned \
  NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/01_sine_observation.yaml,experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml,experiments/nonlinear/05_zero_sine_observation.yaml,experiments/nonlinear/06_random_normal_sine_observation.yaml \
  NONLINEAR_LEARNED_MODELS=structured_elbo,direct_elbo,BEST_UNSUPERVISED_OBJECTIVE,direct_moment_distilled,structured_moment_rollout_h4 \
  NONLINEAR_LEARNED_SEEDS=321,322,323 \
  NONLINEAR_LEARNED_STEPS=1000 \
  NONLINEAR_LEARNED_DIR=outputs/nonlinear_unsupervised_objective_robustness_1000
```

Replace `BEST_UNSUPERVISED_OBJECTIVE` with the selected model key.

### Acceptance criteria

- The best unsupervised objective improves weak/intermittent without breaking zero/random-normal cases.
- The report clearly separates fully unsupervised rows from reference-distilled rows.

## T12 — Final aggregation report

### Goal

Produce a report-ready summary of the new unsupervised objective branch.

### Target file

```text
scripts/aggregate_nonlinear_unsupervised_objective_report.py
```

### Inputs

```text
outputs/nonlinear_unsupervised_objective_pilot_1000/metrics.csv
outputs/nonlinear_unsupervised_combined_objective_1000/metrics.csv
outputs/nonlinear_unsupervised_objective_robustness_1000/metrics.csv
outputs/nonlinear_head_seed_sweep_1000/metrics.csv
```

### Outputs

```text
outputs/nonlinear_unsupervised_objective_final_report/summary.md
outputs/nonlinear_unsupervised_objective_final_report/summary.json
outputs/nonlinear_unsupervised_objective_final_report/summary.csv
```

### Report sections

- Executive summary.
- Baseline failure recap.
- Objective variants tested.
- Fully unsupervised rows only.
- Reference-distilled upper-bound rows.
- State-NLL versus calibration tradeoff.
- Masked-span recovery diagnostics.
- Decision: continue ELBO program, change divergence, or change posterior family.

### Acceptance criteria

- The report can be regenerated from CSV artifacts.
- It does not require opening individual run directories manually.
- It labels each row by training signal.

## Suggested implementation order

Do the work in this order:

1. T0 — freeze baseline interpretation.
2. T1 — predictive log probability.
3. T2 — predictive-y auxiliary.
4. T3 — masked-y semantics.
5. T5 — windowed joint ELBO with `h1` identity check.
6. T7 — sweep keys for the first pilots.
7. T8 — scalar sanity.
8. T9 — nonlinear weak/intermittent pilot.
9. T10 — combined objective pilot.
10. T4, T6, T11, T12 — diagnostics, prefix objective, robustness, and final reporting.

## Minimum viable pilot

If time is tight, implement only:

```text
structured_elbo_predictive_y
structured_joint_elbo_h1
structured_joint_elbo_h4
structured_joint_elbo_h4_predictive_y
```

Then run:

```bash
make sweep-nonlinear-learned \
  NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  NONLINEAR_LEARNED_MODELS=structured_elbo,structured_elbo_predictive_y,structured_joint_elbo_h1,structured_joint_elbo_h4,structured_joint_elbo_h4_predictive_y \
  NONLINEAR_LEARNED_SEEDS=321,322,323 \
  NONLINEAR_LEARNED_STEPS=1000 \
  NONLINEAR_LEARNED_DIR=outputs/nonlinear_unsupervised_minimum_pilot_1000
```

This is enough to answer the core question:

> Does trajectory consistency plus causal observation prediction materially improve the fully unsupervised nonlinear ELBO baseline?

## Decision matrix

| Outcome | Interpretation | Next action |
|---|---|---|
| Joint h4 improves calibration | Local edge objective was too myopic | Continue window/prefix ELBO branch |
| Predictive-y improves calibration | Observation-space prequential signal was missing | Continue masked-y/span objective branch |
| Combined objective wins | Objective repair is working | Run robustness suite and write report |
| Only variance inflates | Calibration is superficial | Add variance-quality diagnostics; reduce weights |
| No unsupervised row improves | Standard ELBO/divergence likely insufficient | Consider IWAE, entropy-regularized ELBO, alpha/Renyi objectives, or richer posterior family |
| Linear sanity fails | Implementation or objective accounting issue | Fix before nonlinear claims |

## Recommended final claim formats

### If successful

> The nonlinear strict Gaussian filter can be trained without reference posterior targets when the objective includes short-horizon trajectory consistency and causal held-out measurement prediction. The improvement specifically addresses the prior under-dispersion and self-fed rollout failure of the local edge ELBO.

### If partially successful

> A trajectory/predictive auxiliary improves the fully unsupervised nonlinear ELBO baseline, but a gap remains to reference moment distillation. The remaining gap likely reflects the divergence/posterior-family tradeoff rather than simple rollout locality.

### If unsuccessful

> The nonlinear failure is not fixed by making the edge ELBO trajectory-consistent or by adding causal observation-space prediction. The next branch should test alternative variational divergences or richer posterior families rather than additional local ELBO tuning.
