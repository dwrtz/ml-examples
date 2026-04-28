# Nonlinear Unsupervised ELBO Recovery: Master Research Plan

Prepared: 2026-04-28

## 1. Executive recommendation

The next nonlinear research branch should stay with the unsupervised ELBO program, but it should stop relying on a purely local edge objective. The strongest next direction is a two-part objective upgrade:

1. **Trajectory-consistent joint/windowed edge ELBO**: assemble a short latent trajectory posterior from the existing filtering marginal and backward edge conditionals, then score the whole latent path under the generative model.
2. **Causal self-supervised measurement prediction**: hide or hold out `y_t` from the update path and require the pre-assimilation belief to assign high probability to the held-out measurement.

This branch should be tested before moving to mixtures, Mamba/context models, or more reference variance penalties. The current evidence says the strict Gaussian posterior family is not the first bottleneck: a moment-matched Gaussian projection of the grid posterior is already close to the grid reference, while the learned unsupervised Gaussian remains much worse. The positive direct moment-distillation and structured horizon-4 rollout-distillation rows show that useful strict Gaussian updates exist when the training signal is right. The remaining open problem is therefore the **unsupervised objective**, not the posterior family alone.

## 2. Current diagnosis

The nonlinear benchmark is:

```text
z_t = z_{t-1} + w_t
y_t = x_t sin(z_t) + v_t
```

The existing strict learned filter carries an explicit Gaussian filtering marginal and backward edge conditional:

```text
q^E_t(z_t, z_{t-1}) = q^F_t(z_t) q^B_t(z_{t-1} | z_t)
```

The current local edge ELBO is useful but underpowered in the nonlinear weak/intermittent setting. Its failure mode is:

```text
local plausible explanation
→ posterior too narrow
→ posterior becomes the next prior
→ self-fed compounding error
→ low coverage and poor state NLL
```

The important empirical conclusions are:

- The grid reference and cache are usable enough for diagnostics.
- Vanilla nonlinear ELBO is badly under-dispersed in weak/intermittent cases.
- More steps, naive resampling, and simple variance penalties did not solve the issue.
- A moment-matched Gaussian reference is close to the grid reference, so single-Gaussian filtering remains a viable target.
- Direct reference moment distillation is a strong state-NLL positive control.
- Structured moment distillation has one-step capacity under teacher forcing but needs horizon rollout training to become stable.
- The best current nonlinear learned rows are reference-assisted, not fully unsupervised.

This plan targets the last point: recover better nonlinear filtering behavior without grid moments, true latent states, or oracle variance targets.

## 3. Research objective

Develop an unsupervised nonlinear training objective that improves state NLL, coverage, and variance calibration while preserving the strict online filtering contract:

```text
q^F_t = update(q^F_{t-1}, x_t, y_t)
```

The target claim is:

> A strict Gaussian learned filter can be trained from the generative model and observations alone when the ELBO objective is made trajectory-consistent and observation-predictive.

This is stronger than the current reference-distilled claim and should be reported separately.

## 4. Non-goals and guardrails

### Non-goals

- Do not attempt to beat `direct_moment_distilled` immediately. That row uses grid-reference targets and is not the fair baseline for this branch.
- Do not prioritize mixtures until the best unsupervised Gaussian objective has been tested.
- Do not prioritize Mamba/context models unless the strict-filter objective branch still fails.
- Do not add more simple variance-ratio penalties as the primary next step.
- Do not make masked `x` reconstruction a core objective unless a generative model for `x_t` is added.

### Guardrails

- The pre-assimilation predictive loss must not see `y_t` before predicting `y_t`.
- Full-sequence objectives must not quietly become smoothing claims unless they are labeled as smoothing diagnostics.
- All headline rows must be classified as one of:
  - **fully unsupervised**: no grid/reference moments or latent states used in training;
  - **reference-distilled diagnostic**: grid/reference information used;
  - **oracle/calibrated diagnostic**: reference variance or oracle statistics used.

## 5. Objective branch A: trajectory-consistent joint/windowed edge ELBO

### 5.1 Motivation

The local edge ELBO asks each time step to explain the current observation and transition. In a nonlinear observation model, this can permit locally good but globally inconsistent edge factors. A trajectory ELBO should make consecutive edge factors jointly accountable for a coherent latent path.

### 5.2 Full-prefix trajectory posterior

Use the existing filtering marginal and backward conditionals to define a trajectory posterior over a prefix:

```text
q_phi(z_{0:T} | y_{1:T}, x_{1:T})
=
q^F_{phi,T}(z_T)
∏_{t=1}^T q^B_{phi,t}(z_{t-1} | z_t)
```

The corresponding joint ELBO is:

```text
E_q [
    log p(z_0)
  + ∑_{t=1}^T log p(z_t | z_{t-1})
  + ∑_{t=1}^T log p(y_t | z_t, x_t)
  - log q^F_T(z_T)
  - ∑_{t=1}^T log q^B_t(z_{t-1} | z_t)
]
```

This remains unsupervised: it uses only the generative model and observations.

### 5.3 Windowed trajectory ELBO

A full-sequence ELBO may overemphasize terminal beliefs. Start with random windows because they directly target rollout stability:

```text
q(z_{s-1:s+H})
=
q^F_{s+H}(z_{s+H})
∏_{t=s}^{s+H} q^B_t(z_{t-1} | z_t)
```

Score the window against the carried prior belief and the generative factors:

```text
E_q [
    log q^F_{s-1}(z_{s-1})
  + ∑_{t=s}^{s+H} log p(z_t | z_{t-1})
  + ∑_{t=s}^{s+H} log p(y_t | z_t, x_t)
  - log q^F_{s+H}(z_{s+H})
  - ∑_{t=s}^{s+H} log q^B_t(z_{t-1} | z_t)
]
```

When `H = 1`, this should reduce to the current edge ELBO up to Monte Carlo estimation details. That identity is the first sanity check.

### 5.4 Random-prefix ELBO

Train intermediate filtering beliefs by sampling a prefix endpoint:

```text
τ ~ Uniform({1, ..., T})
```

Then define:

```text
q(z_{0:τ})
=
q^F_τ(z_τ)
∏_{t=1}^τ q^B_t(z_{t-1} | z_t)
```

This makes each `q^F_t` sometimes act as a terminal marginal of a joint ELBO, adding direct calibration pressure at all times.

### 5.5 Recommended variants

Initial model keys:

```text
structured_joint_elbo_h1
structured_joint_elbo_h2
structured_joint_elbo_h4
structured_joint_elbo_h8
direct_joint_elbo_h4
structured_prefix_joint_elbo
structured_joint_elbo_h4_prefix
```

Primary candidate:

```text
structured_joint_elbo_h4
```

Reason: the successful reference-assisted structured result used horizon-4 rollout distillation, so horizon 4 is the most plausible unsupervised rollout length.

## 6. Objective branch B: causal masked/self-supervised measurement prediction

### 6.1 Motivation

The current failure is not just one-step fitting. The carried belief becomes too narrow and then makes bad future predictions. A prequential measurement objective directly asks:

```text
Before seeing y_t, does q^F_{t-1} imply a predictive distribution that assigns high probability to y_t?
```

This gives observation-space calibration pressure without using grid posterior moments.

### 6.2 Primary target: predict `y`, not `x`

The current state-space model is conditional on `x_t`:

```text
p(z_{0:T}, y_{1:T} | x_{1:T})
```

Therefore, the clean predictive target is:

```text
p(y_t | D_{<t}, x_t)
```

There is no equally clean `p(x_t | history)` objective unless an explicit generative or predictive model for `x_t` is added. Predicting `x_t` should be deferred.

### 6.3 Pre-assimilation predictive distribution

Before assimilating `y_t`, form the transition predictive belief:

```text
q^-_t(z_t)
=
∫ p(z_t | z_{t-1}) q^F_{t-1}(z_{t-1}) dz_{t-1}
```

For the scalar random-walk Gaussian transition:

```text
q^-_t(z_t) = Normal(mu_{t-1}, var_{t-1} + Q)
```

Then score the held-out measurement:

```text
log p(y_t | q^-_t, x_t)
=
log ∫ Normal(y_t ; x_t sin(z_t), R) q^-_t(z_t) dz_t
```

Use quadrature or stable Monte Carlo mixture likelihood. Do not reduce the nonlinear predictive likelihood to only a Gaussian moment approximation for the primary training loss; the nonlinear predictive distribution can be multimodal.

### 6.4 Masking modes

Test three increasingly strong modes:

| Mode | Description | Purpose |
|---|---|---|
| Prequential every-step prediction | Score every `y_t` before assimilation, but assimilate normally afterward | Simple auxiliary calibration signal |
| Random `y` dropout | Hide `y_t` from the update at random timesteps but still score it predictively | Missing-measurement robustness |
| Contiguous `y` spans | Hide spans of length 2, 4, or 8 | Self-fed uncertainty propagation through gaps |

When `y_t` is masked:

```text
q^F_t = q^-_t
```

not:

```text
q^F_t = update(q^F_{t-1}, x_t, y_t)
```

### 6.5 Recommended variants

Initial model keys:

```text
structured_elbo_predictive_y
structured_elbo_masked_y
structured_elbo_masked_y_spans_h2
structured_elbo_masked_y_spans_h4
structured_elbo_masked_y_spans_h8
structured_joint_elbo_h4_predictive_y
direct_elbo_predictive_y
```

Primary candidate:

```text
structured_joint_elbo_h4_predictive_y
```

Reason: it combines trajectory consistency and pre-assimilation measurement calibration.

## 7. Combined objective

The final target objective should be a weighted combination:

```text
maximize
  local_edge_elbo_weight * local_edge_elbo
+ window_joint_elbo_weight * windowed_joint_edge_elbo
+ prefix_joint_elbo_weight * random_prefix_joint_edge_elbo
+ predictive_y_weight * prequential_or_masked_y_log_likelihood
```

Recommended initial weights:

| Term | Starting weights |
|---|---:|
| local edge ELBO | 1.0 |
| windowed joint ELBO | 0.3, 1.0 |
| prefix joint ELBO | 0.0, 0.3 |
| predictive `y` | 0.1, 0.3, 1.0 |

Avoid a large first sweep. Use a small weak/intermittent pilot and only expand after a clear calibration improvement.

## 8. Experimental staging

### Stage 0: preserve baseline interpretation

Freeze the current nonlinear result as a reference-distilled milestone. The current positive rows are not fully unsupervised:

```text
direct_moment_distilled
structured_moment_rollout_h4
```

They remain useful as upper-bound diagnostics, but the fair baseline for this branch is:

```text
structured_elbo
direct_elbo
```

### Stage 1: implementation sanity checks

Sanity checks before running nonlinear pilots:

- `structured_joint_elbo_h1` matches the current local edge ELBO within Monte Carlo noise.
- The pre-assimilation predictive loss receives `q^F_{t-1}`, `x_t`, and `y_t` only as a target.
- Masked `y` timesteps do not call the learned update with the hidden measurement.
- Masked spans increase filter variance during gaps rather than collapsing it.

### Stage 2: scalar linear-Gaussian sanity suite

Run the new objective on the scalar linear-Gaussian weak-observability suite. It should improve over vanilla MC ELBO before being trusted in nonlinear experiments.

Success criteria:

| Metric | Required movement |
|---|---|
| coverage 90 | higher than vanilla MC ELBO |
| variance ratio | moves toward 1.0 |
| state NLL | does not degrade materially |
| predictive NLL | improves or remains close to baseline |
| missing/weak observations | no variance collapse |

### Stage 3: nonlinear weak/intermittent pilot

Run only the difficult nonlinear cases first:

```text
experiments/nonlinear/03_weak_sine_observation.yaml
experiments/nonlinear/04_intermittent_sine_observation.yaml
```

Model set:

```text
structured_elbo
structured_joint_elbo_h2
structured_joint_elbo_h4
structured_joint_elbo_h8
structured_elbo_predictive_y
structured_elbo_masked_y_spans_h4
structured_joint_elbo_h4_predictive_y
direct_elbo
direct_elbo_predictive_y
```

Minimum useful signal:

| Metric | Desired movement |
|---|---|
| coverage 90 | above the current unsupervised ELBO baseline; ideally `> 0.70` |
| variance ratio | meaningfully above `0.10–0.15`; ideally `> 0.50` |
| state NLL | materially below old structured ELBO failure; ideally approaching `3–5` |
| predictive NLL | improves without pure variance inflation |
| masked-gap recovery | better after hidden spans |

### Stage 4: combined objective pilot

If either branch helps, test the combined objective:

```text
structured_joint_elbo_h4_predictive_y
structured_joint_elbo_h4_masked_y_spans_h4
structured_joint_elbo_h4_prefix_predictive_y
```

Use only seeds `321,322,323` and 1000 steps initially.

### Stage 5: robustness and reporting

Only after the weak/intermittent pilot shows a real improvement, run the broader nonlinear stressor set:

```text
experiments/nonlinear/01_sine_observation.yaml
experiments/nonlinear/03_weak_sine_observation.yaml
experiments/nonlinear/04_intermittent_sine_observation.yaml
experiments/nonlinear/05_zero_sine_observation.yaml
experiments/nonlinear/06_random_normal_sine_observation.yaml
```

Defer the long-sequence config unless the final claim needs long-horizon stability.

## 9. Metrics and diagnostics

Primary metrics:

| Metric | Meaning |
|---|---|
| state NLL | Main filtering quality metric against synthetic latent `z_t` |
| coverage 90 | Posterior calibration |
| variance ratio | Learned variance relative to grid reference variance |
| predictive NLL | Observation-space prequential quality |
| reference state NLL | Grid-reference comparison, not a training target |
| masked-gap state NLL | Recovery quality after hidden spans |
| masked-gap coverage 90 | Calibration after hidden spans |

Required diagnostics:

- time-local variance ratio;
- time-local coverage;
- time-local predictive NLL;
- split by observation strength;
- masked-span recovery curves by distance since last observed `y`;
- horizon rollout stability from reference initialization for comparison only.

## 10. Reporting rules

The final report should contain three groups of rows:

| Group | Rows |
|---|---|
| Fully unsupervised | `structured_elbo`, `direct_elbo`, new joint/predictive/masked variants |
| Reference-distilled positives | `direct_moment_distilled`, `structured_moment_rollout_h4` |
| Grid reference | deterministic grid filter and moment-matched Gaussian diagnostic |

Recommended wording if the branch succeeds:

> Windowed trajectory consistency and causal held-out measurement prediction materially reduce the nonlinear unsupervised ELBO under-dispersion failure while preserving the strict online filtering contract.

Recommended wording if the branch fails:

> The failure is not merely edge locality or lack of observation-space prediction. A standard exclusive-KL ELBO over a strict Gaussian posterior remains mode-seeking/under-dispersed in this nonlinear benchmark, motivating alternative variational divergences or richer posterior families.

## 11. Decision gates

### Continue the unsupervised ELBO program if

- weak/intermittent coverage 90 improves by at least 0.20 over vanilla structured ELBO;
- variance ratio moves above 0.50 in at least one hard nonlinear case;
- state NLL improves materially without pure variance inflation;
- the best fully unsupervised row closes a visible fraction of the gap to `direct_moment_distilled`.

### Stop this branch and change objective family if

- `h4` joint ELBO and predictive-y auxiliary both remain severely under-dispersed;
- improvements appear only through variance inflation while NLL worsens;
- `h1` identity or pre-assimilation leakage tests fail and cannot be fixed quickly;
- linear-Gaussian weak-observability sanity regresses against vanilla MC ELBO.

### Defer architecture changes until

- the best strict unsupervised objective has been compared against the current strict baselines;
- failure is localized to posterior family or temporal memory, not objective locality/calibration.

## 12. Recommended repository deliverables

Recommended new committed documents:

```text
docs/results/nonlinear_unsupervised_objective_plan_2026-04-28.md
docs/results/nonlinear_unsupervised_objective_task_pack_2026-04-28.md
```

Recommended new/modified code files:

```text
src/vbf/losses.py
src/vbf/predictive.py
src/vbf/nonlinear.py
scripts/train_nonlinear.py
scripts/sweep_nonlinear_learned.py
scripts/plot_nonlinear.py
scripts/aggregate_nonlinear_unsupervised_objective_report.py
experiments/nonlinear/*.yaml
```

Recommended final generated artifacts:

```text
outputs/nonlinear_unsupervised_objective_pilot_1000/metrics.csv
outputs/nonlinear_unsupervised_objective_pilot_1000/summary.md
outputs/nonlinear_unsupervised_objective_pilot_1000/plots/sweep_comparison.png
outputs/nonlinear_unsupervised_objective_final_report/summary.md
outputs/nonlinear_unsupervised_objective_final_report/summary.json
```

## 13. Bottom line

This branch is worth doing. It directly attacks the current failure mode while preserving the project’s core edge-factorized filtering structure. The best next experiment is not a larger model; it is a better unsupervised objective:

```text
windowed joint edge ELBO
+ causal held-out y predictive likelihood
+ y-dropout/span corruption
```

If that objective materially improves coverage and variance ratio without reference targets, the unsupervised ELBO program is still alive. If it fails, the next branch should change the divergence/objective family or posterior family, not continue local tuning.
