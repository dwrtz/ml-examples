# Variational Bayesian Filtering Modernization: Director Feedback Follow-up

Prepared: 2026-04-25

## 1. Summary

The research director's 2026-04-25 feedback identified that the benchmark had
the right structure but needed tighter evaluation before adding more
architecture. The first cleanup pass has now been implemented.

The main correction is that `state_rmse` now consistently means global RMSE over
all batch and time entries:

```text
sqrt(mean((posterior_mean - z)^2))
```

The previous mean over per-time RMSEs is still reported, but only as the
explicitly named `state_rmse_time_mean`.

## 2. Implemented changes

Evaluation and reporting now include:

- `state_rmse_global`, with `state_rmse` kept as an alias for compatibility;
- `state_rmse_time_mean`;
- learned and oracle `state_nll`;
- `coverage_50`, `coverage_90`, and `coverage_95`;
- oracle coverage metrics;
- `mean_filter_variance`, `oracle_mean_filter_variance`, and `variance_ratio`;
- variance-free closed-form scalar Gaussian edge ELBO diagnostics;
- saved run `config.yaml`;
- saved trained parameters in `params.npz`.

The comparison and sweep scripts were updated so learned and oracle rows use the
same RMSE definition in their main tables. They also surface coverage and
variance-ratio diagnostics so calibration does not depend only on NLL.

## 3. Interpretation update

The earlier statement that learned models can have lower point RMSE than exact
Kalman should be treated as provisional for any run generated before this
change. Older reports mixed RMSE definitions across learned and oracle rows.
New comparison tables should use `state_rmse_global`.

The strict MLP baseline should also be described precisely. It is not a generic
neural filter learned entirely from scratch. `StructuredMLPCell` uses a
Kalman-structured residualized marginal update and learns corrections plus the
backward conditional:

```text
q^E_t(z_t, z_{t-1}) = q^F_t(z_t) q^B_t(z_{t-1} | z_t)
```

This is a useful inductive-bias baseline, but the research narrative should say
"Kalman-structured marginal update with learned edge/backward conditional" until
zero-update, frozen-marginal, and less-structured baselines are run.

## 4. New analytic ELBO diagnostic

For the scalar linear-Gaussian benchmark, the local edge ELBO is now available in
closed form under the factorization:

```text
q^F_t(z_t) q^B_t(z_{t-1} | z_t)
```

This gives a variance-free reference for the Monte Carlo ELBO estimator and
should be used before interpreting ELBO gaps as architectural failures.

## 5. Next recommended experiments

The corrected standard five-seed sweep was rerun to:

```text
outputs/linear_gaussian_sweep_corrected_metrics/
```

Summary:

| Model | Objective | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MLP edge ELBO | unsupervised ELBO | 0.139600 | 0.416881 | 0.579526 | 0.542554 | 0.828984 | 0.615180 | 0.640331 |
| exact Kalman | oracle | 0.000000 | 0.000000 | 0.522426 | 0.401983 | 0.900220 | 1.000000 | 0.600858 |
| MLP supervised edge KL | oracle distillation | 0.228665 | 0.449262 | 0.707524 | 0.629996 | 0.881197 | 1.929025 | 0.708275 |

The corrected RMSE table changes the interpretation: exact Kalman is better on
global state RMSE, state NLL, coverage, and predictive NLL. ELBO still beats the
current supervised baseline on learned-model metrics, but it is under-dispersed
relative to Kalman (`variance_ratio` about `0.62`) and misses 90% coverage.

Before moving to Mamba, nonlinear observations, or other high-capacity context
models, continue the corrected ablations:

```text
uv run python scripts/sweep_elbo_ablation.py --steps 250,1000,3000
```

Then add the director's three diagnostic baselines:

- zero-update initialized model, no training;
- frozen Kalman-structured marginal with learned backward conditional only;
- separate filter and backward heads.

These baselines have been added as runnable configs:

```text
experiments/linear_gaussian/09_zero_init_edge_mlp.yaml
experiments/linear_gaussian/10_frozen_marginal_backward_mlp.yaml
experiments/linear_gaussian/11_supervised_edge_split_mlp.yaml
```

Run them with:

```text
uv run python scripts/sweep_diagnostic_baselines.py
```

The five-seed diagnostic sweep was run to:

```text
outputs/linear_gaussian_diagnostic_baselines/
```

Summary:

| Model | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen marginal backward MLP | 0.000000 | 0.221206 | 0.522426 | 0.401983 | 0.900220 | 1.000006 | 0.600858 |
| split-head supervised MLP | 0.262928 | 0.496513 | 0.740461 | 0.664146 | 0.875427 | 2.031336 | 0.732080 |
| zero-init no training | 0.000000 | 1.050438 | 0.522426 | 0.401983 | 0.900220 | 1.000006 | 0.600858 |

Interpretation:

- The zero-initialized strict MLP already carries an almost exact Kalman
  filtering marginal because of the residualized Kalman update.
- Training only the backward conditional reduces edge KL substantially, from
  about `1.05` to `0.22`, without changing the filtering marginal.
- The separate-hidden-tower split-head supervised baseline is currently worse
  than the original shared-hidden supervised MLP at the same 250-step budget.

## 6. Predictive head milestone

The learned one-step predictive head has been added:

```text
scripts/train_predictive_head.py
experiments/linear_gaussian/06_predictive_head.yaml
src/vbf/models/heads.py
```

The head predicts a Gaussian distribution for `y_t` from only:

```text
q^F_{t-1}.mean, q^F_{t-1}.var, x_t, Q, R
```

The current observation `y_t` is used only as the training target. No-leakage
tests cover both feature construction and output invariance under changes to
current `y_t`.

The first standard seed run was written to:

```text
outputs/linear_gaussian_predictive_head/
```

Result:

| Predictor | predictive NLL | predictive RMSE | variance ratio |
|---|---:|---:|---:|
| learned predictive head | 0.599251 | 0.452217 | 0.990911 |
| exact Kalman predictive | 0.595622 | 0.451505 | 1.000000 |

The five-seed predictive sweep was run to:

```text
outputs/linear_gaussian_predictive_head_sweep/
```

Summary:

| Predictor | predictive NLL | predictive RMSE | variance ratio |
|---|---:|---:|---:|
| learned head on oracle belief | 0.603713 | 0.454580 | 0.991398 |
| exact Kalman predictive | 0.600858 | 0.453947 | 1.000000 |
| learned head on ELBO belief | 0.645217 | n/a | 0.926123 |
| analytic predictive from ELBO belief | 0.640331 | n/a | n/a |
| exact predictive on same ELBO eval | 0.600858 | n/a | 1.000000 |

The head is close to exact when fed oracle/Kalman beliefs. When fed the
ELBO-trained filter beliefs, it is slightly worse than the analytic
model-consistent predictive from the same learned belief. That suggests the next
bottleneck is the learned filtering belief calibration, not a missing predictive
mapping in the scalar linear-Gaussian case.

## 7. Matched objective-budget sweep

The fair supervised-vs-ELBO budget sweep has been added and run:

```text
scripts/sweep_objective_budget.py
outputs/linear_gaussian_objective_budget/
```

The sweep compares supervised edge distillation and ELBO training at matched
training budgets:

```text
250, 1000, 3000 steps
```

All ELBO rows use `32` Monte Carlo samples. Summary:

| Objective | Steps | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELBO | 250 | 0.189909 | 0.726516 | 0.595214 | 0.593159 | 0.821537 | 0.591008 | 0.656687 | -0.941231 |
| ELBO | 1000 | 0.139600 | 0.416881 | 0.579526 | 0.542554 | 0.828984 | 0.615180 | 0.640331 | -0.782532 |
| ELBO | 3000 | 0.090239 | 0.216081 | 0.558504 | 0.492098 | 0.849060 | 0.662955 | 0.622721 | -0.702407 |
| supervised | 250 | 0.228665 | 0.449262 | 0.707524 | 0.629996 | 0.881197 | 1.929025 | 0.708275 | -1.138828 |
| supervised | 1000 | 0.173900 | 0.287539 | 0.694481 | 0.574403 | 0.883887 | 1.769057 | 0.688892 | -0.997207 |
| supervised | 3000 | 0.120872 | 0.212267 | 0.716988 | 0.521150 | 0.897465 | 1.949102 | 0.670420 | -1.533387 |

Interpretation:

- ELBO is not only winning because of the previous 1000-vs-250 step mismatch.
  At matched budgets, ELBO is better on filter KL, global state RMSE, state NLL,
  predictive NLL, and closed-form ELBO.
- Supervised edge distillation catches up on edge KL by 3000 steps, but its
  carried filtering state is still much worse for prediction.
- Supervised remains over-dispersed (`variance_ratio` around `1.8-1.95`),
  while ELBO remains under-dispersed but improves with budget (`0.59 -> 0.66`).
- The 3000-step supervised closed-form ELBO is unstable across seeds, suggesting
  teacher-forced edge matching can produce rolled-out beliefs that score poorly
  under the sequential variational objective.

Next, add a self-fed supervised objective. Current supervised training is
teacher-forced on oracle previous filtering beliefs but evaluated by rolling out
the model's own beliefs. A self-fed supervised variant will separate
teacher-forcing mismatch from objective mismatch.

## 8. Self-fed supervised objective

The self-fed supervised variant has been added:

```text
experiments/linear_gaussian/12_self_fed_supervised_edge_mlp.yaml
scripts/sweep_self_fed_supervised.py
```

Training objective:

```text
roll out q_model using its own q^F_{t-1}
minimize KL(q_oracle_edge || q_model_edge) at each time step
```

This keeps supervised oracle edge targets but removes the teacher-forcing
mismatch from the original supervised baseline.

The five-seed self-fed sweep has now been run to:

```text
outputs/linear_gaussian_self_fed_supervised/
```

Summary at 3000 steps:

| Model | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| self-fed supervised | 0.013449 | 0.052610 | 0.556290 | 0.415215 | 0.899569 | 1.264531 | 0.608135 | -0.684673 |
| ELBO | 0.090239 | 0.216081 | 0.558504 | 0.492098 | 0.849060 | 0.662955 | 0.622721 | -0.702407 |
| teacher-forced supervised | 0.120872 | 0.212267 | 0.716988 | 0.521150 | 0.897465 | 1.949102 | 0.670420 | -1.533387 |

Interpretation:

- The earlier supervised weakness was largely a teacher-forcing mismatch.
- Self-fed supervised is now the strongest learned linear-Gaussian baseline on
  filter KL, edge KL, state NLL, coverage, predictive NLL, and closed-form ELBO.
- ELBO remains the main unsupervised baseline and still improves with training
  budget, but it is under-dispersed relative to Kalman.
- Teacher-forced supervised should remain in reports as a failure-mode baseline,
  not as the best supervised comparison.

## 9. Self-fed variance-ratio regularizer

A diagnostic self-fed supervised variance-ratio regularizer has been added and
run. The useful fixed sweep is:

```text
outputs/linear_gaussian_self_fed_variance_regularizer_fixed/
```

Summary:

| variance weight | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.013449 | 0.052610 | 0.556290 | 0.415215 | 0.899569 | 1.264531 | 0.608135 | -0.684673 |
| 0.1 | 0.012900 | 0.051457 | 0.552749 | 0.415025 | 0.898189 | 1.013291 | 0.607679 | -0.679857 |
| 1 | 0.014044 | 0.054504 | 0.553757 | 0.416383 | 0.897880 | 0.999278 | 0.608232 | -0.683403 |
| 10 | 0.013349 | 0.053560 | 0.550709 | 0.415677 | 0.897091 | 1.000108 | 0.608097 | -0.679481 |

Interpretation:

- The regularizer successfully fixes the mean variance ratio with little cost.
- A weight around `0.1` is a reasonable calibrated supervised baseline for the
  next linear-Gaussian generalization suites.
- The earlier directory
  `outputs/linear_gaussian_self_fed_variance_regularizer/` should be treated as
  stale because those rows were identical across weights.

## 10. Current next steps

The next high-leverage experiments are linear-Gaussian generalization suites,
not Mamba or nonlinear observations yet:

1. Weak observability: compare sinusoidal, weak sinusoidal, intermittent
   sinusoidal, zero, and random observation covariates.
2. Q/R generalization: train/evaluate across multiple process-noise and
   observation-noise settings.
3. Report exact Kalman, ELBO, self-fed supervised, and calibrated self-fed
   supervised rows in the same tables.

Mamba and nonlinear observations should remain postponed until the strict-filter
generalization story is stable.

The five-seed sweep was run to:

```text
outputs/linear_gaussian_self_fed_supervised/
```

Summary:

| Objective | Steps | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| self-fed supervised | 250 | 0.026810 | 0.191091 | 0.556726 | 0.428692 | 0.898551 | 1.513749 | 0.613313 | -1.010395 |
| self-fed supervised | 1000 | 0.020340 | 0.096317 | 0.557577 | 0.421941 | 0.899369 | 1.368491 | 0.610837 | -0.771260 |
| self-fed supervised | 3000 | 0.013449 | 0.052610 | 0.556290 | 0.415215 | 0.899569 | 1.264531 | 0.608135 | -0.684673 |
| ELBO | 3000 | 0.090239 | 0.216081 | 0.558504 | 0.492098 | 0.849060 | 0.662955 | 0.622721 | -0.702407 |
| teacher-forced supervised | 3000 | 0.120872 | 0.212267 | 0.716988 | 0.521150 | 0.897465 | 1.949102 | 0.670420 | -1.533387 |

Interpretation:

- The main supervised failure was teacher-forcing mismatch. When trained
  self-fed, supervised edge distillation becomes much stronger than the
  teacher-forced baseline.
- At 3000 steps, self-fed supervised beats ELBO on filter KL, edge KL, state
  NLL, coverage, predictive NLL, and closed-form ELBO.
- Self-fed supervised is still over-dispersed relative to Kalman
  (`variance_ratio` about `1.26` at 3000 steps), but much less than
  teacher-forced supervised.
- ELBO remains under-dispersed, while self-fed supervised remains
  over-dispersed. This makes calibration the next central issue.

Next useful experiment: add a calibrated/self-fed supervised objective variant
or variance regularizer that targets variance ratio and NLL without degrading
edge KL.

## 9. Self-fed variance calibration sweep

Time-resolved calibration diagnostics are now saved for linear-Gaussian runs:

```text
variance_ratio_over_time
mean_filter_variance_over_time
oracle_mean_filter_variance_over_time
coverage_50_over_time
coverage_90_over_time
coverage_95_over_time
```

A global filtering-variance ratio regularizer was added for self-fed supervised
training:

```text
loss = edge_KL + lambda * log(mean(q_var) / mean(oracle_var))^2
```

Sweep script:

```text
scripts/sweep_self_fed_variance_regularizer.py
```

The first requested small-weight sweep (`0, 0.001, 0.01, 0.05, 0.1`) initially
showed no effect because the regularizer weight was accidentally not passed into
the self-fed branch. After fixing the wiring, an effective sweep was run to:

```text
outputs/linear_gaussian_self_fed_variance_regularizer_fixed/
```

Summary:

| lambda | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.013449 | 0.052610 | 0.556290 | 0.415215 | 0.899569 | 1.264531 | 0.608135 | -0.684673 |
| 0.1 | 0.012900 | 0.051457 | 0.552749 | 0.415025 | 0.898189 | 1.013291 | 0.607679 | -0.679857 |
| 1 | 0.014044 | 0.054504 | 0.553757 | 0.416383 | 0.897880 | 0.999278 | 0.608232 | -0.683403 |
| 10 | 0.013349 | 0.053560 | 0.550709 | 0.415677 | 0.897091 | 1.000108 | 0.608097 | -0.679481 |

Interpretation:

- The global variance regularizer successfully corrects the self-fed variance
  ratio from `1.26` to near `1.0`.
- `lambda=0.1` is the best current tradeoff: it improves variance ratio,
  filter KL, edge KL, global RMSE, predictive NLL, and closed-form ELBO
  relative to no regularizer.
- Coverage remains near 90% and changes only slightly, which suggests aggregate
  variance ratio alone is not the full calibration story.
- The next calibration step should inspect `coverage_*_over_time` and
  `variance_ratio_over_time`; if miscalibration is time-local, move from global
  variance regularization to per-time calibration penalties.
