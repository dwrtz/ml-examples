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

Use the following baseline taxonomy in reports:

| Category | Examples | Interpretation |
|---|---|---|
| Exact oracle | Kalman + exact edge posterior | Gold reference. |
| Analytic-marginal controls | zero-init, frozen-marginal backward MLP | Tests backward/edge learning, not learned filtering. |
| Residualized learned filters | self-fed supervised, ELBO MLP, closed-form ELBO MLP | Learns corrections and edge posteriors around a Kalman-structured marginal. |
| Less-structured learned filters | direct MLP filter ablations | Tests whether the filtering recursion can be learned with less analytic structure. |

The predictive head has the same distinction. The original head is an
`analytic_residual_predictive_head`: it adds learned residuals to the analytic
scalar linear-Gaussian predictive distribution. A `direct_predictive_head`
ablation is now available for non-residualized prediction.

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

## 11. Implemented director follow-up hooks

The code now includes the experiment hooks needed for the next decision points:

- `closed_form_elbo_edge_mlp` trains the structured MLP with the analytic scalar
  Gaussian edge ELBO instead of Monte Carlo samples.
- `direct_elbo_edge_mlp`, `direct_closed_form_elbo_edge_mlp`, and
  `direct_supervised_edge_mlp` provide less-structured filter ablations that do
  not compute the Kalman gain internally.
- `direct_predictive_head` provides the non-residualized predictive-head
  ablation; existing `predictive_head` configs are treated as the backward
  compatible alias for `analytic_residual_predictive_head`.
- `scripts/sweep_diagnostic_baselines.py` accepts `--steps 250,1000,3000` for
  matched-budget frozen-marginal and split-head diagnostics.

## 12. Follow-up experiment results

The matched-budget diagnostic baseline sweep was run to:

```text
outputs/linear_gaussian_diagnostic_baselines_budgeted/
```

Summary:

| Model | Steps | filter KL | edge KL | state NLL | cov 90 | var ratio | pred NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero-init no training | 0 | 0.000000 | 1.050438 | 0.401983 | 0.900220 | 1.000006 | 0.600858 |
| frozen marginal backward MLP | 250 | 0.000000 | 0.221206 | 0.401983 | 0.900220 | 1.000006 | 0.600858 |
| frozen marginal backward MLP | 1000 | 0.000000 | 0.112827 | 0.401983 | 0.900220 | 1.000006 | 0.600858 |
| frozen marginal backward MLP | 3000 | 0.000000 | 0.066231 | 0.401983 | 0.900220 | 1.000006 | 0.600858 |
| split-head supervised MLP | 250 | 0.262928 | 0.496513 | 0.664146 | 0.875427 | 2.031336 | 0.732080 |
| split-head supervised MLP | 1000 | 0.113507 | 0.195412 | 0.513492 | 0.892826 | 1.565684 | 0.664181 |
| split-head supervised MLP | 3000 | 0.086517 | 0.153136 | 0.487062 | 0.900334 | 1.698812 | 0.651496 |

Interpretation:

- The frozen-marginal backward conditional keeps exact Kalman filtering metrics
  and reduces edge KL from `1.05` at zero-init to `0.066` at 3000 steps.
- The split-head supervised baseline improves with budget, but it remains worse
  than frozen-marginal on edge KL and much worse on rolled-out filtering and
  predictive metrics.
- This strengthens the conclusion that the current structured benchmark is very
  effective at isolating backward/edge learning from filtering-marginal
  learning.

The residualization/objective matrix was run in parallel chunks under:

```text
outputs/linear_gaussian_residualization_objective_matrix_split/
```

Selected 3000-step rows:

| Objective | filter KL | edge KL | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |
|---|---:|---:|---:|---:|---:|---:|---:|
| MC ELBO structured | 0.090239 | 0.216081 | 0.492098 | 0.849060 | 0.662955 | 0.622721 | -0.702407 |
| closed-form ELBO structured | 0.076545 | 0.185636 | 0.478342 | 0.855013 | 0.651087 | 0.619151 | -0.699885 |
| MC ELBO direct | 1.712402 | 2.676846 | 2.113122 | 0.690120 | 0.383992 | 0.875051 | -1.139487 |
| closed-form ELBO direct | 1.855637 | 2.879886 | 2.254513 | 0.689697 | 0.379439 | 0.872124 | -1.136147 |
| supervised direct | 4.582399 | 6.266737 | 4.968732 | 0.823372 | 1.994343 | 3.107603 | -20.901517 |

Interpretation:

- Closed-form ELBO modestly improves the structured residualized model relative
  to MC ELBO, but it does not remove the under-dispersion problem. MC variance is
  therefore not the whole ELBO bottleneck.
- The direct non-residualized filter is much worse under both MC and closed-form
  ELBO at this budget. The current scalar benchmark performance depends heavily
  on the analytic Kalman-structured marginal update.
- Direct supervised teacher-forced edge distillation is unstable and should not
  be used as the main less-structured supervised baseline without a self-fed
  variant or stronger optimization controls.

The direct predictive-head sweep was rerun at 3000 steps to:

```text
outputs/linear_gaussian_direct_predictive_head_sweep_3000/
```

At 3000 steps the direct predictive head reaches predictive NLL `0.625870` on
oracle beliefs, versus exact Kalman predictive NLL `0.600858`. On ELBO beliefs,
the direct head gets `0.670554`, worse than the analytic predictive distribution
from those same beliefs (`0.640331`). This supports the previous interpretation:
in the scalar linear-Gaussian benchmark, predictive mapping is not the main
bottleneck; belief quality and calibration are.
