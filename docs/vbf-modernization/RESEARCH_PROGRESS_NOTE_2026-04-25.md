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
