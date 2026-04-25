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

After those baselines are in place, the learned predictive head remains the next
natural modernization milestone.
