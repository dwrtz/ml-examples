# Nonlinear Divergence / Posterior-Family Branch: Experiment Task Pack

Prepared: 2026-04-28

This task pack converts the master research task into implementation and experiment tickets. It assumes the repository already contains the nonlinear grid reference, strict Gaussian nonlinear learned filters, the combined unsupervised objective branch, nonlinear sweep scripts, and the latest aggregation/reporting scripts.

## Task pack overview

| ID | Task | Type | Priority | Main files |
|---|---|---|---|---|
| T0 | Freeze current promoted unsupervised baseline | Reporting | P0 | `docs/results/`, `scripts/aggregate_nonlinear_unsupervised_objective_report.py` |
| T1 | Add divergence/objective config plumbing | Implementation | P0 | `scripts/train_nonlinear.py`, `scripts/sweep_nonlinear_learned.py` |
| T2 | Implement Gaussian windowed IWAE objective | Implementation | P0 | `src/vbf/losses.py`, `scripts/train_nonlinear.py` |
| T3 | Implement Gaussian alpha-Rényi objective | Implementation | P1 | `src/vbf/losses.py`, `scripts/train_nonlinear.py` |
| T4 | Add optional unsupervised entropy regularization | Implementation | P2 | `src/vbf/losses.py` |
| T5 | Implement strict Gaussian mixture belief objects | Implementation | P0 | `src/vbf/distributions.py` |
| T6 | Add mixture update heads and rollout support | Implementation | P0 | `src/vbf/models/heads.py`, `src/vbf/models/cells.py`, `src/vbf/nonlinear.py` |
| T7 | Add mixture-compatible predictive-y and masked-y paths | Implementation | P0 | `src/vbf/predictive.py`, `src/vbf/nonlinear.py`, `src/vbf/losses.py` |
| T8 | Add reference mixture density-projection diagnostic | Diagnostic | P1 | `scripts/diagnose_nonlinear_mixture_projection.py`, `src/vbf/losses.py` |
| T9 | Add model keys to nonlinear sweep | Implementation | P0 | `scripts/sweep_nonlinear_learned.py`, `scripts/plot_nonlinear_sweep.py` |
| T10 | Add unit and identity checks | Testing | P0 | `tests/`, `src/vbf/losses.py`, `src/vbf/distributions.py` |
| T11 | Gaussian divergence pilot | Experiment | P0 | `scripts/sweep_nonlinear_learned.py` |
| T12 | Mixture family pilot | Experiment | P0 | `scripts/sweep_nonlinear_learned.py` |
| T13 | Full 2×2 pilot | Experiment | P0 | `scripts/sweep_nonlinear_learned.py` |
| T14 | Reference-assisted mixture diagnostic | Experiment | P1 | `scripts/diagnose_nonlinear_mixture_projection.py` |
| T15 | Robustness suite for winning row | Experiment | P2 | `scripts/sweep_nonlinear_learned.py` |
| T16 | Final divergence/family aggregation report | Reporting | P1 | `scripts/aggregate_nonlinear_unsupervised_objective_report.py` or new aggregator |

## Shared definitions

### Fully unsupervised rows

Rows are fully unsupervised only if training uses:

```text
x_t, y_t, p(z_t | z_{t-1}), p(y_t | z_t, x_t), p(z_0)
```

and does not use:

```text
grid posterior moments
true latent z_t
reference variance targets
oracle edge posteriors
reference rollout targets
reference density targets
```

### Reference-assisted diagnostics

These are allowed for diagnosis, but not for headline unsupervised claims:

```text
direct_moment_distilled
structured_moment_rollout_h4
direct_mixture_k2_reference_grid_distilled
direct_mixture_k3_reference_grid_distilled
```

### Current unsupervised baseline

All new comparisons must include:

```text
structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
```

Use this as the current best fully unsupervised robustness baseline.

### Primary pilot configs

Use these first:

```text
experiments/nonlinear/03_weak_sine_observation.yaml
experiments/nonlinear/04_intermittent_sine_observation.yaml
```

Use the full robustness set only after the pilot succeeds:

```text
experiments/nonlinear/01_sine_observation.yaml
experiments/nonlinear/03_weak_sine_observation.yaml
experiments/nonlinear/04_intermittent_sine_observation.yaml
experiments/nonlinear/05_zero_sine_observation.yaml
experiments/nonlinear/06_random_normal_sine_observation.yaml
```

## T0 — Freeze current promoted unsupervised baseline

### Goal

Make the current unsupervised objective-repair result the stable baseline for the new divergence/family branch.

### Required actions

- Preserve `docs/results/nonlinear_unsupervised_elbo_t11_status_2026-04-28.md`.
- Preserve `docs/results/nonlinear_unsupervised_elbo_master_research_plan.md` and `docs/results/nonlinear_unsupervised_elbo_experiment_task_pack.md` as the completed prior branch.
- Make sure all new reports include `training_signal` with values:
  - `unsupervised`
  - `reference_distilled`
  - `oracle_calibrated`
- Add a short note in the new report that the current baseline is a partial success, not a solved nonlinear filter.

### Acceptance criteria

- The new branch never compares only against vanilla `structured_elbo` while omitting the promoted unsupervised objective.
- The current promoted row is labeled fully unsupervised.
- Reference-distilled rows remain clearly separated from fully unsupervised rows.

## T1 — Add divergence/objective config plumbing

### Goal

Expose objective-family selection in nonlinear training and sweeps.

### Target files

```text
scripts/train_nonlinear.py
scripts/sweep_nonlinear_learned.py
src/vbf/losses.py
```

### New training config keys

```text
objective_family: elbo
num_importance_samples: 1
renyi_alpha: 1.0
entropy_bonus_weight: 0.0
mixture_components: 1
posterior_family: gaussian
```

Recommended values:

```text
objective_family: elbo | iwae | renyi
posterior_family: gaussian | gaussian_mixture
mixture_components: 1 | 2 | 3
```

### Required behavior

- Existing configs reproduce old behavior when new keys are omitted.
- All objective settings are written to `metrics.json` and `metrics.csv`.
- The sweep summary can group rows by objective family and posterior family.

### Acceptance criteria

- Existing nonlinear learned sweep still runs for `structured_elbo` and `direct_elbo`.
- Existing promoted objective row still reproduces its previous config values.
- `metrics.csv` includes the new fields.

## T2 — Implement Gaussian windowed IWAE objective

### Goal

Test whether the current under-dispersion is caused by the objective/divergence rather than the Gaussian family.

### Target files

```text
src/vbf/losses.py
scripts/train_nonlinear.py
```

### Objective

For each sampled window or trajectory term, estimate a multi-sample bound:

```text
log p(y_window | x_window)
≈
log mean_k exp(log w_k)
```

where each importance weight uses the same generative-model scoring currently used by the windowed joint ELBO:

```text
log w_k = log p(z_path, y_window | carried belief, x_window) - log q(z_path | observations)
```

The implementation should support:

```text
joint_elbo_horizon: 4
num_importance_samples: 8, 16, 32
```

### New model keys

```text
structured_joint_iwae_h4_k8
structured_joint_iwae_h4_k16
structured_joint_iwae_h4_k32
direct_joint_iwae_h4_k16
```

### Guardrails

- Do not accidentally use grid-reference moments.
- Keep the rollout/update path identical to the current strict filter.
- Report whether the objective is computed over local edge terms or windowed trajectory terms.

### Acceptance criteria

- `k1` reproduces the current windowed ELBO convention up to implementation details.
- `k8` and `k16` produce finite losses and gradients on weak/intermittent configs.
- Training does not create NaNs or negative variances.

## T3 — Implement Gaussian alpha-Rényi objective

### Goal

Test whether a less mode-seeking objective improves calibration without changing the posterior family.

### Target files

```text
src/vbf/losses.py
scripts/train_nonlinear.py
```

### New model keys

```text
structured_joint_renyi_h4_alpha_0p3
structured_joint_renyi_h4_alpha_0p5
structured_joint_renyi_h4_alpha_0p7
structured_joint_renyi_h4_alpha_0p9
```

### Required behavior

- Use the same windowed trajectory posterior as the current h4 objective.
- Use `renyi_alpha` only when `objective_family = renyi`.
- Record `renyi_alpha` in metrics and config artifacts.

### Acceptance criteria

- All alpha settings produce finite metrics on a one-seed smoke run.
- Alpha settings are distinguishable in `summary.md` and plots.
- No row is promoted unless it improves calibration without state-NLL regression.

## T4 — Add optional unsupervised entropy regularization

### Goal

Add a low-priority diagnostic for whether fully unsupervised entropy pressure can counter posterior collapse.

### Target file

```text
src/vbf/losses.py
```

### New model keys

```text
structured_joint_entropy_h4_beta_0p001
structured_joint_entropy_h4_beta_0p003
structured_joint_entropy_h4_beta_0p01
```

### Guardrails

- This must not use reference variance.
- This is not a primary branch unless IWAE/Rényi fail or are inconclusive.
- Promotion requires NLL and predictive-NLL checks to rule out pure variance inflation.

### Acceptance criteria

- Entropy value is logged separately from ELBO/IWAE/Rényi terms.
- A row that improves coverage only by worsening NLL is explicitly marked as rejected.

## T5 — Implement strict Gaussian mixture belief objects

### Goal

Add the minimal expressive variational family while preserving explicit filtering marginals.

### Target file

```text
src/vbf/distributions.py
```

### New objects

```text
GaussianMixtureBelief
ConditionalGaussianMixtureBackward
GaussianMixtureEdgePosterior
```

### Required family

```text
q^F_t(z_t)
=
sum_k pi_{t,k} Normal(z_t ; mu_{t,k}, sigma^2_{t,k})
```

```text
q^B_t(z_{t-1} | z_t, k)
=
Normal(z_{t-1} ; a_{t,k} z_t + b_{t,k}, tau^2_{t,k})
```

```text
q^E_t(z_t, z_{t-1})
=
sum_k pi_{t,k}
  Normal(z_t ; mu_{t,k}, sigma^2_{t,k})
  Normal(z_{t-1} ; a_{t,k} z_t + b_{t,k}, tau^2_{t,k})
```

### Required methods

- `sample(num_samples, key)`
- `log_prob(z_t, z_tm1)`
- `filtering_belief()`
- `mean_and_var()` for the filtering marginal
- `edge_mean_cov()` for diagnostics where feasible
- stable log-sum-exp mixture density evaluation

### Acceptance criteria

- K=1 mixture matches the existing Gaussian family within tolerance.
- Mixture weights sum to one and remain finite.
- All variances are positive with the existing `min_var` guardrail.
- Unit tests cover log probability, sampling shape, and marginal mean/variance.

## T6 — Add mixture update heads and rollout support

### Goal

Make direct and structured nonlinear update heads output mixture parameters.

### Target files

```text
src/vbf/models/heads.py
src/vbf/models/cells.py
src/vbf/nonlinear.py
```

### Required behavior

For each component `k`, the head must output:

```text
logit_pi_k
mu_k
logvar_k
a_k
b_k
logtau2_k
```

The filter rollout must carry:

```text
{pi_k, mu_k, var_k}_{k=1..K}
```

as the filtering marginal.

### New model keys

```text
direct_mixture_k2_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
structured_mixture_k2_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
direct_mixture_k3_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
```

### Guardrails

- Start with K=2.
- K=3 is a follow-up, not part of the minimum viable pilot.
- Do not introduce hidden recurrent context.
- The exported filtering belief must be the mixture itself, not only its moment projection.

### Acceptance criteria

- K=1 mixture recovers existing Gaussian metrics on a smoke run.
- K=2 direct mixture trains without NaNs for weak/intermittent one-seed smoke runs.
- Diagnostics save both moment summary and mixture parameters when practical.

## T7 — Add mixture-compatible predictive-y and masked-y paths

### Goal

Keep the current best unsupervised objective components available for mixtures.

### Target files

```text
src/vbf/predictive.py
src/vbf/nonlinear.py
src/vbf/losses.py
```

### Required behavior

For mixture filtering beliefs:

- transition prediction maps each component through the random-walk transition;
- predictive-y likelihood is evaluated as a mixture-integral over components;
- masked-y steps propagate the mixture through transition dynamics without calling the learned update;
- state NLL and coverage use the mixture density or a clearly labeled moment approximation.

### Acceptance criteria

- `predictive_y_nll` is finite for mixture rows.
- Masked spans do not collapse mixture variance.
- Plotting does not fail when mixture diagnostics are present.

## T8 — Add reference mixture density-projection diagnostic

### Goal

Estimate whether the mixture family is useful independent of unsupervised training.

### Target files

```text
scripts/diagnose_nonlinear_mixture_projection.py
src/vbf/losses.py
```

### Diagnostic objective

Fit a K-component mixture to the grid posterior density using cross-entropy:

```text
minimize  - sum_grid p_grid(z_t | D_t) log q_mix(z_t) Δz
```

### New diagnostic rows

```text
direct_mixture_k2_reference_grid_distilled
direct_mixture_k3_reference_grid_distilled
```

### Required outputs

```text
outputs/nonlinear_mixture_reference_projection/metrics.csv
outputs/nonlinear_mixture_reference_projection/summary.md
outputs/nonlinear_mixture_reference_projection/plots/mixture_projection.png
```

### Acceptance criteria

- Report grid reference NLL, moment Gaussian NLL, learned Gaussian NLL, and learned mixture NLL.
- Label these rows `reference_distilled`.
- Do not mix these rows into fully unsupervised comparisons.

## T9 — Add nonlinear sweep model keys

### Goal

Expose all new objective/family rows through the existing sweep entry point.

### Target files

```text
scripts/sweep_nonlinear_learned.py
scripts/plot_nonlinear_sweep.py
```

### Required model keys

```text
structured_joint_iwae_h4_k8
structured_joint_iwae_h4_k16
structured_joint_iwae_h4_k32
direct_joint_iwae_h4_k16
structured_joint_renyi_h4_alpha_0p3
structured_joint_renyi_h4_alpha_0p5
structured_joint_renyi_h4_alpha_0p7
structured_joint_renyi_h4_alpha_0p9
structured_joint_entropy_h4_beta_0p001
structured_joint_entropy_h4_beta_0p003
structured_joint_entropy_h4_beta_0p01
direct_mixture_k2_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
structured_mixture_k2_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
direct_mixture_k2_joint_iwae_h4_k16
structured_mixture_k2_joint_iwae_h4_k16
direct_mixture_k3_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
```

### Acceptance criteria

- Each key writes a config with `posterior_family`, `mixture_components`, `objective_family`, and divergence settings.
- The plot labels are short and readable.
- Unknown keys raise a useful error.

## T10 — Add unit and identity checks

### Goal

Catch implementation errors before running expensive nonlinear sweeps.

### Recommended tests

| Test | Purpose |
|---|---|
| K=1 mixture equals Gaussian | Family compatibility |
| mixture weights normalize | Numerical stability |
| mixture log-prob finite | Density stability |
| h4 IWAE k1 equals h4 ELBO convention | Objective identity |
| predictive-y no leakage | Causal training path |
| masked-y mixture propagation | Correct missing-measurement semantics |
| metrics finite for one batch | Smoke test |

### Acceptance criteria

Run:

```bash
make test
```

and pass all new tests before running T11–T13.

## T11 — Gaussian divergence pilot

### Goal

Test objective/divergence changes before changing posterior family.

### Command

```bash
make sweep-nonlinear-learned \
  NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  NONLINEAR_LEARNED_MODELS=structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4,structured_joint_iwae_h4_k8,structured_joint_iwae_h4_k16,structured_joint_renyi_h4_alpha_0p5,structured_joint_renyi_h4_alpha_0p7 \
  NONLINEAR_LEARNED_SEEDS=321,322,323 \
  NONLINEAR_LEARNED_STEPS=1000 \
  NONLINEAR_LEARNED_DIR=outputs/nonlinear_gaussian_divergence_pilot_1000
```

### Success criteria

| Metric | Gate |
|---|---:|
| variance ratio | `> 0.20` in weak or intermittent |
| coverage 90 | `+0.10` over current promoted baseline |
| state NLL | no material regression |
| predictive NLL | stable or improved |

### Interpretation

- If a Gaussian divergence row passes, prioritize the divergence branch before implementing larger mixtures.
- If all Gaussian divergence rows fail, continue to T12.

## T12 — Mixture family pilot

### Goal

Test whether K=2 mixture expressivity helps under the current promoted objective.

### Command

```bash
make sweep-nonlinear-learned \
  NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  NONLINEAR_LEARNED_MODELS=structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4,direct_mixture_k2_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4,structured_mixture_k2_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4 \
  NONLINEAR_LEARNED_SEEDS=321,322,323 \
  NONLINEAR_LEARNED_STEPS=1000 \
  NONLINEAR_LEARNED_DIR=outputs/nonlinear_mixture_family_pilot_1000
```

### Success criteria

Same as T11, plus:

- mixture rows must not produce unstable component collapse;
- K=2 must improve over the current promoted Gaussian row, not merely over vanilla `structured_elbo`;
- state NLL must be computed using the mixture density or clearly labeled if moment-projected.

### Interpretation

- If K=2 mixture improves under the current objective, posterior family is a major bottleneck.
- If K=2 does not improve, continue to T13 before rejecting mixture expressivity.

## T13 — Full 2×2 pilot

### Goal

Test objective/family interaction directly.

### Command

```bash
make sweep-nonlinear-learned \
  NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  NONLINEAR_LEARNED_MODELS=structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4,structured_joint_iwae_h4_k16,structured_joint_renyi_h4_alpha_0p5,direct_mixture_k2_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4,direct_mixture_k2_joint_iwae_h4_k16,structured_mixture_k2_joint_iwae_h4_k16 \
  NONLINEAR_LEARNED_SEEDS=321,322,323 \
  NONLINEAR_LEARNED_STEPS=1000 \
  NONLINEAR_LEARNED_DIR=outputs/nonlinear_divergence_family_pilot_1000
```

### Required plot

```bash
make plot-nonlinear-sweep \
  NONLINEAR_SWEEP_METRICS=outputs/nonlinear_divergence_family_pilot_1000/metrics.csv \
  NONLINEAR_SWEEP_BASELINE_METRICS=outputs/nonlinear_divergence_family_pilot_1000/metrics.csv \
  NONLINEAR_SWEEP_WEIGHTS=div-family \
  NONLINEAR_SWEEP_PATTERNS=weak_sinusoidal,intermittent_sinusoidal \
  NONLINEAR_SWEEP_PLOT_DIR=outputs/nonlinear_divergence_family_pilot_1000/plots
```

### Acceptance criteria

- At least one row passes the first success gate before T15 robustness.
- The report includes the 2×2 interpretation table.
- Rows are grouped by posterior family and objective family.

## T14 — Reference-assisted mixture diagnostic

### Goal

Estimate the upper-bound value of K=2/K=3 mixtures relative to moment Gaussian projection.

### Command template

Run on representative baseline and candidate run directories:

```bash
uv run python scripts/diagnose_nonlinear_mixture_projection.py \
  --run-dir outputs/nonlinear_divergence_family_pilot_1000/nonlinear_weak_sine_observation_seed_321_direct_mixture_k2_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4 \
  --components 1,2,3
```

### Required summary

Create:

```text
outputs/nonlinear_mixture_reference_projection/summary.md
```

with:

| Projection | Meaning |
|---|---|
| grid | deterministic grid posterior density |
| moment Gaussian | moment-matched Gaussian to grid posterior |
| learned Gaussian | existing learned strict Gaussian |
| reference-fitted K=2 mixture | density-projected mixture diagnostic |
| reference-fitted K=3 mixture | density-projected mixture diagnostic |

### Acceptance criteria

- The diagnostic explains whether mixture expressivity has a real reference-side advantage.
- It is not used as a fully unsupervised result.

## T15 — Robustness suite for winning row

### Goal

Run the best row from T11–T13 on the full nonlinear stressor set.

### Command

```bash
make sweep-nonlinear-learned \
  NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/01_sine_observation.yaml,experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml,experiments/nonlinear/05_zero_sine_observation.yaml,experiments/nonlinear/06_random_normal_sine_observation.yaml \
  NONLINEAR_LEARNED_MODELS=structured_elbo,direct_elbo,structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4,BEST_DIVERGENCE_FAMILY_ROW,direct_moment_distilled,structured_moment_rollout_h4 \
  NONLINEAR_LEARNED_SEEDS=321,322,323 \
  NONLINEAR_LEARNED_STEPS=1000 \
  NONLINEAR_LEARNED_DIR=outputs/nonlinear_divergence_family_robustness_1000
```

Replace:

```text
BEST_DIVERGENCE_FAMILY_ROW
```

with the selected fully unsupervised row.

### Acceptance criteria

- Winning row improves weak/intermittent calibration over the current promoted objective.
- Zero and random-normal do not regress relative to current promoted objective.
- Clean sinusoidal does not catastrophically regress.
- Reference-distilled diagnostics are included only as upper-bound controls.

## T16 — Final divergence/family aggregation report

### Goal

Produce a report-ready summary of the branch.

### Target files

```text
scripts/aggregate_nonlinear_unsupervised_objective_report.py
```

or a new script:

```text
scripts/aggregate_nonlinear_divergence_family_report.py
```

### Inputs

```text
outputs/nonlinear_gaussian_divergence_pilot_1000/metrics.csv
outputs/nonlinear_mixture_family_pilot_1000/metrics.csv
outputs/nonlinear_divergence_family_pilot_1000/metrics.csv
outputs/nonlinear_mixture_reference_projection/metrics.csv
outputs/nonlinear_divergence_family_robustness_1000/metrics.csv
```

### Outputs

```text
outputs/nonlinear_divergence_family_final_report/summary.md
outputs/nonlinear_divergence_family_final_report/summary.json
outputs/nonlinear_divergence_family_final_report/summary.csv
```

### Report sections

- Executive summary.
- Current promoted unsupervised baseline recap.
- 2×2 divergence-versus-family design.
- Fully unsupervised rows only.
- Reference-assisted mixture diagnostics.
- Robustness suite.
- State-NLL versus calibration tradeoff.
- Decision: objective bottleneck, family bottleneck, coupled bottleneck, or neither.

### Acceptance criteria

- The report can be regenerated from CSV artifacts.
- It includes a decision matrix with the observed outcome.
- It does not require manually opening individual run directories.

## Minimum viable pilot

If implementation time is constrained, implement only:

```text
structured_joint_iwae_h4_k16
structured_joint_renyi_h4_alpha_0p5
direct_mixture_k2_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
direct_mixture_k2_joint_iwae_h4_k16
```

Then run:

```bash
make sweep-nonlinear-learned \
  NONLINEAR_LEARNED_CONFIGS=experiments/nonlinear/03_weak_sine_observation.yaml,experiments/nonlinear/04_intermittent_sine_observation.yaml \
  NONLINEAR_LEARNED_MODELS=structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4,structured_joint_iwae_h4_k16,structured_joint_renyi_h4_alpha_0p5,direct_mixture_k2_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4,direct_mixture_k2_joint_iwae_h4_k16 \
  NONLINEAR_LEARNED_SEEDS=321,322,323 \
  NONLINEAR_LEARNED_STEPS=1000 \
  NONLINEAR_LEARNED_DIR=outputs/nonlinear_divergence_family_minimum_pilot_1000
```

This is enough to answer the core question:

> Does the next improvement come from changing the objective/divergence, changing the posterior family, or combining both?

## Stop conditions

Stop or debug before expanding if any of these happen:

- K=1 mixture does not match the Gaussian baseline.
- IWAE `k1` does not match the windowed ELBO convention.
- Mixture log probabilities produce NaNs or extreme underflow.
- Predictive-y scoring accidentally uses post-assimilation beliefs.
- Masked-y steps call the learned update with hidden measurements.
- Coverage improves only through severe state-NLL degradation.
- All rows regress against the current promoted objective on weak and intermittent.

## Decision matrix

| Outcome | Interpretation | Next action |
|---|---|---|
| Gaussian IWAE/Rényi wins | objective/divergence bottleneck | expand divergence robustness; defer mixtures |
| mixture ELBO wins | posterior family bottleneck | run K=2 robustness; optionally K=3 |
| mixture IWAE/Rényi wins | coupled objective/family bottleneck | promote mixture + divergence branch |
| reference mixture wins, unsupervised mixture fails | training objective bottleneck | focus on divergence/optimization |
| no new row improves | amortization/update/optimization bottleneck | inspect rollout stability and model parameterization |
| entropy rows only inflate variance | superficial calibration | reject entropy settings or reduce weights |
| clean sinusoidal regresses badly | poor robustness | do not promote without weighting or objective fix |

## Recommended final claim formats

### If Gaussian divergence succeeds

> The remaining nonlinear strict-filter gap is primarily an objective/divergence issue. A stricter posterior family is not required for the next improvement: a Gaussian filter trained with a less mode-seeking multi-sample or alpha-Rényi objective improves calibration while preserving the strict online belief.

### If mixture succeeds under current objective

> The current nonlinear benchmark exposes a posterior-family bottleneck. A minimal K=2 strict mixture posterior improves the fully unsupervised filter while preserving the explicit filtering marginal.

### If mixture plus divergence succeeds

> The nonlinear failure is coupled: mixture expressivity helps only when paired with a less mode-seeking objective. The next baseline should be a strict K=2 mixture filter trained with the selected IWAE or alpha-Rényi objective.

### If no branch succeeds

> The remaining failure is not explained by edge locality, causal predictive-y supervision, standard multi-sample bounds, alpha-Rényi variants, or a minimal mixture family. The next branch should target update parameterization, rollout stability, or optimization rather than further local ELBO tuning.

## Suggested implementation order

1. T0 — freeze current baseline.
2. T1 — add objective/family config plumbing.
3. T2 — implement Gaussian IWAE and run one-seed smoke test.
4. T3 — implement alpha-Rényi and run one-seed smoke test.
5. T5 — implement K=1/K=2 mixture belief objects.
6. T6/T7 — connect mixture heads, predictive-y, and masked-y paths.
7. T10 — pass unit and identity checks.
8. T11 — run Gaussian divergence pilot.
9. T12 — run mixture family pilot.
10. T13 — run full 2×2 pilot.
11. T14 — run reference mixture diagnostic.
12. T15/T16 — robustness and final report.
