# Nonlinear Divergence / Posterior-Family Branch: Master Research Task

Prepared: 2026-04-28

## Executive recommendation

The next nonlinear research branch should **not** simply make the variational family larger. The current evidence supports a more controlled diagnosis:

> Test whether the remaining nonlinear strict-filter failure is caused by the variational **objective/divergence**, the posterior **family**, or the interaction between the two.

The right next milestone is a compact 2×2 branch:

| Posterior family | Objective / divergence | Purpose |
|---|---|---|
| strict Gaussian | current promoted unsupervised objective | current best fully unsupervised baseline |
| strict Gaussian | IWAE / alpha-Rényi / entropy-regularized objective | tests whether the objective is the bottleneck |
| small strict Gaussian mixture | current promoted unsupervised objective | tests whether the posterior family is the bottleneck |
| small strict Gaussian mixture | IWAE / alpha-Rényi / entropy-regularized objective | tests whether objective and family must change together |

This should be staged as a research diagnosis, not as a broad architecture expansion. The next branch should stay inside the strict online filtering contract:

```text
q^F_t = update(q^F_{t-1}, x_t, y_t)
```

and should keep all headline rows fully unsupervised: no grid moments, true latent states, oracle edge posteriors, or reference variance targets during training.

## Background

The nonlinear benchmark is the scalar random-walk latent model with nonlinear sine observation:

```text
z_t = z_{t-1} + w_t
w_t ~ Normal(0, Q)

y_t = x_t sin(z_t) + v_t
v_t ~ Normal(0, R)
```

The learned strict filter carries an explicit filtering marginal plus an edge/backward conditional:

```text
q^E_t(z_t, z_{t-1}) = q^F_t(z_t) q^B_t(z_{t-1} | z_t)
```

The recent unsupervised objective-repair branch promoted:

```text
structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
```

That row combined:

```text
short-window joint ELBO pressure
+ causal predictive-y scoring
+ masked-y span training
```

It was a useful partial success: it improved degraded-observation robustness relative to vanilla structured ELBO, especially on weak, intermittent, zero, and random-normal observation stressors. But it did not solve the problem. Absolute calibration remained poor, and the best fully unsupervised variance ratios stayed far below the original aspirational `> 0.50` gate.

The crucial interpretation is:

> The local ELBO repair helped enough to justify continuing, but not enough to keep doing local ELBO tuning. The next branch should explicitly compare objective/divergence changes against posterior-family changes.

## Current diagnosis

The current evidence points to three facts that must be respected at the same time.

### 1. The current unsupervised Gaussian is still badly under-dispersed

The promoted unsupervised row improved several stressors, but its variance ratios remained very small. This indicates that the filter is still overconfident and does not yet behave like the grid reference under nonlinear ambiguity.

### 2. A strict Gaussian target is not obviously doomed

Earlier reference-shape diagnostics showed that a moment-matched Gaussian projection of the grid posterior was already much closer to the grid reference than the learned Gaussian. That means the learned unsupervised filter is not reliably reaching the best strict Gaussian target.

This argues against immediately treating mixtures as the only answer.

### 3. The remaining failure may still require a richer family

Even though the moment Gaussian diagnostic cautions against premature mixtures, the sine observation can create aliases and multiple plausible modes. A small mixture family is now worth testing because the current objective-repair row has already reduced some basic rollout/objective failures.

The right conclusion is therefore:

```text
Do not jump directly to a large expressive posterior.
Do test a minimal mixture family against objective/divergence changes.
```

## Research question

Can a fully unsupervised strict learned nonlinear filter close a larger fraction of the gap to the grid/reference-distilled diagnostics by changing:

1. the variational divergence/objective;
2. the posterior family; or
3. both together?

The target claim, if successful, is:

> The nonlinear strict-filter failure is reduced by moving beyond the current exclusive-KL-style ELBO and/or beyond a single Gaussian posterior, while preserving the explicit online filtering marginal.

## Hypotheses

### H1 — Objective/divergence bottleneck

The learned Gaussian family is adequate enough, but the current ELBO-style training signal is too mode-seeking or too locally myopic.

Expected result:

```text
Gaussian + IWAE / alpha-Rényi / entropy objective improves coverage and variance ratio.
Mixture + current objective does not improve much.
```

### H2 — Posterior-family bottleneck

The current objective is adequate enough, but a single Gaussian cannot represent the effective posterior under sine-observation ambiguity.

Expected result:

```text
Mixture + current objective improves calibration and state NLL.
Gaussian + new divergence does not improve much.
```

### H3 — Coupled bottleneck

A mixture family helps only when trained with a less mode-seeking objective.

Expected result:

```text
Only mixture + IWAE / alpha-Rényi / entropy objective clearly improves.
```

### H4 — Amortization / rollout bottleneck

Neither a new divergence nor a small mixture helps. The issue is likely update parameterization, optimization, or self-fed compounding error.

Expected result:

```text
All fully unsupervised variants remain severely under-dispersed.
```

## Non-goals

This branch should not:

- introduce Mamba, GRU, LSTM, Transformer, or hidden-context state as the next primary move;
- use grid-reference moments or true latent states in headline training rows;
- add another simple reference-variance calibration penalty;
- run a large mixture/flow sweep before the K=2 diagnostic is understood;
- report smoothing-style full-sequence objectives as if they were strict online filters;
- promote rows that improve coverage only by inflating variance while worsening state NLL.

## Guardrails

### Training-signal labels

Every row must be labeled as one of:

| Label | Meaning |
|---|---|
| `unsupervised` | Uses only `x`, `y`, known transition, known observation model, and prior |
| `reference_distilled` | Uses grid/reference moments, densities, rollout targets, or posterior shape targets |
| `oracle_calibrated` | Uses reference variance ratios, oracle posteriors, true states, or diagnostic calibration targets |

### Strict filtering contract

The exported belief must remain:

```text
q^F_t(z_t)
```

and downstream filtering metrics must be computed from this belief only.

### Family escalation limit

The first expressive-family diagnostic should be:

```text
K = 2 Gaussian mixture
```

with optional `K = 3` only after K=2 is working and numerically stable.

Do not start with normalizing flows, particles, or large mixtures.

## Proposed posterior families

### Baseline: strict Gaussian edge family

Current family:

```text
q^F_t(z_t) = Normal(mu_t, sigma_t^2)

q^B_t(z_{t-1} | z_t) = Normal(a_t z_t + b_t, tau_t^2)

q^E_t(z_t, z_{t-1}) = q^F_t(z_t) q^B_t(z_{t-1} | z_t)
```

This remains the baseline for the divergence/objective branch.

### New family: strict mixture edge family

Use a small mixture whose filtering marginal remains explicit:

```text
q^E_t(z_t, z_{t-1})
=
sum_k pi_{t,k}
  Normal(z_t ; mu_{t,k}, sigma^2_{t,k})
  Normal(z_{t-1} ; a_{t,k} z_t + b_{t,k}, tau^2_{t,k})
```

The carried filtering marginal is analytically available:

```text
q^F_t(z_t)
=
sum_k pi_{t,k} Normal(z_t ; mu_{t,k}, sigma^2_{t,k})
```

This preserves the project’s edge factorization and strict-filter semantics while permitting multimodal filtering beliefs.

### Mixture implementation requirements

The mixture family must support:

- stable `log_prob` via log-sum-exp;
- reparameterized or score-function-safe sampling for ELBO objectives;
- exact or Monte Carlo filtering marginal metrics;
- predictive-y scoring from mixture transition predictions;
- masked-y transition propagation;
- saved diagnostics compatible with `plot_nonlinear.py` and aggregation scripts.

## Proposed objective / divergence variants

### Current promoted objective baseline

Use the current best fully unsupervised row as the baseline:

```text
structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
```

This is the strongest existing unsupervised robustness baseline and should stay in all comparisons.

### Multi-sample IWAE-style branch

Add a multi-sample bound over the existing edge/window trajectory objective.

Candidate keys:

```text
structured_joint_iwae_h4_k8
structured_joint_iwae_h4_k16
structured_joint_iwae_h4_k32
direct_joint_iwae_h4_k16
```

Start with `k16` as the main diagnostic. Use `k8` for cheap smoke tests and `k32` only if `k16` is promising.

### Alpha-Rényi branch

Add a compact alpha sweep:

```text
structured_joint_renyi_h4_alpha_0p3
structured_joint_renyi_h4_alpha_0p5
structured_joint_renyi_h4_alpha_0p7
structured_joint_renyi_h4_alpha_0p9
```

The alpha values should test whether a less mode-seeking training criterion improves coverage and variance ratio without uncontrolled variance inflation.

### Entropy / calibration branch

Optionally add a fully unsupervised entropy regularizer. It must not use reference variance. It should be interpreted as a prior over posterior entropy, not as a grid-target calibration penalty.

Candidate keys:

```text
structured_joint_entropy_h4_beta_0p001
structured_joint_entropy_h4_beta_0p003
structured_joint_entropy_h4_beta_0p01
```

This is lower priority than IWAE and alpha-Rényi.

## Primary 2×2 pilot design

Use weak and intermittent first:

```text
experiments/nonlinear/03_weak_sine_observation.yaml
experiments/nonlinear/04_intermittent_sine_observation.yaml
```

Use seeds:

```text
321,322,323
```

Use training budget:

```text
1000 steps
```

Recommended pilot rows:

| Row | Purpose |
|---|---|
| `structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4` | current best unsupervised baseline |
| `structured_joint_iwae_h4_k16` | Gaussian + new divergence |
| `structured_joint_renyi_h4_alpha_0p5` | Gaussian + alternative divergence |
| `direct_mixture_k2_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4` | mixture + current objective |
| `direct_mixture_k2_joint_iwae_h4_k16` | mixture + new divergence |
| `structured_mixture_k2_joint_iwae_h4_k16` | structured mixture + new divergence, if implementation is stable |

## Success gates

The first pilot should not require the old aspirational gate of variance ratio `> 0.50` immediately. That gate is still the longer-term target, but the latest fully unsupervised rows are so far below it that the next pilot should use a staged gate.

### First gate: visible movement

A row is worth continuing if it achieves, on weak or intermittent:

| Metric | Gate |
|---|---:|
| variance ratio | `> 0.20` |
| coverage 90 | at least `+0.10` over current promoted unsupervised row |
| state NLL | no material regression versus current promoted unsupervised row |
| predictive NLL | stable or improved |

### Second gate: robustness expansion

Only expand to the full stressor set if the first gate passes.

Full stressor set:

```text
experiments/nonlinear/01_sine_observation.yaml
experiments/nonlinear/03_weak_sine_observation.yaml
experiments/nonlinear/04_intermittent_sine_observation.yaml
experiments/nonlinear/05_zero_sine_observation.yaml
experiments/nonlinear/06_random_normal_sine_observation.yaml
```

### Promotion gate

A row can become the next baseline only if it:

- improves weak/intermittent calibration;
- does not catastrophically regress clean sinusoidal behavior;
- improves or preserves zero-observation uncertainty propagation;
- beats the current promoted unsupervised row on at least two of state NLL, coverage, variance ratio, and predictive NLL;
- remains fully unsupervised.

## Reference-assisted diagnostic

Before trusting unsupervised mixture training, add a reference-assisted density projection diagnostic:

```text
direct_mixture_k2_reference_grid_distilled
direct_mixture_k3_reference_grid_distilled
```

This diagnostic should fit mixture density to the grid posterior by minimizing grid cross-entropy:

```text
minimize  - integral p_grid(z_t | D_t) log q_mix(z_t) dz_t
```

Interpretation:

| Diagnostic outcome | Meaning |
|---|---|
| reference mixture does not beat reference Gaussian | mixture family is probably not buying much |
| reference mixture beats Gaussian but unsupervised mixture fails | objective/divergence remains the bottleneck |
| reference mixture beats Gaussian and unsupervised mixture improves | expressive family is worth promoting |
| reference mixture is unstable | implementation or parameterization issue before research claim |

This diagnostic must be reported separately as `reference_distilled`.

## Recommended staging

### Stage 0 — freeze current baseline

Preserve the current final report and T11 robustness status. Keep the promoted objective as the fully unsupervised baseline for all future comparisons.

### Stage 1 — Gaussian divergence branch

Implement IWAE and alpha-Rényi variants for the current Gaussian strict-filter family. This is the cheapest way to test whether the objective is the bottleneck.

### Stage 2 — minimal mixture family

Implement K=2 strict mixture belief and edge posterior. Do not add K>3 until K=2 is stable.

### Stage 3 — reference mixture diagnostic

Fit mixture densities to grid reference posteriors to estimate the value of the family independent of unsupervised training.

### Stage 4 — 2×2 weak/intermittent pilot

Run the compact pilot with current baseline, Gaussian new objectives, mixture current objective, and mixture new objective.

### Stage 5 — robustness and reporting

Only run the full stressor suite after the pilot shows visible movement.

## Main files

Expected modified files:

```text
src/vbf/distributions.py
src/vbf/losses.py
src/vbf/nonlinear.py
src/vbf/predictive.py
src/vbf/models/heads.py
src/vbf/models/cells.py
scripts/train_nonlinear.py
scripts/sweep_nonlinear_learned.py
scripts/plot_nonlinear.py
scripts/plot_nonlinear_sweep.py
scripts/aggregate_nonlinear_unsupervised_objective_report.py
scripts/diagnose_nonlinear_mixture_projection.py
```

Expected new committed documents:

```text
docs/results/nonlinear_divergence_family_master_research_task_2026-04-28.md
docs/results/nonlinear_divergence_family_experiment_task_pack_2026-04-28.md
```

Expected generated artifacts:

```text
outputs/nonlinear_divergence_family_pilot_1000/metrics.csv
outputs/nonlinear_divergence_family_pilot_1000/summary.md
outputs/nonlinear_divergence_family_pilot_1000/plots/sweep_comparison.png
outputs/nonlinear_mixture_reference_projection/summary.md
outputs/nonlinear_divergence_family_final_report/summary.md
outputs/nonlinear_divergence_family_final_report/summary.json
outputs/nonlinear_divergence_family_final_report/summary.csv
```

## Decision matrix

| Pilot result | Interpretation | Next action |
|---|---|---|
| Gaussian IWAE/Rényi improves | objective/divergence bottleneck | continue divergence branch; do not prioritize mixtures |
| mixture ELBO improves | posterior family bottleneck | expand K=2/K=3 mixture robustness |
| only mixture IWAE/Rényi improves | objective and family coupled | promote mixture + divergence branch |
| reference mixture wins but unsupervised mixture fails | training objective still bottleneck | focus on divergence/optimization |
| no row improves | likely amortization/update/optimization issue | inspect rollout stability and cell parameterization |
| coverage improves but state NLL worsens | variance inflation | reject or downweight entropy/divergence setting |
| clean sinusoidal collapses | overfit to degraded observations | require objective weighting or robustness fix before promotion |

## Bottom line

A more expressive variational family is now worth testing, but only as part of a controlled diagnosis. The next master research task is:

> Run a compact divergence-versus-family experiment that compares the current strict Gaussian objective baseline, alternative unsupervised divergences, a minimal K=2 strict mixture posterior, and the combination of mixture plus new divergence.

This gives a clean answer to the research question instead of making the model larger and hoping the failure disappears.
