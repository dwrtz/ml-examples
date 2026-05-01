# Step 0 And Step 1 K2/K3 Results

Date: 2026-05-01

This note summarizes the GPU Lambda runs used to close Step 0 and Step 1 from
`notes.md`.

## Source Runs

| Run | Output directory | Rows |
|---|---|---:|
| Step 0 family battery | `outputs/cloud_downloads/k2_pareto_lock_family_1000_2026-05-01` | 75 |
| Step 0 stressors | `outputs/cloud_downloads/k2_pareto_lock_stressors_1000_2026-05-01` | 36 |
| Step 1 family battery | `outputs/cloud_downloads/step1_exchangeable_k2_k3_family_1000_2026-05-01` | 75 |
| Step 1 stressors | `outputs/cloud_downloads/step1_exchangeable_k2_k3_stressors_1000_2026-05-01` | 36 |

All runs used 1000 training steps and seeds `321,322,323`. Step 0 used K2
survivors plus the previous structured baseline. Step 1 compared direct K2,
direct K3, explicit exchangeable K2/K3, and Power-EP.

## Executive Takeaway

Promote `direct_mixture_k3_joint_iwae_h4_k32` as the state-density baseline.
Keep `direct_mixture_k2_power_ep_alpha_0p5` as the predictive-y / coverage
comparator. Do not promote the explicit exchangeable component-wise cell yet.

The direct K3 row improved state NLL over direct K2 on both the active family
battery and the harder stressors. Power-EP remains the best predictive-y and
coverage row, but it is not a safe state-density default because it regresses
badly on the random-normal stressor. The exchangeable cell improved component
usage mechanically, but the state-density cost was too large.

## Step 0: K2 Pareto Lock

Step 0 tested the plausible K2 family rows on the active family battery, then
reran the survivors on the stressors.

### Family Battery

| Model | n | State NLL | Pred-y NLL | Coverage 90 |
|---|---:|---:|---:|---:|
| K2 IWAE + pre-update predictive scoring | 15 | 4.614 | 0.971 | 0.599 |
| K2 IWAE | 15 | 4.869 | 1.004 | 0.584 |
| K2 IWAE + local ADF projection w0.3 | 15 | 5.419 | 0.927 | 0.587 |
| K2 Power-EP alpha 0.5 | 15 | 6.764 | 0.841 | 0.640 |
| Structured joint ELBO + predictive-y + masked-y spans | 15 | 4159.987 | 3217.708 | 0.479 |

The previous structured row failed catastrophically, driven by the cubic family,
and should no longer be treated as a promoted nonlinear baseline.

### Stressors

| Model | n | State NLL | Pred-y NLL | Coverage 90 |
|---|---:|---:|---:|---:|
| K2 IWAE | 12 | 6.005 | 0.428 | 0.445 |
| K2 IWAE + pre-update predictive scoring | 12 | 6.049 | 0.427 | 0.444 |
| K2 Power-EP alpha 0.5 | 12 | 8.366 | 0.394 | 0.670 |

Step 0 conclusion: plain K2 IWAE is the state-density lock. The
predictive-consistent K2 row does not buy enough over plain K2 once stressors
are included. Power-EP is retained as a calibration comparator, not as the state
baseline.

## Step 1: Exchangeable K2/K3 Strict Mixtures

Step 1 tested whether K3 and explicit component exchangeability improve the Step
0 K2 baseline. The implementation added an explicit component-wise shared cell
via `mixture_cell: component`, plus these model keys:

```text
direct_exchangeable_mixture_k2_joint_iwae_h4_k32
direct_exchangeable_mixture_k3_joint_iwae_h4_k32
```

### Family Battery

| Model | n | State NLL | Pred-y NLL | Coverage 90 |
|---|---:|---:|---:|---:|
| K3 IWAE | 15 | 4.525 | 0.931 | 0.603 |
| K2 IWAE | 15 | 4.859 | 1.008 | 0.584 |
| K2 Power-EP alpha 0.5 | 15 | 6.764 | 0.841 | 0.640 |
| Exchangeable K2 IWAE | 15 | 8.002 | 1.064 | 0.593 |
| Exchangeable K3 IWAE | 15 | 8.146 | 1.011 | 0.592 |

### Stressors

| Model | n | State NLL | Pred-y NLL | Coverage 90 |
|---|---:|---:|---:|---:|
| K3 IWAE | 12 | 5.965 | 0.423 | 0.440 |
| K2 IWAE | 12 | 6.370 | 0.428 | 0.428 |
| K2 Power-EP alpha 0.5 | 12 | 8.366 | 0.394 | 0.670 |

K3 is a real state-density improvement over K2. It improves the family mean and
the stressor mean, while staying close on predictive-y. Power-EP still owns
predictive-y and coverage, but its state-density failure on random-normal
stressors prevents promotion as the main baseline.

### Component Usage

Mean effective component count from `learned_filter_weights`:

| Model | Effective K | Entropy | Mean max weight |
|---|---:|---:|---:|
| Exchangeable K3 IWAE | 2.507 | 0.916 | 0.514 |
| K2 Power-EP alpha 0.5 | 2.000 | 0.693 | 0.500 |
| Exchangeable K2 IWAE | 1.778 | 0.598 | 0.644 |
| K3 IWAE | 1.614 | 0.491 | 0.786 |
| K2 IWAE | 1.352 | 0.313 | 0.845 |

The exchangeable cell clearly increases component use, especially for K3, but
that did not translate into better state NLL. The current exchangeable
parameterization should be treated as a diagnostic result, not a promoted model.

## Decisions

| Role | Model |
|---|---|
| Main state-density baseline | `direct_mixture_k3_joint_iwae_h4_k32` |
| Previous state baseline retained for comparison | `direct_mixture_k2_joint_iwae_h4_k32` |
| Predictive-y / coverage comparator | `direct_mixture_k2_power_ep_alpha_0p5` |
| Demoted | structured joint ELBO + predictive-y + masked-y spans |
| Not promoted | K2 predictive-consistent IWAE |
| Diagnostic only | explicit exchangeable K2/K3 IWAE |

## Next Experiment

Proceed to Step 2: predictive-consistent objectives, but center the branch on
K3 IWAE with Power-EP guardrails.

Recommended rows:

```text
direct_mixture_k3_joint_iwae_h4_k32
direct_mixture_k2_joint_iwae_h4_k32
direct_mixture_k2_power_ep_alpha_0p5
new: direct_mixture_k3_detached_predictive_consistent_iwae_h4_k32
new: direct_mixture_k3_two_phase_iwae_then_predictive_h4_k32
```

Success criteria:

```text
state NLL stays close to or better than K3 IWAE
predictive-y NLL moves toward Power-EP
coverage improves without a random-normal state-density regression
heteroskedastic and cubic remain stable
```
