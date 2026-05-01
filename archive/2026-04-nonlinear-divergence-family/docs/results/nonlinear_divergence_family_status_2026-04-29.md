# Nonlinear Divergence / Posterior-Family Status

Prepared: 2026-04-29

## Summary

The divergence/family branch now has runnable fully unsupervised rows for:

```text
Gaussian + promoted ELBO repair baseline
Gaussian + IWAE / Renyi
K=2 Gaussian mixture + promoted objective
K=2 Gaussian mixture + IWAE
```

The main result is clear:

> `direct_mixture_k2_joint_iwae_h4_k32` is the strongest state-density and calibration row found so
> far, but it is not yet promotable because predictive-y NLL regresses on nonzero-observation
> stressors.

All headline rows in this report are fully unsupervised. They use only `x`, `y`, the known
transition, the known nonlinear observation model, and the prior. Reference/grid moments and true
states are used only for evaluation metrics.

## Artifacts

```text
outputs/nonlinear_divergence_family_pilot_1000/metrics.csv
outputs/nonlinear_divergence_family_pilot_1000/summary.md
outputs/nonlinear_divergence_family_pilot_1000/plots/sweep_comparison.png

outputs/nonlinear_direct_mixture_iwae_followup_1000/metrics.csv
outputs/nonlinear_direct_mixture_iwae_followup_1000/summary.md
outputs/nonlinear_direct_mixture_iwae_followup_1000/plots/sweep_comparison.png

outputs/nonlinear_direct_mixture_k32_robustness_1000/metrics.csv
outputs/nonlinear_direct_mixture_k32_robustness_1000/summary.md
outputs/nonlinear_direct_mixture_k32_robustness_1000/plots/sweep_comparison.png

outputs/nonlinear_direct_mixture_k32_predy_sweep_1000/metrics.csv
outputs/nonlinear_direct_mixture_k32_predy_sweep_1000/summary.md
outputs/nonlinear_direct_mixture_k32_predy_sweep_1000/plots/sweep_comparison.png

outputs/nonlinear_direct_mixture_k32_predy_schedule_1000/metrics.csv
outputs/nonlinear_direct_mixture_k32_predy_schedule_1000/summary.md
outputs/nonlinear_direct_mixture_k32_predy_schedule_1000/plots/sweep_comparison.png
```

## T13 2x2 Pilot

The 2x2 pilot used weak and intermittent sine observations, seeds `321,322,323`, and 1000 training
steps.

| Pattern | Row | state NLL | coverage 90 | variance ratio | pred-y NLL |
|---|---|---:|---:|---:|---:|
| weak | promoted Gaussian baseline | 14.672 | 0.396 | 0.090 | 0.322 |
| weak | Gaussian IWAE h4 k16 | 16.674 | 0.387 | 0.072 | 0.309 |
| weak | Gaussian Renyi h4 alpha 0.5 | 18.120 | 0.351 | 0.076 | 0.313 |
| weak | direct K2 mixture + promoted objective | 11.487 | 0.347 | 0.054 | 0.319 |
| weak | structured K2 mixture + promoted objective | 15.173 | 0.376 | 0.081 | 0.323 |
| weak | direct K2 mixture IWAE h4 k16 | 4.448 | 0.487 | 0.206 | 0.334 |
| weak | structured K2 mixture IWAE h4 k16 | 21.803 | 0.336 | 0.050 | 0.312 |
| intermittent | promoted Gaussian baseline | 22.992 | 0.371 | 0.060 | 0.374 |
| intermittent | Gaussian IWAE h4 k16 | 36.261 | 0.367 | 0.058 | 0.359 |
| intermittent | Gaussian Renyi h4 alpha 0.5 | 32.587 | 0.353 | 0.035 | 0.356 |
| intermittent | direct K2 mixture + promoted objective | 18.018 | 0.347 | 0.042 | 0.368 |
| intermittent | structured K2 mixture + promoted objective | 20.539 | 0.397 | 0.070 | 0.370 |
| intermittent | direct K2 mixture IWAE h4 k16 | 4.414 | 0.497 | 0.247 | 0.388 |
| intermittent | structured K2 mixture IWAE h4 k16 | 38.181 | 0.311 | 0.036 | 0.365 |

Interpretation:

- Changing only the Gaussian objective did not solve the calibration failure.
- Changing only the family under the promoted objective helped state NLL somewhat, but did not
  improve coverage or variance ratio reliably.
- Direct K2 mixture + IWAE was the only row that strongly improved state NLL, coverage, and
  variance ratio on both pilot patterns.
- Structured K2 mixture + IWAE was unstable and should not be promoted in its current
  parameterization.

This supports a coupled-bottleneck diagnosis: mixture expressivity helps when paired with a
multi-sample objective.

## IWAE K Sweep

The follow-up compared direct K2 mixture IWAE with `k8`, `k16`, and `k32`, plus a fixed predictive-y
auxiliary.

| Pattern | Row | state NLL | coverage 90 | variance ratio | pred-y NLL |
|---|---|---:|---:|---:|---:|
| weak | K2 IWAE h4 k8 | 4.571 | 0.484 | 0.194 | 0.333 |
| weak | K2 IWAE h4 k16 | 4.448 | 0.487 | 0.206 | 0.334 |
| weak | K2 IWAE h4 k32 | 4.220 | 0.512 | 0.232 | 0.334 |
| weak | K2 IWAE h4 k16 + pred-y w1 | 4.788 | 0.466 | 0.184 | 0.333 |
| intermittent | K2 IWAE h4 k8 | 4.700 | 0.472 | 0.252 | 0.389 |
| intermittent | K2 IWAE h4 k16 | 4.414 | 0.497 | 0.247 | 0.388 |
| intermittent | K2 IWAE h4 k32 | 4.335 | 0.501 | 0.254 | 0.389 |
| intermittent | K2 IWAE h4 k16 + pred-y w1 | 4.597 | 0.484 | 0.238 | 0.388 |

Interpretation:

- `k32` is the strongest direct K2 IWAE setting on state NLL, coverage, and variance ratio.
- A fixed predictive-y auxiliary did not repair predictive-y NLL and gave back some calibration.

## Robustness Sweep

The robustness sweep compared the promoted Gaussian baseline against `direct_mixture_k2_joint_iwae_h4_k32`
on the full five-stressor set, seeds `321,322,323`, and 1000 training steps.

| Pattern | Row | state NLL | coverage 90 | variance ratio | pred-y NLL |
|---|---|---:|---:|---:|---:|
| sinusoidal | promoted Gaussian baseline | 54.930 | 0.342 | 0.083 | 0.571 |
| sinusoidal | K2 IWAE h4 k32 | 6.402 | 0.416 | 0.125 | 0.681 |
| weak | promoted Gaussian baseline | 14.672 | 0.396 | 0.090 | 0.322 |
| weak | K2 IWAE h4 k32 | 4.220 | 0.512 | 0.232 | 0.334 |
| intermittent | promoted Gaussian baseline | 22.992 | 0.371 | 0.060 | 0.374 |
| intermittent | K2 IWAE h4 k32 | 4.335 | 0.501 | 0.254 | 0.389 |
| zero | promoted Gaussian baseline | 8.414 | 0.388 | 0.107 | 0.263 |
| zero | K2 IWAE h4 k32 | 5.105 | 0.387 | 0.111 | 0.263 |
| random-normal | promoted Gaussian baseline | 60.109 | 0.358 | 0.040 | 0.614 |
| random-normal | K2 IWAE h4 k32 | 8.452 | 0.377 | 0.087 | 0.726 |

Interpretation:

- K2 IWAE h4 k32 wins state NLL on all five stressors.
- It improves coverage and variance ratio on four of five stressors.
- Zero-observation is mostly neutral on coverage and variance ratio but still improves state NLL.
- Predictive-y NLL regresses on all nonzero-observation stressors.

This row clears the old aspirational variance-ratio gate only on weak and intermittent stressors,
not on all robustness conditions.

## Predictive-y Follow-ups

Two predictive-y repair attempts were run:

1. fixed auxiliary weights `0.05`, `0.1`, and `0.3`;
2. late-start scheduled auxiliaries with start fraction `0.5` and ramp fraction `0.25`.

| Pattern | Row | pred-y NLL | state NLL | coverage 90 | variance ratio |
|---|---|---:|---:|---:|---:|
| sine | promoted Gaussian baseline | 0.571 | 54.930 | 0.342 | 0.083 |
| sine | K2 IWAE h4 k32 | 0.681 | 6.402 | 0.416 | 0.125 |
| sine | K2 IWAE h4 k32 + fixed pred-y w0.05 | 0.676 | 7.101 | 0.379 | 0.121 |
| sine | K2 IWAE h4 k32 + late pred-y w1 | 0.652 | 7.665 | 0.359 | 0.108 |
| intermittent | promoted Gaussian baseline | 0.374 | 22.992 | 0.371 | 0.060 |
| intermittent | K2 IWAE h4 k32 | 0.389 | 4.335 | 0.501 | 0.254 |
| intermittent | K2 IWAE h4 k32 + late pred-y w1 | 0.389 | 4.449 | 0.492 | 0.247 |
| random-normal | promoted Gaussian baseline | 0.614 | 60.109 | 0.358 | 0.040 |
| random-normal | K2 IWAE h4 k32 | 0.726 | 8.452 | 0.377 | 0.087 |
| random-normal | K2 IWAE h4 k32 + fixed pred-y w0.3 | 0.705 | 8.483 | 0.371 | 0.078 |
| random-normal | K2 IWAE h4 k32 + late pred-y w1 | 0.687 | 8.967 | 0.363 | 0.067 |

Interpretation:

- Fixed predictive-y weights did not solve the predictive-y regression.
- Late predictive-y scheduling improves pred-y NLL more than fixed weights on sine and random-normal.
- The gap to the promoted baseline remains large.
- Scheduling weakens state NLL, coverage, and variance ratio relative to plain K2 IWAE h4 k32.

## Current Claim

Supported:

> The remaining nonlinear strict-filter failure is partly a coupled objective/family bottleneck.
> A direct K=2 Gaussian mixture trained with a windowed IWAE objective substantially improves
> fully unsupervised state-density calibration while preserving strict online filtering semantics.

Not supported:

> The current K2 mixture IWAE row is a fully solved nonlinear filter.

The blocker is predictive-y regression. The best state-density row improves the latent filtering
belief but is less good as a one-step observation predictive model on nonzero-observation stressors.

## Recommendation

Keep `direct_mixture_k2_joint_iwae_h4_k32` as the current best fully unsupervised state-density
candidate, but do not promote it as the final headline row.

The next research step should not be another scalar predictive-y weight sweep. The failed fixed and
scheduled auxiliaries suggest the problem is objective geometry, not simply weight magnitude.

Recommended next branch:

```text
Design a predictive-consistent mixture objective that scores one-step y predictions from the
mixture transition density without forcing the same objective term to compete directly with the
IWAE state-density signal throughout training.
```

Concrete options:

- use a two-head objective report: state-density candidate and predictive-y candidate are separate
  until a principled combined objective is found;
- add a held-out objective-selection rule that rejects rows whose predictive-y NLL regresses beyond
  a fixed tolerance;
- explore mixture predictive-y objectives based on transition-propagated component likelihoods,
  possibly with detached filtering parameters for the auxiliary term;
- only after this diagnostic, consider K=3 mixture or a richer update parameterization.

