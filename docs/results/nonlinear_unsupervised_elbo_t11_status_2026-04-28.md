# Nonlinear Unsupervised ELBO T11 Robustness Status

Prepared: 2026-04-28

## Summary

T11 is complete for the selected fully unsupervised objective:

```text
structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
```

The row combines short-window joint ELBO pressure, causal predictive-y scoring, and masked-y span
training. It was compared against `structured_elbo`, `direct_elbo`, `direct_moment_distilled`, and
`structured_moment_rollout_h4` across the five nonlinear observation stressors with seeds
`321,322,323` and 1000 training steps.

Artifacts:

```text
outputs/nonlinear_unsupervised_objective_robustness_full_1000/metrics.csv
outputs/nonlinear_unsupervised_objective_robustness_full_1000/summary.md
outputs/nonlinear_unsupervised_objective_robustness_full_1000/plots/summary.md
outputs/nonlinear_unsupervised_objective_robustness_full_1000/plots/sweep_comparison.png
```

## Fully Unsupervised Result

The promoted objective is a robustness improvement under degraded or non-informative observations,
but it is not a clean-condition Pareto improvement.

| Condition | Structured ELBO NLL | Candidate NLL | Structured ELBO cov 90 | Candidate cov 90 | Structured ELBO var ratio | Candidate var ratio |
|---|---:|---:|---:|---:|---:|---:|
| sinusoidal | 52.989 | 54.930 | 0.347 | 0.342 | 0.041 | 0.083 |
| weak sinusoidal | 20.865 | 14.672 | 0.332 | 0.396 | 0.058 | 0.090 |
| intermittent sinusoidal | 37.853 | 22.992 | 0.327 | 0.371 | 0.038 | 0.060 |
| zero | 13.474 | 8.414 | 0.282 | 0.388 | 0.056 | 0.107 |
| random normal | 113.958 | 60.109 | 0.315 | 0.358 | 0.014 | 0.040 |

Interpretation:

- The candidate materially improves state NLL on weak, intermittent, zero, and random-normal
  observations.
- Coverage and variance ratio also improve on those same stressors, so the gain is not only a
  point-estimate effect.
- The clean sinusoidal condition is a caveat: state NLL and coverage are slightly worse than
  vanilla structured ELBO, although variance ratio improves.
- Absolute calibration is still weak. The best fully unsupervised variance ratios remain below
  `0.11`, far below the original aspirational `> 0.50` gate.

## Diagnostic Rows

The reference-distilled rows are controls, not fair unsupervised baselines.

`direct_moment_distilled` remains the strongest diagnostic in state NLL and coverage across most
conditions, with state NLL near `2.77` and coverage near `0.84`.

`structured_moment_rollout_h4` confirms that structured rollout training can be excellent in weak
and zero observation settings, but it is unstable on random-normal and clean sinusoidal stressors.
Because it uses reference targets, it should be reported as an upper-bound/diagnostic row only.

## Gate Assessment

T11 supports a partial-success claim:

> A combined trajectory-consistent, predictive-y, and masked-y unsupervised objective reduces the
> nonlinear strict-filter failure under weak, intermittent, and non-informative observations.

It does not support the stronger claim that the current ELBO branch solves nonlinear strict
Gaussian filtering. The remaining gap to reference-distilled diagnostics is large, and absolute
coverage/variance calibration is still poor.

## Recommended Next Steps

1. Produce the T12 aggregation report from the CSV artifacts, using this T11 run as the robustness
   input.
2. Add a concise final-claim section that separates:
   - fully unsupervised objective repair;
   - reference-distilled positive controls;
   - remaining calibration gap.
3. Treat the next research step as an objective/divergence question rather than another local ELBO
   tuning pass. Candidate directions are IWAE/multi-sample bounds, alpha/Renyi objectives, or
   entropy/calibration terms that can be justified without reference moments.
4. Keep the current promoted objective as the best fully unsupervised robustness baseline for future
   comparisons.
