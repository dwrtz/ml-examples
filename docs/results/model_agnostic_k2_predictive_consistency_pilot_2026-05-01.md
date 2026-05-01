# Model-Agnostic K2 Predictive-Consistency Pilot

Prepared: 2026-05-01

## Summary

This pilot tested the recommendation from `notes.md`: combine the state-density strength of K2 IWAE
with a more predictive-consistent projection/pre-update scoring signal.

Artifacts:

```text
outputs/model_agnostic_k2_predictive_consistency_pilot_1000/metrics.csv
outputs/model_agnostic_k2_predictive_consistency_pilot_1000/aggregate_by_observation_model.csv
outputs/model_agnostic_k2_predictive_consistency_pilot_1000/summary.md
```

Rows:

```text
structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
direct_mixture_k2_joint_iwae_h4_k32
direct_mixture_k2_power_ep_alpha_0p5
direct_mixture_k2_hybrid_iwae_projection_h4_k16_w01
direct_mixture_k2_hybrid_iwae_projection_h4_k16_w03
direct_mixture_k2_predictive_consistent_iwae_h4_k32
```

## Aggregate Results

| observation | row | state NLL | cov 90 | var ratio | pred-y NLL |
|---|---|---:|---:|---:|---:|
| `x_sine` | baseline | 54.930 | 0.342 | 0.083 | 0.571 |
| `x_sine` | K2 IWAE | 6.402 | 0.416 | 0.125 | 0.681 |
| `x_sine` | K2 Power-EP | 15.642 | 0.470 | 0.050 | 0.518 |
| `x_sine` | hybrid w0.1 | 7.033 | 0.401 | 0.088 | 0.642 |
| `x_sine` | hybrid w0.3 | 5.388 | 0.468 | 0.101 | 0.653 |
| `x_sine` | pre-update IWAE | 7.004 | 0.381 | 0.121 | 0.676 |
| `student_t` | baseline | 25.165 | 0.379 | 0.067 | 0.852 |
| `student_t` | K2 IWAE | 8.114 | 0.357 | 0.109 | 0.877 |
| `student_t` | K2 Power-EP | 8.666 | 0.578 | 0.110 | 0.844 |
| `student_t` | hybrid w0.1 | 8.599 | 0.364 | 0.050 | 0.830 |
| `student_t` | hybrid w0.3 | 8.680 | 0.361 | 0.042 | 0.826 |
| `student_t` | pre-update IWAE | 6.985 | 0.442 | 0.082 | 0.870 |
| `x_tanh` | baseline | 10.559 | 0.469 | 0.185 | 0.417 |
| `x_tanh` | K2 IWAE | 3.158 | 0.671 | 0.438 | 0.462 |
| `x_tanh` | K2 Power-EP | 3.198 | 0.680 | 0.337 | 0.407 |
| `x_tanh` | hybrid w0.1 | 3.428 | 0.662 | 0.359 | 0.431 |
| `x_tanh` | hybrid w0.3 | 3.368 | 0.671 | 0.360 | 0.427 |
| `x_tanh` | pre-update IWAE | 3.168 | 0.668 | 0.427 | 0.459 |
| `x_cubic` | baseline | 5680.423 | 0.716 | 2.369 | 665.041 |
| `x_cubic` | K2 IWAE | 2.016 | 0.878 | 4.270 | 2.137 |
| `x_cubic` | K2 Power-EP | 1.880 | 0.788 | 1.360 | 2.284 |
| `x_cubic` | hybrid w0.1 | 1.993 | 0.857 | 3.625 | 1.814 |
| `x_cubic` | hybrid w0.3 | 1.947 | 0.874 | 3.489 | 2.113 |
| `x_cubic` | pre-update IWAE | 1.999 | 0.882 | 4.275 | 2.147 |

## Interpretation

No single row cleanly dominates across all four observation families.

The hybrid objective is useful, but not as a universal replacement for K2 IWAE. `hybrid_w03`
improves state NLL and coverage on `x_sine` relative to plain K2 IWAE, while `hybrid_w01` gives the
best predictive-y NLL on `x_cubic`. On `student_t` and `x_tanh`, however, the hybrids mostly trade
away some state-density performance.

The pre-update predictive IWAE variant is the strongest Student-t state-density row in this pilot:
state NLL `6.985` versus `8.114` for plain K2 IWAE and `8.666` for Power-EP. It also improves
Student-t coverage relative to plain K2 IWAE, but does not fix predictive-y enough to become the
global promotion candidate.

K2 Power-EP remains the best balanced predictive row on `x_sine` and `x_tanh`, and it has the best
state NLL on `x_cubic`; however, its `x_sine` state NLL is much worse than IWAE/hybrid.

## Recommendation

The next best experiment is not another broad objective sweep. The evidence now points to a
family-specific selection problem:

```text
state-density leg: K2 IWAE / hybrid_w03
predictive-consistency leg: K2 Power-EP / hybrid_w01
heavy-tail leg: pre-update IWAE
```

The most aligned next branch is a small Pareto/selection pass, not a new architecture:

1. Train only the three competitive K2 rows: `hybrid_w03`, `k2_power_ep`, and `preupdate_iwae`.
2. Add `heteroskedastic_gaussian` as the fifth observation family.
3. Evaluate a fixed promotion rule using both state NLL and predictive-y tolerance.

If one row must be chosen for a single follow-up, choose `hybrid_w03`: it is the only variant in this
pilot that beats plain K2 IWAE state NLL on `x_sine` while preserving strong non-sine state-density
gains.
