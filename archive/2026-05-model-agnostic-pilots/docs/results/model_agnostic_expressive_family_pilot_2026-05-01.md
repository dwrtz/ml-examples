# Model-Agnostic Expressive-Family Pilot

Prepared: 2026-05-01

## Summary

The pilot ran the model-agnostic expressive-family matrix from `notes.md` across four observation
families, seeds `321,322,323`, and 1000 training steps:

```text
x_sine
student_t
x_tanh
x_cubic
```

Rows:

```text
structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4
direct_mixture_k2_joint_iwae_h4_k32
direct_mixture_k3_joint_iwae_h4_k32
direct_mixture_k2_fivo_n32
direct_mixture_k2_power_ep_alpha_0p5
```

Artifacts:

```text
outputs/model_agnostic_expressive_family_pilot_1000/metrics.csv
outputs/model_agnostic_expressive_family_pilot_1000/summary.md
```

## Aggregate Results

| observation | row | state NLL | cov 90 | var ratio | pred-y NLL |
|---|---|---:|---:|---:|---:|
| `x_sine` | baseline | 54.930 | 0.342 | 0.083 | 0.571 |
| `x_sine` | K2 IWAE | 6.402 | 0.416 | 0.125 | 0.681 |
| `x_sine` | K3 IWAE | 6.706 | 0.402 | 0.130 | 0.677 |
| `x_sine` | K2 FIVO | 31.952 | 0.278 | 0.011 | 0.500 |
| `x_sine` | K2 Power-EP | 15.642 | 0.470 | 0.050 | 0.518 |
| `student_t` | baseline | 25.165 | 0.379 | 0.067 | 0.852 |
| `student_t` | K2 IWAE | 8.114 | 0.357 | 0.109 | 0.877 |
| `student_t` | K3 IWAE | 9.569 | 0.343 | 0.101 | 0.873 |
| `student_t` | K2 FIVO | 31.184 | 0.271 | 0.010 | 0.786 |
| `student_t` | K2 Power-EP | 8.666 | 0.578 | 0.110 | 0.844 |
| `x_tanh` | baseline | 10.559 | 0.469 | 0.185 | 0.417 |
| `x_tanh` | K2 IWAE | 3.158 | 0.671 | 0.438 | 0.462 |
| `x_tanh` | K3 IWAE | 3.124 | 0.675 | 0.392 | 0.433 |
| `x_tanh` | K2 FIVO | 13.082 | 0.395 | 0.062 | 0.389 |
| `x_tanh` | K2 Power-EP | 3.198 | 0.680 | 0.337 | 0.407 |
| `x_cubic` | baseline | 5680.423 | 0.716 | 2.369 | 665.041 |
| `x_cubic` | K2 IWAE | 2.016 | 0.878 | 4.270 | 2.137 |
| `x_cubic` | K3 IWAE | 1.946 | 0.901 | 4.279 | 2.300 |
| `x_cubic` | K2 FIVO | 2.191 | 0.886 | 2.840 | 33.436 |
| `x_cubic` | K2 Power-EP | 1.880 | 0.788 | 1.360 | 2.284 |

## Interpretation

The pilot supports the main pivot in `notes.md`: the K2/K3 non-Gaussian strict-family improvement
is not sine-only. K2/K3 IWAE and K2 Power-EP all reduce state NLL substantially on the non-periodic
`x_tanh` and `x_cubic` families, and K2 IWAE / Power-EP reduce state NLL on `student_t`.

K3 did not clearly dominate K2. It improves coverage on `x_tanh` and `x_cubic`, but gives back state
NLL on `x_sine` and `student_t`. Treat K3 as a capacity diagnostic, not the next promoted row.

K2 Power-EP is the most balanced row in this pilot. It is not always the best state-density row, but
it keeps predictive-y NLL closer to the best predictive row on `x_sine`, `student_t`, and `x_tanh`,
while improving coverage strongly on `student_t` and `x_tanh`.

K2 FIVO did not look competitive as currently parameterized. It often improves predictive-y NLL but
usually loses state NLL or coverage, so it is better treated as a diagnostic/reference direction than
as the immediate promotion candidate.

## Recommendation

Promote the next branch around generic K2 mixture projection / IWAE hybrids, not alias-aware K5 and
not plain FIVO. The next concrete experiment should combine the state-density strength of K2 IWAE
with the predictive consistency of K2 Power-EP:

```text
direct_mixture_k2_hybrid_iwae_projection_h4_k16_w01
direct_mixture_k2_hybrid_iwae_projection_h4_k16_w03
direct_mixture_k2_predictive_consistent_iwae_h4_k32
```

Run those against the same four-family battery and keep the promotion rule from `notes.md`: improve
state NLL or coverage on at least two nonlinear likelihood families while keeping predictive-y NLL
within tolerance of the best predictive row.
