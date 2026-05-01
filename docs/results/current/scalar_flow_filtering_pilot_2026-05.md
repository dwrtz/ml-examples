# Scalar Flow Filtering Pilot, May 2026

This report tests the Step 4 hypothesis: if mixtures and FIVO diagnostics still
leave a gap, try a modest online scalar flow posterior rather than a broad
sequence model.

## Implementation

The pilot adds `posterior_family: scalar_flow`, a strict online scalar
monotone-spline flow:

```text
u ~ Normal(0, 1)
z_t = loc_t + scale_t * S_t(u)
q^F_t(z_t) = q_u(u) |du / dz_t|
```

The update emits current flow parameters from the previous filtering moments
and current observation only. It keeps the filtering contract
`q^F_t = update(q^F_{t-1}, x_t, y_t)` and uses a Gaussian backward conditional
for existing edge/window objectives.

## Runs

- Family: `outputs/cloud_downloads/step4_scalar_flow_family_1000_2026-05-02`
- Seeds: `321,322,323`
- Steps: `1000`
- Models: K2 IWAE, K2 FIVO bridge, scalar-flow IWAE, scalar-flow FIVO bridge

## Family Summary

| model | state NLL | pred-y NLL | cov90 | state RMSE | var ratio |
|---|---:|---:|---:|---:|---:|
| scalar-flow FIVO bridge n32 | 4.136 | 4.283 | 0.755 | 5.263 | 22.392 |
| K2 mixture FIVO bridge n32 | 4.296 | 1.670 | 0.570 | 3.680 | 0.924 |
| K2 mixture IWAE h4 k32 | 5.578 | 1.159 | 0.545 | 3.637 | 0.952 |
| scalar-flow IWAE h4 k32 | 16.206 | 3.149 | 0.438 | 4.723 | 1.316 |

## Family Breakdown

| observation | scalar-flow FIVO state NLL | K2 FIVO state NLL | scalar-flow FIVO pred-y | K2 FIVO pred-y |
|---|---:|---:|---:|---:|
| student_t | 3.165 | 3.149 | 1.076 | 1.000 |
| x_cubic | 4.443 | 4.744 | 14.555 | 4.117 |
| x_sine | 3.021 | 3.155 | 0.802 | 0.781 |
| x_tanh | 5.916 | 6.134 | 0.698 | 0.782 |

## Interpretation

The flow family is not a free win. Flow+FIVO bridge improves average state NLL
slightly and substantially improves moment-based coverage, but it over-inflates
variance and badly hurts predictive-y on cubic. Flow+IWAE is unstable. This
suggests the remaining issue is not simply "use a richer marginal"; objective
and predictive-normalizer compatibility still matter.

## Decision

Do not promote scalar flow yet. The next flow work, if any, should be narrow:
regularize flow variance or add predictive-normalizer constraints, then rerun
only the flow+FIVO bridge row against K2 FIVO bridge. Otherwise, move to
synthesis and publication.
