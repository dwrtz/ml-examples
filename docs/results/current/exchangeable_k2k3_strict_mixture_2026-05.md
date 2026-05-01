# Exchangeable K2/K3 Strict Mixtures, May 2026

This report tests whether the K2 frontier is limited by component count or by a
non-exchangeable parameterization.

## Runs

- Family: `outputs/cloud_downloads/step1_exchangeable_k2_k3_family_1000_2026-05-01`
- Stressors: `outputs/cloud_downloads/step1_exchangeable_k2_k3_stressors_1000_2026-05-01`
- Seeds: `321,322,323`
- Steps: `1000`

## Family Summary

| model | state NLL | pred-y NLL | cov90 | state RMSE | var ratio |
|---|---:|---:|---:|---:|---:|
| K3 mixture IWAE h4 k32 | 4.525 | 0.931 | 0.603 | 3.042 | 0.909 |
| K2 mixture IWAE h4 k32 | 4.859 | 1.008 | 0.584 | 3.303 | 0.922 |
| K2 generic Power-EP alpha 0.5 | 6.764 | 0.841 | 0.640 | 2.838 | 0.507 |
| exchangeable K2 mixture IWAE h4 k32 | 8.002 | 1.064 | 0.593 | 3.365 | 2.209 |
| exchangeable K3 mixture IWAE h4 k32 | 8.146 | 1.011 | 0.592 | 3.626 | 3.398 |

## Stressor Summary

| model | state NLL | pred-y NLL | cov90 | state RMSE | var ratio |
|---|---:|---:|---:|---:|---:|
| K3 mixture IWAE h4 k32 | 5.965 | 0.423 | 0.440 | 4.545 | 0.159 |
| K2 mixture IWAE h4 k32 | 6.370 | 0.428 | 0.428 | 4.579 | 0.149 |
| K2 generic Power-EP alpha 0.5 | 8.366 | 0.394 | 0.670 | 3.905 | 0.362 |

## Interpretation

K3 improves state density and predictive-y modestly over K2, both on the family
grid and on stressors. This says component count still matters. The explicitly
exchangeable parameterization, however, is not an immediate win: it improves
neither state NLL nor predictive-y and inflates variance ratio.

## Decision

K3 is a valid Pareto candidate. The exchangeable parameterization should be
treated as a diagnostic negative result, not a promoted model. Further K growth
should be justified by a specific diagnostic rather than broadened by default.
