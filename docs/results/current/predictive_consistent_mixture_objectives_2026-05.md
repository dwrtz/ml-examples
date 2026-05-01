# Predictive-Consistent Mixture Objectives, May 2026

This report asks whether explicitly scoring predictive normalizers can improve
the K2 mixture frontier without sacrificing filtering-state density.

## Runs

- Family: `outputs/cloud_downloads/step2_predictive_consistent_family_1000_2026-05-01_rerun`
- Stressors: `outputs/cloud_downloads/step2_predictive_consistent_stressors_1000_2026-05-01`
- Seeds: `321,322,323`
- Steps: `1000`

## Family Summary

| model | state NLL | pred-y NLL | cov90 | state RMSE | var ratio |
|---|---:|---:|---:|---:|---:|
| K2 IWAE + pre-update predictive scoring | 4.548 | 0.998 | 0.602 | 3.072 | 0.928 |
| K2 IWAE h4 k32 | 4.557 | 0.987 | 0.601 | 3.070 | 0.907 |
| K2 IWAE + late predictive-y w0.3 | 4.925 | 1.001 | 0.577 | 3.275 | 0.904 |
| K2 IWAE + detached pre-update predictive scoring | 4.988 | 1.002 | 0.578 | 3.351 | 0.890 |
| K2 IWAE h4 k16 + local ADF projection w0.3 | 5.456 | 0.916 | 0.586 | 3.099 | 0.800 |
| K2 generic Power-EP alpha 0.5 | 6.764 | 0.841 | 0.640 | 2.838 | 0.507 |

## Stressor Summary

| model | state NLL | pred-y NLL | cov90 | state RMSE | var ratio |
|---|---:|---:|---:|---:|---:|
| K2 IWAE h4 k16 + local ADF projection w0.3 | 4.382 | 0.431 | 0.588 | 3.972 | 0.249 |
| K2 IWAE h4 k32 | 6.044 | 0.428 | 0.444 | 4.511 | 0.156 |
| K2 IWAE + pre-update predictive scoring | 6.077 | 0.427 | 0.442 | 4.519 | 0.155 |
| K2 generic Power-EP alpha 0.5 | 8.366 | 0.394 | 0.670 | 3.905 | 0.362 |

## Interpretation

Pre-update predictive scoring ties the baseline on the family grid, but it does
not improve predictive-y and slightly regresses the stressor state NLL. Detached
and late predictive-y variants are worse on family state density. The local ADF
hybrid remains interesting on stressors, but it is not a simple predictive
consistency fix for the family grid.

## Decision

Do not promote a predictive-consistent objective from this pass. The predictive
normalizer gap is real, but direct predictive-y pressure mostly moves the model
along the same tradeoff rather than resolving it.
