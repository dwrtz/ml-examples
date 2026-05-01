# K2 Pareto Lock, May 2026

This report locks the starting K2 nonlinear-filter frontier before the
exchangeability, predictive-consistency, VSMC/FIVO, and flow experiments.

## Runs

- Family: `outputs/cloud_downloads/k2_pareto_lock_family_1000_2026-05-01`
- Stressors: `outputs/cloud_downloads/k2_pareto_lock_stressors_1000_2026-05-01`
- Config families: active nonlinear family configs plus weak/intermittent/zero/random-normal stressors.
- Seeds: `321,322,323`
- Steps: `1000`

## Family Summary

| model | state NLL | pred-y NLL | cov90 | state RMSE | var ratio |
|---|---:|---:|---:|---:|---:|
| K2 IWAE h4 k32 + pre-update predictive scoring | 4.614 | 0.971 | 0.599 | 3.052 | 0.895 |
| K2 IWAE h4 k32 | 4.869 | 1.004 | 0.584 | 3.316 | 0.918 |
| K2 IWAE h4 k16 + local ADF projection w0.3 | 5.419 | 0.927 | 0.587 | 3.090 | 0.809 |
| K2 generic Power-EP alpha 0.5 | 6.764 | 0.841 | 0.640 | 2.838 | 0.507 |
| promoted strict Gaussian baseline | 4159.987 | 3217.708 | 0.479 | 3.337 | 0.401 |

## Stressor Summary

| model | state NLL | pred-y NLL | cov90 | state RMSE | var ratio |
|---|---:|---:|---:|---:|---:|
| K2 IWAE h4 k32 | 6.005 | 0.428 | 0.445 | 4.507 | 0.156 |
| K2 IWAE h4 k32 + pre-update predictive scoring | 6.049 | 0.427 | 0.444 | 4.503 | 0.156 |
| K2 generic Power-EP alpha 0.5 | 8.366 | 0.394 | 0.670 | 3.905 | 0.362 |

## Interpretation

The K2 mixture IWAE row is the clean baseline to carry forward. It is
reference-free, stable across the stressors, and dramatically better than the
strict Gaussian baseline on nonlinear family state density. Power-EP remains a
useful predictive/coverage comparator, but it gives up too much state density.
Pre-update predictive scoring is worth testing, but at Step 0 it is not yet a
separate promoted objective.

## Decision

Carry `direct_mixture_k2_joint_iwae_h4_k32` forward as the baseline. Keep
Power-EP as a comparator. Treat pre-update predictive scoring as an objective
variant, not as a locked default.
