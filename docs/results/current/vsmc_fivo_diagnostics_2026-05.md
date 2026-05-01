# VSMC/FIVO Diagnostics, May 2026

This report treats FIVO as diagnostic instrumentation first and as a contender
second. The key question is whether proposal/resampling diagnostics explain a
failure mode that K2/K3 IWAE cannot.

## Runs

- Family: `outputs/cloud_downloads/step3_fivo_diagnostics_family_1000_2026-05-01`
- Stressors: `outputs/cloud_downloads/step3_fivo_diagnostics_stressors_1000_2026-05-01`
- Seeds: `321,322,323`
- Steps: `1000`

## Family Summary

| model | state NLL | pred-y NLL | cov90 | state RMSE | mean ESS | mean log-w var |
|---|---:|---:|---:|---:|---:|---:|
| K4 FIVO fixed-lag twist h4 | 4.026 | 1.505 | 0.643 | 4.238 | 23.058 | 425.288 |
| K2 FIVO bridge n32 | 4.299 | 1.669 | 0.562 | 3.681 | 23.012 | 303.241 |
| K4 FIVO bridge n32 | 4.382 | 1.991 | 0.589 | 4.496 | 23.480 | 475.579 |
| K2 IWAE h4 k32 | 5.495 | 1.103 | 0.543 | 3.613 | n/a | n/a |
| K2 FIVO n32 | 16.986 | 0.901 | 0.441 | 3.107 | 12.174 | 1182852.000 |
| K4 auxiliary FIVO bridge n32 | 23.844 | 2.769 | 0.206 | 10.552 | 16.693 | 10057.560 |

## Stressor Summary

| model | state NLL | pred-y NLL | cov90 | state RMSE | mean ESS | mean log-w var |
|---|---:|---:|---:|---:|---:|---:|
| K2 FIVO bridge n32 | 3.111 | 0.367 | 0.697 | 3.903 | 28.936 | 0.638 |
| K4 FIVO fixed-lag twist h4 | 3.359 | 0.368 | 0.643 | 4.107 | 27.798 | 0.348 |
| K2 IWAE h4 k32 | 5.623 | 0.367 | 0.458 | 4.985 | n/a | n/a |
| K2 FIVO n32 | 33.581 | 0.343 | 0.226 | 3.988 | 14.044 | 69.736 |

## Interpretation

Plain FIVO improves predictive-y only by collapsing state density. FIVO bridge
is the important row: it improves state density and coverage on the stressors
without a predictive-y penalty, and it keeps ESS high. On the family grid, FIVO
bridge improves state NLL but pays a predictive-y cost. The twist row improves
average family state NLL, but its variance ratio and state RMSE suggest the
improvement is not a clean replacement for K2 bridge.

## Decision

Use FIVO bridge as a diagnostic and a stressor candidate. Do not promote plain
FIVO. The diagnostics justify one modest flow pilot because proposal mismatch
and posterior-shape mismatch remain plausible explanations.
