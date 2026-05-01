# Archive

This directory contains retired research branches, historical experiment configs,
one-off reports, and scripts that are preserved for reproducibility but are no
longer part of the active repo surface.

Archived code is not expected to pass current lint, tests, or dependency checks.
The active implementation lives under `src/vbf/`, active entry points live under
`scripts/`, and active configs live under `experiments/`.

## Branches

| Directory | Status | Contents | Notes |
|---|---|---|---|
| `2020_variational_filtering_original/` | historical | TensorFlow scripts, paper notes | Original variational filtering experiment |
| `2026-04-vbf-modernization-planning/` | historical | planning docs | Original modernization plans and handoff notes |
| `2026-04-linear-gaussian-ablation-sweeps/` | retired | configs, sweeps, aggregators | Scalar ablation, predictive-head, calibration, and Q/R studies |
| `2026-04-nonlinear-strict-filter/` | retired | reports | Reference-distilled nonlinear strict-filter milestone |
| `2026-04-nonlinear-unsupervised-objective/` | retired | reports, configs, aggregators | ELBO objective-repair branch |
| `2026-04-nonlinear-divergence-family/` | retired | reports, diagnostics | IWAE, Renyi, and mixture-family branch |
| `2026-04-quadrature-adf-fivo/` | retired | configs, reports, scripts | Quadrature, ADF, FIVO, and predictive-normalizer experiments |
| `2026-05-model-agnostic-pilots/` | historical/current-adjacent | pilot reports | Move back to `docs/results/current/` if this becomes the active branch |
