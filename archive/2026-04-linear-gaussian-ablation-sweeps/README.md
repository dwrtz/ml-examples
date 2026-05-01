# 2026-04 Linear-Gaussian Ablation Sweeps

- Why this branch existed: explore scalar filtering ablations, predictive heads,
  weak observability, self-fed training, and Q/R generalization.
- What was concluded: the self-fed supervised edge objective was the strongest
  learned linear-Gaussian baseline, while the ELBO path remained the key
  unsupervised benchmark.
- Files moved: branch-specific linear-Gaussian configs, sweep scripts,
  aggregators, predictive-head scripts, and placeholder recurrent configs.
- Superseded by: `docs/results/current/linear_gaussian_scalar_report.md`.
