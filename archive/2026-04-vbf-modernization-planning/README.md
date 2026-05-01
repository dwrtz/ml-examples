# 2026-04 VBF Modernization Planning

- Why this branch existed: plan the migration from the original TensorFlow
  experiment into the modern `src/vbf/` package.
- What was concluded: the modern repo should center the explicit edge posterior,
  linear-Gaussian oracle checks, and reusable JAX implementation.
- Files moved: `docs/vbf-modernization/`.
- Superseded by: the active package layout in `src/vbf/` and the root README.
