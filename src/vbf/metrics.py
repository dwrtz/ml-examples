"""Evaluation metrics for learned filters."""

from __future__ import annotations

import jax.numpy as jnp


def scalar_gaussian_kl(
    mean_p: jnp.ndarray,
    var_p: jnp.ndarray,
    mean_q: jnp.ndarray,
    var_q: jnp.ndarray,
) -> jnp.ndarray:
    """Return `KL(N_p || N_q)` for scalar Gaussian arrays."""

    return 0.5 * (jnp.log(var_q / var_p) + (var_p + (mean_p - mean_q) ** 2) / var_q - 1.0)


def mean_over_batch(value: jnp.ndarray) -> jnp.ndarray:
    """Average a batch-major time series over the batch axis."""

    return jnp.mean(value, axis=0)


def rmse_over_batch(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Return per-time RMSE for batch-major trajectories."""

    return jnp.sqrt(jnp.mean((pred - target) ** 2, axis=0))
