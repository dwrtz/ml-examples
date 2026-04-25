"""Evaluation metrics for learned filters."""

from __future__ import annotations

import jax.numpy as jnp

LOG_2PI = jnp.log(2.0 * jnp.pi)


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


def rmse_global(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Return one RMSE over all batch and time entries."""

    return jnp.sqrt(jnp.mean((pred - target) ** 2))


def rmse_time_mean(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Return the mean over time of per-time batch RMSEs."""

    return jnp.mean(rmse_over_batch(pred, target))


def scalar_gaussian_nll(value: jnp.ndarray, mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
    """Return scalar Gaussian negative log likelihood elementwise."""

    return 0.5 * (LOG_2PI + jnp.log(var) + (value - mean) ** 2 / var)


def gaussian_interval_coverage(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    var: jnp.ndarray,
    *,
    z_score: float,
) -> jnp.ndarray:
    """Return empirical central Gaussian interval coverage."""

    half_width = z_score * jnp.sqrt(var)
    return jnp.mean((value >= mean - half_width) & (value <= mean + half_width))
