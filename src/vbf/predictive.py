"""Posterior predictive distribution helpers."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp  # noqa: E402

from vbf.data import EpisodeBatch, LinearGaussianParams, broadcast_param_like  # noqa: E402


class PredictiveMoments(NamedTuple):
    mean: jax.Array
    var: jax.Array


def linear_gaussian_predictive_from_filter(
    filter_mean: jax.Array,
    filter_var: jax.Array,
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
) -> PredictiveMoments:
    """Return `p(y_t | q^F_{t-1}, x_t)` under the scalar transition/measurement model."""

    prev_mean, prev_var = previous_filter_beliefs(filter_mean, filter_var, state_params)
    q = broadcast_param_like(state_params.q, batch.x)
    r = broadcast_param_like(state_params.r, batch.x)
    pred_state_var = prev_var + q
    return PredictiveMoments(
        mean=batch.x * prev_mean,
        var=batch.x**2 * pred_state_var + r,
    )


def previous_filter_beliefs(
    filter_mean: jax.Array,
    filter_var: jax.Array,
    state_params: LinearGaussianParams,
) -> tuple[jax.Array, jax.Array]:
    """Return batch-major previous filtering beliefs aligned with each current time step."""

    initial_mean = jnp.full((filter_mean.shape[0], 1), state_params.m0, dtype=filter_mean.dtype)
    initial_var = jnp.full((filter_var.shape[0], 1), state_params.p0, dtype=filter_var.dtype)
    return (
        jnp.concatenate((initial_mean, filter_mean[:, :-1]), axis=1),
        jnp.concatenate((initial_var, filter_var[:, :-1]), axis=1),
    )
