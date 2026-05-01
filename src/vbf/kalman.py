"""Exact Kalman filtering and edge-posterior oracles."""

from __future__ import annotations

from dataclasses import dataclass

import jax

from vbf.dtypes import DEFAULT_DTYPE  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from vbf.data import EpisodeBatch, LinearGaussianParams, broadcast_param_like  # noqa: E402


@dataclass(frozen=True)
class KalmanOutputs:
    pred_mean: jax.Array
    pred_var: jax.Array
    filter_mean: jax.Array
    filter_var: jax.Array
    predictive_mean: jax.Array
    predictive_var: jax.Array


@dataclass(frozen=True)
class EdgeOracleOutputs:
    edge_mean: jax.Array
    edge_cov: jax.Array
    filter_mean: jax.Array
    filter_var: jax.Array


@dataclass(frozen=True)
class PredictiveOutputs:
    mean: jax.Array
    var: jax.Array


def kalman_filter_scalar(batch: EpisodeBatch, params: LinearGaussianParams) -> KalmanOutputs:
    """Run the exact scalar Kalman filter over a batch of episodes."""

    _assert_batch_shapes(batch)
    _assert_positive_params(params)

    x_bt = batch.x.T
    y_bt = batch.y.T

    def step(carry: tuple[jax.Array, jax.Array], obs: tuple[jax.Array, jax.Array]):
        mean_prev, var_prev = carry
        x_t, y_t = obs

        pred_mean = mean_prev
        q = broadcast_param_like(params.q, x_t)
        r = broadcast_param_like(params.r, x_t)
        pred_var = var_prev + q
        predictive_mean = x_t * pred_mean
        predictive_var = x_t**2 * pred_var + r
        gain = pred_var * x_t / predictive_var
        filter_mean = pred_mean + gain * (y_t - predictive_mean)
        filter_var = (1.0 - gain * x_t) * pred_var

        outputs = (
            pred_mean,
            pred_var,
            filter_mean,
            filter_var,
            predictive_mean,
            predictive_var,
        )
        return (filter_mean, filter_var), outputs

    batch_size = batch.x.shape[0]
    init = (
        jnp.full((batch_size,), params.m0, dtype=DEFAULT_DTYPE),
        jnp.full((batch_size,), params.p0, dtype=DEFAULT_DTYPE),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt))

    return KalmanOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def kalman_edge_posterior_scalar(
    batch: EpisodeBatch,
    params: LinearGaussianParams,
) -> EdgeOracleOutputs:
    """Compute the exact two-state edge posterior for each time step.

    The edge state is ordered as `[z_t, z_tm1]`.
    """

    _assert_batch_shapes(batch)
    _assert_positive_params(params)

    x_bt = batch.x.T
    y_bt = batch.y.T

    def step(carry: tuple[jax.Array, jax.Array], obs: tuple[jax.Array, jax.Array]):
        mean_prev, var_prev = carry
        x_t, y_t = obs

        q = broadcast_param_like(params.q, x_t)
        r = broadcast_param_like(params.r, x_t)
        prior_var_zt = var_prev + q
        prior_cov = var_prev
        innovation_var = x_t**2 * prior_var_zt + r
        innovation = y_t - x_t * mean_prev
        gain_zt = prior_var_zt * x_t / innovation_var
        gain_z_tm1 = prior_cov * x_t / innovation_var

        edge_mean_zt = mean_prev + gain_zt * innovation
        edge_mean_z_tm1 = mean_prev + gain_z_tm1 * innovation

        cov_00 = prior_var_zt - gain_zt * innovation_var * gain_zt
        cov_01 = prior_cov - gain_zt * innovation_var * gain_z_tm1
        cov_11 = var_prev - gain_z_tm1 * innovation_var * gain_z_tm1

        edge_mean = jnp.stack((edge_mean_zt, edge_mean_z_tm1), axis=-1)
        row_0 = jnp.stack((cov_00, cov_01), axis=-1)
        row_1 = jnp.stack((cov_01, cov_11), axis=-1)
        edge_cov = jnp.stack((row_0, row_1), axis=-2)

        return (edge_mean_zt, cov_00), (edge_mean, edge_cov, edge_mean_zt, cov_00)

    batch_size = batch.x.shape[0]
    init = (
        jnp.full((batch_size,), params.m0, dtype=DEFAULT_DTYPE),
        jnp.full((batch_size,), params.p0, dtype=DEFAULT_DTYPE),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt))
    edge_mean, edge_cov, filter_mean, filter_var = outputs

    return EdgeOracleOutputs(
        edge_mean=_time_major_to_batch_major(edge_mean),
        edge_cov=_time_major_to_batch_major(edge_cov),
        filter_mean=_time_major_to_batch_major(filter_mean),
        filter_var=_time_major_to_batch_major(filter_var),
    )


def measurement_predictive_scalar(
    batch: EpisodeBatch,
    params: LinearGaussianParams,
) -> PredictiveOutputs:
    """Return `p(y_t | D_{1:t-1}, x_t)` for the scalar model."""

    outputs = kalman_filter_scalar(batch, params)
    return PredictiveOutputs(mean=outputs.predictive_mean, var=outputs.predictive_var)


def _time_major_to_batch_major(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, 0, 1)


def _assert_batch_shapes(batch: EpisodeBatch) -> None:
    if batch.x.shape != batch.y.shape or batch.x.shape != batch.z.shape:
        raise ValueError("x, y, and z must have matching [batch, time] shapes")
    if len(batch.x.shape) != 2:
        raise ValueError("x, y, and z must be rank-2 [batch, time] arrays")


def _assert_positive_params(params: LinearGaussianParams) -> None:
    if jnp.any(jnp.asarray(params.q) <= 0) or jnp.any(jnp.asarray(params.r) <= 0):
        raise ValueError("q and r must be positive")
    if params.p0 <= 0:
        raise ValueError("q, r, and p0 must be positive")
