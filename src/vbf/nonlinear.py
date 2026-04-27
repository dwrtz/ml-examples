"""Nonlinear scalar state-space benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.scipy as jsp

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vbf.data import (  # noqa: E402
    EpisodeBatch,
    LinearGaussianDataConfig,
    LinearGaussianParams,
    make_observation_covariates,
)


LOG_2PI = jnp.log(2.0 * jnp.pi)


@dataclass(frozen=True)
class NonlinearDataConfig:
    """Configuration for scalar nonlinear dynamic regression episodes."""

    batch_size: int = 128
    time_steps: int = 96
    x_pattern: str = "sinusoidal"
    x_cycles: float = 3.0
    x_amplitude: float = 1.0
    x_constant: float = 1.0
    x_missing_period: int = 4
    observation: str = "x_sine"


@dataclass(frozen=True)
class GridReferenceConfig:
    """Configuration for deterministic grid filtering diagnostics."""

    grid_min: float = -12.0
    grid_max: float = 12.0
    num_grid: int = 1201


class NonlinearReferenceOutputs(NamedTuple):
    filter_mean: jax.Array
    filter_var: jax.Array
    predictive_mean: jax.Array
    predictive_var: jax.Array


def make_nonlinear_batch(
    config: NonlinearDataConfig,
    params: LinearGaussianParams,
    seed: int,
) -> EpisodeBatch:
    """Generate nonlinear scalar episodes with explicit observed covariate `x_t`.

    The default observation model is:

    ```text
    z_t = z_{t-1} + w_t
    y_t = x_t * sin(z_t) + v_t
    ```

    Keeping `x_t` explicit preserves the weak-observability stress case: when
    `x_t` is near zero, observations carry little information about `z_t`.
    """

    if params.q <= 0 or params.r <= 0 or params.p0 <= 0:
        raise ValueError("q, r, and p0 must be positive")
    data_config = LinearGaussianDataConfig(
        batch_size=config.batch_size,
        time_steps=config.time_steps,
        x_pattern=config.x_pattern,
        x_cycles=config.x_cycles,
        x_amplitude=config.x_amplitude,
        x_constant=config.x_constant,
        x_missing_period=config.x_missing_period,
    )
    key = jax.random.PRNGKey(seed)
    key_z0, key_w, key_v = jax.random.split(key, 3)
    x = make_observation_covariates(data_config, jax.random.fold_in(key, 42))
    z_initial = params.m0 + jnp.sqrt(params.p0) * jax.random.normal(
        key_z0,
        shape=(config.batch_size,),
        dtype=jnp.float64,
    )
    innovations = jnp.sqrt(params.q) * jax.random.normal(
        key_w,
        shape=(config.batch_size, config.time_steps),
        dtype=jnp.float64,
    )
    z = z_initial[:, None] + jnp.cumsum(innovations, axis=1)
    y_mean = nonlinear_observation_mean(z, x, config.observation)
    y = y_mean + jnp.sqrt(params.r) * jax.random.normal(
        key_v,
        shape=(config.batch_size, config.time_steps),
        dtype=jnp.float64,
    )
    return EpisodeBatch(x=x, y=y, z=z)


def nonlinear_grid_filter(
    batch: EpisodeBatch,
    params: LinearGaussianParams,
    *,
    data_config: NonlinearDataConfig,
    grid_config: GridReferenceConfig = GridReferenceConfig(),
) -> NonlinearReferenceOutputs:
    """Approximate nonlinear filtering with a deterministic 1D grid."""

    if grid_config.num_grid < 3:
        raise ValueError("num_grid must be at least 3")
    if grid_config.grid_max <= grid_config.grid_min:
        raise ValueError("grid_max must be greater than grid_min")
    grid = jnp.linspace(
        grid_config.grid_min,
        grid_config.grid_max,
        grid_config.num_grid,
        dtype=jnp.float64,
    )
    dz = (grid_config.grid_max - grid_config.grid_min) / (grid_config.num_grid - 1)
    log_dz = jnp.log(dz)
    prior_log_mass = _normal_log_prob(grid, params.m0, params.p0) + log_dz
    prior_log_mass = prior_log_mass - jsp.special.logsumexp(prior_log_mass)
    transition_log_mass = _normal_log_prob(grid[None, :], grid[:, None], params.q) + log_dz
    transition_log_mass = transition_log_mass - jsp.special.logsumexp(
        transition_log_mass,
        axis=1,
        keepdims=True,
    )

    x_tb = batch.x.T
    y_tb = batch.y.T

    def step(prev_log_mass: jax.Array, obs: tuple[jax.Array, jax.Array]):
        x_t, y_t = obs
        pred_log_mass = jsp.special.logsumexp(
            prev_log_mass[:, :, None] + transition_log_mass[None, :, :],
            axis=1,
        )
        pred_mass = jnp.exp(pred_log_mass)
        obs_mean = nonlinear_observation_mean(
            grid[None, :],
            x_t[:, None],
            data_config.observation,
        )
        predictive_mean = jnp.sum(pred_mass * obs_mean, axis=1)
        predictive_var = jnp.sum(
            pred_mass * ((obs_mean - predictive_mean[:, None]) ** 2 + params.r),
            axis=1,
        )
        filter_log_mass = pred_log_mass + _normal_log_prob(y_t[:, None], obs_mean, params.r)
        filter_log_mass = filter_log_mass - jsp.special.logsumexp(
            filter_log_mass,
            axis=1,
            keepdims=True,
        )
        filter_mass = jnp.exp(filter_log_mass)
        filter_mean = jnp.sum(filter_mass * grid[None, :], axis=1)
        filter_var = jnp.sum(filter_mass * (grid[None, :] - filter_mean[:, None]) ** 2, axis=1)
        return filter_log_mass, (filter_mean, filter_var, predictive_mean, predictive_var)

    init_log_mass = jnp.broadcast_to(prior_log_mass[None, :], (batch.x.shape[0], grid.shape[0]))
    _, outputs = jax.lax.scan(step, init_log_mass, (x_tb, y_tb))
    return NonlinearReferenceOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def nonlinear_observation_mean(
    z: jax.Array,
    x: jax.Array,
    observation: str = "x_sine",
) -> jax.Array:
    """Return `h(z_t, x_t)` for supported nonlinear observation models."""

    if observation == "x_sine":
        return x * jnp.sin(z)
    if observation == "sine_product":
        return jnp.sin(x * z)
    raise ValueError(f"Unsupported nonlinear observation: {observation}")


def nonlinear_predictive_moments_from_filter(
    filter_mean: jax.Array,
    filter_var: jax.Array,
    x: jax.Array,
    params: LinearGaussianParams,
    *,
    observation: str = "x_sine",
) -> tuple[jax.Array, jax.Array]:
    """Return Gaussian moment approximation for `p(y_t | q^F_{t-1}, x_t)`.

    For the default `x_sine` observation, the moments are analytic because
    `z_t` is Gaussian under the random-walk transition.
    """

    if observation != "x_sine":
        raise ValueError(f"Unsupported predictive moments for observation: {observation}")

    initial_mean = jnp.full((filter_mean.shape[0], 1), params.m0, dtype=filter_mean.dtype)
    initial_var = jnp.full((filter_var.shape[0], 1), params.p0, dtype=filter_var.dtype)
    prev_mean = jnp.concatenate((initial_mean, filter_mean[:, :-1]), axis=1)
    prev_var = jnp.concatenate((initial_var, filter_var[:, :-1]), axis=1)
    pred_state_var = prev_var + params.q
    mean_sin = jnp.exp(-0.5 * pred_state_var) * jnp.sin(prev_mean)
    mean_cos_2z = jnp.exp(-2.0 * pred_state_var) * jnp.cos(2.0 * prev_mean)
    mean_sin_sq = 0.5 * (1.0 - mean_cos_2z)
    var_sin = jnp.maximum(mean_sin_sq - mean_sin**2, 0.0)
    return x * mean_sin, x**2 * var_sin + params.r


def _normal_log_prob(
    value: jax.Array, mean: jax.Array | float, var: jax.Array | float
) -> jax.Array:
    return -0.5 * (LOG_2PI + jnp.log(var) + (value - mean) ** 2 / var)


def _time_major_to_batch_major(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, 0, 1)
