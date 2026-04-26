"""Nonlinear scalar state-space benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax

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
    prior_mass = jnp.exp(_normal_log_prob(grid, params.m0, params.p0)) * dz
    prior_mass = prior_mass / jnp.sum(prior_mass)
    transition = jnp.exp(_normal_log_prob(grid[None, :], grid[:, None], params.q)) * dz
    transition = transition / jnp.sum(transition, axis=1, keepdims=True)

    x_tb = batch.x.T
    y_tb = batch.y.T

    def step(prev_mass: jax.Array, obs: tuple[jax.Array, jax.Array]):
        x_t, y_t = obs
        pred_mass = prev_mass @ transition
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
        likelihood = jnp.exp(_normal_log_prob(y_t[:, None], obs_mean, params.r))
        filter_mass = pred_mass * likelihood
        filter_mass = filter_mass / jnp.sum(filter_mass, axis=1, keepdims=True)
        filter_mean = jnp.sum(filter_mass * grid[None, :], axis=1)
        filter_var = jnp.sum(filter_mass * (grid[None, :] - filter_mean[:, None]) ** 2, axis=1)
        return filter_mass, (filter_mean, filter_var, predictive_mean, predictive_var)

    init_mass = jnp.broadcast_to(prior_mass[None, :], (batch.x.shape[0], grid.shape[0]))
    _, outputs = jax.lax.scan(step, init_mass, (x_tb, y_tb))
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


def _normal_log_prob(
    value: jax.Array, mean: jax.Array | float, var: jax.Array | float
) -> jax.Array:
    return -0.5 * (LOG_2PI + jnp.log(var) + (value - mean) ** 2 / var)


def _time_major_to_batch_major(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, 0, 1)
