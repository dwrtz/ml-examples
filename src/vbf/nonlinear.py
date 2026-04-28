"""Nonlinear scalar state-space benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.scipy as jsp
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vbf.data import (  # noqa: E402
    EpisodeBatch,
    LinearGaussianDataConfig,
    LinearGaussianParams,
    make_observation_covariates,
)
from vbf.models.cells import StructuredMLPOutputs, _mlp_features  # noqa: E402


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


class NonlinearReferenceShapeOutputs(NamedTuple):
    entropy: jax.Array
    normalized_entropy: jax.Array
    peak_count: jax.Array
    max_mass: jax.Array
    credible_width_90: jax.Array


class NonlinearReferenceGridOutputs(NamedTuple):
    grid: jax.Array
    filter_mass: jax.Array


def make_y_observed_mask(
    *,
    batch_size: int,
    time_steps: int,
    probability: float = 0.0,
    span_probability: float = 0.0,
    span_length: int = 1,
    seed: int,
) -> jax.Array:
    """Return a boolean mask where true entries expose `y_t` to the update."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if not 0.0 <= span_probability <= 1.0:
        raise ValueError("span_probability must be in [0, 1]")
    if span_length <= 0:
        raise ValueError("span_length must be positive")

    key = jax.random.PRNGKey(seed)
    key_point, key_span = jax.random.split(key)
    point_masked = jax.random.bernoulli(
        key_point,
        probability,
        shape=(batch_size, time_steps),
    )
    span_starts = jax.random.bernoulli(
        key_span,
        span_probability,
        shape=(batch_size, time_steps),
    )
    span_offsets = jnp.arange(span_length)
    span_indices = jnp.arange(time_steps)[:, None] - span_offsets[None, :]
    span_indices = jnp.clip(span_indices, 0, time_steps - 1)
    span_masked = jnp.any(jnp.take(span_starts, span_indices, axis=1), axis=-1)
    return ~(point_masked | span_masked)


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


def nonlinear_grid_filter_shape_diagnostics(
    batch: EpisodeBatch,
    params: LinearGaussianParams,
    *,
    data_config: NonlinearDataConfig,
    grid_config: GridReferenceConfig = GridReferenceConfig(),
    peak_fraction: float = 0.1,
) -> NonlinearReferenceShapeOutputs:
    """Return grid posterior shape diagnostics for the nonlinear reference filter."""

    if grid_config.num_grid < 3:
        raise ValueError("num_grid must be at least 3")
    if grid_config.grid_max <= grid_config.grid_min:
        raise ValueError("grid_max must be greater than grid_min")
    if peak_fraction <= 0.0:
        raise ValueError("peak_fraction must be positive")

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
        obs_mean = nonlinear_observation_mean(
            grid[None, :],
            x_t[:, None],
            data_config.observation,
        )
        filter_log_mass = pred_log_mass + _normal_log_prob(y_t[:, None], obs_mean, params.r)
        filter_log_mass = filter_log_mass - jsp.special.logsumexp(
            filter_log_mass,
            axis=1,
            keepdims=True,
        )
        filter_mass = jnp.exp(filter_log_mass)
        entropy = -jnp.sum(filter_mass * filter_log_mass, axis=1)
        max_mass = jnp.max(filter_mass, axis=1)
        local_peak = (
            (filter_mass[:, 1:-1] > filter_mass[:, :-2])
            & (filter_mass[:, 1:-1] >= filter_mass[:, 2:])
            & (filter_mass[:, 1:-1] >= peak_fraction * max_mass[:, None])
        )
        cdf = jnp.cumsum(filter_mass, axis=1)
        lower_index = jnp.argmax(cdf >= 0.05, axis=1)
        upper_index = jnp.argmax(cdf >= 0.95, axis=1)
        credible_width_90 = grid[upper_index] - grid[lower_index]
        outputs = (
            entropy,
            entropy / jnp.log(grid_config.num_grid),
            jnp.sum(local_peak, axis=1),
            max_mass,
            credible_width_90,
        )
        return filter_log_mass, outputs

    init_log_mass = jnp.broadcast_to(prior_log_mass[None, :], (batch.x.shape[0], grid.shape[0]))
    _, outputs = jax.lax.scan(step, init_log_mass, (x_tb, y_tb))
    return NonlinearReferenceShapeOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def nonlinear_grid_filter_masses(
    batch: EpisodeBatch,
    params: LinearGaussianParams,
    *,
    data_config: NonlinearDataConfig,
    grid_config: GridReferenceConfig = GridReferenceConfig(),
) -> NonlinearReferenceGridOutputs:
    """Return full grid posterior masses for reference-only diagnostics."""

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
        obs_mean = nonlinear_observation_mean(
            grid[None, :],
            x_t[:, None],
            data_config.observation,
        )
        filter_log_mass = pred_log_mass + _normal_log_prob(y_t[:, None], obs_mean, params.r)
        filter_log_mass = filter_log_mass - jsp.special.logsumexp(
            filter_log_mass,
            axis=1,
            keepdims=True,
        )
        return filter_log_mass, jnp.exp(filter_log_mass)

    init_log_mass = jnp.broadcast_to(prior_log_mass[None, :], (batch.x.shape[0], grid.shape[0]))
    _, mass_tbg = jax.lax.scan(step, init_log_mass, (x_tb, y_tb))
    return NonlinearReferenceGridOutputs(
        grid=grid,
        filter_mass=jnp.swapaxes(mass_tbg, 0, 1),
    )


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


def nonlinear_preassimilation_log_prob_y(
    prev_filter_mean: jax.Array,
    prev_filter_var: jax.Array,
    x: jax.Array,
    y: jax.Array,
    params: LinearGaussianParams,
    *,
    observation: str = "x_sine",
    num_points: int = 32,
) -> jax.Array:
    """Return `log p(y_t | q^F_{t-1}, x_t)` using Gauss-Hermite quadrature."""

    if observation != "x_sine":
        raise ValueError(f"Unsupported nonlinear predictive likelihood: {observation}")
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    nodes_np, weights_np = np.polynomial.hermite.hermgauss(num_points)
    nodes = jnp.asarray(nodes_np, dtype=prev_filter_mean.dtype)
    log_weights = jnp.log(jnp.asarray(weights_np, dtype=prev_filter_mean.dtype))
    pred_state_var = prev_filter_var + params.q
    z = prev_filter_mean[..., None] + jnp.sqrt(2.0 * pred_state_var[..., None]) * nodes
    obs_mean = x[..., None] * jnp.sin(z)
    log_likelihood = _normal_log_prob(y[..., None], obs_mean, params.r)
    return jsp.special.logsumexp(log_weights + log_likelihood, axis=-1) - 0.5 * jnp.log(jnp.pi)


def run_nonlinear_structured_mlp_filter(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    observation: str = "x_sine",
    min_var: float = 1e-6,
    y_observed: jax.Array | None = None,
) -> StructuredMLPOutputs:
    """Run an EKF-residualized strict nonlinear filter over batch-major episodes."""

    x_bt = batch.x.T
    y_bt = batch.y.T
    if y_observed is None:
        y_observed = jnp.ones_like(batch.y, dtype=bool)
    observed_bt = y_observed.T

    def step(carry: tuple[jax.Array, jax.Array], obs: tuple[jax.Array, jax.Array, jax.Array]):
        prev_mean, prev_var = carry
        x_t, y_t, observed_t = obs
        update_outputs = nonlinear_structured_mlp_step(
            mlp_params,
            prev_mean,
            prev_var,
            x_t,
            y_t,
            state_params,
            observation=observation,
            min_var=min_var,
        )
        transition_outputs = transition_prediction_outputs(prev_mean, prev_var, state_params)
        outputs = _where_outputs(observed_t, update_outputs, transition_outputs)
        return (outputs.filter_mean, outputs.filter_var), outputs

    batch_size = batch.x.shape[0]
    init = (
        jnp.full((batch_size,), state_params.m0, dtype=jnp.float64),
        jnp.full((batch_size,), state_params.p0, dtype=jnp.float64),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt, observed_bt))
    return StructuredMLPOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def run_nonlinear_structured_mlp_teacher_forced(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    target_filter_mean: jax.Array,
    target_filter_var: jax.Array,
    *,
    observation: str = "x_sine",
    min_var: float = 1e-6,
) -> StructuredMLPOutputs:
    """Run nonlinear structured updates using target previous beliefs as inputs."""

    initial_mean = jnp.full((batch.x.shape[0], 1), state_params.m0, dtype=jnp.float64)
    initial_var = jnp.full((batch.x.shape[0], 1), state_params.p0, dtype=jnp.float64)
    prev_mean = jnp.concatenate((initial_mean, target_filter_mean[:, :-1]), axis=1)
    prev_var = jnp.concatenate((initial_var, target_filter_var[:, :-1]), axis=1)
    return nonlinear_structured_mlp_step(
        mlp_params,
        prev_mean,
        prev_var,
        batch.x,
        batch.y,
        state_params,
        observation=observation,
        min_var=min_var,
    )


def nonlinear_structured_mlp_step(
    mlp_params: dict[str, jax.Array],
    prev_mean: jax.Array,
    prev_var: jax.Array,
    x_t: jax.Array,
    y_t: jax.Array,
    state_params: LinearGaussianParams,
    *,
    observation: str = "x_sine",
    min_var: float = 1e-6,
) -> StructuredMLPOutputs:
    """Compute one EKF-residualized nonlinear filter update."""

    if observation != "x_sine":
        raise ValueError(f"Unsupported structured nonlinear observation: {observation}")

    features = _mlp_features(prev_mean, prev_var, x_t, y_t, state_params)
    hidden = jnp.tanh(features @ mlp_params["w1"] + mlp_params["b1"])
    raw = hidden @ mlp_params["w2"] + mlp_params["b2"]
    pred_var = prev_var + state_params.q
    obs_mean = x_t * jnp.sin(prev_mean)
    obs_jacobian = x_t * jnp.cos(prev_mean)
    innovation = y_t - obs_mean
    innovation_var = obs_jacobian**2 * pred_var + state_params.r
    base_gain = pred_var * obs_jacobian / innovation_var
    gain_scale = 2.0 * jax.nn.sigmoid(raw[..., 0])
    filter_mean = prev_mean + gain_scale * base_gain * innovation
    base_filter_var = pred_var * state_params.r / innovation_var
    filter_var = base_filter_var * jnp.exp(jnp.clip(raw[..., 1], -5.0, 5.0)) + min_var
    backward_a = raw[..., 2]
    return StructuredMLPOutputs(
        filter_mean=filter_mean,
        filter_var=filter_var,
        backward_a=backward_a,
        backward_b=prev_mean - backward_a * filter_mean + raw[..., 3],
        backward_var=jax.nn.softplus(raw[..., 4]) + min_var,
    )


def transition_prediction_outputs(
    prev_mean: jax.Array,
    prev_var: jax.Array,
    state_params: LinearGaussianParams,
) -> StructuredMLPOutputs:
    """Return the exact random-walk transition update for a masked measurement."""

    pred_var = prev_var + state_params.q
    backward_a = prev_var / pred_var
    return StructuredMLPOutputs(
        filter_mean=prev_mean,
        filter_var=pred_var,
        backward_a=backward_a,
        backward_b=prev_mean - backward_a * prev_mean,
        backward_var=prev_var * state_params.q / pred_var,
    )


def _where_outputs(
    condition: jax.Array,
    true_outputs: StructuredMLPOutputs,
    false_outputs: StructuredMLPOutputs,
) -> StructuredMLPOutputs:
    condition = condition.astype(bool)
    return StructuredMLPOutputs(
        *(
            jnp.where(condition, true_value, false_value)
            for true_value, false_value in zip(true_outputs, false_outputs, strict=True)
        )
    )


def _normal_log_prob(
    value: jax.Array, mean: jax.Array | float, var: jax.Array | float
) -> jax.Array:
    return -0.5 * (LOG_2PI + jnp.log(var) + (value - mean) ** 2 / var)


def _time_major_to_batch_major(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, 0, 1)
