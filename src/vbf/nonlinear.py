"""Nonlinear scalar state-space benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.scipy as jsp
import numpy as np

from vbf.dtypes import DEFAULT_DTYPE  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from vbf.data import (  # noqa: E402
    EpisodeBatch,
    LinearGaussianDataConfig,
    LinearGaussianParams,
    make_observation_covariates,
)
from vbf.models.cells import (  # noqa: E402
    GaussianMixtureMLPOutputs,
    StructuredMLPOutputs,
    _mlp_features,
    mixture_mean_and_var,
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


class NonlinearReferenceShapeOutputs(NamedTuple):
    entropy: jax.Array
    normalized_entropy: jax.Array
    peak_count: jax.Array
    max_mass: jax.Array
    credible_width_90: jax.Array


class NonlinearReferenceGridOutputs(NamedTuple):
    grid: jax.Array
    filter_mass: jax.Array


class NonlinearParticleFilterOutputs(NamedTuple):
    filter_mean: jax.Array
    filter_var: jax.Array
    predictive_mean: jax.Array
    predictive_var: jax.Array
    predictive_log_prob_y: jax.Array
    filter_log_prob_z: jax.Array
    mean_ess: jax.Array


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
    key_z0, key_w, key_v, key_t = jax.random.split(key, 4)
    x = make_observation_covariates(data_config, jax.random.fold_in(key, 42))
    z_initial = params.m0 + jnp.sqrt(params.p0) * jax.random.normal(
        key_z0,
        shape=(config.batch_size,),
        dtype=DEFAULT_DTYPE,
    )
    innovations = jnp.sqrt(params.q) * jax.random.normal(
        key_w,
        shape=(config.batch_size, config.time_steps),
        dtype=DEFAULT_DTYPE,
    )
    z = z_initial[:, None] + jnp.cumsum(innovations, axis=1)
    y_mean = nonlinear_observation_mean(z, x, config.observation)
    if observation_noise_family(config.observation) == "student_t":
        df = observation_student_t_df(config.observation)
        normal = jax.random.normal(
            key_v,
            shape=(config.batch_size, config.time_steps),
            dtype=DEFAULT_DTYPE,
        )
        chi2 = 2.0 * jax.random.gamma(
            key_t,
            0.5 * df,
            shape=(config.batch_size, config.time_steps),
            dtype=DEFAULT_DTYPE,
        )
        y = y_mean + jnp.sqrt(params.r) * normal / jnp.sqrt(chi2 / df)
    else:
        y = y_mean + jnp.sqrt(
            nonlinear_observation_var(z, x, params, config.observation)
        ) * jax.random.normal(
            key_v,
            shape=(config.batch_size, config.time_steps),
            dtype=DEFAULT_DTYPE,
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
        dtype=DEFAULT_DTYPE,
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
        obs_var = nonlinear_observation_var(
            grid[None, :],
            x_t[:, None],
            params,
            data_config.observation,
        )
        predictive_var = jnp.sum(
            pred_mass * ((obs_mean - predictive_mean[:, None]) ** 2 + obs_var),
            axis=1,
        )
        filter_log_mass = pred_log_mass + nonlinear_observation_log_prob(
            y_t[:, None],
            grid[None, :],
            x_t[:, None],
            params,
            data_config.observation,
        )
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
        dtype=DEFAULT_DTYPE,
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
        filter_log_mass = pred_log_mass + nonlinear_observation_log_prob(
            y_t[:, None],
            grid[None, :],
            x_t[:, None],
            params,
            data_config.observation,
        )
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
        dtype=DEFAULT_DTYPE,
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
        filter_log_mass = pred_log_mass + nonlinear_observation_log_prob(
            y_t[:, None],
            grid[None, :],
            x_t[:, None],
            params,
            data_config.observation,
        )
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


def nonlinear_bootstrap_particle_filter(
    batch: EpisodeBatch,
    params: LinearGaussianParams,
    *,
    data_config: NonlinearDataConfig,
    num_particles: int = 128,
    seed: int = 0,
    kde_bandwidth_scale: float = 1.0,
) -> NonlinearParticleFilterOutputs:
    """Run a bootstrap particle filter for nonlinear reference diagnostics."""

    if num_particles <= 0:
        raise ValueError("num_particles must be positive")
    if kde_bandwidth_scale <= 0.0:
        raise ValueError("kde_bandwidth_scale must be positive")

    batch_size = batch.x.shape[0]
    key = jax.random.PRNGKey(seed)
    init_particles = params.m0 + jnp.sqrt(params.p0) * jax.random.normal(
        key,
        shape=(batch_size, num_particles),
        dtype=batch.x.dtype,
    )

    x_tb = batch.x.T
    y_tb = batch.y.T
    z_tb = batch.z.T
    step_keys = jax.random.split(jax.random.fold_in(key, 1), batch.x.shape[1])

    def step(
        particles: jax.Array,
        obs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        x_t, y_t, z_t_true, step_key = obs
        transition_key, resample_key = jax.random.split(step_key)
        pred_particles = particles + jnp.sqrt(params.q) * jax.random.normal(
            transition_key,
            shape=particles.shape,
            dtype=particles.dtype,
        )
        obs_mean = nonlinear_observation_mean(
            pred_particles,
            x_t[:, None],
            data_config.observation,
        )
        predictive_mean = jnp.mean(obs_mean, axis=1)
        obs_var = nonlinear_observation_var(
            pred_particles,
            x_t[:, None],
            params,
            data_config.observation,
        )
        predictive_var = jnp.mean(
            (obs_mean - predictive_mean[:, None]) ** 2 + obs_var,
            axis=1,
        )
        log_likelihood = nonlinear_observation_log_prob(
            y_t[:, None],
            pred_particles,
            x_t[:, None],
            params,
            data_config.observation,
        )
        predictive_log_prob_y = jsp.special.logsumexp(log_likelihood, axis=1) - jnp.log(
            num_particles
        )
        log_weights = log_likelihood - jsp.special.logsumexp(
            log_likelihood,
            axis=1,
            keepdims=True,
        )
        weights = jnp.exp(log_weights)
        filter_mean = jnp.sum(weights * pred_particles, axis=1)
        filter_var = jnp.sum(weights * (pred_particles - filter_mean[:, None]) ** 2, axis=1)
        bandwidth_var = _particle_kde_bandwidth_var(pred_particles, weights, kde_bandwidth_scale)
        filter_log_prob_z = _particle_kde_log_prob(
            z_t_true,
            pred_particles,
            weights,
            bandwidth_var,
        )
        ess = 1.0 / jnp.sum(weights**2, axis=1)
        resample_keys = jax.random.split(resample_key, batch_size)
        indices = jax.vmap(
            lambda row_key, row_logits: jax.random.categorical(
                row_key,
                logits=row_logits,
                shape=(num_particles,),
            )
        )(resample_keys, log_weights)
        next_particles = jnp.take_along_axis(pred_particles, indices, axis=1)
        return next_particles, (
            filter_mean,
            filter_var,
            predictive_mean,
            predictive_var,
            predictive_log_prob_y,
            filter_log_prob_z,
            ess,
        )

    _, outputs = jax.lax.scan(step, init_particles, (x_tb, y_tb, z_tb, step_keys))
    batch_major = tuple(_time_major_to_batch_major(item) for item in outputs)
    return NonlinearParticleFilterOutputs(
        *batch_major[:-1],
        mean_ess=jnp.mean(batch_major[-1]),
    )


def nonlinear_observation_mean(
    z: jax.Array,
    x: jax.Array,
    observation: str = "x_sine",
) -> jax.Array:
    """Return `h(z_t, x_t)` for supported nonlinear observation models."""

    if observation in {"x_sine", "student_t"}:
        return x * jnp.sin(z)
    if observation == "sine_product":
        return jnp.sin(x * z)
    if observation == "x_tanh":
        return x * jnp.tanh(z)
    if observation == "x_cubic":
        return x * (z**3) / 25.0
    if observation == "x_quadratic_signed":
        return x * jnp.sign(z) * (z**2) / 8.0
    if observation == "heteroskedastic_gaussian":
        return x * jnp.tanh(z)
    raise ValueError(f"Unsupported nonlinear observation: {observation}")


def observation_noise_family(observation: str) -> str:
    """Return the scalar observation-noise family for a benchmark name."""

    if observation == "student_t":
        return "student_t"
    if observation in {
        "x_sine",
        "sine_product",
        "x_tanh",
        "x_cubic",
        "x_quadratic_signed",
        "heteroskedastic_gaussian",
    }:
        return "gaussian"
    raise ValueError(f"Unsupported nonlinear observation: {observation}")


def observation_student_t_df(observation: str) -> float:
    """Return degrees of freedom for heavy-tailed observation benchmarks."""

    if observation == "student_t":
        return 3.0
    raise ValueError(f"Observation does not use Student-t noise: {observation}")


def nonlinear_observation_var(
    z: jax.Array,
    x: jax.Array,
    params: LinearGaussianParams,
    observation: str = "x_sine",
) -> jax.Array:
    """Return conditional observation variance for supported benchmarks."""

    base = jnp.asarray(params.r, dtype=z.dtype) + jnp.zeros_like(z + x)
    if observation == "heteroskedastic_gaussian":
        return base * (0.5 + jax.nn.softplus(0.5 * z))
    if observation == "student_t":
        df = observation_student_t_df(observation)
        return base * df / (df - 2.0)
    observation_noise_family(observation)
    return base


def nonlinear_observation_jacobian(
    z: jax.Array,
    x: jax.Array,
    observation: str = "x_sine",
) -> jax.Array:
    """Return elementwise derivative of observation mean with respect to `z`."""

    def scalar_jacobian(z_scalar: jax.Array, x_scalar: jax.Array) -> jax.Array:
        return jax.grad(
            lambda z_value: nonlinear_observation_mean(z_value, x_scalar, observation)
        )(z_scalar)

    return jax.vmap(scalar_jacobian)(jnp.ravel(z), jnp.ravel(x)).reshape(z.shape)


def nonlinear_observation_log_prob(
    y: jax.Array,
    z: jax.Array,
    x: jax.Array,
    params: LinearGaussianParams,
    observation: str = "x_sine",
) -> jax.Array:
    """Return `log p(y_t | z_t, x_t)` for supported nonlinear observations."""

    mean = nonlinear_observation_mean(z, x, observation)
    if observation_noise_family(observation) == "student_t":
        df = observation_student_t_df(observation)
        scale = jnp.sqrt(jnp.asarray(params.r, dtype=z.dtype))
        standardized = (y - mean) / scale
        return (
            jsp.special.gammaln(0.5 * (df + 1.0))
            - jsp.special.gammaln(0.5 * df)
            - 0.5 * jnp.log(df * jnp.pi)
            - jnp.log(scale)
            - 0.5 * (df + 1.0) * jnp.log1p((standardized**2) / df)
        )
    return _normal_log_prob(
        y,
        mean,
        nonlinear_observation_var(z, x, params, observation),
    )


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

    initial_mean = jnp.full((filter_mean.shape[0], 1), params.m0, dtype=filter_mean.dtype)
    initial_var = jnp.full((filter_var.shape[0], 1), params.p0, dtype=filter_var.dtype)
    prev_mean = jnp.concatenate((initial_mean, filter_mean[:, :-1]), axis=1)
    prev_var = jnp.concatenate((initial_var, filter_var[:, :-1]), axis=1)
    pred_state_var = prev_var + params.q
    if observation != "x_sine":
        nodes, log_weights = _hermgauss(32, filter_mean.dtype)
        z = prev_mean[..., None] + jnp.sqrt(2.0 * pred_state_var[..., None]) * nodes
        obs_mean = nonlinear_observation_mean(z, x[..., None], observation)
        weights = jnp.exp(log_weights - 0.5 * jnp.log(jnp.pi))
        predictive_mean = jnp.sum(weights * obs_mean, axis=-1)
        obs_var = nonlinear_observation_var(z, x[..., None], params, observation)
        predictive_var = jnp.sum(
            weights * ((obs_mean - predictive_mean[..., None]) ** 2 + obs_var),
            axis=-1,
        )
        return predictive_mean, predictive_var

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

    if num_points <= 0:
        raise ValueError("num_points must be positive")

    nodes, log_weights = _hermgauss(num_points, prev_filter_mean.dtype)
    pred_state_var = prev_filter_var + params.q
    z = prev_filter_mean[..., None] + jnp.sqrt(2.0 * pred_state_var[..., None]) * nodes
    log_likelihood = nonlinear_observation_log_prob(
        y[..., None],
        z,
        x[..., None],
        params,
        observation,
    )
    return jsp.special.logsumexp(log_weights + log_likelihood, axis=-1) - 0.5 * jnp.log(jnp.pi)


def nonlinear_preupdate_predictive_normalizer_loss(
    outputs: StructuredMLPOutputs | GaussianMixtureMLPOutputs,
    batch: EpisodeBatch,
    params: LinearGaussianParams,
    *,
    observation: str = "x_sine",
    num_points: int = 32,
    min_var: float = 1e-6,
    stop_filter_gradient: bool = False,
) -> jax.Array:
    """Return `-log p_q(y_t | D_<t, x_t)` for the carried transition prediction.

    For Gaussian-mixture filters this is an exact mixture-of-quadrature
    normalizer: each previous component is propagated through the random-walk
    transition and scored under the nonlinear observation likelihood.
    """

    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if stop_filter_gradient:
        outputs = jax.tree_util.tree_map(jax.lax.stop_gradient, outputs)
    if isinstance(outputs, GaussianMixtureMLPOutputs):
        log_prob = _mixture_preupdate_log_prob_y(
            outputs,
            batch.x,
            batch.y,
            params,
            observation=observation,
            num_points=num_points,
            min_var=min_var,
        )
    else:
        prev_mean, prev_var = _previous_filter_beliefs(
            outputs.filter_mean,
            outputs.filter_var,
            params,
        )
        log_prob = nonlinear_preassimilation_log_prob_y(
            prev_mean,
            prev_var,
            batch.x,
            batch.y,
            params,
            observation=observation,
            num_points=num_points,
        )
    return -log_prob


def nonlinear_tilted_projection_loss(
    outputs: StructuredMLPOutputs | GaussianMixtureMLPOutputs,
    batch: EpisodeBatch,
    params: LinearGaussianParams,
    *,
    observation: str = "x_sine",
    num_points: int = 32,
    likelihood_power: float = 1.0,
    divergence: str = "forward_kl",
    alpha: float = 0.5,
    min_var: float = 1e-6,
    stop_target: bool = True,
) -> jax.Array:
    """Forward-KL style ADF projection loss from the local tilted posterior.

    The target is generated only from the carried belief, transition model, and
    current observation:

    `tilde p(z_t) proportional p(y_t | z_t, x_t) int p(z_t | z_tm1) q^F_{t-1}(z_tm1) dz_tm1`.

    The returned array has batch-time shape.
    """

    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if likelihood_power <= 0.0:
        raise ValueError("likelihood_power must be positive")
    if divergence not in {"forward_kl", "alpha"}:
        raise ValueError("divergence must be one of: forward_kl, alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if isinstance(outputs, GaussianMixtureMLPOutputs):
        return _mixture_tilted_projection_loss(
            outputs,
            batch,
            params,
            observation=observation,
            num_points=num_points,
            likelihood_power=likelihood_power,
            divergence=divergence,
            alpha=alpha,
            min_var=min_var,
            stop_target=stop_target,
        )
    return _gaussian_tilted_projection_loss(
        outputs,
        batch,
        params,
        observation=observation,
        num_points=num_points,
        likelihood_power=likelihood_power,
        divergence=divergence,
        alpha=alpha,
        min_var=min_var,
        stop_target=stop_target,
    )


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
        jnp.full((batch_size,), state_params.m0, dtype=DEFAULT_DTYPE),
        jnp.full((batch_size,), state_params.p0, dtype=DEFAULT_DTYPE),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt, observed_bt))
    return StructuredMLPOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def run_nonlinear_structured_mixture_mlp_filter(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    num_components: int,
    observation: str = "x_sine",
    min_var: float = 1e-6,
    y_observed: jax.Array | None = None,
) -> GaussianMixtureMLPOutputs:
    """Run an EKF-residualized strict Gaussian-mixture nonlinear filter."""

    x_bt = batch.x.T
    y_bt = batch.y.T
    if y_observed is None:
        y_observed = jnp.ones_like(batch.y, dtype=bool)
    observed_bt = y_observed.T

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        obs: tuple[jax.Array, jax.Array, jax.Array],
    ):
        prev_weights, prev_mean, prev_var = carry
        x_t, y_t, observed_t = obs
        update_outputs = nonlinear_structured_mixture_mlp_step(
            mlp_params,
            prev_weights,
            prev_mean,
            prev_var,
            x_t,
            y_t,
            state_params,
            num_components=num_components,
            observation=observation,
            min_var=min_var,
        )
        transition_outputs = mixture_transition_prediction_outputs(
            prev_weights,
            prev_mean,
            prev_var,
            state_params,
        )
        outputs = _where_outputs(observed_t, update_outputs, transition_outputs)
        return (outputs.filter_weights, outputs.component_mean, outputs.component_var), outputs

    batch_size = batch.x.shape[0]
    init = (
        jnp.full((batch_size, num_components), 1.0 / num_components, dtype=DEFAULT_DTYPE),
        jnp.full((batch_size, num_components), state_params.m0, dtype=DEFAULT_DTYPE),
        jnp.full((batch_size, num_components), state_params.p0, dtype=DEFAULT_DTYPE),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt, observed_bt))
    return GaussianMixtureMLPOutputs(*(_time_major_to_batch_major(item) for item in outputs))


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

    initial_mean = jnp.full((batch.x.shape[0], 1), state_params.m0, dtype=DEFAULT_DTYPE)
    initial_var = jnp.full((batch.x.shape[0], 1), state_params.p0, dtype=DEFAULT_DTYPE)
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

    features = _mlp_features(prev_mean, prev_var, x_t, y_t, state_params)
    hidden = jnp.tanh(features @ mlp_params["w1"] + mlp_params["b1"])
    raw = hidden @ mlp_params["w2"] + mlp_params["b2"]
    pred_var = prev_var + state_params.q
    obs_mean = nonlinear_observation_mean(prev_mean, x_t, observation)
    obs_jacobian = nonlinear_observation_jacobian(prev_mean, x_t, observation)
    obs_var = nonlinear_observation_var(prev_mean, x_t, state_params, observation)
    innovation = y_t - obs_mean
    innovation_var = obs_jacobian**2 * pred_var + obs_var
    base_gain = pred_var * obs_jacobian / innovation_var
    gain_scale = 2.0 * jax.nn.sigmoid(raw[..., 0])
    filter_mean = prev_mean + gain_scale * base_gain * innovation
    base_filter_var = pred_var * obs_var / innovation_var
    filter_var = base_filter_var * jnp.exp(jnp.clip(raw[..., 1], -5.0, 5.0)) + min_var
    backward_a = raw[..., 2]
    return StructuredMLPOutputs(
        filter_mean=filter_mean,
        filter_var=filter_var,
        backward_a=backward_a,
        backward_b=prev_mean - backward_a * filter_mean + raw[..., 3],
        backward_var=jax.nn.softplus(raw[..., 4]) + min_var,
    )


def nonlinear_structured_mixture_mlp_step(
    mlp_params: dict[str, jax.Array],
    prev_weights: jax.Array,
    prev_component_mean: jax.Array,
    prev_component_var: jax.Array,
    x_t: jax.Array,
    y_t: jax.Array,
    state_params: LinearGaussianParams,
    *,
    num_components: int,
    observation: str = "x_sine",
    min_var: float = 1e-6,
) -> GaussianMixtureMLPOutputs:
    """Compute one EKF-residualized Gaussian-mixture nonlinear update."""

    prev_mean, prev_var = mixture_mean_and_var(
        prev_weights,
        prev_component_mean,
        prev_component_var,
    )
    features = _mlp_features(prev_mean, prev_var, x_t, y_t, state_params)
    hidden = jnp.tanh(features @ mlp_params["w1"] + mlp_params["b1"])
    raw = hidden @ mlp_params["w2"] + mlp_params["b2"]
    raw = jnp.reshape(raw, raw.shape[:-1] + (num_components, 6))
    pred_var = prev_var + state_params.q
    obs_mean = nonlinear_observation_mean(prev_mean, x_t, observation)
    obs_jacobian = nonlinear_observation_jacobian(prev_mean, x_t, observation)
    obs_var = nonlinear_observation_var(prev_mean, x_t, state_params, observation)
    innovation = y_t - obs_mean
    innovation_var = obs_jacobian**2 * pred_var + obs_var
    base_gain = pred_var * obs_jacobian / innovation_var
    gain_scale = 2.0 * jax.nn.sigmoid(raw[..., 1])
    component_mean = prev_mean[..., None] + gain_scale * base_gain[..., None] * innovation[
        ..., None
    ]
    base_filter_var = pred_var * obs_var / innovation_var
    component_var = base_filter_var[..., None] * jnp.exp(jnp.clip(raw[..., 2], -5.0, 5.0))
    component_var = component_var + min_var
    backward_a = raw[..., 3]
    return GaussianMixtureMLPOutputs(
        filter_weights=jax.nn.softmax(raw[..., 0], axis=-1),
        component_mean=component_mean,
        component_var=component_var,
        backward_a=backward_a,
        backward_b=prev_mean[..., None] - backward_a * component_mean + raw[..., 4],
        backward_var=jax.nn.softplus(raw[..., 5]) + min_var,
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


def mixture_transition_prediction_outputs(
    prev_weights: jax.Array,
    prev_component_mean: jax.Array,
    prev_component_var: jax.Array,
    state_params: LinearGaussianParams,
) -> GaussianMixtureMLPOutputs:
    """Return componentwise random-walk transition update for masked measurements."""

    pred_var = prev_component_var + state_params.q
    backward_a = prev_component_var / pred_var
    return GaussianMixtureMLPOutputs(
        filter_weights=prev_weights,
        component_mean=prev_component_mean,
        component_var=pred_var,
        backward_a=backward_a,
        backward_b=prev_component_mean - backward_a * prev_component_mean,
        backward_var=prev_component_var * state_params.q / pred_var,
    )


def _where_outputs(
    condition: jax.Array,
    true_outputs: StructuredMLPOutputs | GaussianMixtureMLPOutputs,
    false_outputs: StructuredMLPOutputs | GaussianMixtureMLPOutputs,
) -> StructuredMLPOutputs | GaussianMixtureMLPOutputs:
    condition = condition.astype(bool)
    return type(true_outputs)(
        *(
            jnp.where(
                _expand_condition_for_value(condition, true_value),
                true_value,
                false_value,
            )
            for true_value, false_value in zip(true_outputs, false_outputs, strict=True)
        )
    )


def _expand_condition_for_value(condition: jax.Array, value: jax.Array) -> jax.Array:
    extra_dims = value.ndim - condition.ndim
    if extra_dims <= 0:
        return condition
    return jnp.reshape(condition, condition.shape + (1,) * extra_dims)


def _normal_log_prob(
    value: jax.Array, mean: jax.Array | float, var: jax.Array | float
) -> jax.Array:
    return -0.5 * (LOG_2PI + jnp.log(var) + (value - mean) ** 2 / var)


def _gaussian_tilted_projection_loss(
    outputs: StructuredMLPOutputs,
    batch: EpisodeBatch,
    params: LinearGaussianParams,
    *,
    observation: str,
    num_points: int,
    likelihood_power: float,
    divergence: str,
    alpha: float,
    min_var: float,
    stop_target: bool,
) -> jax.Array:
    prev_mean, prev_var = _previous_filter_beliefs(
        outputs.filter_mean,
        outputs.filter_var,
        params,
    )
    nodes, log_weights = _hermgauss(num_points, outputs.filter_mean.dtype)
    pred_var = prev_var + params.q
    z = prev_mean[..., None] + jnp.sqrt(2.0 * pred_var[..., None]) * nodes
    log_likelihood = nonlinear_observation_log_prob(
        batch.y[..., None],
        z,
        batch.x[..., None],
        params,
        observation,
    )
    log_base_weights = log_weights - 0.5 * jnp.log(jnp.pi)
    log_target_normalizer = jsp.special.logsumexp(
        log_base_weights + likelihood_power * log_likelihood,
        axis=-1,
        keepdims=True,
    )
    log_target_weights = log_base_weights + likelihood_power * log_likelihood
    log_target_weights = log_target_weights - log_target_normalizer
    log_target_density = (
        _normal_log_prob(z, prev_mean[..., None], pred_var[..., None])
        + likelihood_power * log_likelihood
        - log_target_normalizer
    )
    log_target_weights = log_target_weights - jsp.special.logsumexp(
        log_target_weights,
        axis=-1,
        keepdims=True,
    )
    target_weights = jnp.exp(log_target_weights)
    if stop_target:
        z = jax.lax.stop_gradient(z)
        target_weights = jax.lax.stop_gradient(target_weights)
        log_target_density = jax.lax.stop_gradient(log_target_density)
        log_base_weights = jax.lax.stop_gradient(log_base_weights)
        pred_log_prob = jax.lax.stop_gradient(
            _normal_log_prob(z, prev_mean[..., None], pred_var[..., None])
        )
    else:
        pred_log_prob = _normal_log_prob(z, prev_mean[..., None], pred_var[..., None])
    filter_var = jnp.maximum(outputs.filter_var, min_var)
    log_q = _normal_log_prob(z, outputs.filter_mean[..., None], filter_var[..., None])
    if divergence == "forward_kl":
        return -jnp.sum(target_weights * log_q, axis=-1)
    log_affinity = jsp.special.logsumexp(
        log_base_weights
        + alpha * log_target_density
        + (1.0 - alpha) * log_q
        - pred_log_prob,
        axis=-1,
    )
    return -log_affinity / (1.0 - alpha)


def _mixture_tilted_projection_loss(
    outputs: GaussianMixtureMLPOutputs,
    batch: EpisodeBatch,
    params: LinearGaussianParams,
    *,
    observation: str,
    num_points: int,
    likelihood_power: float,
    divergence: str,
    alpha: float,
    min_var: float,
    stop_target: bool,
) -> jax.Array:
    prev_weights, prev_mean, prev_var = _previous_mixture_filter_beliefs(outputs, params)
    nodes, log_weights = _hermgauss(num_points, outputs.component_mean.dtype)
    pred_var = prev_var + params.q
    z = prev_mean[..., None] + jnp.sqrt(2.0 * pred_var[..., None]) * nodes
    log_likelihood = nonlinear_observation_log_prob(
        batch.y[..., None, None],
        z,
        batch.x[..., None, None],
        params,
        observation,
    )
    log_prev_weights = jnp.log(jnp.clip(prev_weights, min_var))
    log_pred_density = jsp.special.logsumexp(
        log_prev_weights[..., None, None, :]
        + _normal_log_prob(
            z[..., None],
            prev_mean[..., None, None, :],
            pred_var[..., None, None, :],
        ),
        axis=-1,
    )
    log_base_weights = log_prev_weights[..., None] + log_weights - 0.5 * jnp.log(jnp.pi)
    log_target_normalizer = jsp.special.logsumexp(
        log_base_weights + likelihood_power * log_likelihood,
        axis=(-2, -1),
        keepdims=True,
    )
    log_target_weights = log_base_weights + likelihood_power * log_likelihood
    log_target_weights = log_target_weights - log_target_normalizer
    log_target_density = (
        log_pred_density + likelihood_power * log_likelihood - log_target_normalizer
    )
    log_target_weights = log_target_weights - jsp.special.logsumexp(
        log_target_weights,
        axis=(-2, -1),
        keepdims=True,
    )
    target_weights = jnp.exp(log_target_weights)
    if stop_target:
        z = jax.lax.stop_gradient(z)
        target_weights = jax.lax.stop_gradient(target_weights)
        log_base_weights = jax.lax.stop_gradient(log_base_weights)
        log_target_density = jax.lax.stop_gradient(log_target_density)
        log_pred_density = jax.lax.stop_gradient(log_pred_density)

    output_weights = jnp.maximum(outputs.filter_weights, min_var)
    output_weights = output_weights / jnp.sum(output_weights, axis=-1, keepdims=True)
    output_var = jnp.maximum(outputs.component_var, min_var)
    log_q = jsp.special.logsumexp(
        jnp.log(output_weights[..., None, None, :])
        + _normal_log_prob(
            z[..., None],
            outputs.component_mean[..., None, None, :],
            output_var[..., None, None, :],
        ),
        axis=-1,
    )
    if divergence == "forward_kl":
        return -jnp.sum(target_weights * log_q, axis=(-2, -1))
    log_affinity = jsp.special.logsumexp(
        log_base_weights
        + alpha * log_target_density
        + (1.0 - alpha) * log_q
        - log_pred_density,
        axis=(-2, -1),
    )
    return -log_affinity / (1.0 - alpha)


def _mixture_preupdate_log_prob_y(
    outputs: GaussianMixtureMLPOutputs,
    x: jax.Array,
    y: jax.Array,
    params: LinearGaussianParams,
    *,
    observation: str,
    num_points: int,
    min_var: float,
) -> jax.Array:
    prev_weights, prev_mean, prev_var = _previous_mixture_filter_beliefs(outputs, params)
    nodes, log_weights = _hermgauss(num_points, outputs.component_mean.dtype)
    pred_state_var = jnp.maximum(prev_var + params.q, min_var)
    z = prev_mean[..., None] + jnp.sqrt(2.0 * pred_state_var[..., None]) * nodes
    log_likelihood = nonlinear_observation_log_prob(
        y[..., None, None],
        z,
        x[..., None, None],
        params,
        observation,
    )
    component_log_prob = (
        jnp.log(jnp.clip(prev_weights, min_var))
        + jsp.special.logsumexp(log_weights + log_likelihood, axis=-1)
        - 0.5 * jnp.log(jnp.pi)
    )
    return jsp.special.logsumexp(component_log_prob, axis=-1)


def _previous_filter_beliefs(
    filter_mean: jax.Array,
    filter_var: jax.Array,
    params: LinearGaussianParams,
) -> tuple[jax.Array, jax.Array]:
    initial_mean = jnp.full((filter_mean.shape[0], 1), params.m0, dtype=filter_mean.dtype)
    initial_var = jnp.full((filter_var.shape[0], 1), params.p0, dtype=filter_var.dtype)
    return (
        jnp.concatenate((initial_mean, filter_mean[:, :-1]), axis=1),
        jnp.concatenate((initial_var, filter_var[:, :-1]), axis=1),
    )


def _previous_mixture_filter_beliefs(
    outputs: GaussianMixtureMLPOutputs,
    params: LinearGaussianParams,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_size = outputs.component_mean.shape[0]
    num_components = outputs.component_mean.shape[-1]
    initial_weights = jnp.full(
        (batch_size, 1, num_components),
        1.0 / num_components,
        dtype=outputs.filter_weights.dtype,
    )
    initial_mean = jnp.full(
        (batch_size, 1, num_components),
        params.m0,
        dtype=outputs.component_mean.dtype,
    )
    initial_var = jnp.full(
        (batch_size, 1, num_components),
        params.p0,
        dtype=outputs.component_var.dtype,
    )
    return (
        jnp.concatenate((initial_weights, outputs.filter_weights[:, :-1]), axis=1),
        jnp.concatenate((initial_mean, outputs.component_mean[:, :-1]), axis=1),
        jnp.concatenate((initial_var, outputs.component_var[:, :-1]), axis=1),
    )


def _hermgauss(num_points: int, dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(num_points)
    nodes = jnp.asarray(nodes_np, dtype=dtype)
    log_weights = jnp.log(jnp.asarray(weights_np, dtype=dtype))
    return nodes, log_weights


def _particle_kde_bandwidth_var(
    particles: jax.Array,
    weights: jax.Array,
    bandwidth_scale: float,
) -> jax.Array:
    mean = jnp.sum(weights * particles, axis=1)
    var = jnp.sum(weights * (particles - mean[:, None]) ** 2, axis=1)
    effective_n = 1.0 / jnp.sum(weights**2, axis=1)
    bandwidth = bandwidth_scale * 1.06 * jnp.sqrt(jnp.maximum(var, 1e-12)) * effective_n ** (-0.2)
    return jnp.maximum(bandwidth**2, 1e-8)


def _particle_kde_log_prob(
    value: jax.Array,
    particles: jax.Array,
    weights: jax.Array,
    bandwidth_var: jax.Array,
) -> jax.Array:
    return jsp.special.logsumexp(
        jnp.log(weights) + _normal_log_prob(value[:, None], particles, bandwidth_var[:, None]),
        axis=1,
    )


def _time_major_to_batch_major(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, 0, 1)
