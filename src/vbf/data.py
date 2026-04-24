"""Synthetic data generation for filtering experiments."""

from __future__ import annotations

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


@dataclass(frozen=True)
class LinearGaussianDataConfig:
    """Configuration for scalar dynamic regression episodes."""

    batch_size: int = 128
    time_steps: int = 96
    x_pattern: str = "sinusoidal"
    x_cycles: float = 3.0


@dataclass(frozen=True)
class LinearGaussianParams:
    """Scalar state-space parameters."""

    q: float = 0.1
    r: float = 0.1
    m0: float = 1.0
    p0: float = 10.0


@dataclass(frozen=True)
class EpisodeBatch:
    """Batch-major synthetic episodes."""

    x: jax.Array
    y: jax.Array
    z: jax.Array


def make_linear_gaussian_batch(
    config: LinearGaussianDataConfig,
    params: LinearGaussianParams,
    seed: int,
) -> EpisodeBatch:
    """Generate scalar linear-Gaussian dynamic regression episodes.

    Shapes are batch-major: `x`, `y`, and latent `z` are all `[batch, time]`.
    The latent trajectory starts from `z_{-1} ~ Normal(m0, p0)` and then
    follows `z_t = z_{t-1} + w_t`.
    """

    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if params.q <= 0 or params.r <= 0 or params.p0 <= 0:
        raise ValueError("q, r, and p0 must be positive")

    key = jax.random.PRNGKey(seed)
    key_z0, key_w, key_v = jax.random.split(key, 3)

    x_single = _make_x(config)
    x = jnp.broadcast_to(x_single, (config.batch_size, config.time_steps))

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
    y = x * z + jnp.sqrt(params.r) * jax.random.normal(
        key_v,
        shape=(config.batch_size, config.time_steps),
        dtype=jnp.float64,
    )

    return EpisodeBatch(x=x, y=y, z=z)


def _make_x(config: LinearGaussianDataConfig) -> jax.Array:
    if config.x_pattern != "sinusoidal":
        raise ValueError(f"Unsupported x_pattern: {config.x_pattern}")

    time = jnp.arange(config.time_steps, dtype=jnp.float64)
    phase = 2.0 * jnp.pi * config.x_cycles * time / config.time_steps
    return jnp.sin(phase)
