"""Strict and recurrent learned filtering cells."""

from __future__ import annotations

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vbf.data import EpisodeBatch, LinearGaussianParams  # noqa: E402


class StructuredMLPOutputs(NamedTuple):
    filter_mean: jax.Array
    filter_var: jax.Array
    backward_a: jax.Array
    backward_b: jax.Array
    backward_var: jax.Array


def init_structured_mlp_params(
    key: jax.Array,
    *,
    hidden_dim: int = 32,
    input_dim: int = 6,
) -> dict[str, jax.Array]:
    """Initialize a one-hidden-layer strict filtering MLP."""

    key_w1, _ = jax.random.split(key)
    w1 = jax.random.normal(key_w1, shape=(input_dim, hidden_dim), dtype=jnp.float64)
    w1 = w1 * jnp.sqrt(2.0 / input_dim)
    return {
        "w1": w1,
        "b1": jnp.zeros((hidden_dim,), dtype=jnp.float64),
        "w2": jnp.zeros((hidden_dim, 5), dtype=jnp.float64),
        "b2": jnp.zeros((5,), dtype=jnp.float64),
    }


def run_structured_mlp_filter(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    min_var: float = 1e-6,
) -> StructuredMLPOutputs:
    """Run a strict learned filter over batch-major episodes."""

    x_bt = batch.x.T
    y_bt = batch.y.T

    def step(carry: tuple[jax.Array, jax.Array], obs: tuple[jax.Array, jax.Array]):
        prev_mean, prev_var = carry
        x_t, y_t = obs
        outputs = structured_mlp_step(
            mlp_params,
            prev_mean,
            prev_var,
            x_t,
            y_t,
            state_params,
            min_var=min_var,
        )
        next_carry = (outputs.filter_mean, outputs.filter_var)
        return next_carry, outputs

    batch_size = batch.x.shape[0]
    init = (
        jnp.full((batch_size,), state_params.m0, dtype=jnp.float64),
        jnp.full((batch_size,), state_params.p0, dtype=jnp.float64),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt))
    return StructuredMLPOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def run_structured_mlp_teacher_forced(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    target_filter_mean: jax.Array,
    target_filter_var: jax.Array,
    *,
    min_var: float = 1e-6,
) -> StructuredMLPOutputs:
    """Run independent updates using target previous beliefs as inputs."""

    initial_mean = jnp.full((batch.x.shape[0], 1), state_params.m0, dtype=jnp.float64)
    initial_var = jnp.full((batch.x.shape[0], 1), state_params.p0, dtype=jnp.float64)
    prev_mean = jnp.concatenate((initial_mean, target_filter_mean[:, :-1]), axis=1)
    prev_var = jnp.concatenate((initial_var, target_filter_var[:, :-1]), axis=1)
    return structured_mlp_step(
        mlp_params,
        prev_mean,
        prev_var,
        batch.x,
        batch.y,
        state_params,
        min_var=min_var,
    )


def structured_mlp_step(
    mlp_params: dict[str, jax.Array],
    prev_mean: jax.Array,
    prev_var: jax.Array,
    x_t: jax.Array,
    y_t: jax.Array,
    state_params: LinearGaussianParams,
    *,
    min_var: float = 1e-6,
) -> StructuredMLPOutputs:
    """Compute one strict filter update from previous belief and current observation."""

    features = jnp.stack(
        (
            prev_mean,
            jnp.log(prev_var),
            x_t,
            y_t,
            jnp.full_like(x_t, jnp.log(state_params.q)),
            jnp.full_like(x_t, jnp.log(state_params.r)),
        ),
        axis=-1,
    )
    hidden = jnp.tanh(features @ mlp_params["w1"] + mlp_params["b1"])
    raw = hidden @ mlp_params["w2"] + mlp_params["b2"]
    pred_var = prev_var + state_params.q
    innovation = y_t - x_t * prev_mean
    innovation_var = x_t**2 * pred_var + state_params.r
    base_gain = pred_var * x_t / innovation_var
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


def edge_mean_cov_from_outputs(outputs: StructuredMLPOutputs) -> tuple[jax.Array, jax.Array]:
    """Return joint edge moments in `[z_t, z_tm1]` order."""

    mean_z_t = outputs.filter_mean
    var_z_t = outputs.filter_var
    mean_z_tm1 = outputs.backward_a * mean_z_t + outputs.backward_b
    cov = outputs.backward_a * var_z_t
    var_z_tm1 = outputs.backward_a**2 * var_z_t + outputs.backward_var
    edge_mean = jnp.stack((mean_z_t, mean_z_tm1), axis=-1)
    row_0 = jnp.stack((var_z_t, cov), axis=-1)
    row_1 = jnp.stack((cov, var_z_tm1), axis=-1)
    edge_cov = jnp.stack((row_0, row_1), axis=-2)
    return edge_mean, edge_cov


def _time_major_to_batch_major(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, 0, 1)
