"""Strict and recurrent learned filtering cells."""

from __future__ import annotations

from typing import NamedTuple

import jax

from vbf.dtypes import DEFAULT_DTYPE  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from vbf.data import EpisodeBatch, LinearGaussianParams, broadcast_param_like  # noqa: E402


class StructuredMLPOutputs(NamedTuple):
    filter_mean: jax.Array
    filter_var: jax.Array
    backward_a: jax.Array
    backward_b: jax.Array
    backward_var: jax.Array


class GaussianMixtureMLPOutputs(NamedTuple):
    filter_weights: jax.Array
    component_mean: jax.Array
    component_var: jax.Array
    backward_a: jax.Array
    backward_b: jax.Array
    backward_var: jax.Array

    @property
    def filter_mean(self) -> jax.Array:
        return jnp.sum(self.filter_weights * self.component_mean, axis=-1)

    @property
    def filter_var(self) -> jax.Array:
        second_moment = jnp.sum(
            self.filter_weights * (self.component_var + self.component_mean**2),
            axis=-1,
        )
        return jnp.maximum(second_moment - self.filter_mean**2, 0.0)


class ScalarFlowMLPOutputs(NamedTuple):
    filter_mean: jax.Array
    filter_var: jax.Array
    flow_loc: jax.Array
    flow_log_scale: jax.Array
    flow_bin_logits: jax.Array
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
    w1 = jax.random.normal(key_w1, shape=(input_dim, hidden_dim), dtype=DEFAULT_DTYPE)
    w1 = w1 * jnp.sqrt(2.0 / input_dim)
    return {
        "w1": w1,
        "b1": jnp.zeros((hidden_dim,), dtype=DEFAULT_DTYPE),
        "w2": jnp.zeros((hidden_dim, 5), dtype=DEFAULT_DTYPE),
        "b2": jnp.zeros((5,), dtype=DEFAULT_DTYPE),
    }


def init_split_head_mlp_params(
    key: jax.Array,
    *,
    hidden_dim: int = 32,
    input_dim: int = 6,
) -> dict[str, jax.Array]:
    """Initialize an MLP with separate filter and backward output heads."""

    key_filter, key_backward = jax.random.split(key)
    w_filter1 = jax.random.normal(key_filter, shape=(input_dim, hidden_dim), dtype=DEFAULT_DTYPE)
    w_backward1 = jax.random.normal(
        key_backward, shape=(input_dim, hidden_dim), dtype=DEFAULT_DTYPE
    )
    scale = jnp.sqrt(2.0 / input_dim)
    return {
        "w_filter1": w_filter1 * scale,
        "b_filter1": jnp.zeros((hidden_dim,), dtype=DEFAULT_DTYPE),
        "w_filter2": jnp.zeros((hidden_dim, 2), dtype=DEFAULT_DTYPE),
        "b_filter2": jnp.zeros((2,), dtype=DEFAULT_DTYPE),
        "w_backward1": w_backward1 * scale,
        "b_backward1": jnp.zeros((hidden_dim,), dtype=DEFAULT_DTYPE),
        "w_backward2": jnp.zeros((hidden_dim, 3), dtype=DEFAULT_DTYPE),
        "b_backward2": jnp.zeros((3,), dtype=DEFAULT_DTYPE),
    }


def init_direct_mlp_params(
    key: jax.Array,
    *,
    hidden_dim: int = 32,
    input_dim: int = 6,
) -> dict[str, jax.Array]:
    """Initialize a less-structured filtering MLP.

    Unlike `StructuredMLPCell`, this parameterization does not compute an
    analytic Kalman gain internally. The filter head emits a direct mean delta
    and variance for `q^F_t`.
    """

    return init_structured_mlp_params(key, hidden_dim=hidden_dim, input_dim=input_dim)


def init_direct_mixture_mlp_params(
    key: jax.Array,
    *,
    hidden_dim: int = 32,
    input_dim: int = 6,
    num_components: int = 2,
    component_mean_init_span: float = 0.0,
) -> dict[str, jax.Array]:
    """Initialize a direct strict Gaussian-mixture filtering MLP."""

    if num_components <= 0:
        raise ValueError("num_components must be positive")
    key_w1, _ = jax.random.split(key)
    w1 = jax.random.normal(key_w1, shape=(input_dim, hidden_dim), dtype=DEFAULT_DTYPE)
    w1 = w1 * jnp.sqrt(2.0 / input_dim)
    b2 = jnp.zeros((num_components, 6), dtype=DEFAULT_DTYPE)
    if component_mean_init_span != 0.0:
        component_offsets = jnp.linspace(
            -0.5 * component_mean_init_span,
            0.5 * component_mean_init_span,
            num_components,
            dtype=DEFAULT_DTYPE,
        )
        b2 = b2.at[:, 1].set(component_offsets)
    return {
        "w1": w1,
        "b1": jnp.zeros((hidden_dim,), dtype=DEFAULT_DTYPE),
        "w2": jnp.zeros((hidden_dim, 6 * num_components), dtype=DEFAULT_DTYPE),
        "b2": jnp.reshape(b2, (6 * num_components,)),
    }


def init_scalar_flow_mlp_params(
    key: jax.Array,
    *,
    hidden_dim: int = 32,
    input_dim: int = 6,
    flow_bins: int = 8,
) -> dict[str, jax.Array]:
    """Initialize a strict scalar filtering MLP with a monotone spline marginal."""

    if flow_bins <= 0:
        raise ValueError("flow_bins must be positive")
    key_w1, _ = jax.random.split(key)
    w1 = jax.random.normal(key_w1, shape=(input_dim, hidden_dim), dtype=DEFAULT_DTYPE)
    w1 = w1 * jnp.sqrt(2.0 / input_dim)
    return {
        "w1": w1,
        "b1": jnp.zeros((hidden_dim,), dtype=DEFAULT_DTYPE),
        "w2": jnp.zeros((hidden_dim, 5 + flow_bins), dtype=DEFAULT_DTYPE),
        "b2": jnp.zeros((5 + flow_bins,), dtype=DEFAULT_DTYPE),
    }


def init_component_mixture_mlp_params(
    key: jax.Array,
    *,
    hidden_dim: int = 32,
    input_dim: int = 7,
    num_components: int = 2,
    component_mean_init_span: float = 0.0,
) -> dict[str, jax.Array]:
    """Initialize a component-aware strict Gaussian-mixture filtering MLP."""

    if num_components <= 0:
        raise ValueError("num_components must be positive")
    key_w1, _ = jax.random.split(key)
    w1 = jax.random.normal(key_w1, shape=(input_dim, hidden_dim), dtype=DEFAULT_DTYPE)
    w1 = w1 * jnp.sqrt(2.0 / input_dim)
    return {
        "w1": w1,
        "b1": jnp.zeros((hidden_dim,), dtype=DEFAULT_DTYPE),
        "w2": jnp.zeros((hidden_dim, 6), dtype=DEFAULT_DTYPE),
        "b2": jnp.zeros((num_components, 6), dtype=DEFAULT_DTYPE),
    }


def init_structured_mixture_mlp_params(
    key: jax.Array,
    *,
    hidden_dim: int = 32,
    input_dim: int = 6,
    num_components: int = 2,
    component_mean_init_span: float = 0.0,
) -> dict[str, jax.Array]:
    """Initialize an EKF-residualized strict Gaussian-mixture filtering MLP."""

    return init_direct_mixture_mlp_params(
        key,
        hidden_dim=hidden_dim,
        input_dim=input_dim,
        num_components=num_components,
        component_mean_init_span=component_mean_init_span,
    )


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
        jnp.full((batch_size,), state_params.m0, dtype=DEFAULT_DTYPE),
        jnp.full((batch_size,), state_params.p0, dtype=DEFAULT_DTYPE),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt))
    return StructuredMLPOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def run_direct_mlp_filter(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    min_var: float = 1e-6,
) -> StructuredMLPOutputs:
    """Run a less-structured learned filter over batch-major episodes."""

    x_bt = batch.x.T
    y_bt = batch.y.T

    def step(carry: tuple[jax.Array, jax.Array], obs: tuple[jax.Array, jax.Array]):
        prev_mean, prev_var = carry
        x_t, y_t = obs
        outputs = direct_mlp_step(
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
        jnp.full((batch_size,), state_params.m0, dtype=DEFAULT_DTYPE),
        jnp.full((batch_size,), state_params.p0, dtype=DEFAULT_DTYPE),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt))
    return StructuredMLPOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def run_direct_mixture_mlp_filter(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    num_components: int,
    min_var: float = 1e-6,
) -> GaussianMixtureMLPOutputs:
    """Run a strict direct Gaussian-mixture learned filter over batch-major episodes."""

    x_bt = batch.x.T
    y_bt = batch.y.T

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        obs: tuple[jax.Array, jax.Array],
    ):
        prev_weights, prev_mean, prev_var = carry
        x_t, y_t = obs
        outputs = direct_mixture_mlp_step(
            mlp_params,
            prev_weights,
            prev_mean,
            prev_var,
            x_t,
            y_t,
            state_params,
            num_components=num_components,
            min_var=min_var,
        )
        next_carry = (outputs.filter_weights, outputs.component_mean, outputs.component_var)
        return next_carry, outputs

    batch_size = batch.x.shape[0]
    init = (
        jnp.full((batch_size, num_components), 1.0 / num_components, dtype=DEFAULT_DTYPE),
        jnp.full((batch_size, num_components), state_params.m0, dtype=DEFAULT_DTYPE),
        jnp.full((batch_size, num_components), state_params.p0, dtype=DEFAULT_DTYPE),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt))
    return GaussianMixtureMLPOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def run_scalar_flow_mlp_filter(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    flow_bins: int = 8,
    flow_bound: float = 5.0,
    min_var: float = 1e-6,
) -> ScalarFlowMLPOutputs:
    """Run a strict scalar monotone-spline flow filter over batch-major episodes."""

    x_bt = batch.x.T
    y_bt = batch.y.T

    def step(carry: tuple[jax.Array, jax.Array], obs: tuple[jax.Array, jax.Array]):
        prev_mean, prev_var = carry
        x_t, y_t = obs
        outputs = scalar_flow_mlp_step(
            mlp_params,
            prev_mean,
            prev_var,
            x_t,
            y_t,
            state_params,
            flow_bins=flow_bins,
            flow_bound=flow_bound,
            min_var=min_var,
        )
        return (outputs.filter_mean, outputs.filter_var), outputs

    batch_size = batch.x.shape[0]
    init = (
        jnp.full((batch_size,), state_params.m0, dtype=DEFAULT_DTYPE),
        jnp.full((batch_size,), state_params.p0, dtype=DEFAULT_DTYPE),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt))
    return ScalarFlowMLPOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def run_component_mixture_mlp_filter(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    num_components: int,
    min_var: float = 1e-6,
    component_mean_init_span: float = 0.0,
) -> GaussianMixtureMLPOutputs:
    """Run a component-aware Gaussian-mixture filter over batch-major episodes."""

    x_bt = batch.x.T
    y_bt = batch.y.T

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        obs: tuple[jax.Array, jax.Array],
    ):
        prev_weights, prev_mean, prev_var = carry
        x_t, y_t = obs
        outputs = component_mixture_mlp_step(
            mlp_params,
            prev_weights,
            prev_mean,
            prev_var,
            x_t,
            y_t,
            state_params,
            min_var=min_var,
        )
        next_carry = (outputs.filter_weights, outputs.component_mean, outputs.component_var)
        return next_carry, outputs

    batch_size = batch.x.shape[0]
    init = (
        jnp.full((batch_size, num_components), 1.0 / num_components, dtype=DEFAULT_DTYPE),
        jnp.full((batch_size, num_components), state_params.m0, dtype=DEFAULT_DTYPE)
        + _component_offsets(num_components, component_mean_init_span)[None, :],
        jnp.full((batch_size, num_components), state_params.p0, dtype=DEFAULT_DTYPE),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt))
    return GaussianMixtureMLPOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def run_split_head_mlp_filter(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    min_var: float = 1e-6,
) -> StructuredMLPOutputs:
    """Run a strict split-head learned filter over batch-major episodes."""

    x_bt = batch.x.T
    y_bt = batch.y.T

    def step(carry: tuple[jax.Array, jax.Array], obs: tuple[jax.Array, jax.Array]):
        prev_mean, prev_var = carry
        x_t, y_t = obs
        outputs = split_head_mlp_step(
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
        jnp.full((batch_size,), state_params.m0, dtype=DEFAULT_DTYPE),
        jnp.full((batch_size,), state_params.p0, dtype=DEFAULT_DTYPE),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt))
    return StructuredMLPOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def run_direct_mlp_teacher_forced(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    target_filter_mean: jax.Array,
    target_filter_var: jax.Array,
    *,
    min_var: float = 1e-6,
) -> StructuredMLPOutputs:
    """Run direct updates using target previous beliefs as inputs."""

    initial_mean = jnp.full((batch.x.shape[0], 1), state_params.m0, dtype=DEFAULT_DTYPE)
    initial_var = jnp.full((batch.x.shape[0], 1), state_params.p0, dtype=DEFAULT_DTYPE)
    prev_mean = jnp.concatenate((initial_mean, target_filter_mean[:, :-1]), axis=1)
    prev_var = jnp.concatenate((initial_var, target_filter_var[:, :-1]), axis=1)
    return direct_mlp_step(
        mlp_params,
        prev_mean,
        prev_var,
        batch.x,
        batch.y,
        state_params,
        min_var=min_var,
    )


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

    initial_mean = jnp.full((batch.x.shape[0], 1), state_params.m0, dtype=DEFAULT_DTYPE)
    initial_var = jnp.full((batch.x.shape[0], 1), state_params.p0, dtype=DEFAULT_DTYPE)
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


def run_split_head_mlp_teacher_forced(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    target_filter_mean: jax.Array,
    target_filter_var: jax.Array,
    *,
    min_var: float = 1e-6,
) -> StructuredMLPOutputs:
    """Run independent split-head updates using target previous beliefs as inputs."""

    initial_mean = jnp.full((batch.x.shape[0], 1), state_params.m0, dtype=DEFAULT_DTYPE)
    initial_var = jnp.full((batch.x.shape[0], 1), state_params.p0, dtype=DEFAULT_DTYPE)
    prev_mean = jnp.concatenate((initial_mean, target_filter_mean[:, :-1]), axis=1)
    prev_var = jnp.concatenate((initial_var, target_filter_var[:, :-1]), axis=1)
    return split_head_mlp_step(
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
            jnp.log(broadcast_param_like(state_params.q, x_t)),
            jnp.log(broadcast_param_like(state_params.r, x_t)),
        ),
        axis=-1,
    )
    hidden = jnp.tanh(features @ mlp_params["w1"] + mlp_params["b1"])
    raw = hidden @ mlp_params["w2"] + mlp_params["b2"]
    q = broadcast_param_like(state_params.q, x_t)
    r = broadcast_param_like(state_params.r, x_t)
    pred_var = prev_var + q
    innovation = y_t - x_t * prev_mean
    innovation_var = x_t**2 * pred_var + r
    base_gain = pred_var * x_t / innovation_var
    gain_scale = 2.0 * jax.nn.sigmoid(raw[..., 0])
    filter_mean = prev_mean + gain_scale * base_gain * innovation
    base_filter_var = pred_var * r / innovation_var
    filter_var = base_filter_var * jnp.exp(jnp.clip(raw[..., 1], -5.0, 5.0)) + min_var
    backward_a = raw[..., 2]

    return StructuredMLPOutputs(
        filter_mean=filter_mean,
        filter_var=filter_var,
        backward_a=backward_a,
        backward_b=prev_mean - backward_a * filter_mean + raw[..., 3],
        backward_var=jax.nn.softplus(raw[..., 4]) + min_var,
    )


def direct_mlp_step(
    mlp_params: dict[str, jax.Array],
    prev_mean: jax.Array,
    prev_var: jax.Array,
    x_t: jax.Array,
    y_t: jax.Array,
    state_params: LinearGaussianParams,
    *,
    min_var: float = 1e-6,
) -> StructuredMLPOutputs:
    """Compute one non-residualized MLP filter update."""

    hidden = _mlp_hidden(mlp_params, prev_mean, prev_var, x_t, y_t, state_params)
    raw = hidden @ mlp_params["w2"] + mlp_params["b2"]
    filter_mean = prev_mean + raw[..., 0]
    filter_var = jax.nn.softplus(raw[..., 1]) + min_var
    backward_a = raw[..., 2]
    return StructuredMLPOutputs(
        filter_mean=filter_mean,
        filter_var=filter_var,
        backward_a=backward_a,
        backward_b=prev_mean - backward_a * filter_mean + raw[..., 3],
        backward_var=jax.nn.softplus(raw[..., 4]) + min_var,
    )


def direct_mixture_mlp_step(
    mlp_params: dict[str, jax.Array],
    prev_weights: jax.Array,
    prev_component_mean: jax.Array,
    prev_component_var: jax.Array,
    x_t: jax.Array,
    y_t: jax.Array,
    state_params: LinearGaussianParams,
    *,
    num_components: int,
    min_var: float = 1e-6,
) -> GaussianMixtureMLPOutputs:
    """Compute one direct Gaussian-mixture filter update."""

    prev_mean, prev_var = mixture_mean_and_var(
        prev_weights,
        prev_component_mean,
        prev_component_var,
    )
    hidden = _mlp_hidden(mlp_params, prev_mean, prev_var, x_t, y_t, state_params)
    raw = hidden @ mlp_params["w2"] + mlp_params["b2"]
    raw = jnp.reshape(raw, raw.shape[:-1] + (num_components, 6))
    filter_weights = jax.nn.softmax(raw[..., 0], axis=-1)
    component_mean = prev_mean[..., None] + raw[..., 1]
    component_var = jax.nn.softplus(raw[..., 2]) + min_var
    backward_a = raw[..., 3]
    backward_b = prev_mean[..., None] - backward_a * component_mean + raw[..., 4]
    backward_var = jax.nn.softplus(raw[..., 5]) + min_var
    return GaussianMixtureMLPOutputs(
        filter_weights=filter_weights,
        component_mean=component_mean,
        component_var=component_var,
        backward_a=backward_a,
        backward_b=backward_b,
        backward_var=backward_var,
    )


def scalar_flow_mlp_step(
    mlp_params: dict[str, jax.Array],
    prev_mean: jax.Array,
    prev_var: jax.Array,
    x_t: jax.Array,
    y_t: jax.Array,
    state_params: LinearGaussianParams,
    *,
    flow_bins: int = 8,
    flow_bound: float = 5.0,
    min_var: float = 1e-6,
) -> ScalarFlowMLPOutputs:
    """Compute one direct scalar-flow filtering update.

    The filter marginal is `z = loc + scale * S(u)`, where `u` is standard
    normal and `S` is a monotone piecewise-linear spline with fixed base knots
    and learned positive output-bin heights. This gives exact density
    evaluation while preserving the online filtering update contract.
    """

    hidden = _mlp_hidden(mlp_params, prev_mean, prev_var, x_t, y_t, state_params)
    raw = hidden @ mlp_params["w2"] + mlp_params["b2"]
    flow_loc = prev_mean + raw[..., 0]
    flow_log_scale = jnp.clip(raw[..., 1], -5.0, 5.0)
    flow_bin_logits = raw[..., 2 : 2 + flow_bins]
    filter_mean, filter_var = scalar_flow_moments(
        flow_loc,
        flow_log_scale,
        flow_bin_logits,
        bound=flow_bound,
        min_var=min_var,
    )
    backward_raw = raw[..., 2 + flow_bins :]
    backward_a = backward_raw[..., 0]
    return ScalarFlowMLPOutputs(
        filter_mean=filter_mean,
        filter_var=filter_var,
        flow_loc=flow_loc,
        flow_log_scale=flow_log_scale,
        flow_bin_logits=flow_bin_logits,
        backward_a=backward_a,
        backward_b=prev_mean - backward_a * filter_mean + backward_raw[..., 1],
        backward_var=jax.nn.softplus(backward_raw[..., 2]) + min_var,
    )


def component_mixture_mlp_step(
    mlp_params: dict[str, jax.Array],
    prev_weights: jax.Array,
    prev_component_mean: jax.Array,
    prev_component_var: jax.Array,
    x_t: jax.Array,
    y_t: jax.Array,
    state_params: LinearGaussianParams,
    *,
    min_var: float = 1e-6,
) -> GaussianMixtureMLPOutputs:
    """Compute one component-aware Gaussian-mixture filter update."""

    q = broadcast_param_like(state_params.q, x_t)
    r = broadcast_param_like(state_params.r, x_t)
    features = jnp.stack(
        (
            jnp.log(jnp.clip(prev_weights, min_var)),
            prev_component_mean,
            jnp.log(prev_component_var),
            jnp.broadcast_to(x_t[..., None], prev_component_mean.shape),
            jnp.broadcast_to(y_t[..., None], prev_component_mean.shape),
            jnp.broadcast_to(jnp.log(q)[..., None], prev_component_mean.shape),
            jnp.broadcast_to(jnp.log(r)[..., None], prev_component_mean.shape),
        ),
        axis=-1,
    )
    hidden = jnp.tanh(features @ mlp_params["w1"] + mlp_params["b1"])
    raw = hidden @ mlp_params["w2"] + mlp_params["b2"]
    filter_weights = jax.nn.softmax(raw[..., 0], axis=-1)
    component_mean = prev_component_mean + raw[..., 1]
    component_var = jax.nn.softplus(raw[..., 2]) + min_var
    backward_a = raw[..., 3]
    backward_b = prev_component_mean - backward_a * component_mean + raw[..., 4]
    backward_var = jax.nn.softplus(raw[..., 5]) + min_var
    return GaussianMixtureMLPOutputs(
        filter_weights=filter_weights,
        component_mean=component_mean,
        component_var=component_var,
        backward_a=backward_a,
        backward_b=backward_b,
        backward_var=backward_var,
    )


def split_head_mlp_step(
    mlp_params: dict[str, jax.Array],
    prev_mean: jax.Array,
    prev_var: jax.Array,
    x_t: jax.Array,
    y_t: jax.Array,
    state_params: LinearGaussianParams,
    *,
    min_var: float = 1e-6,
) -> StructuredMLPOutputs:
    """Compute one split-head filter update from previous belief and observation."""

    features = _mlp_features(prev_mean, prev_var, x_t, y_t, state_params)
    filter_hidden = jnp.tanh(features @ mlp_params["w_filter1"] + mlp_params["b_filter1"])
    backward_hidden = jnp.tanh(features @ mlp_params["w_backward1"] + mlp_params["b_backward1"])
    raw_filter = filter_hidden @ mlp_params["w_filter2"] + mlp_params["b_filter2"]
    raw_backward = backward_hidden @ mlp_params["w_backward2"] + mlp_params["b_backward2"]
    return _outputs_from_raw(
        raw_filter,
        raw_backward,
        prev_mean,
        prev_var,
        x_t,
        y_t,
        state_params,
        min_var=min_var,
    )


def edge_mean_cov_from_outputs(outputs: StructuredMLPOutputs) -> tuple[jax.Array, jax.Array]:
    """Return joint edge moments in `[z_t, z_tm1]` order."""

    if isinstance(outputs, GaussianMixtureMLPOutputs):
        return mixture_edge_mean_cov_from_outputs(outputs)

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


def mixture_mean_and_var(
    weights: jax.Array,
    component_mean: jax.Array,
    component_var: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mean = jnp.sum(weights * component_mean, axis=-1)
    second_moment = jnp.sum(weights * (component_var + component_mean**2), axis=-1)
    return mean, jnp.maximum(second_moment - mean**2, 0.0)


def mixture_edge_mean_cov_from_outputs(
    outputs: GaussianMixtureMLPOutputs,
) -> tuple[jax.Array, jax.Array]:
    """Return moment-projected mixture edge moments in `[z_t, z_tm1]` order."""

    mean_z_t_k = outputs.component_mean
    var_z_t_k = outputs.component_var
    mean_z_tm1_k = outputs.backward_a * mean_z_t_k + outputs.backward_b
    cov_k = outputs.backward_a * var_z_t_k
    var_z_tm1_k = outputs.backward_a**2 * var_z_t_k + outputs.backward_var

    weights = outputs.filter_weights
    mean_z_t = jnp.sum(weights * mean_z_t_k, axis=-1)
    mean_z_tm1 = jnp.sum(weights * mean_z_tm1_k, axis=-1)
    edge_mean = jnp.stack((mean_z_t, mean_z_tm1), axis=-1)

    second_z_t = jnp.sum(weights * (var_z_t_k + mean_z_t_k**2), axis=-1)
    second_z_tm1 = jnp.sum(weights * (var_z_tm1_k + mean_z_tm1_k**2), axis=-1)
    cross = jnp.sum(weights * (cov_k + mean_z_t_k * mean_z_tm1_k), axis=-1)
    var_z_t = jnp.maximum(second_z_t - mean_z_t**2, 0.0)
    var_z_tm1 = jnp.maximum(second_z_tm1 - mean_z_tm1**2, 0.0)
    cov = cross - mean_z_t * mean_z_tm1
    row_0 = jnp.stack((var_z_t, cov), axis=-1)
    row_1 = jnp.stack((cov, var_z_tm1), axis=-1)
    return edge_mean, jnp.stack((row_0, row_1), axis=-2)


def scalar_flow_sample(
    key: jax.Array,
    loc: jax.Array,
    log_scale: jax.Array,
    bin_logits: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
    bound: float = 5.0,
) -> jax.Array:
    base_shape = sample_shape + loc.shape
    u = jax.random.normal(key, shape=base_shape, dtype=loc.dtype)
    return scalar_flow_forward(u, loc, log_scale, bin_logits, bound=bound)


def scalar_flow_log_prob(
    value: jax.Array,
    loc: jax.Array,
    log_scale: jax.Array,
    bin_logits: jax.Array,
    *,
    bound: float = 5.0,
) -> jax.Array:
    u, log_dz_du = scalar_flow_inverse(value, loc, log_scale, bin_logits, bound=bound)
    return -0.5 * (LOG_2PI + u**2) - log_dz_du


def scalar_flow_forward(
    u: jax.Array,
    loc: jax.Array,
    log_scale: jax.Array,
    bin_logits: jax.Array,
    *,
    bound: float = 5.0,
) -> jax.Array:
    target_shape = jnp.broadcast_shapes(u.shape, loc.shape, log_scale.shape, bin_logits.shape[:-1])
    u = jnp.broadcast_to(u, target_shape)
    spline, _ = _piecewise_linear_spline_forward(u, bin_logits, bound=bound)
    return loc + jnp.exp(log_scale) * spline


def scalar_flow_inverse(
    value: jax.Array,
    loc: jax.Array,
    log_scale: jax.Array,
    bin_logits: jax.Array,
    *,
    bound: float = 5.0,
) -> tuple[jax.Array, jax.Array]:
    target_shape = jnp.broadcast_shapes(
        value.shape, loc.shape, log_scale.shape, bin_logits.shape[:-1]
    )
    value = jnp.broadcast_to(value, target_shape)
    spline_value = (value - loc) * jnp.exp(-log_scale)
    u, log_ds_du = _piecewise_linear_spline_inverse(spline_value, bin_logits, bound=bound)
    return u, log_scale + log_ds_du


def scalar_flow_moments(
    loc: jax.Array,
    log_scale: jax.Array,
    bin_logits: jax.Array,
    *,
    bound: float = 5.0,
    min_var: float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    nodes = jnp.asarray(
        [
            -3.4361591188377374,
            -2.5327316742327897,
            -1.7566836492998819,
            -1.0366108297895136,
            -0.3429013272237046,
            0.3429013272237046,
            1.0366108297895136,
            1.7566836492998819,
            2.5327316742327897,
            3.4361591188377374,
        ],
        dtype=loc.dtype,
    )
    weights = jnp.asarray(
        [
            0.00000764043285523262,
            0.0013436457467812327,
            0.033874394455481064,
            0.2401386110823147,
            0.6108626337353258,
            0.6108626337353258,
            0.2401386110823147,
            0.033874394455481064,
            0.0013436457467812327,
            0.00000764043285523262,
        ],
        dtype=loc.dtype,
    ) / jnp.sqrt(jnp.pi)
    z = scalar_flow_forward(
        jnp.sqrt(2.0) * nodes.reshape((10,) + (1,) * loc.ndim),
        loc,
        log_scale,
        bin_logits,
        bound=bound,
    )
    weights = weights.reshape((10,) + (1,) * loc.ndim)
    mean = jnp.sum(weights * z, axis=0)
    second = jnp.sum(weights * z**2, axis=0)
    return mean, jnp.maximum(second - mean**2, min_var)


def _component_offsets(num_components: int, component_mean_init_span: float) -> jax.Array:
    if component_mean_init_span == 0.0:
        return jnp.zeros((num_components,), dtype=DEFAULT_DTYPE)
    return jnp.linspace(
        -0.5 * component_mean_init_span,
        0.5 * component_mean_init_span,
        num_components,
        dtype=DEFAULT_DTYPE,
    )


def _mlp_hidden(
    mlp_params: dict[str, jax.Array],
    prev_mean: jax.Array,
    prev_var: jax.Array,
    x_t: jax.Array,
    y_t: jax.Array,
    state_params: LinearGaussianParams,
) -> jax.Array:
    features = _mlp_features(prev_mean, prev_var, x_t, y_t, state_params)
    return jnp.tanh(features @ mlp_params["w1"] + mlp_params["b1"])


def _mlp_features(
    prev_mean: jax.Array,
    prev_var: jax.Array,
    x_t: jax.Array,
    y_t: jax.Array,
    state_params: LinearGaussianParams,
) -> jax.Array:
    return jnp.stack(
        (
            prev_mean,
            jnp.log(prev_var),
            x_t,
            y_t,
            jnp.log(broadcast_param_like(state_params.q, x_t)),
            jnp.log(broadcast_param_like(state_params.r, x_t)),
        ),
        axis=-1,
    )


LOG_2PI = jnp.log(2.0 * jnp.pi)


def _piecewise_linear_spline_forward(
    u: jax.Array,
    bin_logits: jax.Array,
    *,
    bound: float,
) -> tuple[jax.Array, jax.Array]:
    bin_logits = _broadcast_flow_bin_logits(bin_logits, u)
    num_bins = bin_logits.shape[-1]
    dtype = u.dtype
    x_edges = jnp.linspace(-bound, bound, num_bins + 1, dtype=dtype)
    bin_width = jnp.asarray(2.0 * bound / num_bins, dtype=dtype)
    heights = jax.nn.softmax(bin_logits, axis=-1) * (2.0 * bound)
    y_edges = jnp.concatenate(
        (
            jnp.full(bin_logits.shape[:-1] + (1,), -bound, dtype=dtype),
            -bound + jnp.cumsum(heights, axis=-1),
        ),
        axis=-1,
    )
    slopes = heights / bin_width

    bin_index = jnp.clip(jnp.floor((u - x_edges[0]) / bin_width).astype(jnp.int32), 0, num_bins - 1)
    x_left = x_edges[bin_index]
    y_left = jnp.take_along_axis(y_edges, bin_index[..., None], axis=-1)[..., 0]
    slope = jnp.take_along_axis(slopes, bin_index[..., None], axis=-1)[..., 0]
    inside = (u >= -bound) & (u <= bound)
    y_inside = y_left + slope * (u - x_left)
    left_slope = slopes[..., 0]
    right_slope = slopes[..., -1]
    y_left_tail = -bound + left_slope * (u + bound)
    y_right_tail = bound + right_slope * (u - bound)
    y = jnp.where(inside, y_inside, jnp.where(u < -bound, y_left_tail, y_right_tail))
    log_slope = jnp.log(jnp.where(inside, slope, jnp.where(u < -bound, left_slope, right_slope)))
    return y, log_slope


def _piecewise_linear_spline_inverse(
    y: jax.Array,
    bin_logits: jax.Array,
    *,
    bound: float,
) -> tuple[jax.Array, jax.Array]:
    bin_logits = _broadcast_flow_bin_logits(bin_logits, y)
    num_bins = bin_logits.shape[-1]
    dtype = y.dtype
    x_edges = jnp.linspace(-bound, bound, num_bins + 1, dtype=dtype)
    bin_width = jnp.asarray(2.0 * bound / num_bins, dtype=dtype)
    heights = jax.nn.softmax(bin_logits, axis=-1) * (2.0 * bound)
    y_edges = jnp.concatenate(
        (
            jnp.full(bin_logits.shape[:-1] + (1,), -bound, dtype=dtype),
            -bound + jnp.cumsum(heights, axis=-1),
        ),
        axis=-1,
    )
    slopes = heights / bin_width
    bin_index = jnp.sum(y[..., None] >= y_edges[..., 1:], axis=-1)
    bin_index = jnp.clip(bin_index.astype(jnp.int32), 0, num_bins - 1)
    x_left = x_edges[bin_index]
    y_left = jnp.take_along_axis(y_edges, bin_index[..., None], axis=-1)[..., 0]
    slope = jnp.take_along_axis(slopes, bin_index[..., None], axis=-1)[..., 0]
    inside = (y >= -bound) & (y <= bound)
    u_inside = x_left + (y - y_left) / slope
    left_slope = slopes[..., 0]
    right_slope = slopes[..., -1]
    u_left_tail = -bound + (y + bound) / left_slope
    u_right_tail = bound + (y - bound) / right_slope
    u = jnp.where(inside, u_inside, jnp.where(y < -bound, u_left_tail, u_right_tail))
    log_slope = jnp.log(jnp.where(inside, slope, jnp.where(y < -bound, left_slope, right_slope)))
    return u, log_slope


def _broadcast_flow_bin_logits(bin_logits: jax.Array, value: jax.Array) -> jax.Array:
    sample_ndim = value.ndim - (bin_logits.ndim - 1)
    if sample_ndim < 0:
        raise ValueError("value shape is not compatible with flow bin logits")
    reshaped = jnp.reshape(bin_logits, (1,) * sample_ndim + bin_logits.shape)
    return jnp.broadcast_to(reshaped, value.shape + (bin_logits.shape[-1],))


def _outputs_from_raw(
    raw_filter: jax.Array,
    raw_backward: jax.Array,
    prev_mean: jax.Array,
    prev_var: jax.Array,
    x_t: jax.Array,
    y_t: jax.Array,
    state_params: LinearGaussianParams,
    *,
    min_var: float,
) -> StructuredMLPOutputs:
    q = broadcast_param_like(state_params.q, x_t)
    r = broadcast_param_like(state_params.r, x_t)
    pred_var = prev_var + q
    innovation = y_t - x_t * prev_mean
    innovation_var = x_t**2 * pred_var + r
    base_gain = pred_var * x_t / innovation_var
    gain_scale = 2.0 * jax.nn.sigmoid(raw_filter[..., 0])
    filter_mean = prev_mean + gain_scale * base_gain * innovation
    base_filter_var = pred_var * r / innovation_var
    filter_var = base_filter_var * jnp.exp(jnp.clip(raw_filter[..., 1], -5.0, 5.0)) + min_var
    backward_a = raw_backward[..., 0]

    return StructuredMLPOutputs(
        filter_mean=filter_mean,
        filter_var=filter_var,
        backward_a=backward_a,
        backward_b=prev_mean - backward_a * filter_mean + raw_backward[..., 1],
        backward_var=jax.nn.softplus(raw_backward[..., 2]) + min_var,
    )


def _time_major_to_batch_major(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, 0, 1)
