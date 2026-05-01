"""Output heads for posterior and predictive distributions."""

from __future__ import annotations

from typing import NamedTuple

import jax

from vbf.dtypes import DEFAULT_DTYPE  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from vbf.data import LinearGaussianParams, broadcast_param_like  # noqa: E402
from vbf.predictive import PredictiveMoments  # noqa: E402


class PredictiveHeadOutputs(NamedTuple):
    """Gaussian predictive distribution parameters for `y_t`."""

    mean: jax.Array
    var: jax.Array


def init_predictive_mlp_params(
    key: jax.Array,
    *,
    hidden_dim: int = 32,
    input_dim: int = 5,
) -> dict[str, jax.Array]:
    """Initialize a Gaussian predictive MLP head."""

    key_w1 = key
    w1 = jax.random.normal(key_w1, shape=(input_dim, hidden_dim), dtype=DEFAULT_DTYPE)
    w1 = w1 * jnp.sqrt(2.0 / input_dim)
    return {
        "w1": w1,
        "b1": jnp.zeros((hidden_dim,), dtype=DEFAULT_DTYPE),
        "w2": jnp.zeros((hidden_dim, 2), dtype=DEFAULT_DTYPE),
        "b2": jnp.zeros((2,), dtype=DEFAULT_DTYPE),
    }


def predictive_head_features(
    prev_filter_mean: jax.Array,
    prev_filter_var: jax.Array,
    x_t: jax.Array,
    state_params: LinearGaussianParams,
) -> jax.Array:
    """Return pre-assimilation predictive features.

    The current observation `y_t` is deliberately not an argument.
    """

    return jnp.stack(
        (
            prev_filter_mean,
            jnp.log(prev_filter_var),
            x_t,
            jnp.log(broadcast_param_like(state_params.q, x_t)),
            jnp.log(broadcast_param_like(state_params.r, x_t)),
        ),
        axis=-1,
    )


def run_predictive_mlp_head(
    params: dict[str, jax.Array],
    prev_filter_mean: jax.Array,
    prev_filter_var: jax.Array,
    x_t: jax.Array,
    state_params: LinearGaussianParams,
    *,
    min_var: float = 1e-6,
) -> PredictiveHeadOutputs:
    """Predict `y_t` from `q^F_{t-1}`, `x_t`, and known model scales only."""

    features = predictive_head_features(prev_filter_mean, prev_filter_var, x_t, state_params)
    hidden = jnp.tanh(features @ params["w1"] + params["b1"])
    raw = hidden @ params["w2"] + params["b2"]
    analytic = _analytic_predictive(prev_filter_mean, prev_filter_var, x_t, state_params)
    return PredictiveHeadOutputs(
        mean=analytic.mean + raw[..., 0],
        var=analytic.var * jnp.exp(jnp.clip(raw[..., 1], -5.0, 5.0)) + min_var,
    )


def run_direct_predictive_mlp_head(
    params: dict[str, jax.Array],
    prev_filter_mean: jax.Array,
    prev_filter_var: jax.Array,
    x_t: jax.Array,
    state_params: LinearGaussianParams,
    *,
    min_var: float = 1e-6,
) -> PredictiveHeadOutputs:
    """Predict `y_t` directly without an analytic predictive baseline."""

    features = predictive_head_features(prev_filter_mean, prev_filter_var, x_t, state_params)
    hidden = jnp.tanh(features @ params["w1"] + params["b1"])
    raw = hidden @ params["w2"] + params["b2"]
    return PredictiveHeadOutputs(
        mean=raw[..., 0],
        var=jax.nn.softplus(raw[..., 1]) + min_var,
    )


def _analytic_predictive(
    prev_filter_mean: jax.Array,
    prev_filter_var: jax.Array,
    x_t: jax.Array,
    state_params: LinearGaussianParams,
) -> PredictiveMoments:
    q = broadcast_param_like(state_params.q, x_t)
    r = broadcast_param_like(state_params.r, x_t)
    pred_state_var = prev_filter_var + q
    return PredictiveMoments(
        mean=x_t * prev_filter_mean,
        var=x_t**2 * pred_state_var + r,
    )
