"""Tests that predictive heads cannot access current observations as inputs."""

import jax
import numpy as np
from dataclasses import replace

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.kalman import kalman_edge_posterior_scalar
from vbf.models.heads import (
    init_predictive_mlp_params,
    predictive_head_features,
    run_predictive_mlp_head,
)
from vbf.predictive import previous_filter_beliefs


def test_predictive_head_features_do_not_depend_on_current_y() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=30,
    )
    oracle = kalman_edge_posterior_scalar(batch, state_params)
    prev_mean, prev_var = previous_filter_beliefs(
        oracle.filter_mean,
        oracle.filter_var,
        state_params,
    )

    original = predictive_head_features(prev_mean, prev_var, batch.x, state_params)
    changed_y_batch = replace(batch, y=batch.y + 1000.0)
    changed_y = predictive_head_features(prev_mean, prev_var, changed_y_batch.x, state_params)

    np.testing.assert_allclose(np.asarray(original), np.asarray(changed_y), atol=0.0)


def test_predictive_head_outputs_do_not_depend_on_current_y() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=31,
    )
    oracle = kalman_edge_posterior_scalar(batch, state_params)
    prev_mean, prev_var = previous_filter_beliefs(
        oracle.filter_mean,
        oracle.filter_var,
        state_params,
    )
    params = init_predictive_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    outputs = run_predictive_mlp_head(params, prev_mean, prev_var, batch.x, state_params)
    changed_y_batch = replace(batch, y=batch.y - 1000.0)
    changed_outputs = run_predictive_mlp_head(
        params,
        prev_mean,
        prev_var,
        changed_y_batch.x,
        state_params,
    )

    np.testing.assert_allclose(np.asarray(outputs.mean), np.asarray(changed_outputs.mean), atol=0.0)
    np.testing.assert_allclose(np.asarray(outputs.var), np.asarray(changed_outputs.var), atol=0.0)
