"""Tests for model-consistent linear-Gaussian predictive moments."""

import jax.numpy as jnp

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.kalman import measurement_predictive_scalar, kalman_edge_posterior_scalar
from vbf.predictive import linear_gaussian_predictive_from_filter


def test_predictive_from_oracle_filter_matches_kalman_predictive() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=19,
    )
    oracle = kalman_edge_posterior_scalar(batch, state_params)
    predictive = linear_gaussian_predictive_from_filter(
        oracle.filter_mean,
        oracle.filter_var,
        batch,
        state_params,
    )
    kalman_predictive = measurement_predictive_scalar(batch, state_params)

    assert jnp.allclose(predictive.mean, kalman_predictive.mean)
    assert jnp.allclose(predictive.var, kalman_predictive.var)
