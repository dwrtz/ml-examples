"""Tests for exact two-state edge posterior oracles."""

import numpy as np

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.kalman import kalman_edge_posterior_scalar, kalman_filter_scalar

FLOAT_ATOL = 1e-6


def test_edge_posterior_marginal_matches_kalman_filter() -> None:
    data_config = LinearGaussianDataConfig(batch_size=8, time_steps=11)
    params = LinearGaussianParams(q=0.05, r=0.2, m0=0.3, p0=4.0)
    batch = make_linear_gaussian_batch(data_config, params, seed=5)

    kalman = kalman_filter_scalar(batch, params)
    edge = kalman_edge_posterior_scalar(batch, params)

    np.testing.assert_allclose(
        np.asarray(edge.edge_mean[..., 0]),
        np.asarray(kalman.filter_mean),
        atol=FLOAT_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(edge.edge_cov[..., 0, 0]),
        np.asarray(kalman.filter_var),
        atol=FLOAT_ATOL,
    )


def test_edge_covariance_is_symmetric() -> None:
    data_config = LinearGaussianDataConfig(batch_size=4, time_steps=6)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(data_config, params, seed=6)

    edge = kalman_edge_posterior_scalar(batch, params)

    np.testing.assert_allclose(
        np.asarray(edge.edge_cov[..., 0, 1]),
        np.asarray(edge.edge_cov[..., 1, 0]),
        atol=FLOAT_ATOL,
    )
    assert np.all(np.asarray(edge.edge_cov[..., 0, 0]) > 0)
    assert np.all(np.asarray(edge.edge_cov[..., 1, 1]) > 0)
