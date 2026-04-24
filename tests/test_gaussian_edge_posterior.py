"""Tests for Gaussian edge posterior distribution helpers."""

import jax
import jax.numpy as jnp
import numpy as np

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.distributions import GaussianEdgePosterior
from vbf.kalman import kalman_edge_posterior_scalar


def test_gaussian_edge_posterior_reconstructs_oracle_moments() -> None:
    data_config = LinearGaussianDataConfig(batch_size=3, time_steps=5)
    params = LinearGaussianParams(q=0.1, r=0.2, m0=1.0, p0=2.0)
    batch = make_linear_gaussian_batch(data_config, params, seed=7)
    oracle = kalman_edge_posterior_scalar(batch, params)

    q_edge = GaussianEdgePosterior.from_mean_cov(oracle.edge_mean, oracle.edge_cov)

    np.testing.assert_allclose(
        np.asarray(q_edge.joint_mean()),
        np.asarray(oracle.edge_mean),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(q_edge.joint_cov()),
        np.asarray(oracle.edge_cov),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(q_edge.filter_marginal.mean),
        np.asarray(oracle.filter_mean),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(q_edge.filter_marginal.var),
        np.asarray(oracle.filter_var),
        atol=1e-12,
    )


def test_gaussian_edge_log_prob_matches_bivariate_normal() -> None:
    edge_mean = jnp.array([[0.5, -0.2], [1.0, 0.8]], dtype=jnp.float64)
    edge_cov = jnp.array(
        [
            [[0.7, 0.2], [0.2, 0.5]],
            [[1.5, -0.4], [-0.4, 0.9]],
        ],
        dtype=jnp.float64,
    )
    z = jnp.array([[0.1, -0.4], [1.3, 0.2]], dtype=jnp.float64)
    q_edge = GaussianEdgePosterior.from_mean_cov(edge_mean, edge_cov)

    actual = q_edge.log_prob(z[..., 0], z[..., 1])
    expected = _bivariate_log_prob(z, edge_mean, edge_cov)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-12)


def test_gaussian_edge_sampling_shapes() -> None:
    edge_mean = jnp.zeros((2, 4, 2), dtype=jnp.float64)
    edge_cov = jnp.broadcast_to(
        jnp.array([[1.0, 0.25], [0.25, 0.5]], dtype=jnp.float64),
        (2, 4, 2, 2),
    )
    q_edge = GaussianEdgePosterior.from_mean_cov(edge_mean, edge_cov)

    z_t, z_tm1 = q_edge.sample(jax.random.PRNGKey(0), sample_shape=(6,))

    assert z_t.shape == (6, 2, 4)
    assert z_tm1.shape == (6, 2, 4)


def _bivariate_log_prob(value: jax.Array, mean: jax.Array, cov: jax.Array) -> jax.Array:
    centered = value - mean
    solve = jnp.linalg.solve(cov, centered[..., None])[..., 0]
    quadratic = jnp.sum(centered * solve, axis=-1)
    log_det = jnp.linalg.slogdet(cov)[1]
    return -0.5 * (2.0 * jnp.log(2.0 * jnp.pi) + log_det + quadratic)
