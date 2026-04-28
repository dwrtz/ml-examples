"""Tests for Gaussian edge posterior distribution helpers."""

import jax
import jax.numpy as jnp
import numpy as np

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.distributions import (
    ConditionalGaussianMixtureBackward,
    GaussianEdgePosterior,
    GaussianMixtureBelief,
    GaussianMixtureEdgePosterior,
)
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


def test_k1_mixture_edge_matches_gaussian_edge() -> None:
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
    q_mix = GaussianMixtureEdgePosterior.from_gaussian_edge(q_edge)

    np.testing.assert_allclose(
        np.asarray(q_mix.log_prob(z[..., 0], z[..., 1])),
        np.asarray(q_edge.log_prob(z[..., 0], z[..., 1])),
        atol=1e-12,
    )
    mix_mean, mix_cov = q_mix.edge_mean_cov()
    np.testing.assert_allclose(np.asarray(mix_mean), np.asarray(q_edge.joint_mean()), atol=1e-12)
    np.testing.assert_allclose(np.asarray(mix_cov), np.asarray(q_edge.joint_cov()), atol=1e-12)


def test_mixture_belief_weights_normalize_and_moments() -> None:
    belief = GaussianMixtureBelief(
        weights=jnp.array([[0.25, 0.75]], dtype=jnp.float64),
        mean=jnp.array([[0.0, 2.0]], dtype=jnp.float64),
        var=jnp.array([[1.0, 3.0]], dtype=jnp.float64),
    )

    mean, var = belief.mean_and_var()

    np.testing.assert_allclose(np.asarray(jnp.sum(belief.weights, axis=-1)), np.ones((1,)))
    np.testing.assert_allclose(np.asarray(mean), np.array([1.5]))
    np.testing.assert_allclose(np.asarray(var), np.array([3.25]))


def test_mixture_edge_log_prob_and_sampling_shapes() -> None:
    q_mix = GaussianMixtureEdgePosterior(
        q_filter=GaussianMixtureBelief(
            weights=jnp.array([[0.4, 0.6], [0.2, 0.8]], dtype=jnp.float64),
            mean=jnp.array([[-1.0, 1.0], [0.0, 2.0]], dtype=jnp.float64),
            var=jnp.array([[0.5, 0.7], [1.0, 1.5]], dtype=jnp.float64),
        ),
        q_backward=ConditionalGaussianMixtureBackward(
            a=jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float64),
            b=jnp.array([[0.0, 0.5], [-0.5, 0.2]], dtype=jnp.float64),
            var=jnp.array([[0.8, 0.9], [1.1, 1.2]], dtype=jnp.float64),
        ),
    )
    z_t = jnp.array([0.2, 1.1], dtype=jnp.float64)
    z_tm1 = jnp.array([-0.1, 0.4], dtype=jnp.float64)

    log_prob = q_mix.log_prob(z_t, z_tm1)
    samples_t, samples_tm1, components = q_mix.sample(jax.random.PRNGKey(3), sample_shape=(5,))

    assert log_prob.shape == (2,)
    assert jnp.all(jnp.isfinite(log_prob))
    assert samples_t.shape == (5, 2)
    assert samples_tm1.shape == (5, 2)
    assert components.shape == (5, 2)


def _bivariate_log_prob(value: jax.Array, mean: jax.Array, cov: jax.Array) -> jax.Array:
    centered = value - mean
    solve = jnp.linalg.solve(cov, centered[..., None])[..., 0]
    quadratic = jnp.sum(centered * solve, axis=-1)
    log_det = jnp.linalg.slogdet(cov)[1]
    return -0.5 * (2.0 * jnp.log(2.0 * jnp.pi) + log_det + quadratic)
