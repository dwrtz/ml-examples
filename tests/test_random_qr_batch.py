import jax.numpy as jnp

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.kalman import kalman_edge_posterior_scalar
from vbf.models.cells import init_structured_mlp_params, run_structured_mlp_filter
from vbf.predictive import linear_gaussian_predictive_from_filter


def test_linear_gaussian_batch_accepts_per_episode_qr() -> None:
    config = LinearGaussianDataConfig(batch_size=4, time_steps=8)
    params = LinearGaussianParams(
        q=jnp.asarray([0.03, 0.1, 0.3, 0.1]),
        r=jnp.asarray([0.3, 0.1, 0.03, 0.1]),
        m0=1.0,
        p0=10.0,
    )

    batch = make_linear_gaussian_batch(config, params, seed=123)
    oracle = kalman_edge_posterior_scalar(batch, params)

    assert batch.x.shape == (4, 8)
    assert oracle.filter_mean.shape == (4, 8)
    assert oracle.edge_cov.shape == (4, 8, 2, 2)
    assert jnp.all(oracle.filter_var > 0.0)


def test_structured_mlp_and_predictive_support_per_episode_qr() -> None:
    config = LinearGaussianDataConfig(batch_size=3, time_steps=6)
    params = LinearGaussianParams(
        q=jnp.asarray([0.03, 0.1, 0.3]),
        r=jnp.asarray([0.3, 0.1, 0.03]),
        m0=1.0,
        p0=10.0,
    )
    batch = make_linear_gaussian_batch(config, params, seed=124)
    mlp_params = init_structured_mlp_params(jnp.asarray([0, 1], dtype=jnp.uint32))

    outputs = run_structured_mlp_filter(mlp_params, batch, params)
    predictive = linear_gaussian_predictive_from_filter(
        outputs.filter_mean,
        outputs.filter_var,
        batch,
        params,
    )

    assert outputs.filter_mean.shape == (3, 6)
    assert predictive.mean.shape == (3, 6)
    assert jnp.all(outputs.filter_var > 0.0)
    assert jnp.all(predictive.var > 0.0)
