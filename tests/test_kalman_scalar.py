import numpy as np

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.kalman import kalman_filter_scalar, measurement_predictive_scalar
from vbf.models.cells import init_structured_mlp_params, run_structured_mlp_filter


def test_vbf_package_imports() -> None:
    import vbf

    assert "kalman" in vbf.__all__


def test_kalman_scalar_shapes() -> None:
    data_config = LinearGaussianDataConfig(batch_size=4, time_steps=7)
    params = LinearGaussianParams(q=0.1, r=0.2, m0=1.0, p0=2.0)
    batch = make_linear_gaussian_batch(data_config, params, seed=0)

    outputs = kalman_filter_scalar(batch, params)

    assert outputs.filter_mean.shape == (4, 7)
    assert outputs.filter_var.shape == (4, 7)
    assert outputs.pred_mean.shape == (4, 7)
    assert outputs.pred_var.shape == (4, 7)
    assert outputs.predictive_mean.shape == (4, 7)
    assert outputs.predictive_var.shape == (4, 7)
    assert np.all(np.asarray(outputs.filter_var) > 0)


def test_measurement_predictive_variance_formula() -> None:
    data_config = LinearGaussianDataConfig(batch_size=3, time_steps=5)
    params = LinearGaussianParams(q=0.1, r=0.2, m0=1.0, p0=2.0)
    batch = make_linear_gaussian_batch(data_config, params, seed=1)

    kalman = kalman_filter_scalar(batch, params)
    predictive = measurement_predictive_scalar(batch, params)

    expected = batch.x**2 * kalman.pred_var + params.r
    np.testing.assert_allclose(np.asarray(predictive.var), np.asarray(expected), atol=1e-12)


def test_structured_mlp_zero_init_forward_matches_kalman() -> None:
    import jax

    data_config = LinearGaussianDataConfig(batch_size=4, time_steps=7)
    params = LinearGaussianParams(q=0.1, r=0.2, m0=1.0, p0=2.0)
    batch = make_linear_gaussian_batch(data_config, params, seed=2)
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    kalman = kalman_filter_scalar(batch, params)
    outputs = run_structured_mlp_filter(mlp_params, batch, params, min_var=0.0)

    np.testing.assert_allclose(
        np.asarray(outputs.filter_mean),
        np.asarray(kalman.filter_mean),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(outputs.filter_var),
        np.asarray(kalman.filter_var),
        atol=1e-6,
    )
