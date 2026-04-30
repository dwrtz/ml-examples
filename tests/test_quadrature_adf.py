import importlib.util
import sys
from pathlib import Path

import numpy as np

from vbf.data import LinearGaussianParams
from vbf.nonlinear import NonlinearDataConfig, make_nonlinear_batch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sweep_quadrature_adf.py"
SPEC = importlib.util.spec_from_file_location("sweep_quadrature_adf", MODULE_PATH)
assert SPEC is not None
quadrature_adf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = quadrature_adf
SPEC.loader.exec_module(quadrature_adf)
run_quadrature_adf_filter = quadrature_adf.run_quadrature_adf_filter


def test_quadrature_adf_gaussian_outputs_are_finite() -> None:
    config = NonlinearDataConfig(batch_size=3, time_steps=5)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=201)

    outputs = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        components=1,
        likelihood_power=1.0,
        init_span=0.0,
        num_points=16,
        em_steps=5,
    )

    assert outputs.filter_mean.shape == (3, 5)
    assert outputs.filter_var.shape == (3, 5)
    assert outputs.weights.shape == (3, 5, 1)
    assert np.all(np.isfinite(outputs.predictive_y_log_prob))
    assert np.all(outputs.filter_var > 0.0)


def test_quadrature_adf_mixture_weights_are_normalized() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=4)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=202)

    outputs = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        components=4,
        likelihood_power=0.5,
        init_span=2.0 * np.pi,
        num_points=16,
        em_steps=5,
    )

    assert outputs.weights.shape == (2, 4, 4)
    assert np.allclose(np.sum(outputs.weights, axis=-1), 1.0)
    assert np.all(outputs.component_var > 0.0)


def test_zero_x_predictive_y_matches_observation_noise() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=4, x_pattern="zero")
    params = LinearGaussianParams(q=0.1, r=0.3, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=203)

    outputs = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        components=2,
        likelihood_power=1.0,
        init_span=0.0,
        num_points=16,
        em_steps=5,
    )
    expected = -0.5 * (
        np.log(2.0 * np.pi * float(params.r)) + np.asarray(batch.y) ** 2 / float(params.r)
    )

    assert np.allclose(outputs.predictive_y_log_prob, expected)
