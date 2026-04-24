"""Tests for deterministic data generation and training setup."""

import numpy as np

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.metrics import rmse_over_batch


def test_linear_gaussian_batch_reproducible_for_seed() -> None:
    data_config = LinearGaussianDataConfig(batch_size=5, time_steps=9)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)

    first = make_linear_gaussian_batch(data_config, params, seed=123)
    second = make_linear_gaussian_batch(data_config, params, seed=123)

    np.testing.assert_array_equal(np.asarray(first.x), np.asarray(second.x))
    np.testing.assert_array_equal(np.asarray(first.y), np.asarray(second.y))
    np.testing.assert_array_equal(np.asarray(first.z), np.asarray(second.z))


def test_linear_gaussian_batch_shapes() -> None:
    data_config = LinearGaussianDataConfig(batch_size=5, time_steps=9)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)

    batch = make_linear_gaussian_batch(data_config, params, seed=123)

    assert batch.x.shape == (5, 9)
    assert batch.y.shape == (5, 9)
    assert batch.z.shape == (5, 9)


def test_rmse_over_batch_shape() -> None:
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.0, 1.0], [1.0, 2.0]])

    rmse = rmse_over_batch(pred, target)

    assert rmse.shape == (2,)
