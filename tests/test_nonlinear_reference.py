import jax.numpy as jnp

from vbf.data import LinearGaussianParams
from vbf.nonlinear import (
    GridReferenceConfig,
    NonlinearDataConfig,
    make_nonlinear_batch,
    nonlinear_grid_filter,
    nonlinear_observation_mean,
)


def test_nonlinear_batch_generates_x_y_and_z() -> None:
    config = NonlinearDataConfig(batch_size=4, time_steps=8, observation="x_sine")
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)

    batch = make_nonlinear_batch(config, params, seed=123)

    assert batch.x.shape == (4, 8)
    assert batch.y.shape == (4, 8)
    assert batch.z.shape == (4, 8)


def test_x_sine_observation_keeps_weak_observability_role_for_x() -> None:
    z = jnp.asarray([0.0, 1.0, 2.0])
    x = jnp.asarray([0.0, 2.0, 0.0])

    mean = nonlinear_observation_mean(z, x, observation="x_sine")

    assert mean[0] == 0.0
    assert mean[2] == 0.0
    assert mean[1] != 0.0


def test_nonlinear_grid_filter_shapes_and_positive_variance() -> None:
    config = NonlinearDataConfig(batch_size=3, time_steps=6, observation="x_sine")
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_nonlinear_batch(config, params, seed=124)

    reference = nonlinear_grid_filter(
        batch,
        params,
        data_config=config,
        grid_config=GridReferenceConfig(grid_min=-12.0, grid_max=12.0, num_grid=401),
    )

    assert reference.filter_mean.shape == (3, 6)
    assert reference.filter_var.shape == (3, 6)
    assert reference.predictive_mean.shape == (3, 6)
    assert reference.predictive_var.shape == (3, 6)
    assert jnp.all(reference.filter_var > 0.0)
    assert jnp.all(reference.predictive_var > 0.0)


def test_zero_x_reference_variance_grows_from_transition_noise() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=6, x_pattern="zero")
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=125)

    reference = nonlinear_grid_filter(
        batch,
        params,
        data_config=config,
        grid_config=GridReferenceConfig(grid_min=-10.0, grid_max=10.0, num_grid=501),
    )

    mean_var_t = jnp.mean(reference.filter_var, axis=0)
    assert mean_var_t[-1] > mean_var_t[0]
