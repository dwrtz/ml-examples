import jax
import jax.numpy as jnp

from vbf.data import LinearGaussianParams
from vbf.models.cells import init_structured_mlp_params
from vbf.nonlinear import (
    GridReferenceConfig,
    NonlinearDataConfig,
    make_nonlinear_batch,
    nonlinear_grid_filter,
    nonlinear_grid_filter_masses,
    nonlinear_grid_filter_shape_diagnostics,
    nonlinear_observation_mean,
    nonlinear_preassimilation_log_prob_y,
    nonlinear_predictive_moments_from_filter,
    run_nonlinear_structured_mlp_filter,
    run_nonlinear_structured_mlp_teacher_forced,
)
from vbf.nonlinear_cache import load_or_compute_nonlinear_reference


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


def test_nonlinear_grid_filter_stays_finite_with_sharp_likelihood() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=10, observation="x_sine")
    params = LinearGaussianParams(q=0.03, r=0.001, m0=1.0, p0=10.0)
    batch = make_nonlinear_batch(config, params, seed=126)

    reference = nonlinear_grid_filter(
        batch,
        params,
        data_config=config,
        grid_config=GridReferenceConfig(grid_min=-18.0, grid_max=18.0, num_grid=901),
    )

    assert jnp.all(jnp.isfinite(reference.filter_mean))
    assert jnp.all(jnp.isfinite(reference.filter_var))
    assert jnp.all(jnp.isfinite(reference.predictive_mean))
    assert jnp.all(jnp.isfinite(reference.predictive_var))


def test_nonlinear_grid_filter_shape_diagnostics_are_finite() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=5, observation="x_sine")
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=130)

    shape = nonlinear_grid_filter_shape_diagnostics(
        batch,
        params,
        data_config=config,
        grid_config=GridReferenceConfig(grid_min=-8.0, grid_max=8.0, num_grid=101),
    )

    assert shape.entropy.shape == (2, 5)
    assert shape.peak_count.shape == (2, 5)
    assert jnp.all(jnp.isfinite(shape.entropy))
    assert jnp.all(shape.normalized_entropy >= 0.0)
    assert jnp.all(shape.normalized_entropy <= 1.0)
    assert jnp.all(shape.peak_count >= 0.0)
    assert jnp.all(shape.credible_width_90 >= 0.0)


def test_nonlinear_grid_filter_masses_are_normalized() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=4, observation="x_sine")
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=131)

    grid_outputs = nonlinear_grid_filter_masses(
        batch,
        params,
        data_config=config,
        grid_config=GridReferenceConfig(grid_min=-8.0, grid_max=8.0, num_grid=101),
    )

    assert grid_outputs.grid.shape == (101,)
    assert grid_outputs.filter_mass.shape == (2, 4, 101)
    assert jnp.allclose(jnp.sum(grid_outputs.filter_mass, axis=-1), 1.0)


def test_x_sine_predictive_moments_are_finite_and_positive() -> None:
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    filter_mean = jnp.zeros((3, 5))
    filter_var = jnp.ones((3, 5))
    x = jnp.ones((3, 5))

    mean, var = nonlinear_predictive_moments_from_filter(
        filter_mean,
        filter_var,
        x,
        params,
        observation="x_sine",
    )

    assert mean.shape == (3, 5)
    assert var.shape == (3, 5)
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(var > 0.0)


def test_nonlinear_preassimilation_log_prob_y_is_finite_and_batched() -> None:
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    prev_mean = jnp.zeros((3, 5))
    prev_var = jnp.ones((3, 5))
    x = jnp.ones((3, 5))
    y = jnp.zeros((3, 5))

    log_prob = nonlinear_preassimilation_log_prob_y(
        prev_mean,
        prev_var,
        x,
        y,
        params,
        observation="x_sine",
        num_points=16,
    )

    assert log_prob.shape == (3, 5)
    assert jnp.all(jnp.isfinite(log_prob))


def test_nonlinear_preassimilation_log_prob_y_zero_x_matches_observation_noise() -> None:
    params = LinearGaussianParams(q=0.1, r=0.3, m0=1.0, p0=2.0)
    prev_mean = jnp.zeros((2, 4))
    prev_var = jnp.ones((2, 4))
    x = jnp.zeros((2, 4))
    y = jnp.asarray([[0.0, 0.5, -1.0, 1.5], [0.1, -0.2, 0.3, -0.4]])

    log_prob = nonlinear_preassimilation_log_prob_y(
        prev_mean,
        prev_var,
        x,
        y,
        params,
        observation="x_sine",
        num_points=16,
    )
    expected = -0.5 * (jnp.log(2.0 * jnp.pi * params.r) + y**2 / params.r)

    assert jnp.allclose(log_prob, expected)


def test_nonlinear_structured_mlp_filter_outputs_positive_variances() -> None:
    config = NonlinearDataConfig(batch_size=3, time_steps=7, observation="x_sine")
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=127)
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(128), hidden_dim=8)

    outputs = run_nonlinear_structured_mlp_filter(
        mlp_params,
        batch,
        params,
        observation="x_sine",
    )

    assert outputs.filter_mean.shape == (3, 7)
    assert outputs.filter_var.shape == (3, 7)
    assert outputs.backward_var.shape == (3, 7)
    assert jnp.all(outputs.filter_var > 0.0)
    assert jnp.all(outputs.backward_var > 0.0)


def test_nonlinear_structured_mlp_teacher_forced_outputs_positive_variances() -> None:
    config = NonlinearDataConfig(batch_size=3, time_steps=7, observation="x_sine")
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=132)
    reference = nonlinear_grid_filter(
        batch,
        params,
        data_config=config,
        grid_config=GridReferenceConfig(grid_min=-8.0, grid_max=8.0, num_grid=101),
    )
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(133), hidden_dim=8)

    outputs = run_nonlinear_structured_mlp_teacher_forced(
        mlp_params,
        batch,
        params,
        reference.filter_mean,
        reference.filter_var,
        observation="x_sine",
    )

    assert outputs.filter_mean.shape == (3, 7)
    assert outputs.filter_var.shape == (3, 7)
    assert jnp.all(outputs.filter_var > 0.0)


def test_nonlinear_reference_cache_hits_on_second_load(tmp_path) -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=4, observation="x_sine")
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    grid_config = GridReferenceConfig(grid_min=-8.0, grid_max=8.0, num_grid=101)

    first = load_or_compute_nonlinear_reference(
        config,
        params,
        seed=129,
        grid_config=grid_config,
        cache_dir=tmp_path,
    )
    second = load_or_compute_nonlinear_reference(
        config,
        params,
        seed=129,
        grid_config=grid_config,
        cache_dir=tmp_path,
    )

    assert not first.cache_hit
    assert second.cache_hit
    assert first.cache_path == second.cache_path
    assert jnp.allclose(first.batch.z, second.batch.z)
    assert jnp.allclose(first.reference.filter_var, second.reference.filter_var)
