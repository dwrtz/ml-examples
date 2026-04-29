"""Shape tests for local edge ELBO calculations."""

import jax
import numpy as np

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.kalman import kalman_edge_posterior_scalar
from vbf.losses import (
    edge_elbo_closed_form_terms_from_outputs,
    edge_elbo_closed_form_loss,
    edge_elbo_loss,
    edge_elbo_terms,
    filter_variance_ratio_penalty,
    filter_variance_ratio_over_time_penalty,
    low_observation_filter_variance_ratio_penalty,
    oracle_edge_elbo_closed_form_terms,
    oracle_edge_elbo_terms,
    regime_filter_variance_ratio_penalty,
    supervised_edge_kl_loss,
    transition_consistency_penalty,
)
from vbf.models.cells import (
    edge_mean_cov_from_outputs,
    init_direct_mixture_mlp_params,
    init_direct_mlp_params,
    init_split_head_mlp_params,
    init_structured_mlp_params,
    run_direct_mixture_mlp_filter,
    run_direct_mlp_filter,
    run_direct_mlp_teacher_forced,
    run_split_head_mlp_filter,
    run_split_head_mlp_teacher_forced,
    run_structured_mlp_filter,
    run_structured_mlp_teacher_forced,
)


def test_structured_mlp_edge_shapes() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=12,
    )
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    outputs = run_structured_mlp_filter(mlp_params, batch, state_params)
    edge_mean, edge_cov = edge_mean_cov_from_outputs(outputs)

    assert edge_mean.shape == (3, 5, 2)
    assert edge_cov.shape == (3, 5, 2, 2)


def test_supervised_edge_kl_loss_is_scalar() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=13,
    )
    oracle = kalman_edge_posterior_scalar(batch, state_params)
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    loss = supervised_edge_kl_loss(mlp_params, batch, state_params, oracle)

    assert loss.shape == ()


def test_edge_elbo_loss_is_scalar() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=15,
    )
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    loss = edge_elbo_loss(
        mlp_params,
        batch,
        state_params,
        jax.random.PRNGKey(1),
        num_samples=4,
    )

    assert loss.shape == ()


def test_direct_edge_elbo_loss_is_scalar() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=151,
    )
    mlp_params = init_direct_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    loss = edge_elbo_loss(
        mlp_params,
        batch,
        state_params,
        jax.random.PRNGKey(1),
        num_samples=4,
        direct=True,
    )

    assert loss.shape == ()


def test_closed_form_edge_elbo_loss_is_scalar() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=152,
    )
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    loss = edge_elbo_closed_form_loss(mlp_params, batch, state_params)

    assert loss.shape == ()


def test_edge_elbo_terms_are_batch_time_shaped() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=16,
    )
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    terms = edge_elbo_terms(
        mlp_params,
        batch,
        state_params,
        jax.random.PRNGKey(1),
        num_samples=4,
    )

    assert terms.log_likelihood.shape == (3, 5)
    assert terms.log_transition.shape == (3, 5)
    assert terms.log_prev_filter.shape == (3, 5)
    assert terms.neg_log_current_filter.shape == (3, 5)
    assert terms.neg_log_backward.shape == (3, 5)
    assert terms.elbo.shape == (3, 5)


def test_oracle_edge_elbo_terms_are_batch_time_shaped() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=17,
    )
    oracle = kalman_edge_posterior_scalar(batch, state_params)

    terms = oracle_edge_elbo_terms(
        oracle,
        batch,
        state_params,
        jax.random.PRNGKey(1),
        num_samples=4,
    )

    assert terms.log_likelihood.shape == (3, 5)
    assert terms.log_transition.shape == (3, 5)
    assert terms.log_prev_filter.shape == (3, 5)
    assert terms.neg_log_current_filter.shape == (3, 5)
    assert terms.neg_log_backward.shape == (3, 5)
    assert terms.elbo.shape == (3, 5)


def test_closed_form_edge_elbo_terms_are_batch_time_shaped() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=19,
    )
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)
    outputs = run_structured_mlp_filter(mlp_params, batch, state_params)

    terms = edge_elbo_closed_form_terms_from_outputs(outputs, batch, state_params)

    assert terms.log_likelihood.shape == (3, 5)
    assert terms.log_transition.shape == (3, 5)
    assert terms.log_prev_filter.shape == (3, 5)
    assert terms.neg_log_current_filter.shape == (3, 5)
    assert terms.neg_log_backward.shape == (3, 5)
    assert terms.elbo.shape == (3, 5)


def test_oracle_closed_form_edge_elbo_matches_large_mc_estimate() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=2, time_steps=3),
        state_params,
        seed=20,
    )
    oracle = kalman_edge_posterior_scalar(batch, state_params)

    closed_form = oracle_edge_elbo_closed_form_terms(oracle, batch, state_params)
    mc = oracle_edge_elbo_terms(
        oracle,
        batch,
        state_params,
        jax.random.PRNGKey(1),
        num_samples=200_000,
    )

    np.testing.assert_allclose(np.asarray(mc.elbo), np.asarray(closed_form.elbo), atol=7e-3)


def test_transition_consistency_penalty_is_scalar() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=18,
    )
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)
    outputs = run_structured_mlp_filter(mlp_params, batch, state_params)

    penalty = transition_consistency_penalty(outputs, state_params)

    assert penalty.shape == ()


def test_filter_variance_ratio_penalty() -> None:
    filter_var = jax.numpy.array([2.0, 2.0])
    oracle_var = jax.numpy.array([1.0, 1.0])

    penalty = filter_variance_ratio_penalty(filter_var, oracle_var)

    np.testing.assert_allclose(float(penalty), float(jax.numpy.log(2.0) ** 2))


def test_filter_variance_ratio_over_time_penalty() -> None:
    filter_var = jax.numpy.array([[2.0, 1.0], [2.0, 4.0]])
    oracle_var = jax.numpy.array([[1.0, 1.0], [1.0, 1.0]])

    penalty = filter_variance_ratio_over_time_penalty(filter_var, oracle_var)

    expected = jax.numpy.mean(jax.numpy.log(jax.numpy.array([2.0, 2.5])) ** 2)
    np.testing.assert_allclose(float(penalty), float(expected))


def test_regime_filter_variance_ratio_penalty() -> None:
    filter_var = jax.numpy.array([[2.0, 2.0], [4.0, 4.0]])
    oracle_var = jax.numpy.array([[1.0, 1.0], [2.0, 2.0]])

    penalty = regime_filter_variance_ratio_penalty(filter_var, oracle_var)

    np.testing.assert_allclose(float(penalty), float(jax.numpy.log(2.0) ** 2))


def test_low_observation_filter_variance_ratio_penalty_weights_low_x() -> None:
    filter_var = jax.numpy.array([[2.0, 2.0], [2.0, 2.0]])
    oracle_var = jax.numpy.array([[1.0, 1.0], [1.0, 1.0]])
    x = jax.numpy.array([[0.0, 10.0], [0.0, 10.0]])

    penalty = low_observation_filter_variance_ratio_penalty(filter_var, oracle_var, x)

    np.testing.assert_allclose(float(penalty), float(jax.numpy.log(2.0) ** 2))


def test_elbo_variance_ratio_penalties_are_scalar() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=153,
    )
    oracle = kalman_edge_posterior_scalar(batch, state_params)
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    loss = edge_elbo_loss(
        mlp_params,
        batch,
        state_params,
        jax.random.PRNGKey(1),
        num_samples=4,
        oracle=oracle,
        variance_ratio_weight=0.1,
        time_variance_ratio_weight=0.1,
        low_observation_variance_ratio_weight=0.1,
        regime_variance_ratio_weight=0.1,
    )

    assert loss.shape == ()


def test_structured_mlp_teacher_forced_shapes() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=14,
    )
    oracle = kalman_edge_posterior_scalar(batch, state_params)
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    outputs = run_structured_mlp_teacher_forced(
        mlp_params,
        batch,
        state_params,
        oracle.filter_mean,
        oracle.filter_var,
    )

    assert outputs.filter_mean.shape == (3, 5)
    assert outputs.filter_var.shape == (3, 5)


def test_split_head_mlp_filter_shapes() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=21,
    )
    oracle = kalman_edge_posterior_scalar(batch, state_params)
    mlp_params = init_split_head_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    rollout_outputs = run_split_head_mlp_filter(mlp_params, batch, state_params)
    teacher_outputs = run_split_head_mlp_teacher_forced(
        mlp_params,
        batch,
        state_params,
        oracle.filter_mean,
        oracle.filter_var,
    )

    assert rollout_outputs.filter_mean.shape == (3, 5)
    assert rollout_outputs.filter_var.shape == (3, 5)
    assert teacher_outputs.filter_mean.shape == (3, 5)
    assert teacher_outputs.backward_var.shape == (3, 5)


def test_direct_mlp_filter_shapes() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=22,
    )
    oracle = kalman_edge_posterior_scalar(batch, state_params)
    mlp_params = init_direct_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    rollout_outputs = run_direct_mlp_filter(mlp_params, batch, state_params)
    teacher_outputs = run_direct_mlp_teacher_forced(
        mlp_params,
        batch,
        state_params,
        oracle.filter_mean,
        oracle.filter_var,
    )

    assert rollout_outputs.filter_mean.shape == (3, 5)
    assert rollout_outputs.filter_var.shape == (3, 5)
    assert teacher_outputs.filter_mean.shape == (3, 5)
    assert teacher_outputs.backward_var.shape == (3, 5)


def test_direct_mixture_mlp_filter_shapes_and_moments() -> None:
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=3, time_steps=5),
        state_params,
        seed=23,
    )
    mlp_params = init_direct_mixture_mlp_params(
        jax.random.PRNGKey(0),
        hidden_dim=8,
        num_components=2,
    )

    outputs = run_direct_mixture_mlp_filter(
        mlp_params,
        batch,
        state_params,
        num_components=2,
    )
    edge_mean, edge_cov = edge_mean_cov_from_outputs(outputs)

    assert outputs.filter_weights.shape == (3, 5, 2)
    assert outputs.component_mean.shape == (3, 5, 2)
    assert outputs.component_var.shape == (3, 5, 2)
    assert outputs.filter_mean.shape == (3, 5)
    assert outputs.filter_var.shape == (3, 5)
    assert edge_mean.shape == (3, 5, 2)
    assert edge_cov.shape == (3, 5, 2, 2)
    np.testing.assert_allclose(
        np.asarray(outputs.filter_weights.sum(axis=-1)),
        np.ones((3, 5)),
        atol=1e-12,
    )


def test_direct_mixture_mlp_component_mean_spread_initialization() -> None:
    params = init_direct_mixture_mlp_params(
        jax.random.PRNGKey(0),
        hidden_dim=8,
        num_components=4,
        component_mean_init_span=6.0,
    )
    b2 = np.asarray(params["b2"]).reshape(4, 6)

    np.testing.assert_allclose(b2[:, 1], np.linspace(-3.0, 3.0, 4), atol=1e-12)
