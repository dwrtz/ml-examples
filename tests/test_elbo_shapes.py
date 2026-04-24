"""Shape tests for local edge ELBO calculations."""

import jax

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.kalman import kalman_edge_posterior_scalar
from vbf.losses import edge_elbo_loss, edge_elbo_terms, supervised_edge_kl_loss
from vbf.models.cells import (
    edge_mean_cov_from_outputs,
    init_structured_mlp_params,
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
