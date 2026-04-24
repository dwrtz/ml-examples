"""Shape tests for local edge ELBO calculations."""

import jax

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.kalman import kalman_edge_posterior_scalar
from vbf.losses import supervised_edge_kl_loss
from vbf.models.cells import edge_mean_cov_from_outputs, init_structured_mlp_params, run_structured_mlp_filter


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
