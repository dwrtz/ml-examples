"""Training objectives for variational filtering."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vbf.data import EpisodeBatch, LinearGaussianParams  # noqa: E402
from vbf.kalman import EdgeOracleOutputs  # noqa: E402
from vbf.models.cells import edge_mean_cov_from_outputs, run_structured_mlp_filter  # noqa: E402


def supervised_edge_kl_loss(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    oracle: EdgeOracleOutputs,
    *,
    min_var: float = 1e-6,
) -> jax.Array:
    """Mean `KL(q_oracle_edge || q_learned_edge)` over batch and time."""

    outputs = run_structured_mlp_filter(mlp_params, batch, state_params, min_var=min_var)
    pred_mean, pred_cov = edge_mean_cov_from_outputs(outputs)
    kl = gaussian_kl(oracle.edge_mean, oracle.edge_cov, pred_mean, pred_cov)
    return jnp.mean(kl)


def gaussian_kl(
    mean_p: jax.Array,
    cov_p: jax.Array,
    mean_q: jax.Array,
    cov_q: jax.Array,
) -> jax.Array:
    """Closed-form `KL(N_p || N_q)` for 2D Gaussian arrays."""

    delta = mean_q - mean_p
    solve_cov = jnp.linalg.solve(cov_q, cov_p)
    solve_delta = jnp.linalg.solve(cov_q, delta[..., None])[..., 0]
    trace_term = jnp.trace(solve_cov, axis1=-2, axis2=-1)
    quad_term = jnp.sum(delta * solve_delta, axis=-1)
    logdet_p = jnp.linalg.slogdet(cov_p)[1]
    logdet_q = jnp.linalg.slogdet(cov_q)[1]
    return 0.5 * (logdet_q - logdet_p - 2.0 + trace_term + quad_term)
