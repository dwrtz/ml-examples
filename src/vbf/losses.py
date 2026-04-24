"""Training objectives for variational filtering."""

from __future__ import annotations

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vbf.data import EpisodeBatch, LinearGaussianParams  # noqa: E402
from vbf.kalman import EdgeOracleOutputs  # noqa: E402
from vbf.models.cells import (  # noqa: E402
    edge_mean_cov_from_outputs,
    run_structured_mlp_filter,
    run_structured_mlp_teacher_forced,
)


LOG_2PI = jnp.log(2.0 * jnp.pi)


class EdgeElboTerms(NamedTuple):
    """Sample-averaged local edge ELBO terms with batch-time shape."""

    log_likelihood: jax.Array
    log_transition: jax.Array
    log_prev_filter: jax.Array
    neg_log_current_filter: jax.Array
    neg_log_backward: jax.Array
    elbo: jax.Array


def supervised_edge_kl_loss(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    oracle: EdgeOracleOutputs,
    *,
    min_var: float = 1e-6,
) -> jax.Array:
    """Mean teacher-forced `KL(q_oracle_edge || q_learned_edge)`.

    Supervised distillation trains the local update with oracle previous
    filtering beliefs as inputs. Self-fed rollout is kept for evaluation.
    """

    outputs = run_structured_mlp_teacher_forced(
        mlp_params,
        batch,
        state_params,
        oracle.filter_mean,
        oracle.filter_var,
        min_var=min_var,
    )
    pred_mean, pred_cov = edge_mean_cov_from_outputs(outputs)
    kl = gaussian_kl(oracle.edge_mean, oracle.edge_cov, pred_mean, pred_cov)
    return jnp.mean(kl)


def edge_elbo_loss(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    num_samples: int = 8,
    min_var: float = 1e-6,
) -> jax.Array:
    """Negative mean local edge ELBO using reparameterized samples.

    The learned filter is rolled out strictly, so the previous filtering belief
    in the ELBO is the model's own carried belief rather than an oracle target.
    """

    terms = edge_elbo_terms(
        mlp_params,
        batch,
        state_params,
        key,
        num_samples=num_samples,
        min_var=min_var,
    )
    return -jnp.mean(terms.elbo)


def edge_elbo_terms(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    num_samples: int = 8,
    min_var: float = 1e-6,
) -> EdgeElboTerms:
    """Return sample-averaged local edge ELBO terms for diagnostics."""

    outputs = run_structured_mlp_filter(mlp_params, batch, state_params, min_var=min_var)
    prev_filter_mean, prev_filter_var = _previous_filter_beliefs(outputs.filter_mean, outputs.filter_var, state_params)
    return edge_elbo_terms_from_factors(
        batch,
        state_params,
        key,
        filter_mean=outputs.filter_mean,
        filter_var=outputs.filter_var,
        backward_a=outputs.backward_a,
        backward_b=outputs.backward_b,
        backward_var=outputs.backward_var,
        prev_filter_mean=prev_filter_mean,
        prev_filter_var=prev_filter_var,
        num_samples=num_samples,
    )


def oracle_edge_elbo_terms(
    oracle: EdgeOracleOutputs,
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    num_samples: int = 8,
    min_var: float = 1e-9,
) -> EdgeElboTerms:
    """Return local edge ELBO terms under the exact Kalman edge posterior."""

    edge_cov = oracle.edge_cov
    filter_var = jnp.maximum(oracle.filter_var, min_var)
    backward_a = edge_cov[..., 0, 1] / filter_var
    backward_b = oracle.edge_mean[..., 1] - backward_a * oracle.filter_mean
    backward_var = jnp.maximum(edge_cov[..., 1, 1] - edge_cov[..., 0, 1] ** 2 / filter_var, min_var)
    prev_filter_mean, prev_filter_var = _previous_filter_beliefs(
        oracle.filter_mean,
        oracle.filter_var,
        state_params,
    )
    return edge_elbo_terms_from_factors(
        batch,
        state_params,
        key,
        filter_mean=oracle.filter_mean,
        filter_var=filter_var,
        backward_a=backward_a,
        backward_b=backward_b,
        backward_var=backward_var,
        prev_filter_mean=prev_filter_mean,
        prev_filter_var=prev_filter_var,
        num_samples=num_samples,
    )


def edge_elbo_terms_from_factors(
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    filter_mean: jax.Array,
    filter_var: jax.Array,
    backward_a: jax.Array,
    backward_b: jax.Array,
    backward_var: jax.Array,
    prev_filter_mean: jax.Array,
    prev_filter_var: jax.Array,
    num_samples: int = 8,
) -> EdgeElboTerms:
    """Return local edge ELBO terms from `q^F_t q^B_t` factors."""

    eps_t_key, eps_tm1_key = jax.random.split(key)
    sample_shape = (num_samples,) + filter_mean.shape
    eps_t = jax.random.normal(eps_t_key, shape=sample_shape, dtype=filter_mean.dtype)
    eps_tm1 = jax.random.normal(eps_tm1_key, shape=sample_shape, dtype=filter_mean.dtype)

    filter_std = jnp.sqrt(filter_var)
    backward_std = jnp.sqrt(backward_var)
    z_t = filter_mean[None, ...] + filter_std[None, ...] * eps_t
    backward_mean = backward_a[None, ...] * z_t + backward_b[None, ...]
    z_tm1 = backward_mean + backward_std[None, ...] * eps_tm1

    log_likelihood = _normal_log_prob(
        batch.y[None, ...],
        batch.x[None, ...] * z_t,
        jnp.asarray(state_params.r, dtype=z_t.dtype),
    )
    log_transition = _normal_log_prob(
        z_t,
        z_tm1,
        jnp.asarray(state_params.q, dtype=z_t.dtype),
    )
    log_prev_filter = _normal_log_prob(
        z_tm1,
        prev_filter_mean[None, ...],
        prev_filter_var[None, ...],
    )
    log_current_filter = _normal_log_prob(
        z_t,
        filter_mean[None, ...],
        filter_var[None, ...],
    )
    log_backward = _normal_log_prob(
        z_tm1,
        backward_mean,
        backward_var[None, ...],
    )

    neg_log_current_filter = -log_current_filter
    neg_log_backward = -log_backward
    elbo = log_likelihood + log_transition + log_prev_filter + neg_log_current_filter + neg_log_backward

    return EdgeElboTerms(
        log_likelihood=jnp.mean(log_likelihood, axis=0),
        log_transition=jnp.mean(log_transition, axis=0),
        log_prev_filter=jnp.mean(log_prev_filter, axis=0),
        neg_log_current_filter=jnp.mean(neg_log_current_filter, axis=0),
        neg_log_backward=jnp.mean(neg_log_backward, axis=0),
        elbo=jnp.mean(elbo, axis=0),
    )


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


def _previous_filter_beliefs(
    filter_mean: jax.Array,
    filter_var: jax.Array,
    state_params: LinearGaussianParams,
) -> tuple[jax.Array, jax.Array]:
    initial_mean = jnp.full((filter_mean.shape[0], 1), state_params.m0, dtype=filter_mean.dtype)
    initial_var = jnp.full((filter_var.shape[0], 1), state_params.p0, dtype=filter_var.dtype)
    return (
        jnp.concatenate((initial_mean, filter_mean[:, :-1]), axis=1),
        jnp.concatenate((initial_var, filter_var[:, :-1]), axis=1),
    )


def _normal_log_prob(value: jax.Array, mean: jax.Array, var: jax.Array) -> jax.Array:
    return -0.5 * (LOG_2PI + jnp.log(var) + (value - mean) ** 2 / var)
