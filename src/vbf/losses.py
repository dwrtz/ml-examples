"""Training objectives for variational filtering."""

from __future__ import annotations

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from vbf.data import EpisodeBatch, LinearGaussianParams  # noqa: E402
from vbf.kalman import EdgeOracleOutputs  # noqa: E402
from vbf.models.cells import (  # noqa: E402
    StructuredMLPOutputs,
    edge_mean_cov_from_outputs,
    run_direct_mlp_filter,
    run_structured_mlp_filter,
    run_structured_mlp_teacher_forced,
)
from vbf.predictive import previous_filter_beliefs  # noqa: E402


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


def self_fed_supervised_edge_kl_loss(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    oracle: EdgeOracleOutputs,
    *,
    min_var: float = 1e-6,
    variance_ratio_weight: float = 0.0,
) -> jax.Array:
    """Mean rollout `KL(q_oracle_edge || q_learned_edge)`.

    Unlike `supervised_edge_kl_loss`, this trains against oracle edge targets
    while feeding the update cell its own previous filtering beliefs.
    """

    outputs = run_structured_mlp_filter(mlp_params, batch, state_params, min_var=min_var)
    pred_mean, pred_cov = edge_mean_cov_from_outputs(outputs)
    kl = gaussian_kl(oracle.edge_mean, oracle.edge_cov, pred_mean, pred_cov)
    loss = jnp.mean(kl)
    if variance_ratio_weight != 0.0:
        loss = loss + variance_ratio_weight * filter_variance_ratio_penalty(
            outputs.filter_var,
            oracle.filter_var,
        )
    return loss


def filter_variance_ratio_penalty(
    filter_var: jax.Array,
    oracle_filter_var: jax.Array,
) -> jax.Array:
    """Squared log-ratio penalty for mean filtering variance calibration."""

    ratio = jnp.mean(filter_var) / jnp.mean(oracle_filter_var)
    return jnp.log(ratio) ** 2


def edge_elbo_loss(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    num_samples: int = 8,
    min_var: float = 1e-6,
    oracle: EdgeOracleOutputs | None = None,
    edge_kl_weight: float = 0.0,
    transition_consistency_weight: float = 0.0,
    direct: bool = False,
) -> jax.Array:
    """Negative mean local edge ELBO using reparameterized samples.

    The learned filter is rolled out strictly, so the previous filtering belief
    in the ELBO is the model's own carried belief rather than an oracle target.
    """

    if direct:
        outputs = run_direct_mlp_filter(mlp_params, batch, state_params, min_var=min_var)
    else:
        outputs = run_structured_mlp_filter(mlp_params, batch, state_params, min_var=min_var)
    terms = edge_elbo_terms_from_outputs(
        outputs,
        batch,
        state_params,
        key,
        num_samples=num_samples,
    )
    loss = -jnp.mean(terms.elbo)
    if edge_kl_weight != 0.0:
        if oracle is None:
            raise ValueError("oracle is required when edge_kl_weight is nonzero")
        pred_mean, pred_cov = edge_mean_cov_from_outputs(outputs)
        loss = loss + edge_kl_weight * jnp.mean(
            gaussian_kl(oracle.edge_mean, oracle.edge_cov, pred_mean, pred_cov)
        )
    if transition_consistency_weight != 0.0:
        loss = loss + transition_consistency_weight * transition_consistency_penalty(
            outputs,
            state_params,
        )
    return loss


def edge_elbo_closed_form_loss(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    min_var: float = 1e-6,
    direct: bool = False,
) -> jax.Array:
    """Negative mean closed-form scalar Gaussian edge ELBO."""

    if direct:
        outputs = run_direct_mlp_filter(mlp_params, batch, state_params, min_var=min_var)
    else:
        outputs = run_structured_mlp_filter(mlp_params, batch, state_params, min_var=min_var)
    terms = edge_elbo_closed_form_terms_from_outputs(outputs, batch, state_params)
    return -jnp.mean(terms.elbo)


def transition_consistency_penalty(
    outputs: StructuredMLPOutputs,
    state_params: LinearGaussianParams,
) -> jax.Array:
    """Moment penalty encouraging edge residuals to match transition noise."""

    residual_mean = (1.0 - outputs.backward_a) * outputs.filter_mean - outputs.backward_b
    residual_var = (1.0 - outputs.backward_a) ** 2 * outputs.filter_var + outputs.backward_var
    q = jnp.asarray(state_params.q, dtype=outputs.filter_mean.dtype)
    mean_penalty = residual_mean**2 / q
    second_moment = residual_var + residual_mean**2
    scale_penalty = (second_moment / q - 1.0) ** 2
    return jnp.mean(mean_penalty + scale_penalty)


def edge_elbo_terms(
    mlp_params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    num_samples: int = 8,
    min_var: float = 1e-6,
    direct: bool = False,
) -> EdgeElboTerms:
    """Return sample-averaged local edge ELBO terms for diagnostics."""

    if direct:
        outputs = run_direct_mlp_filter(mlp_params, batch, state_params, min_var=min_var)
    else:
        outputs = run_structured_mlp_filter(mlp_params, batch, state_params, min_var=min_var)
    return edge_elbo_terms_from_outputs(
        outputs,
        batch,
        state_params,
        key,
        num_samples=num_samples,
    )


def edge_elbo_terms_from_outputs(
    outputs: StructuredMLPOutputs,
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    num_samples: int = 8,
) -> EdgeElboTerms:
    """Return local edge ELBO terms from structured MLP outputs."""

    prev_filter_mean, prev_filter_var = previous_filter_beliefs(
        outputs.filter_mean, outputs.filter_var, state_params
    )
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


def edge_elbo_closed_form_terms_from_outputs(
    outputs: StructuredMLPOutputs,
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
) -> EdgeElboTerms:
    """Return closed-form scalar Gaussian edge ELBO terms from model outputs."""

    prev_filter_mean, prev_filter_var = previous_filter_beliefs(
        outputs.filter_mean,
        outputs.filter_var,
        state_params,
    )
    return edge_elbo_closed_form_terms_from_factors(
        batch,
        state_params,
        filter_mean=outputs.filter_mean,
        filter_var=outputs.filter_var,
        backward_a=outputs.backward_a,
        backward_b=outputs.backward_b,
        backward_var=outputs.backward_var,
        prev_filter_mean=prev_filter_mean,
        prev_filter_var=prev_filter_var,
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
    prev_filter_mean, prev_filter_var = previous_filter_beliefs(
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


def oracle_edge_elbo_closed_form_terms(
    oracle: EdgeOracleOutputs,
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    min_var: float = 1e-9,
) -> EdgeElboTerms:
    """Return closed-form scalar Gaussian edge ELBO terms under the exact posterior."""

    edge_cov = oracle.edge_cov
    filter_var = jnp.maximum(oracle.filter_var, min_var)
    backward_a = edge_cov[..., 0, 1] / filter_var
    backward_b = oracle.edge_mean[..., 1] - backward_a * oracle.filter_mean
    backward_var = jnp.maximum(edge_cov[..., 1, 1] - edge_cov[..., 0, 1] ** 2 / filter_var, min_var)
    prev_filter_mean, prev_filter_var = previous_filter_beliefs(
        oracle.filter_mean,
        oracle.filter_var,
        state_params,
    )
    return edge_elbo_closed_form_terms_from_factors(
        batch,
        state_params,
        filter_mean=oracle.filter_mean,
        filter_var=filter_var,
        backward_a=backward_a,
        backward_b=backward_b,
        backward_var=backward_var,
        prev_filter_mean=prev_filter_mean,
        prev_filter_var=prev_filter_var,
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
    elbo = (
        log_likelihood
        + log_transition
        + log_prev_filter
        + neg_log_current_filter
        + neg_log_backward
    )

    return EdgeElboTerms(
        log_likelihood=jnp.mean(log_likelihood, axis=0),
        log_transition=jnp.mean(log_transition, axis=0),
        log_prev_filter=jnp.mean(log_prev_filter, axis=0),
        neg_log_current_filter=jnp.mean(neg_log_current_filter, axis=0),
        neg_log_backward=jnp.mean(neg_log_backward, axis=0),
        elbo=jnp.mean(elbo, axis=0),
    )


def edge_elbo_closed_form_terms_from_factors(
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    filter_mean: jax.Array,
    filter_var: jax.Array,
    backward_a: jax.Array,
    backward_b: jax.Array,
    backward_var: jax.Array,
    prev_filter_mean: jax.Array,
    prev_filter_var: jax.Array,
) -> EdgeElboTerms:
    """Return analytic local edge ELBO terms for scalar Gaussian factors.

    This is a variance-free reference for `edge_elbo_terms_from_factors`.
    """

    z_tm1_mean = backward_a * filter_mean + backward_b
    z_tm1_var = backward_a**2 * filter_var + backward_var
    z_t_z_tm1_cov = backward_a * filter_var

    likelihood_var_term = (batch.y - batch.x * filter_mean) ** 2 + batch.x**2 * filter_var
    log_likelihood = _expected_normal_log_prob(likelihood_var_term, state_params.r)

    residual_mean = filter_mean - z_tm1_mean
    residual_var = filter_var + z_tm1_var - 2.0 * z_t_z_tm1_cov
    log_transition = _expected_normal_log_prob(residual_mean**2 + residual_var, state_params.q)

    prev_residual = (z_tm1_mean - prev_filter_mean) ** 2 + z_tm1_var
    log_prev_filter = _expected_normal_log_prob(prev_residual, prev_filter_var)

    neg_log_current_filter = 0.5 * (LOG_2PI + jnp.log(filter_var) + 1.0)
    neg_log_backward = 0.5 * (LOG_2PI + jnp.log(backward_var) + 1.0)
    elbo = (
        log_likelihood
        + log_transition
        + log_prev_filter
        + neg_log_current_filter
        + neg_log_backward
    )

    return EdgeElboTerms(
        log_likelihood=log_likelihood,
        log_transition=log_transition,
        log_prev_filter=log_prev_filter,
        neg_log_current_filter=neg_log_current_filter,
        neg_log_backward=neg_log_backward,
        elbo=elbo,
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


def _normal_log_prob(value: jax.Array, mean: jax.Array, var: jax.Array) -> jax.Array:
    return -0.5 * (LOG_2PI + jnp.log(var) + (value - mean) ** 2 / var)


def _expected_normal_log_prob(
    expected_squared_error: jax.Array, var: jax.Array | float
) -> jax.Array:
    return -0.5 * (LOG_2PI + jnp.log(var) + expected_squared_error / var)
