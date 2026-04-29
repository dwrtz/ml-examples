"""Train learned filters on nonlinear benchmark variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from vbf.data import EpisodeBatch, LinearGaussianParams
from vbf.metrics import gaussian_interval_coverage, rmse_global, scalar_gaussian_nll
from vbf.models.cells import (
    GaussianMixtureMLPOutputs,
    direct_mlp_step,
    direct_mixture_mlp_step,
    edge_mean_cov_from_outputs,
    init_direct_mixture_mlp_params,
    init_direct_mlp_params,
    init_structured_mixture_mlp_params,
    init_structured_mlp_params,
    run_direct_mlp_filter,
    run_direct_mixture_mlp_filter,
    run_direct_mlp_teacher_forced,
)
from vbf.nonlinear import (
    GridReferenceConfig,
    NonlinearDataConfig,
    make_y_observed_mask,
    make_nonlinear_batch,
    nonlinear_observation_mean,
    nonlinear_preassimilation_log_prob_y,
    nonlinear_predictive_moments_from_filter,
    nonlinear_tilted_projection_loss,
    nonlinear_structured_mlp_step,
    run_nonlinear_structured_mixture_mlp_filter,
    run_nonlinear_structured_mlp_filter,
    run_nonlinear_structured_mlp_teacher_forced,
    mixture_transition_prediction_outputs,
    transition_prediction_outputs,
)
from vbf.nonlinear_cache import load_or_compute_nonlinear_reference
from vbf.predictive import previous_filter_beliefs
from vbf.train import adam_update, init_adam


LOG_2PI = jnp.log(2.0 * jnp.pi)
SUPPORTED_MODELS = {"direct_elbo_sine_mlp", "structured_elbo_sine_mlp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", default="outputs/cache/nonlinear_reference")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)

    if config["model"] not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported nonlinear training model: {config['model']}")

    data_config = NonlinearDataConfig(**config["data"])
    state_params = LinearGaussianParams(**config["state_space"])
    training_config = config["training"]
    evaluation_config = config.get("evaluation", {})
    reference_config = GridReferenceConfig(**config.get("reference", {}))
    min_var = float(training_config.get("min_var", 1e-6))
    objective_family = str(training_config.get("objective_family", "elbo"))
    num_importance_samples = int(
        training_config.get(
            "num_importance_samples",
            training_config.get("joint_elbo_num_samples", 16),
        )
    )
    renyi_alpha = float(training_config.get("renyi_alpha", 1.0))
    entropy_bonus_weight = float(training_config.get("entropy_bonus_weight", 0.0))
    posterior_family = str(training_config.get("posterior_family", "gaussian"))
    mixture_components = int(training_config.get("mixture_components", 1))
    default_elbo_weight = 0.0 if objective_family == "local_projection" else 1.0
    elbo_weight = float(training_config.get("elbo_weight", default_elbo_weight))
    joint_elbo_weight = float(training_config.get("joint_elbo_weight", 0.0))
    joint_elbo_horizon = int(training_config.get("joint_elbo_horizon", 1))
    joint_elbo_num_samples = int(training_config.get("joint_elbo_num_samples", 16))
    joint_elbo_num_windows = int(training_config.get("joint_elbo_num_windows", 8))
    joint_elbo_window_seed_offset = int(training_config.get("joint_elbo_window_seed_offset", 80_000))
    fivo_num_particles = int(training_config.get("fivo_num_particles", num_importance_samples))
    fivo_resampling = str(training_config.get("fivo_resampling", "every_step"))
    predictive_y_weight = float(training_config.get("predictive_y_weight", 0.0))
    predictive_y_start_fraction = float(training_config.get("predictive_y_start_fraction", 0.0))
    predictive_y_ramp_fraction = float(training_config.get("predictive_y_ramp_fraction", 0.0))
    predictive_y_num_samples = int(training_config.get("predictive_y_num_samples", 32))
    predictive_y_estimator = str(training_config.get("predictive_y_estimator", "quadrature"))
    default_local_projection_weight = 1.0 if objective_family == "local_projection" else 0.0
    local_projection_weight = float(
        training_config.get("local_projection_weight", default_local_projection_weight)
    )
    local_projection_num_points = int(training_config.get("local_projection_num_points", 32))
    local_projection_likelihood_power = float(
        training_config.get("local_projection_likelihood_power", 1.0)
    )
    local_projection_stop_target = bool(training_config.get("local_projection_stop_target", True))
    reference_mean_weight = float(training_config.get("reference_mean_weight", 0.0))
    reference_rollout_weight = float(training_config.get("reference_rollout_weight", 0.0))
    reference_rollout_horizon = int(training_config.get("reference_rollout_horizon", 1))
    reference_variance_ratio_weight = float(
        training_config.get("reference_variance_ratio_weight", 0.0)
    )
    reference_time_variance_ratio_weight = float(
        training_config.get("reference_time_variance_ratio_weight", 0.0)
    )
    reference_log_variance_weight = float(training_config.get("reference_log_variance_weight", 0.0))
    reference_low_observation_variance_ratio_weight = float(
        training_config.get("reference_low_observation_variance_ratio_weight", 0.0)
    )
    low_observation_eps = float(training_config.get("low_observation_eps", 1e-3))
    teacher_forced = bool(training_config.get("teacher_forced", False))
    resample_batch = bool(training_config.get("resample_batch", False))
    batch_seed_stride = int(training_config.get("batch_seed_stride", 1))
    mask_y_probability = float(training_config.get("mask_y_probability", 0.0))
    mask_y_span_probability = float(training_config.get("mask_y_span_probability", 0.0))
    mask_y_span_length = int(training_config.get("mask_y_span_length", 1))
    mask_y_seed_offset = int(training_config.get("mask_y_seed_offset", 70_000))
    uses_reference_calibration = (
        reference_mean_weight != 0.0
        or reference_rollout_weight != 0.0
        or reference_variance_ratio_weight != 0.0
        or reference_time_variance_ratio_weight != 0.0
        or reference_log_variance_weight != 0.0
        or reference_low_observation_variance_ratio_weight != 0.0
    )
    if resample_batch and uses_reference_calibration:
        raise ValueError("resample_batch is only supported for non-reference-calibrated training")
    if teacher_forced and not uses_reference_calibration:
        raise ValueError("teacher_forced requires reference targets")
    if reference_rollout_horizon <= 0:
        raise ValueError("reference_rollout_horizon must be positive")
    if reference_rollout_horizon > data_config.time_steps:
        raise ValueError("reference_rollout_horizon cannot exceed data time_steps")
    if joint_elbo_horizon <= 0:
        raise ValueError("joint_elbo_horizon must be positive")
    if joint_elbo_horizon > data_config.time_steps:
        raise ValueError("joint_elbo_horizon cannot exceed data time_steps")
    if joint_elbo_num_samples <= 0:
        raise ValueError("joint_elbo_num_samples must be positive")
    if joint_elbo_num_windows <= 0:
        raise ValueError("joint_elbo_num_windows must be positive")
    if fivo_num_particles <= 0:
        raise ValueError("fivo_num_particles must be positive")
    if fivo_resampling != "every_step":
        raise ValueError("Only fivo_resampling='every_step' is supported")
    if predictive_y_num_samples <= 0:
        raise ValueError("predictive_y_num_samples must be positive")
    if local_projection_weight < 0.0:
        raise ValueError("local_projection_weight must be nonnegative")
    if local_projection_num_points <= 0:
        raise ValueError("local_projection_num_points must be positive")
    if local_projection_likelihood_power <= 0.0:
        raise ValueError("local_projection_likelihood_power must be positive")
    if not 0.0 <= predictive_y_start_fraction <= 1.0:
        raise ValueError("predictive_y_start_fraction must be in [0, 1]")
    if not 0.0 <= predictive_y_ramp_fraction <= 1.0:
        raise ValueError("predictive_y_ramp_fraction must be in [0, 1]")
    if objective_family not in {"elbo", "iwae", "renyi", "local_projection", "fivo"}:
        raise ValueError(
            "objective_family must be one of: elbo, iwae, renyi, local_projection, fivo"
        )
    if num_importance_samples <= 0:
        raise ValueError("num_importance_samples must be positive")
    if not 0.0 < renyi_alpha <= 1.0:
        raise ValueError("renyi_alpha must be in (0, 1]")
    if entropy_bonus_weight < 0.0:
        raise ValueError("entropy_bonus_weight must be nonnegative")
    if posterior_family not in {"gaussian", "gaussian_mixture"}:
        raise ValueError("posterior_family must be one of: gaussian, gaussian_mixture")
    if posterior_family == "gaussian" and mixture_components != 1:
        raise ValueError("mixture_components must be 1 for posterior_family='gaussian'")
    if posterior_family == "gaussian_mixture" and mixture_components <= 1:
        raise ValueError("gaussian_mixture requires mixture_components > 1")
    if predictive_y_estimator != "quadrature":
        raise ValueError("Only predictive_y_estimator='quadrature' is supported")
    if not 0.0 <= mask_y_probability <= 1.0:
        raise ValueError("mask_y_probability must be in [0, 1]")
    if not 0.0 <= mask_y_span_probability <= 1.0:
        raise ValueError("mask_y_span_probability must be in [0, 1]")
    if mask_y_span_length <= 0:
        raise ValueError("mask_y_span_length must be positive")
    uses_y_mask = mask_y_probability != 0.0 or mask_y_span_probability != 0.0
    if teacher_forced and uses_y_mask:
        raise ValueError("masked-y training is not supported with teacher_forced")

    train_batch = make_nonlinear_batch(data_config, state_params, seed=int(config["seed"]))
    train_y_observed = _make_y_observed_mask(
        data_config,
        seed=int(config["seed"]) + mask_y_seed_offset,
        probability=mask_y_probability,
        span_probability=mask_y_span_probability,
        span_length=mask_y_span_length,
    )
    train_reference = None
    train_cached = None
    if uses_reference_calibration:
        train_cached = load_or_compute_nonlinear_reference(
            data_config,
            state_params,
            seed=int(config["seed"]),
            grid_config=reference_config,
            cache_dir=Path(args.cache_dir),
            use_cache=not args.no_cache,
        )
        train_batch = train_cached.batch
        train_reference = train_cached.reference
    reference_mean_filter_var = (
        jnp.mean(train_reference.filter_var) if train_reference is not None else None
    )
    reference_filter_var_t = (
        jnp.mean(train_reference.filter_var, axis=0) if train_reference is not None else None
    )
    low_observation_weights_t = None
    if train_reference is not None:
        mean_x2_t = jnp.mean(train_batch.x**2, axis=0)
        low_observation_weights_t = 1.0 / (mean_x2_t + low_observation_eps)
        low_observation_weights_t = low_observation_weights_t / jnp.mean(low_observation_weights_t)
    eval_data_config = NonlinearDataConfig(
        **{**config["data"], **evaluation_config.get("data", {})}
    )
    eval_seed = int(config["seed"]) + int(evaluation_config.get("seed_offset", 10_000))
    eval_batch = make_nonlinear_batch(
        eval_data_config,
        state_params,
        seed=eval_seed,
    )
    eval_y_observed = _make_y_observed_mask(
        eval_data_config,
        seed=eval_seed + mask_y_seed_offset,
        probability=mask_y_probability,
        span_probability=mask_y_span_probability,
        span_length=mask_y_span_length,
    )

    params = _init_model_params(
        str(config["model"]),
        jax.random.PRNGKey(int(config["seed"]) + 1),
        hidden_dim=int(training_config["hidden_dim"]),
        posterior_family=posterior_family,
        mixture_components=mixture_components,
    )
    opt_state = init_adam(params)

    def loss_fn(
        current_params: dict[str, jax.Array],
        batch_x: jax.Array,
        batch_y: jax.Array,
        batch_z: jax.Array,
        y_observed: jax.Array,
        key: jax.Array,
        step_index: jax.Array,
    ) -> jax.Array:
        batch = EpisodeBatch(x=batch_x, y=batch_y, z=batch_z)
        outputs = (
            _run_model_teacher_forced(
                str(config["model"]),
                current_params,
                batch,
                state_params,
                train_reference.filter_mean,
                train_reference.filter_var,
                observation=data_config.observation,
                min_var=min_var,
                posterior_family=posterior_family,
                mixture_components=mixture_components,
            )
            if teacher_forced
            else _run_model_filter(
                str(config["model"]),
                current_params,
                batch,
                state_params,
                observation=data_config.observation,
                min_var=min_var,
                y_observed=y_observed,
                posterior_family=posterior_family,
                mixture_components=mixture_components,
            )
        )
        loss = jnp.asarray(0.0, dtype=batch_x.dtype)
        if elbo_weight != 0.0:
            loss = loss - elbo_weight * jnp.mean(
                _nonlinear_edge_elbo(
                    outputs,
                    batch,
                    state_params,
                    key,
                    observation=data_config.observation,
                    num_samples=int(training_config.get("num_elbo_samples", 8)),
                )
            )
        if joint_elbo_weight != 0.0:
            if objective_family == "local_projection":
                raise ValueError("local_projection objective_family cannot be used for joint ELBO")
            joint_key = jax.random.fold_in(key, joint_elbo_window_seed_offset)
            joint_objective = (
                _nonlinear_fivo_objective(
                    outputs,
                    batch,
                    state_params,
                    joint_key,
                    observation=data_config.observation,
                    num_particles=fivo_num_particles,
                )
                if objective_family == "fivo"
                else _nonlinear_windowed_joint_objective(
                    outputs,
                    batch,
                    state_params,
                    joint_key,
                    observation=data_config.observation,
                    horizon=joint_elbo_horizon,
                    num_samples=(
                        num_importance_samples
                        if objective_family in {"iwae", "renyi"}
                        else joint_elbo_num_samples
                    ),
                    num_windows=joint_elbo_num_windows,
                    objective_family=objective_family,
                    renyi_alpha=renyi_alpha,
                )
            )
            loss = loss - joint_elbo_weight * jnp.mean(joint_objective)
        if entropy_bonus_weight != 0.0:
            loss = loss - entropy_bonus_weight * jnp.mean(
                _gaussian_filter_entropy(outputs.filter_var)
            )
        effective_predictive_y_weight = predictive_y_weight * _scheduled_weight(
            step_index,
            total_steps=int(training_config["steps"]),
            start_fraction=predictive_y_start_fraction,
            ramp_fraction=predictive_y_ramp_fraction,
        )
        if predictive_y_weight != 0.0:
            prev_mean, prev_var = previous_filter_beliefs(
                outputs.filter_mean,
                outputs.filter_var,
                state_params,
            )
            predictive_y_log_prob = (
                _nonlinear_mixture_preassimilation_log_prob_y(
                    outputs,
                    batch.x,
                    batch.y,
                    state_params,
                    observation=data_config.observation,
                    num_points=predictive_y_num_samples,
                )
                if _is_mixture_outputs(outputs)
                else nonlinear_preassimilation_log_prob_y(
                    prev_mean,
                    prev_var,
                    batch.x,
                    batch.y,
                    state_params,
                    observation=data_config.observation,
                    num_points=predictive_y_num_samples,
                )
            )
            loss = loss - effective_predictive_y_weight * jnp.mean(predictive_y_log_prob)
        if local_projection_weight != 0.0:
            projection_loss = nonlinear_tilted_projection_loss(
                outputs,
                batch,
                state_params,
                observation=data_config.observation,
                num_points=local_projection_num_points,
                likelihood_power=local_projection_likelihood_power,
                min_var=min_var,
                stop_target=local_projection_stop_target,
            )
            loss = loss + local_projection_weight * jnp.mean(projection_loss)
        if reference_mean_weight != 0.0:
            if train_reference is None:
                raise ValueError("train_reference is required for reference mean distillation")
            loss = loss + reference_mean_weight * jnp.mean(
                (outputs.filter_mean - train_reference.filter_mean) ** 2
                / jnp.maximum(train_reference.filter_var, min_var)
            )
        if reference_rollout_weight != 0.0:
            if train_reference is None:
                raise ValueError("train_reference is required for reference rollout distillation")
            loss = loss + reference_rollout_weight * _reference_rollout_moment_loss(
                str(config["model"]),
                current_params,
                batch,
                state_params,
                train_reference,
                horizon=reference_rollout_horizon,
                observation=data_config.observation,
                min_var=min_var,
            )
        if reference_variance_ratio_weight != 0.0:
            if reference_mean_filter_var is None:
                raise ValueError("reference_mean_filter_var is required for reference calibration")
            ratio = jnp.mean(outputs.filter_var) / reference_mean_filter_var
            loss = loss + reference_variance_ratio_weight * jnp.log(ratio) ** 2
        if reference_time_variance_ratio_weight != 0.0:
            if reference_filter_var_t is None:
                raise ValueError("reference_filter_var_t is required for reference calibration")
            learned_t = jnp.mean(outputs.filter_var, axis=0)
            loss = loss + reference_time_variance_ratio_weight * jnp.mean(
                jnp.log(learned_t / reference_filter_var_t) ** 2
            )
        if reference_log_variance_weight != 0.0:
            if train_reference is None:
                raise ValueError("train_reference is required for log-variance calibration")
            loss = loss + reference_log_variance_weight * jnp.mean(
                (jnp.log(outputs.filter_var) - jnp.log(train_reference.filter_var)) ** 2
            )
        if reference_low_observation_variance_ratio_weight != 0.0:
            if reference_filter_var_t is None or low_observation_weights_t is None:
                raise ValueError("time-local targets are required for reference calibration")
            learned_t = jnp.mean(outputs.filter_var, axis=0)
            loss = loss + reference_low_observation_variance_ratio_weight * jnp.mean(
                low_observation_weights_t * jnp.log(learned_t / reference_filter_var_t) ** 2
            )
        return loss

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    history: list[tuple[int, float]] = []
    train_key = jax.random.PRNGKey(int(config["seed"]) + 2)
    for step in range(1, int(training_config["steps"]) + 1):
        train_key, step_key = jax.random.split(train_key)
        step_batch = (
            make_nonlinear_batch(
                data_config,
                state_params,
                seed=int(config["seed"]) + batch_seed_stride * step,
            )
            if resample_batch
            else train_batch
        )
        step_y_observed = (
            _make_y_observed_mask(
                data_config,
                seed=int(config["seed"]) + mask_y_seed_offset + batch_seed_stride * step,
                probability=mask_y_probability,
                span_probability=mask_y_span_probability,
                span_length=mask_y_span_length,
            )
            if resample_batch
            else train_y_observed
        )
        loss_value, grads = value_and_grad(
            params,
            step_batch.x,
            step_batch.y,
            step_batch.z,
            step_y_observed,
            step_key,
            jnp.asarray(step, dtype=jnp.float64),
        )
        params, opt_state = adam_update(
            params,
            grads,
            opt_state,
            learning_rate=float(training_config["learning_rate"]),
        )
        if step == 1 or step % int(training_config["log_every"]) == 0:
            history.append((step, float(loss_value)))

    final_loss = float(
        loss_fn(
            params,
            train_batch.x,
            train_batch.y,
            train_batch.z,
            train_y_observed,
            jax.random.PRNGKey(int(config["seed"]) + 3),
            jnp.asarray(training_config["steps"], dtype=jnp.float64),
        )
    )
    outputs = _run_model_filter(
        str(config["model"]),
        params,
        eval_batch,
        state_params,
        observation=eval_data_config.observation,
        min_var=min_var,
        y_observed=eval_y_observed,
        posterior_family=posterior_family,
        mixture_components=mixture_components,
    )
    eval_cached = load_or_compute_nonlinear_reference(
        eval_data_config,
        state_params,
        seed=eval_seed,
        grid_config=reference_config,
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
    )
    eval_batch = eval_cached.batch
    reference = eval_cached.reference
    teacher_outputs = None
    if teacher_forced:
        teacher_outputs = _run_model_teacher_forced(
            str(config["model"]),
            params,
            eval_batch,
            state_params,
            reference.filter_mean,
            reference.filter_var,
            observation=eval_data_config.observation,
            min_var=min_var,
            posterior_family=posterior_family,
            mixture_components=mixture_components,
        )
    learned_predictive_mean, learned_predictive_var = nonlinear_predictive_moments_from_filter(
        outputs.filter_mean,
        outputs.filter_var,
        eval_batch.x,
        state_params,
        observation=eval_data_config.observation,
    )
    learned_state_nll = _state_nll(outputs, eval_batch.z)
    reference_state_nll = scalar_gaussian_nll(
        eval_batch.z,
        reference.filter_mean,
        reference.filter_var,
    )
    learned_predictive_nll = scalar_gaussian_nll(
        eval_batch.y,
        learned_predictive_mean,
        learned_predictive_var,
    )
    eval_prev_mean, eval_prev_var = previous_filter_beliefs(
        outputs.filter_mean,
        outputs.filter_var,
        state_params,
    )
    learned_predictive_y_nll = -(
        _nonlinear_mixture_preassimilation_log_prob_y(
            outputs,
            eval_batch.x,
            eval_batch.y,
            state_params,
            observation=eval_data_config.observation,
            num_points=predictive_y_num_samples,
        )
        if _is_mixture_outputs(outputs)
        else nonlinear_preassimilation_log_prob_y(
            eval_prev_mean,
            eval_prev_var,
            eval_batch.x,
            eval_batch.y,
            state_params,
            observation=eval_data_config.observation,
            num_points=predictive_y_num_samples,
        )
    )
    reference_predictive_nll = scalar_gaussian_nll(
        eval_batch.y,
        reference.predictive_mean,
        reference.predictive_var,
    )
    _, edge_cov = edge_mean_cov_from_outputs(outputs)
    metrics = {
        "benchmark": "nonlinear",
        "objective": config["model"],
        "seed": int(config["seed"]),
        "observation": eval_data_config.observation,
        "x_pattern": eval_data_config.x_pattern,
        "training_steps": int(training_config["steps"]),
        "num_elbo_samples": int(training_config.get("num_elbo_samples", 8)),
        "objective_family": objective_family,
        "num_importance_samples": num_importance_samples,
        "renyi_alpha": renyi_alpha,
        "entropy_bonus_weight": entropy_bonus_weight,
        "posterior_family": posterior_family,
        "mixture_components": mixture_components,
        "elbo_weight": elbo_weight,
        "joint_elbo_weight": joint_elbo_weight,
        "joint_elbo_horizon": joint_elbo_horizon,
        "joint_elbo_num_samples": joint_elbo_num_samples,
        "joint_elbo_num_windows": joint_elbo_num_windows,
        "joint_elbo_window_seed_offset": joint_elbo_window_seed_offset,
        "fivo_num_particles": fivo_num_particles,
        "fivo_resampling": fivo_resampling,
        "predictive_y_weight": predictive_y_weight,
        "predictive_y_start_fraction": predictive_y_start_fraction,
        "predictive_y_ramp_fraction": predictive_y_ramp_fraction,
        "predictive_y_num_samples": predictive_y_num_samples,
        "predictive_y_estimator": predictive_y_estimator,
        "local_projection_weight": local_projection_weight,
        "local_projection_num_points": local_projection_num_points,
        "local_projection_likelihood_power": local_projection_likelihood_power,
        "local_projection_stop_target": local_projection_stop_target,
        "state_nll_estimator": "mixture_density" if _is_mixture_outputs(outputs) else "gaussian",
        "coverage_estimator": "moment_gaussian",
        "reference_mean_weight": reference_mean_weight,
        "reference_rollout_weight": reference_rollout_weight,
        "reference_rollout_horizon": reference_rollout_horizon,
        "teacher_forced": teacher_forced,
        "resample_batch": resample_batch,
        "batch_seed_stride": batch_seed_stride,
        "mask_y_probability": mask_y_probability,
        "mask_y_span_probability": mask_y_span_probability,
        "mask_y_span_length": mask_y_span_length,
        "mask_y_seed_offset": mask_y_seed_offset,
        "train_y_observed_fraction": float(jnp.mean(train_y_observed.astype(jnp.float64))),
        "eval_y_observed_fraction": float(jnp.mean(eval_y_observed.astype(jnp.float64))),
        "reference_variance_ratio_weight": reference_variance_ratio_weight,
        "reference_time_variance_ratio_weight": reference_time_variance_ratio_weight,
        "reference_log_variance_weight": reference_log_variance_weight,
        "reference_low_observation_variance_ratio_weight": (
            reference_low_observation_variance_ratio_weight
        ),
        "low_observation_eps": low_observation_eps,
        "train_reference_cache_hit": None if train_cached is None else train_cached.cache_hit,
        "train_reference_cache_path": None
        if train_cached is None
        else str(train_cached.cache_path),
        "eval_reference_cache_hit": eval_cached.cache_hit,
        "eval_reference_cache_path": str(eval_cached.cache_path),
        "final_loss": final_loss,
        "state_rmse": float(rmse_global(outputs.filter_mean, eval_batch.z)),
        "reference_state_rmse": float(rmse_global(reference.filter_mean, eval_batch.z)),
        "state_nll": float(jnp.mean(learned_state_nll)),
        "reference_state_nll": float(jnp.mean(reference_state_nll)),
        "predictive_nll": float(jnp.mean(learned_predictive_nll)),
        "predictive_y_nll": float(jnp.mean(learned_predictive_y_nll)),
        "reference_predictive_nll": float(jnp.mean(reference_predictive_nll)),
        "coverage_90": float(
            gaussian_interval_coverage(
                eval_batch.z,
                outputs.filter_mean,
                outputs.filter_var,
                z_score=1.6448536269514722,
            )
        ),
        "reference_coverage_90": float(
            gaussian_interval_coverage(
                eval_batch.z,
                reference.filter_mean,
                reference.filter_var,
                z_score=1.6448536269514722,
            )
        ),
        "mean_filter_variance": float(jnp.mean(outputs.filter_var)),
        "reference_mean_filter_variance": float(jnp.mean(reference.filter_var)),
        "variance_ratio": float(jnp.mean(outputs.filter_var) / jnp.mean(reference.filter_var)),
        "mean_edge_covariance_trace": float(jnp.mean(jnp.trace(edge_cov, axis1=-2, axis2=-1))),
    }
    if teacher_outputs is not None:
        teacher_state_nll = _state_nll(teacher_outputs, eval_batch.z)
        metrics.update(
            {
                "teacher_forced_state_nll": float(jnp.mean(teacher_state_nll)),
                "teacher_forced_coverage_90": float(
                    gaussian_interval_coverage(
                        eval_batch.z,
                        teacher_outputs.filter_mean,
                        teacher_outputs.filter_var,
                        z_score=1.6448536269514722,
                    )
                ),
                "teacher_forced_variance_ratio": float(
                    jnp.mean(teacher_outputs.filter_var) / jnp.mean(reference.filter_var)
                ),
            }
        )

    output_dir = Path(config.get("output_dir", "outputs/nonlinear_direct_elbo_sine_mlp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_loss_history(output_dir / "loss_history.csv", history)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(
        output_dir / "params.npz", **{name: np.asarray(value) for name, value in params.items()}
    )
    diagnostics = {
        "x": np.asarray(eval_batch.x),
        "y": np.asarray(eval_batch.y),
        "z": np.asarray(eval_batch.z),
        "learned_filter_mean": np.asarray(outputs.filter_mean),
        "learned_filter_var": np.asarray(outputs.filter_var),
        "reference_filter_mean": np.asarray(reference.filter_mean),
        "reference_filter_var": np.asarray(reference.filter_var),
        "learned_predictive_mean": np.asarray(learned_predictive_mean),
        "learned_predictive_var": np.asarray(learned_predictive_var),
        "reference_predictive_mean": np.asarray(reference.predictive_mean),
        "reference_predictive_var": np.asarray(reference.predictive_var),
        "y_observed_mask": np.asarray(eval_y_observed),
        "loss_history_step": np.asarray([step for step, _ in history], dtype=np.int64),
        "loss_history_loss": np.asarray([loss for _, loss in history], dtype=np.float64),
    }
    if _is_mixture_outputs(outputs):
        diagnostics.update(
            {
                "learned_filter_weights": np.asarray(outputs.filter_weights),
                "learned_component_mean": np.asarray(outputs.component_mean),
                "learned_component_var": np.asarray(outputs.component_var),
                "learned_backward_a": np.asarray(outputs.backward_a),
                "learned_backward_b": np.asarray(outputs.backward_b),
                "learned_backward_var": np.asarray(outputs.backward_var),
            }
        )
    np.savez(output_dir / "diagnostics.npz", **diagnostics)
    summary_path = output_dir / "evaluation_summary.md"
    summary_path.write_text(_render_summary(config["name"], metrics, history), encoding="utf-8")
    print(f"Wrote {summary_path}")
    if train_cached is not None:
        train_cache_status = "hit" if train_cached.cache_hit else "miss"
        print(f"Train reference cache {train_cache_status}: {train_cached.cache_path}")
    eval_cache_status = "hit" if eval_cached.cache_hit else "miss"
    print(f"Eval reference cache {eval_cache_status}: {eval_cached.cache_path}")


def _init_model_params(
    model: str,
    key: jax.Array,
    *,
    hidden_dim: int,
    posterior_family: str = "gaussian",
    mixture_components: int = 1,
) -> dict[str, jax.Array]:
    if posterior_family == "gaussian_mixture":
        init_fn = (
            init_structured_mixture_mlp_params
            if model == "structured_elbo_sine_mlp"
            else init_direct_mixture_mlp_params
        )
        return init_fn(
            key,
            hidden_dim=hidden_dim,
            num_components=mixture_components,
        )
    if model == "structured_elbo_sine_mlp":
        return init_structured_mlp_params(key, hidden_dim=hidden_dim)
    return init_direct_mlp_params(key, hidden_dim=hidden_dim)


def _run_model_filter(
    model: str,
    params: dict[str, jax.Array],
    batch,
    state_params: LinearGaussianParams,
    *,
    observation: str,
    min_var: float,
    y_observed: jax.Array | None = None,
    posterior_family: str = "gaussian",
    mixture_components: int = 1,
):
    if posterior_family == "gaussian_mixture":
        if model == "structured_elbo_sine_mlp":
            return run_nonlinear_structured_mixture_mlp_filter(
                params,
                batch,
                state_params,
                num_components=mixture_components,
                observation=observation,
                min_var=min_var,
                y_observed=y_observed,
            )
        return _run_direct_mixture_mlp_filter(
            params,
            batch,
            state_params,
            num_components=mixture_components,
            min_var=min_var,
            y_observed=y_observed,
        )
    if model == "structured_elbo_sine_mlp":
        return run_nonlinear_structured_mlp_filter(
            params,
            batch,
            state_params,
            observation=observation,
            min_var=min_var,
            y_observed=y_observed,
        )
    return _run_direct_mlp_filter(
        params,
        batch,
        state_params,
        min_var=min_var,
        y_observed=y_observed,
    )


def _run_direct_mlp_filter(
    params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    min_var: float,
    y_observed: jax.Array | None,
):
    if y_observed is None:
        return run_direct_mlp_filter(params, batch, state_params, min_var=min_var)

    x_bt = batch.x.T
    y_bt = batch.y.T
    observed_bt = y_observed.T

    def step(carry: tuple[jax.Array, jax.Array], obs: tuple[jax.Array, jax.Array, jax.Array]):
        prev_mean, prev_var = carry
        x_t, y_t, observed_t = obs
        update_outputs = direct_mlp_step(
            params,
            prev_mean,
            prev_var,
            x_t,
            y_t,
            state_params,
            min_var=min_var,
        )
        transition_outputs = transition_prediction_outputs(prev_mean, prev_var, state_params)
        outputs = _where_outputs(observed_t, update_outputs, transition_outputs)
        return (outputs.filter_mean, outputs.filter_var), outputs

    batch_size = batch.x.shape[0]
    init = (
        jnp.full((batch_size,), state_params.m0, dtype=jnp.float64),
        jnp.full((batch_size,), state_params.p0, dtype=jnp.float64),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt, observed_bt))
    return type(outputs)(*(_time_major_to_batch_major(item) for item in outputs))


def _run_direct_mixture_mlp_filter(
    params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    num_components: int,
    min_var: float,
    y_observed: jax.Array | None,
):
    if y_observed is None:
        return run_direct_mixture_mlp_filter(
            params,
            batch,
            state_params,
            num_components=num_components,
            min_var=min_var,
        )

    x_bt = batch.x.T
    y_bt = batch.y.T
    observed_bt = y_observed.T

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        obs: tuple[jax.Array, jax.Array, jax.Array],
    ):
        prev_weights, prev_mean, prev_var = carry
        x_t, y_t, observed_t = obs
        update_outputs = direct_mixture_mlp_step(
            params,
            prev_weights,
            prev_mean,
            prev_var,
            x_t,
            y_t,
            state_params,
            num_components=num_components,
            min_var=min_var,
        )
        transition_outputs = mixture_transition_prediction_outputs(
            prev_weights,
            prev_mean,
            prev_var,
            state_params,
        )
        outputs = _where_outputs(observed_t, update_outputs, transition_outputs)
        return (outputs.filter_weights, outputs.component_mean, outputs.component_var), outputs

    batch_size = batch.x.shape[0]
    init = (
        jnp.full((batch_size, num_components), 1.0 / num_components, dtype=jnp.float64),
        jnp.full((batch_size, num_components), state_params.m0, dtype=jnp.float64),
        jnp.full((batch_size, num_components), state_params.p0, dtype=jnp.float64),
    )
    _, outputs = jax.lax.scan(step, init, (x_bt, y_bt, observed_bt))
    return GaussianMixtureMLPOutputs(*(_time_major_to_batch_major(item) for item in outputs))


def _where_outputs(condition, true_outputs, false_outputs):
    condition = condition.astype(bool)
    return type(true_outputs)(
        *(
            jnp.where(
                _expand_condition_for_value(condition, true_value),
                true_value,
                false_value,
            )
            for true_value, false_value in zip(true_outputs, false_outputs, strict=True)
        )
    )


def _expand_condition_for_value(condition: jax.Array, value: jax.Array) -> jax.Array:
    extra_dims = value.ndim - condition.ndim
    if extra_dims <= 0:
        return condition
    return jnp.reshape(condition, condition.shape + (1,) * extra_dims)


def _time_major_to_batch_major(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, 0, 1)


def _run_model_teacher_forced(
    model: str,
    params: dict[str, jax.Array],
    batch,
    state_params: LinearGaussianParams,
    target_filter_mean: jax.Array,
    target_filter_var: jax.Array,
    *,
    observation: str,
    min_var: float,
    posterior_family: str = "gaussian",
    mixture_components: int = 1,
):
    if posterior_family == "gaussian_mixture":
        raise ValueError("teacher-forced gaussian_mixture training is not implemented")
    if model == "structured_elbo_sine_mlp":
        return run_nonlinear_structured_mlp_teacher_forced(
            params,
            batch,
            state_params,
            target_filter_mean,
            target_filter_var,
            observation=observation,
            min_var=min_var,
        )
    return run_direct_mlp_teacher_forced(
        params,
        batch,
        state_params,
        target_filter_mean,
        target_filter_var,
        min_var=min_var,
    )


def _run_model_step(
    model: str,
    params: dict[str, jax.Array],
    prev_mean: jax.Array,
    prev_var: jax.Array,
    x_t: jax.Array,
    y_t: jax.Array,
    state_params: LinearGaussianParams,
    *,
    observation: str,
    min_var: float,
):
    if model == "structured_elbo_sine_mlp":
        return nonlinear_structured_mlp_step(
            params,
            prev_mean,
            prev_var,
            x_t,
            y_t,
            state_params,
            observation=observation,
            min_var=min_var,
        )
    return direct_mlp_step(
        params,
        prev_mean,
        prev_var,
        x_t,
        y_t,
        state_params,
        min_var=min_var,
    )


def _reference_rollout_moment_loss(
    model: str,
    params: dict[str, jax.Array],
    batch,
    state_params: LinearGaussianParams,
    reference,
    *,
    horizon: int,
    observation: str,
    min_var: float,
) -> jax.Array:
    initial_mean = jnp.full((batch.x.shape[0], 1), state_params.m0, dtype=batch.x.dtype)
    initial_var = jnp.full((batch.x.shape[0], 1), state_params.p0, dtype=batch.x.dtype)
    prev_mean = jnp.concatenate((initial_mean, reference.filter_mean[:, :-1]), axis=1)
    prev_var = jnp.concatenate((initial_var, reference.filter_var[:, :-1]), axis=1)
    loss = jnp.asarray(0.0, dtype=batch.x.dtype)
    terms = 0
    for offset in range(horizon):
        outputs = _run_model_step(
            model,
            params,
            prev_mean,
            prev_var,
            batch.x[:, offset:],
            batch.y[:, offset:],
            state_params,
            observation=observation,
            min_var=min_var,
        )
        target_mean = reference.filter_mean[:, offset:]
        target_var = reference.filter_var[:, offset:]
        target_var = jnp.maximum(target_var, min_var)
        loss = loss + jnp.mean((outputs.filter_mean - target_mean) ** 2 / target_var)
        loss = loss + jnp.mean((jnp.log(outputs.filter_var) - jnp.log(target_var)) ** 2)
        terms += 2
        if offset != horizon - 1:
            prev_mean = outputs.filter_mean[:, :-1]
            prev_var = outputs.filter_var[:, :-1]
    return loss / terms


def _make_y_observed_mask(
    data_config: NonlinearDataConfig,
    *,
    seed: int,
    probability: float,
    span_probability: float,
    span_length: int,
) -> jax.Array:
    return make_y_observed_mask(
        batch_size=data_config.batch_size,
        time_steps=data_config.time_steps,
        probability=probability,
        span_probability=span_probability,
        span_length=span_length,
        seed=seed,
    )


def _nonlinear_edge_elbo(
    outputs,
    batch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    observation: str,
    num_samples: int,
) -> jax.Array:
    log_weights = _nonlinear_edge_log_weights(
        outputs,
        batch,
        state_params,
        key,
        observation=observation,
        num_samples=num_samples,
    )
    return jnp.mean(log_weights, axis=0)


def _nonlinear_edge_log_weights(
    outputs,
    batch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    observation: str,
    num_samples: int,
) -> jax.Array:
    if _is_mixture_outputs(outputs):
        return _nonlinear_mixture_edge_log_weights(
            outputs,
            batch,
            state_params,
            key,
            observation=observation,
            num_samples=num_samples,
        )

    eps_t_key, eps_tm1_key = jax.random.split(key)
    sample_shape = (num_samples,) + outputs.filter_mean.shape
    eps_t = jax.random.normal(eps_t_key, shape=sample_shape, dtype=outputs.filter_mean.dtype)
    eps_tm1 = jax.random.normal(eps_tm1_key, shape=sample_shape, dtype=outputs.filter_mean.dtype)
    z_t = outputs.filter_mean[None, ...] + jnp.sqrt(outputs.filter_var)[None, ...] * eps_t
    backward_mean = outputs.backward_a[None, ...] * z_t + outputs.backward_b[None, ...]
    z_tm1 = backward_mean + jnp.sqrt(outputs.backward_var)[None, ...] * eps_tm1
    prev_mean, prev_var = _previous_filter_beliefs(
        outputs.filter_mean, outputs.filter_var, state_params
    )
    observation_mean = nonlinear_observation_mean(z_t, batch.x[None, ...], observation)
    return (
        _normal_log_prob(batch.y[None, ...], observation_mean, state_params.r)
        + _normal_log_prob(z_t, z_tm1, state_params.q)
        + _normal_log_prob(z_tm1, prev_mean[None, ...], prev_var[None, ...])
        - _normal_log_prob(z_t, outputs.filter_mean[None, ...], outputs.filter_var[None, ...])
        - _normal_log_prob(z_tm1, backward_mean, outputs.backward_var[None, ...])
    )


def _nonlinear_windowed_joint_elbo(
    outputs,
    batch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    observation: str,
    horizon: int,
    num_samples: int,
    num_windows: int,
) -> jax.Array:
    return _nonlinear_windowed_joint_objective(
        outputs,
        batch,
        state_params,
        key,
        observation=observation,
        horizon=horizon,
        num_samples=num_samples,
        num_windows=num_windows,
        objective_family="elbo",
        renyi_alpha=1.0,
    )


def _nonlinear_windowed_joint_objective(
    outputs,
    batch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    observation: str,
    horizon: int,
    num_samples: int,
    num_windows: int,
    objective_family: str,
    renyi_alpha: float,
) -> jax.Array:
    log_weights = _nonlinear_windowed_joint_log_weights(
        outputs,
        batch,
        state_params,
        key,
        observation=observation,
        horizon=horizon,
        num_samples=num_samples,
        num_windows=num_windows,
    )
    if objective_family == "elbo":
        return jnp.mean(log_weights, axis=0)
    if objective_family == "iwae":
        return jax.nn.logsumexp(log_weights, axis=0) - jnp.log(num_samples)
    if objective_family == "renyi":
        if renyi_alpha == 1.0:
            return jnp.mean(log_weights, axis=0)
        scale = 1.0 - renyi_alpha
        return (jax.nn.logsumexp(scale * log_weights, axis=0) - jnp.log(num_samples)) / scale
    raise ValueError(f"Unsupported objective_family: {objective_family}")


def _nonlinear_windowed_joint_log_weights(
    outputs,
    batch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    observation: str,
    horizon: int,
    num_samples: int,
    num_windows: int,
) -> jax.Array:
    if horizon == 1:
        return _nonlinear_edge_log_weights(
            outputs,
            batch,
            state_params,
            key,
            observation=observation,
            num_samples=num_samples,
        )
    if _is_mixture_outputs(outputs):
        return _nonlinear_mixture_windowed_joint_log_weights(
            outputs,
            batch,
            state_params,
            key,
            observation=observation,
            horizon=horizon,
            num_samples=num_samples,
            num_windows=num_windows,
        )

    time_steps = batch.x.shape[1]
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if horizon > time_steps:
        raise ValueError("horizon cannot exceed batch time_steps")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_windows <= 0:
        raise ValueError("num_windows must be positive")

    possible_windows = time_steps - horizon + 1
    end_indices = (
        jnp.arange(horizon - 1, time_steps)
        if num_windows >= possible_windows
        else jax.random.randint(
            key,
            shape=(num_windows,),
            minval=horizon - 1,
            maxval=time_steps,
        )
    )
    sample_key, *backward_keys = jax.random.split(key, horizon + 1)
    end_mean = jnp.take(outputs.filter_mean, end_indices, axis=1)
    end_var = jnp.take(outputs.filter_var, end_indices, axis=1)
    eps_end = jax.random.normal(
        sample_key,
        shape=(num_samples,) + end_mean.shape,
        dtype=outputs.filter_mean.dtype,
    )
    z_t = end_mean[None, ...] + jnp.sqrt(end_var)[None, ...] * eps_end
    log_q = _normal_log_prob(z_t, end_mean[None, ...], end_var[None, ...])
    log_score = jnp.zeros_like(log_q)

    for offset in range(horizon):
        t_indices = end_indices - offset
        x_t = jnp.take(batch.x, t_indices, axis=1)
        y_t = jnp.take(batch.y, t_indices, axis=1)
        obs_mean = nonlinear_observation_mean(z_t, x_t[None, ...], observation)
        log_score = log_score + _normal_log_prob(y_t[None, ...], obs_mean, state_params.r)

        backward_a = jnp.take(outputs.backward_a, t_indices, axis=1)
        backward_b = jnp.take(outputs.backward_b, t_indices, axis=1)
        backward_var = jnp.take(outputs.backward_var, t_indices, axis=1)
        backward_mean = backward_a[None, ...] * z_t + backward_b[None, ...]
        eps_prev = jax.random.normal(
            backward_keys[offset],
            shape=z_t.shape,
            dtype=outputs.filter_mean.dtype,
        )
        z_prev = backward_mean + jnp.sqrt(backward_var)[None, ...] * eps_prev
        log_score = log_score + _normal_log_prob(z_t, z_prev, state_params.q)
        log_q = log_q + _normal_log_prob(z_prev, backward_mean, backward_var[None, ...])
        z_t = z_prev

    first_transition_indices = end_indices - horizon + 1
    prev_mean, prev_var = _previous_filter_beliefs(
        outputs.filter_mean,
        outputs.filter_var,
        state_params,
    )
    initial_mean = jnp.take(prev_mean, first_transition_indices, axis=1)
    initial_var = jnp.take(prev_var, first_transition_indices, axis=1)
    log_score = log_score + _normal_log_prob(z_t, initial_mean[None, ...], initial_var[None, ...])
    return log_score - log_q


def _nonlinear_fivo_objective(
    outputs,
    batch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    observation: str,
    num_particles: int,
) -> jax.Array:
    """Sequential particle-filter marginal likelihood objective."""

    if observation != "x_sine":
        raise ValueError(f"Unsupported FIVO observation: {observation}")
    if num_particles <= 0:
        raise ValueError("num_particles must be positive")

    batch_size = batch.x.shape[0]
    init_key, scan_key = jax.random.split(key)
    prev_particles = state_params.m0 + jnp.sqrt(state_params.p0) * jax.random.normal(
        init_key,
        shape=(batch_size, num_particles),
        dtype=batch.x.dtype,
    )
    step_keys = jax.random.split(scan_key, batch.x.shape[1])
    proposal_params = (
        (
            outputs.filter_weights.transpose(1, 0, 2),
            outputs.component_mean.transpose(1, 0, 2),
            outputs.component_var.transpose(1, 0, 2),
        )
        if _is_mixture_outputs(outputs)
        else (outputs.filter_mean.T, outputs.filter_var.T)
    )

    def step(carry: jax.Array, obs):
        prev_z = carry
        if _is_mixture_outputs(outputs):
            x_t, y_t, weights_t, mean_t, var_t, step_key = obs
            sample_key, resample_key = jax.random.split(step_key)
            z_t, _ = _sample_mixture_marginal(
                sample_key,
                weights_t,
                mean_t,
                var_t,
                sample_shape=(num_particles,),
            )
            z_t = jnp.swapaxes(z_t, 0, 1)
            log_q = _mixture_log_prob(
                z_t,
                weights_t[:, None, :],
                mean_t[:, None, :],
                var_t[:, None, :],
            )
        else:
            x_t, y_t, mean_t, var_t, step_key = obs
            sample_key, resample_key = jax.random.split(step_key)
            eps = jax.random.normal(
                sample_key,
                shape=(batch_size, num_particles),
                dtype=batch.x.dtype,
            )
            z_t = mean_t[:, None] + jnp.sqrt(var_t[:, None]) * eps
            log_q = _normal_log_prob(z_t, mean_t[:, None], var_t[:, None])

        obs_mean = nonlinear_observation_mean(z_t, x_t[:, None], observation)
        log_weights = (
            _normal_log_prob(y_t[:, None], obs_mean, state_params.r)
            + _normal_log_prob(z_t, prev_z, state_params.q)
            - log_q
        )
        increment = jax.nn.logsumexp(log_weights, axis=1) - jnp.log(num_particles)
        normalized_log_weights = log_weights - jax.nn.logsumexp(
            log_weights,
            axis=1,
            keepdims=True,
        )
        resample_keys = jax.random.split(resample_key, batch_size)
        indices = jax.vmap(
            lambda row_key, row_logits: jax.random.categorical(
                row_key,
                logits=row_logits,
                shape=(num_particles,),
            )
        )(resample_keys, normalized_log_weights)
        next_z = jnp.take_along_axis(z_t, indices, axis=1)
        return next_z, increment

    _, increments = jax.lax.scan(
        step,
        prev_particles,
        (batch.x.T, batch.y.T, *proposal_params, step_keys),
    )
    return jnp.sum(jnp.swapaxes(increments, 0, 1), axis=1)


def _gaussian_filter_entropy(filter_var: jax.Array) -> jax.Array:
    return 0.5 * (LOG_2PI + 1.0 + jnp.log(filter_var))


def _scheduled_weight(
    step_index: jax.Array,
    *,
    total_steps: int,
    start_fraction: float,
    ramp_fraction: float,
) -> jax.Array:
    progress = step_index / float(total_steps)
    if ramp_fraction == 0.0:
        return jnp.where(progress >= start_fraction, 1.0, 0.0)
    ramp = (progress - start_fraction) / ramp_fraction
    return jnp.clip(ramp, 0.0, 1.0)


def _nonlinear_mixture_edge_log_weights(
    outputs: GaussianMixtureMLPOutputs,
    batch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    observation: str,
    num_samples: int,
) -> jax.Array:
    z_key, backward_key = jax.random.split(key)
    z_t, _ = _sample_mixture_marginal(
        z_key,
        outputs.filter_weights,
        outputs.component_mean,
        outputs.component_var,
        sample_shape=(num_samples,),
    )
    z_tm1, _ = _sample_mixture_backward_conditional(
        backward_key,
        z_t,
        outputs.filter_weights,
        outputs.component_mean,
        outputs.component_var,
        outputs.backward_a,
        outputs.backward_b,
        outputs.backward_var,
    )
    prev_weights, prev_mean, prev_var = _previous_mixture_filter_beliefs(
        outputs,
        state_params,
    )
    observation_mean = nonlinear_observation_mean(z_t, batch.x[None, ...], observation)
    return (
        _normal_log_prob(batch.y[None, ...], observation_mean, state_params.r)
        + _normal_log_prob(z_t, z_tm1, state_params.q)
        + _mixture_log_prob(z_tm1, prev_weights, prev_mean, prev_var)
        - _mixture_log_prob(
            z_t,
            outputs.filter_weights,
            outputs.component_mean,
            outputs.component_var,
        )
        - _mixture_backward_log_prob(
            z_tm1,
            z_t,
            outputs.filter_weights,
            outputs.component_mean,
            outputs.component_var,
            outputs.backward_a,
            outputs.backward_b,
            outputs.backward_var,
        )
    )


def _nonlinear_mixture_windowed_joint_log_weights(
    outputs: GaussianMixtureMLPOutputs,
    batch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    observation: str,
    horizon: int,
    num_samples: int,
    num_windows: int,
) -> jax.Array:
    time_steps = batch.x.shape[1]
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if horizon > time_steps:
        raise ValueError("horizon cannot exceed batch time_steps")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_windows <= 0:
        raise ValueError("num_windows must be positive")

    possible_windows = time_steps - horizon + 1
    end_indices = (
        jnp.arange(horizon - 1, time_steps)
        if num_windows >= possible_windows
        else jax.random.randint(
            key,
            shape=(num_windows,),
            minval=horizon - 1,
            maxval=time_steps,
        )
    )
    sample_key, *backward_keys = jax.random.split(key, horizon + 1)
    end_weights = jnp.take(outputs.filter_weights, end_indices, axis=1)
    end_mean = jnp.take(outputs.component_mean, end_indices, axis=1)
    end_var = jnp.take(outputs.component_var, end_indices, axis=1)
    z_t, _ = _sample_mixture_marginal(
        sample_key,
        end_weights,
        end_mean,
        end_var,
        sample_shape=(num_samples,),
    )
    log_q = _mixture_log_prob(z_t, end_weights, end_mean, end_var)
    log_score = jnp.zeros_like(log_q)

    for offset in range(horizon):
        t_indices = end_indices - offset
        x_t = jnp.take(batch.x, t_indices, axis=1)
        y_t = jnp.take(batch.y, t_indices, axis=1)
        obs_mean = nonlinear_observation_mean(z_t, x_t[None, ...], observation)
        log_score = log_score + _normal_log_prob(y_t[None, ...], obs_mean, state_params.r)

        weights_t = jnp.take(outputs.filter_weights, t_indices, axis=1)
        mean_t = jnp.take(outputs.component_mean, t_indices, axis=1)
        var_t = jnp.take(outputs.component_var, t_indices, axis=1)
        backward_a = jnp.take(outputs.backward_a, t_indices, axis=1)
        backward_b = jnp.take(outputs.backward_b, t_indices, axis=1)
        backward_var = jnp.take(outputs.backward_var, t_indices, axis=1)
        z_prev, _ = _sample_mixture_backward_conditional(
            backward_keys[offset],
            z_t,
            weights_t,
            mean_t,
            var_t,
            backward_a,
            backward_b,
            backward_var,
        )
        log_score = log_score + _normal_log_prob(z_t, z_prev, state_params.q)
        log_q = log_q + _mixture_backward_log_prob(
            z_prev,
            z_t,
            weights_t,
            mean_t,
            var_t,
            backward_a,
            backward_b,
            backward_var,
        )
        z_t = z_prev

    first_transition_indices = end_indices - horizon + 1
    prev_weights, prev_mean, prev_var = _previous_mixture_filter_beliefs(outputs, state_params)
    initial_weights = jnp.take(prev_weights, first_transition_indices, axis=1)
    initial_mean = jnp.take(prev_mean, first_transition_indices, axis=1)
    initial_var = jnp.take(prev_var, first_transition_indices, axis=1)
    log_score = log_score + _mixture_log_prob(z_t, initial_weights, initial_mean, initial_var)
    return log_score - log_q


def _state_nll(outputs, z: jax.Array) -> jax.Array:
    if _is_mixture_outputs(outputs):
        return -_mixture_log_prob(
            z,
            outputs.filter_weights,
            outputs.component_mean,
            outputs.component_var,
        )
    return scalar_gaussian_nll(z, outputs.filter_mean, outputs.filter_var)


def _nonlinear_mixture_preassimilation_log_prob_y(
    outputs: GaussianMixtureMLPOutputs,
    x: jax.Array,
    y: jax.Array,
    params: LinearGaussianParams,
    *,
    observation: str,
    num_points: int,
) -> jax.Array:
    if observation != "x_sine":
        raise ValueError(f"Unsupported nonlinear predictive likelihood: {observation}")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    prev_weights, prev_mean, prev_var = _previous_mixture_filter_beliefs(outputs, params)
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(num_points)
    nodes = jnp.asarray(nodes_np, dtype=prev_mean.dtype)
    log_quadrature_weights = jnp.log(jnp.asarray(weights_np, dtype=prev_mean.dtype))
    pred_state_var = prev_var + params.q
    z = prev_mean[..., None] + jnp.sqrt(2.0 * pred_state_var[..., None]) * nodes
    obs_mean = x[..., None, None] * jnp.sin(z)
    log_likelihood = _normal_log_prob(y[..., None, None], obs_mean, params.r)
    component_log_prob = (
        jnp.log(prev_weights)
        + jax.nn.logsumexp(log_quadrature_weights + log_likelihood, axis=-1)
        - 0.5 * jnp.log(jnp.pi)
    )
    return jax.nn.logsumexp(component_log_prob, axis=-1)


def _sample_mixture_marginal(
    key: jax.Array,
    weights: jax.Array,
    mean: jax.Array,
    var: jax.Array,
    *,
    sample_shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    key_component, key_noise = jax.random.split(key)
    component = jax.random.categorical(
        key_component,
        logits=jnp.log(weights),
        axis=-1,
        shape=sample_shape + weights.shape[:-1],
    )
    selected_mean = _take_component(mean, component)
    selected_var = _take_component(var, component)
    noise = jax.random.normal(key_noise, shape=selected_mean.shape, dtype=mean.dtype)
    return selected_mean + jnp.sqrt(selected_var) * noise, component


def _sample_mixture_backward_conditional(
    key: jax.Array,
    z_t: jax.Array,
    weights: jax.Array,
    mean: jax.Array,
    var: jax.Array,
    backward_a: jax.Array,
    backward_b: jax.Array,
    backward_var: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    key_component, key_noise = jax.random.split(key)
    component_logits = jnp.log(weights) + _normal_log_prob(z_t[..., None], mean, var)
    component = jax.random.categorical(key_component, logits=component_logits, axis=-1)
    selected_a = _take_component(backward_a, component)
    selected_b = _take_component(backward_b, component)
    selected_var = _take_component(backward_var, component)
    backward_mean = selected_a * z_t + selected_b
    noise = jax.random.normal(key_noise, shape=z_t.shape, dtype=z_t.dtype)
    return backward_mean + jnp.sqrt(selected_var) * noise, component


def _mixture_backward_log_prob(
    z_tm1: jax.Array,
    z_t: jax.Array,
    weights: jax.Array,
    mean: jax.Array,
    var: jax.Array,
    backward_a: jax.Array,
    backward_b: jax.Array,
    backward_var: jax.Array,
) -> jax.Array:
    log_edge = jax.nn.logsumexp(
        jnp.log(weights)
        + _normal_log_prob(z_t[..., None], mean, var)
        + _normal_log_prob(z_tm1[..., None], backward_a * z_t[..., None] + backward_b, backward_var),
        axis=-1,
    )
    return log_edge - _mixture_log_prob(z_t, weights, mean, var)


def _mixture_log_prob(
    value: jax.Array,
    weights: jax.Array,
    mean: jax.Array,
    var: jax.Array,
) -> jax.Array:
    return jax.nn.logsumexp(
        jnp.log(weights) + _normal_log_prob(value[..., None], mean, var),
        axis=-1,
    )


def _previous_mixture_filter_beliefs(
    outputs: GaussianMixtureMLPOutputs,
    state_params: LinearGaussianParams,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_size = outputs.component_mean.shape[0]
    num_components = outputs.component_mean.shape[-1]
    initial_weights = jnp.full(
        (batch_size, 1, num_components),
        1.0 / num_components,
        dtype=outputs.filter_weights.dtype,
    )
    initial_mean = jnp.full(
        (batch_size, 1, num_components),
        state_params.m0,
        dtype=outputs.component_mean.dtype,
    )
    initial_var = jnp.full(
        (batch_size, 1, num_components),
        state_params.p0,
        dtype=outputs.component_var.dtype,
    )
    return (
        jnp.concatenate((initial_weights, outputs.filter_weights[:, :-1]), axis=1),
        jnp.concatenate((initial_mean, outputs.component_mean[:, :-1]), axis=1),
        jnp.concatenate((initial_var, outputs.component_var[:, :-1]), axis=1),
    )


def _take_component(values: jax.Array, component: jax.Array) -> jax.Array:
    sample_ndim = component.ndim - (values.ndim - 1)
    if sample_ndim < 0:
        raise ValueError("component shape is not compatible with values")
    broadcast_values = jnp.reshape(values, (1,) * sample_ndim + values.shape)
    broadcast_values = jnp.broadcast_to(broadcast_values, component.shape + values.shape[-1:])
    return jnp.take_along_axis(broadcast_values, component[..., None], axis=-1)[..., 0]


def _is_mixture_outputs(outputs) -> bool:
    return isinstance(outputs, GaussianMixtureMLPOutputs)


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


def _normal_log_prob(
    value: jax.Array, mean: jax.Array | float, var: jax.Array | float
) -> jax.Array:
    return -0.5 * (LOG_2PI + jnp.log(var) + (value - mean) ** 2 / var)


def _write_loss_history(path: Path, history: list[tuple[int, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "loss"))
        writer.writerows(history)


def _render_summary(
    name: str,
    metrics: dict[str, float | int | str],
    history: list[tuple[int, float]],
) -> str:
    lines = [
        f"# {name}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Loss History", "", "| Step | Loss |", "|---:|---:|"])
    lines.extend(f"| {step} | {loss:.6f} |" for step, loss in history)
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `loss_history.csv`",
            "- `metrics.json`",
            "- `config.yaml`",
            "- `params.npz`",
            "- `diagnostics.npz`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
