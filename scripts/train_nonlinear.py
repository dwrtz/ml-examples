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
    direct_mlp_step,
    edge_mean_cov_from_outputs,
    init_direct_mlp_params,
    init_structured_mlp_params,
    run_direct_mlp_filter,
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
    nonlinear_structured_mlp_step,
    run_nonlinear_structured_mlp_filter,
    run_nonlinear_structured_mlp_teacher_forced,
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
    elbo_weight = float(training_config.get("elbo_weight", 1.0))
    joint_elbo_weight = float(training_config.get("joint_elbo_weight", 0.0))
    joint_elbo_horizon = int(training_config.get("joint_elbo_horizon", 1))
    joint_elbo_num_samples = int(training_config.get("joint_elbo_num_samples", 16))
    joint_elbo_num_windows = int(training_config.get("joint_elbo_num_windows", 8))
    joint_elbo_window_seed_offset = int(training_config.get("joint_elbo_window_seed_offset", 80_000))
    predictive_y_weight = float(training_config.get("predictive_y_weight", 0.0))
    predictive_y_num_samples = int(training_config.get("predictive_y_num_samples", 32))
    predictive_y_estimator = str(training_config.get("predictive_y_estimator", "quadrature"))
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
    if predictive_y_num_samples <= 0:
        raise ValueError("predictive_y_num_samples must be positive")
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
    )
    opt_state = init_adam(params)

    def loss_fn(
        current_params: dict[str, jax.Array],
        batch_x: jax.Array,
        batch_y: jax.Array,
        batch_z: jax.Array,
        y_observed: jax.Array,
        key: jax.Array,
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
            joint_key = jax.random.fold_in(key, joint_elbo_window_seed_offset)
            loss = loss - joint_elbo_weight * jnp.mean(
                _nonlinear_windowed_joint_elbo(
                    outputs,
                    batch,
                    state_params,
                    joint_key,
                    observation=data_config.observation,
                    horizon=joint_elbo_horizon,
                    num_samples=joint_elbo_num_samples,
                    num_windows=joint_elbo_num_windows,
                )
            )
        if predictive_y_weight != 0.0:
            prev_mean, prev_var = previous_filter_beliefs(
                outputs.filter_mean,
                outputs.filter_var,
                state_params,
            )
            predictive_y_log_prob = nonlinear_preassimilation_log_prob_y(
                prev_mean,
                prev_var,
                batch.x,
                batch.y,
                state_params,
                observation=data_config.observation,
                num_points=predictive_y_num_samples,
            )
            loss = loss - predictive_y_weight * jnp.mean(predictive_y_log_prob)
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
        )
    learned_predictive_mean, learned_predictive_var = nonlinear_predictive_moments_from_filter(
        outputs.filter_mean,
        outputs.filter_var,
        eval_batch.x,
        state_params,
        observation=eval_data_config.observation,
    )
    learned_state_nll = scalar_gaussian_nll(eval_batch.z, outputs.filter_mean, outputs.filter_var)
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
    learned_predictive_y_nll = -nonlinear_preassimilation_log_prob_y(
        eval_prev_mean,
        eval_prev_var,
        eval_batch.x,
        eval_batch.y,
        state_params,
        observation=eval_data_config.observation,
        num_points=predictive_y_num_samples,
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
        "elbo_weight": elbo_weight,
        "joint_elbo_weight": joint_elbo_weight,
        "joint_elbo_horizon": joint_elbo_horizon,
        "joint_elbo_num_samples": joint_elbo_num_samples,
        "joint_elbo_num_windows": joint_elbo_num_windows,
        "joint_elbo_window_seed_offset": joint_elbo_window_seed_offset,
        "predictive_y_weight": predictive_y_weight,
        "predictive_y_num_samples": predictive_y_num_samples,
        "predictive_y_estimator": predictive_y_estimator,
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
        teacher_state_nll = scalar_gaussian_nll(
            eval_batch.z,
            teacher_outputs.filter_mean,
            teacher_outputs.filter_var,
        )
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
    np.savez(
        output_dir / "diagnostics.npz",
        x=np.asarray(eval_batch.x),
        y=np.asarray(eval_batch.y),
        z=np.asarray(eval_batch.z),
        learned_filter_mean=np.asarray(outputs.filter_mean),
        learned_filter_var=np.asarray(outputs.filter_var),
        reference_filter_mean=np.asarray(reference.filter_mean),
        reference_filter_var=np.asarray(reference.filter_var),
        learned_predictive_mean=np.asarray(learned_predictive_mean),
        learned_predictive_var=np.asarray(learned_predictive_var),
        reference_predictive_mean=np.asarray(reference.predictive_mean),
        reference_predictive_var=np.asarray(reference.predictive_var),
        y_observed_mask=np.asarray(eval_y_observed),
        loss_history_step=np.asarray([step for step, _ in history], dtype=np.int64),
        loss_history_loss=np.asarray([loss for _, loss in history], dtype=np.float64),
    )
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
) -> dict[str, jax.Array]:
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
):
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


def _where_outputs(condition, true_outputs, false_outputs):
    condition = condition.astype(bool)
    return type(true_outputs)(
        *(
            jnp.where(condition, true_value, false_value)
            for true_value, false_value in zip(true_outputs, false_outputs, strict=True)
        )
    )


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
):
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
    elbo = (
        _normal_log_prob(batch.y[None, ...], observation_mean, state_params.r)
        + _normal_log_prob(z_t, z_tm1, state_params.q)
        + _normal_log_prob(z_tm1, prev_mean[None, ...], prev_var[None, ...])
        - _normal_log_prob(z_t, outputs.filter_mean[None, ...], outputs.filter_var[None, ...])
        - _normal_log_prob(z_tm1, backward_mean, outputs.backward_var[None, ...])
    )
    return jnp.mean(elbo, axis=0)


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
    if horizon == 1:
        return _nonlinear_edge_elbo(
            outputs,
            batch,
            state_params,
            key,
            observation=observation,
            num_samples=num_samples,
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
    return jnp.mean(log_score - log_q, axis=0)


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
