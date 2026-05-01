"""Train a nonlinear mixture filter by distilling quadrature ADF targets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import yaml

from vbf.dtypes import DEFAULT_DTYPE
import jax.numpy as jnp
from sweep_quadrature_adf import (
    _logsumexp as _np_logsumexp,
    _mixture_log_prob as _np_mixture_log_prob,
    _mixture_mean_var as _np_mixture_mean_var,
    _normal_log_prob as _np_normal_log_prob,
    _project_mixture_em,
    run_quadrature_adf_filter,
)
from train_nonlinear import _nonlinear_mixture_preassimilation_log_prob_y, _state_nll
from vbf.data import LinearGaussianParams
from vbf.metrics import gaussian_interval_coverage, rmse_global, scalar_gaussian_nll
from vbf.models.cells import (
    component_mixture_mlp_step,
    direct_mixture_mlp_step,
    init_component_mixture_mlp_params,
    init_direct_mixture_mlp_params,
    run_component_mixture_mlp_filter,
    run_direct_mixture_mlp_filter,
)
from vbf.nonlinear import (
    GridReferenceConfig,
    NonlinearDataConfig,
    make_nonlinear_batch,
    nonlinear_predictive_moments_from_filter,
)
from vbf.nonlinear_cache import load_or_compute_nonlinear_reference
from vbf.train import adam_update, init_adam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default="outputs/cache/nonlinear_reference")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    data_config = NonlinearDataConfig(**config["data"])
    eval_data_config = NonlinearDataConfig(
        **{**config["data"], **config.get("evaluation", {}).get("data", {})}
    )
    state_params = LinearGaussianParams(**config["state_space"])
    reference_config = GridReferenceConfig(**config.get("reference", {}))
    training_config = config["training"]

    components = int(training_config.get("mixture_components", 4))
    if components <= 1:
        raise ValueError("quadrature ADF distillation expects mixture_components > 1")
    cell_type = str(training_config.get("cell_type", "direct_mixture"))
    if cell_type not in {"direct_mixture", "component_mixture"}:
        raise ValueError("cell_type must be one of: direct_mixture, component_mixture")
    steps = int(training_config.get("steps", 250))
    learning_rate = float(training_config.get("learning_rate", 1e-3))
    hidden_dim = int(training_config.get("hidden_dim", 32))
    min_var = float(training_config.get("min_var", 1e-6))
    log_every = int(training_config.get("log_every", 50))
    init_span = float(training_config.get("mixture_component_mean_init_span", 2.0 * np.pi))
    target_likelihood_power = float(training_config.get("target_likelihood_power", 0.5))
    target_projection = str(training_config.get("target_projection", "em"))
    target_alias_spacing = float(training_config.get("target_alias_spacing", 0.0))
    target_initial_weighting = str(training_config.get("target_initial_weighting", "uniform"))
    target_max_active_aliases = int(training_config.get("target_max_active_aliases", 0))
    target_alias_mean_shrink = float(training_config.get("target_alias_mean_shrink", 1.0))
    target_alias_entropy_threshold = float(
        training_config.get("target_alias_entropy_threshold", 0.0)
    )
    target_num_points = int(training_config.get("target_num_points", 64))
    target_em_steps = int(training_config.get("target_em_steps", 30))
    target_density_num_points = int(training_config.get("target_density_num_points", 16))
    density_loss_weight = float(training_config.get("density_loss_weight", 1.0))
    weight_loss_weight = float(training_config.get("weight_loss_weight", 0.0))
    mean_loss_weight = float(training_config.get("mean_loss_weight", 0.0))
    logvar_loss_weight = float(training_config.get("logvar_loss_weight", 0.0))
    moment_loss_weight = float(training_config.get("moment_loss_weight", 0.1))
    predictive_y_weight = float(training_config.get("predictive_y_weight", 0.0))
    predictive_y_num_samples = int(training_config.get("predictive_y_num_samples", 32))
    predictive_carry_weight = float(training_config.get("predictive_carry_weight", 0.0))
    predictive_normalizer_target_weight = float(
        training_config.get("predictive_normalizer_target_weight", 0.0)
    )
    predictive_normalizer_num_points = int(
        training_config.get("predictive_normalizer_num_points", predictive_y_num_samples)
    )
    rollout_distillation_weight = float(training_config.get("rollout_distillation_weight", 0.0))
    rollout_distillation_horizon = int(training_config.get("rollout_distillation_horizon", 1))
    rollout_distillation_stride = int(
        training_config.get("rollout_distillation_stride", rollout_distillation_horizon)
    )
    component_stability_weight = float(training_config.get("component_stability_weight", 0.0))
    component_stability_mean_weight = float(
        training_config.get("component_stability_mean_weight", 1.0)
    )
    component_stability_logvar_weight = float(
        training_config.get("component_stability_logvar_weight", 0.1)
    )
    component_stability_weight_weight = float(
        training_config.get("component_stability_weight_weight", 0.1)
    )
    hybrid_refinement_steps = int(training_config.get("hybrid_refinement_steps", 0))
    if predictive_normalizer_target_weight < 0.0:
        raise ValueError("predictive_normalizer_target_weight must be nonnegative")
    if predictive_normalizer_num_points <= 0:
        raise ValueError("predictive_normalizer_num_points must be positive")
    if rollout_distillation_weight < 0.0:
        raise ValueError("rollout_distillation_weight must be nonnegative")
    if rollout_distillation_horizon <= 0:
        raise ValueError("rollout_distillation_horizon must be positive")
    if rollout_distillation_horizon > data_config.time_steps:
        raise ValueError("rollout_distillation_horizon cannot exceed data time_steps")
    if rollout_distillation_stride <= 0:
        raise ValueError("rollout_distillation_stride must be positive")
    if component_stability_weight < 0.0:
        raise ValueError("component_stability_weight must be nonnegative")
    if component_stability_mean_weight < 0.0:
        raise ValueError("component_stability_mean_weight must be nonnegative")
    if component_stability_logvar_weight < 0.0:
        raise ValueError("component_stability_logvar_weight must be nonnegative")
    if component_stability_weight_weight < 0.0:
        raise ValueError("component_stability_weight_weight must be nonnegative")

    train_batch = make_nonlinear_batch(data_config, state_params, seed=int(config["seed"]))
    target = run_quadrature_adf_filter(
        np.asarray(train_batch.x),
        np.asarray(train_batch.y),
        state_params,
        components=components,
        likelihood_power=target_likelihood_power,
        init_span=init_span,
        projection=target_projection,
        alias_spacing=target_alias_spacing,
        initial_weighting=target_initial_weighting,
        max_active_aliases=target_max_active_aliases,
        alias_mean_shrink=target_alias_mean_shrink,
        alias_entropy_threshold=target_alias_entropy_threshold,
        num_points=target_num_points,
        em_steps=target_em_steps,
        min_var=min_var,
    )
    target_weights = jnp.asarray(target.weights)
    target_mean = jnp.asarray(target.component_mean)
    target_var = jnp.asarray(target.component_var)
    target_filter_mean = jnp.asarray(target.filter_mean)
    target_filter_var = jnp.asarray(target.filter_var)
    target_predictive_weights = jnp.asarray(target.predictive_weights)
    target_predictive_mean = jnp.asarray(target.predictive_component_mean)
    target_predictive_var = jnp.asarray(target.predictive_component_var)
    target_predictive_y_log_prob = jnp.asarray(target.predictive_y_log_prob)
    density_nodes_np, density_weights_np = np.polynomial.hermite.hermgauss(
        target_density_num_points
    )
    density_nodes = jnp.asarray(density_nodes_np, dtype=target_mean.dtype)
    density_log_weights = (
        jnp.log(jnp.asarray(density_weights_np, dtype=target_mean.dtype))
        - 0.5 * jnp.log(jnp.pi)
    )

    filter_key, predictive_key = jax.random.split(jax.random.PRNGKey(int(config["seed"]) + 1))
    params = {
        "filter": _init_params(
            cell_type,
            filter_key,
            hidden_dim=hidden_dim,
            num_components=components,
            component_mean_init_span=init_span,
        )
    }
    if predictive_carry_weight != 0.0:
        params["predictive"] = _init_predictive_carry_params(
            predictive_key,
            hidden_dim=hidden_dim,
            num_components=components,
        )
    opt_state = init_adam(params)
    train_x = train_batch.x
    train_y = train_batch.y
    train_z = train_batch.z

    def loss_fn(current_params: dict[str, jax.Array]) -> jax.Array:
        batch = type(train_batch)(x=train_x, y=train_y, z=train_z)
        outputs = _run_filter(
            cell_type,
            current_params["filter"],
            batch,
            state_params,
            num_components=components,
            component_mean_init_span=init_span,
            min_var=min_var,
        )
        pred_weights = jnp.clip(outputs.filter_weights, min_var, 1.0)
        pred_var = jnp.maximum(outputs.component_var, min_var)
        safe_target_var = jnp.maximum(target_var, min_var)
        target_z = (
            target_mean[..., None]
            + jnp.sqrt(2.0 * safe_target_var[..., None]) * density_nodes
        )
        target_log_mass = (
            jnp.log(jnp.clip(target_weights, min_var, 1.0))[..., None] + density_log_weights
        )
        log_q_at_target = _mixture_log_prob(
            target_z,
            pred_weights[..., None, None, :],
            outputs.component_mean[..., None, None, :],
            pred_var[..., None, None, :],
        )
        density_loss = -jnp.mean(
            jnp.sum(jnp.exp(target_log_mass) * log_q_at_target, axis=(-2, -1))
        )
        weight_loss = -jnp.mean(jnp.sum(target_weights * jnp.log(pred_weights), axis=-1))
        mean_loss = jnp.mean(
            jnp.sum(
                target_weights * (outputs.component_mean - target_mean) ** 2 / safe_target_var,
                axis=-1,
            )
        )
        logvar_loss = jnp.mean(
            jnp.sum(
                target_weights * (jnp.log(pred_var) - jnp.log(safe_target_var)) ** 2,
                axis=-1,
            )
        )
        moment_loss = jnp.mean((outputs.filter_mean - target_filter_mean) ** 2 / target_filter_var)
        moment_loss = moment_loss + jnp.mean(
            (jnp.log(outputs.filter_var) - jnp.log(target_filter_var)) ** 2
        )
        loss = (
            density_loss_weight * density_loss
            + weight_loss_weight * weight_loss
            + mean_loss_weight * mean_loss
            + logvar_loss_weight * logvar_loss
            + moment_loss_weight * moment_loss
        )
        if predictive_carry_weight != 0.0:
            pred_weights_carry, pred_mean_carry, pred_var_carry = _run_predictive_carry_head(
                current_params["predictive"],
                outputs,
                batch.x,
                state_params,
                num_components=components,
                component_mean_init_span=init_span,
                min_var=min_var,
            )
            predictive_carry_loss = _mixture_density_projection_loss(
                target_predictive_weights,
                target_predictive_mean,
                target_predictive_var,
                pred_weights_carry,
                pred_mean_carry,
                pred_var_carry,
                density_nodes,
                density_log_weights,
                min_var=min_var,
            )
            loss = loss + predictive_carry_weight * predictive_carry_loss
        if predictive_y_weight != 0.0:
            predictive_y_log_prob = _nonlinear_mixture_preassimilation_log_prob_y(
                outputs,
                batch.x,
                batch.y,
                state_params,
                observation=data_config.observation,
                num_points=predictive_y_num_samples,
            )
            loss = loss - predictive_y_weight * jnp.mean(predictive_y_log_prob)
        if predictive_normalizer_target_weight != 0.0:
            predictive_log_prob = _mixture_preassimilation_log_prob_y_from_components(
                * _previous_mixture_filter_beliefs(
                    outputs,
                    state_params,
                    num_components=components,
                    component_mean_init_span=init_span,
                ),
                batch.x,
                batch.y,
                state_params,
                num_points=predictive_normalizer_num_points,
                min_var=min_var,
            )
            normalizer_loss = jnp.mean((predictive_log_prob - target_predictive_y_log_prob) ** 2)
            loss = loss + predictive_normalizer_target_weight * normalizer_loss
        if rollout_distillation_weight != 0.0:
            rollout_loss = _rollout_distillation_loss(
                cell_type,
                current_params["filter"],
                batch,
                state_params,
                target_weights,
                target_mean,
                target_var,
                target_predictive_weights,
                target_predictive_mean,
                target_predictive_var,
                density_nodes,
                density_log_weights,
                num_components=components,
                horizon=rollout_distillation_horizon,
                stride=rollout_distillation_stride,
                min_var=min_var,
            )
            loss = loss + rollout_distillation_weight * rollout_loss
        if component_stability_weight != 0.0:
            stability_loss = _component_stability_loss(
                outputs,
                state_params,
                num_components=components,
                component_mean_init_span=init_span,
                mean_weight=component_stability_mean_weight,
                logvar_weight=component_stability_logvar_weight,
                weight_weight=component_stability_weight_weight,
                min_var=min_var,
            )
            loss = loss + component_stability_weight * stability_loss
        return loss

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    history: list[tuple[int, float]] = []
    for step in range(1, steps + 1):
        loss_value, grads = value_and_grad(params)
        params, opt_state = adam_update(params, grads, opt_state, learning_rate=learning_rate)
        if step == 1 or step % log_every == 0:
            history.append((step, float(loss_value)))

    eval_seed = int(config["seed"]) + int(config.get("evaluation", {}).get("seed_offset", 10000))
    eval_cached = load_or_compute_nonlinear_reference(
        eval_data_config,
        state_params,
        seed=eval_seed,
        grid_config=reference_config,
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
    )
    eval_batch = eval_cached.batch
    outputs = _run_filter(
        cell_type,
        params["filter"],
        eval_batch,
        state_params,
        num_components=components,
        component_mean_init_span=init_span,
        min_var=min_var,
    )
    learned_predictive_mean, learned_predictive_var = nonlinear_predictive_moments_from_filter(
        outputs.filter_mean,
        outputs.filter_var,
        eval_batch.x,
        state_params,
        observation=eval_data_config.observation,
    )
    learned_state_nll = _state_nll(outputs, eval_batch.z)
    learned_predictive_y_nll = -_nonlinear_mixture_preassimilation_log_prob_y(
        outputs,
        eval_batch.x,
        eval_batch.y,
        state_params,
        observation=eval_data_config.observation,
        num_points=predictive_y_num_samples,
    )
    predictive_carry_y_nll = None
    predictive_carry_weights = None
    predictive_carry_mean = None
    predictive_carry_var = None
    if predictive_carry_weight != 0.0:
        predictive_carry_weights, predictive_carry_mean, predictive_carry_var = (
            _run_predictive_carry_head(
                params["predictive"],
                outputs,
                eval_batch.x,
                state_params,
                num_components=components,
                component_mean_init_span=init_span,
                min_var=min_var,
            )
        )
        predictive_carry_y_nll = -_mixture_preassimilation_log_prob_y_from_components(
            predictive_carry_weights,
            predictive_carry_mean,
            predictive_carry_var,
            eval_batch.x,
            eval_batch.y,
            state_params,
            num_points=predictive_y_num_samples,
            min_var=min_var,
        )
    hybrid_outputs = None
    if hybrid_refinement_steps > 0:
        hybrid_outputs = _run_hybrid_refined_filter(
            params["filter"],
            cell_type,
            eval_batch,
            state_params,
            num_components=components,
            component_mean_init_span=init_span,
            likelihood_power=target_likelihood_power,
            num_points=target_num_points,
            refinement_steps=hybrid_refinement_steps,
            min_var=min_var,
        )
        hybrid_state_nll = -_np_mixture_log_prob(
            np.asarray(eval_batch.z),
            hybrid_outputs["weights"],
            hybrid_outputs["component_mean"],
            hybrid_outputs["component_var"],
        )
        hybrid_predictive_y_nll = -hybrid_outputs["predictive_y_log_prob"]
    reference_state_nll = scalar_gaussian_nll(
        eval_batch.z,
        eval_cached.reference.filter_mean,
        eval_cached.reference.filter_var,
    )
    reference_predictive_nll = scalar_gaussian_nll(
        eval_batch.y,
        eval_cached.reference.predictive_mean,
        eval_cached.reference.predictive_var,
    )
    metrics = {
        "benchmark": "nonlinear",
        "objective": "quadrature_adf_distillation",
        "seed": int(config["seed"]),
        "observation": eval_data_config.observation,
        "x_pattern": eval_data_config.x_pattern,
        "training_steps": steps,
        "cell_type": cell_type,
        "posterior_family": "gaussian_mixture",
        "mixture_components": components,
        "mixture_component_mean_init_span": init_span,
        "target_likelihood_power": target_likelihood_power,
        "target_projection": target_projection,
        "target_alias_spacing": target_alias_spacing,
        "target_initial_weighting": target_initial_weighting,
        "target_max_active_aliases": target_max_active_aliases,
        "target_alias_mean_shrink": target_alias_mean_shrink,
        "target_alias_entropy_threshold": target_alias_entropy_threshold,
        "target_num_points": target_num_points,
        "target_em_steps": target_em_steps,
        "target_density_num_points": target_density_num_points,
        "density_loss_weight": density_loss_weight,
        "weight_loss_weight": weight_loss_weight,
        "mean_loss_weight": mean_loss_weight,
        "logvar_loss_weight": logvar_loss_weight,
        "moment_loss_weight": moment_loss_weight,
        "predictive_y_weight": predictive_y_weight,
        "predictive_y_num_samples": predictive_y_num_samples,
        "predictive_carry_weight": predictive_carry_weight,
        "predictive_normalizer_target_weight": predictive_normalizer_target_weight,
        "predictive_normalizer_num_points": predictive_normalizer_num_points,
        "rollout_distillation_weight": rollout_distillation_weight,
        "rollout_distillation_horizon": rollout_distillation_horizon,
        "rollout_distillation_stride": rollout_distillation_stride,
        "component_stability_weight": component_stability_weight,
        "component_stability_mean_weight": component_stability_mean_weight,
        "component_stability_logvar_weight": component_stability_logvar_weight,
        "component_stability_weight_weight": component_stability_weight_weight,
        "hybrid_refinement_steps": hybrid_refinement_steps,
        "eval_reference_cache_hit": eval_cached.cache_hit,
        "eval_reference_cache_path": str(eval_cached.cache_path),
        "final_loss": float(loss_fn(params)),
        "state_rmse": float(rmse_global(outputs.filter_mean, eval_batch.z)),
        "reference_state_rmse": float(
            rmse_global(eval_cached.reference.filter_mean, eval_batch.z)
        ),
        "state_nll": float(jnp.mean(learned_state_nll)),
        "reference_state_nll": float(jnp.mean(reference_state_nll)),
        "predictive_nll": float(
            jnp.mean(
                scalar_gaussian_nll(
                    eval_batch.y,
                    learned_predictive_mean,
                    learned_predictive_var,
                )
            )
        ),
        "predictive_y_nll": float(jnp.mean(learned_predictive_y_nll)),
        "predictive_carry_y_nll": None
        if predictive_carry_y_nll is None
        else float(jnp.mean(predictive_carry_y_nll)),
        "hybrid_state_nll": None
        if hybrid_outputs is None
        else float(np.mean(hybrid_state_nll)),
        "hybrid_predictive_y_nll": None
        if hybrid_outputs is None
        else float(np.mean(hybrid_predictive_y_nll)),
        "hybrid_coverage_90": None
        if hybrid_outputs is None
        else float(
            gaussian_interval_coverage(
                eval_batch.z,
                jnp.asarray(hybrid_outputs["filter_mean"]),
                jnp.asarray(hybrid_outputs["filter_var"]),
                z_score=1.6448536269514722,
            )
        ),
        "hybrid_variance_ratio": None
        if hybrid_outputs is None
        else float(np.mean(hybrid_outputs["filter_var"]) / np.mean(eval_cached.reference.filter_var)),
        "pareto_state_nll": float(jnp.mean(learned_state_nll)),
        "pareto_predictive_y_nll": None
        if hybrid_outputs is None
        else float(np.mean(hybrid_predictive_y_nll)),
        "pareto_coverage_90": float(
            gaussian_interval_coverage(
                eval_batch.z,
                outputs.filter_mean,
                outputs.filter_var,
                z_score=1.6448536269514722,
            )
        ),
        "pareto_variance_ratio": float(
            jnp.mean(outputs.filter_var) / jnp.mean(eval_cached.reference.filter_var)
        ),
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
                eval_cached.reference.filter_mean,
                eval_cached.reference.filter_var,
                z_score=1.6448536269514722,
            )
        ),
        "mean_filter_variance": float(jnp.mean(outputs.filter_var)),
        "reference_mean_filter_variance": float(jnp.mean(eval_cached.reference.filter_var)),
        "variance_ratio": float(
            jnp.mean(outputs.filter_var) / jnp.mean(eval_cached.reference.filter_var)
        ),
    }

    output_dir = Path(
        args.output_dir or config.get("output_dir", "outputs/nonlinear_quadrature_adf_distilled")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_loss_history(output_dir / "loss_history.csv", history)
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(
        output_dir / "params.npz",
        **_flatten_params_for_npz(params),
    )
    np.savez(
        output_dir / "diagnostics.npz",
        x=np.asarray(eval_batch.x),
        y=np.asarray(eval_batch.y),
        z=np.asarray(eval_batch.z),
        learned_filter_weights=np.asarray(outputs.filter_weights),
        learned_component_mean=np.asarray(outputs.component_mean),
        learned_component_var=np.asarray(outputs.component_var),
        learned_filter_mean=np.asarray(outputs.filter_mean),
        learned_filter_var=np.asarray(outputs.filter_var),
        reference_filter_mean=np.asarray(eval_cached.reference.filter_mean),
        reference_filter_var=np.asarray(eval_cached.reference.filter_var),
        target_filter_weights=np.asarray(target.weights),
        target_component_mean=np.asarray(target.component_mean),
        target_component_var=np.asarray(target.component_var),
        target_predictive_weights=np.asarray(target.predictive_weights),
        target_predictive_component_mean=np.asarray(target.predictive_component_mean),
        target_predictive_component_var=np.asarray(target.predictive_component_var),
        loss_history_step=np.asarray([step for step, _ in history], dtype=np.int64),
        loss_history_loss=np.asarray([loss for _, loss in history], dtype=np.float64),
        **(
            {}
            if predictive_carry_weights is None
            else {
                "predictive_carry_weights": np.asarray(predictive_carry_weights),
                "predictive_carry_component_mean": np.asarray(predictive_carry_mean),
                "predictive_carry_component_var": np.asarray(predictive_carry_var),
            }
        ),
        **(
            {}
            if hybrid_outputs is None
            else {f"hybrid_{name}": value for name, value in hybrid_outputs.items()}
        ),
    )
    summary_path = output_dir / "evaluation_summary.md"
    summary_path.write_text(_render_summary(config["name"], metrics, history), encoding="utf-8")
    print(f"Wrote {summary_path}")


def _write_loss_history(path: Path, history: list[tuple[int, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "loss"])
        writer.writerows(history)


def _init_predictive_carry_params(
    key: jax.Array,
    *,
    hidden_dim: int,
    num_components: int,
) -> dict[str, jax.Array]:
    input_dim = 6
    key_w1, _ = jax.random.split(key)
    w1 = jax.random.normal(key_w1, shape=(input_dim, hidden_dim), dtype=DEFAULT_DTYPE)
    w1 = w1 * jnp.sqrt(2.0 / input_dim)
    return {
        "w1": w1,
        "b1": jnp.zeros((hidden_dim,), dtype=DEFAULT_DTYPE),
        "w2": jnp.zeros((hidden_dim, 3), dtype=DEFAULT_DTYPE),
        "b2": jnp.zeros((num_components, 3), dtype=DEFAULT_DTYPE),
    }


def _init_params(
    cell_type: str,
    key: jax.Array,
    *,
    hidden_dim: int,
    num_components: int,
    component_mean_init_span: float,
) -> dict[str, jax.Array]:
    if cell_type == "component_mixture":
        return init_component_mixture_mlp_params(
            key,
            hidden_dim=hidden_dim,
            num_components=num_components,
            component_mean_init_span=component_mean_init_span,
        )
    return init_direct_mixture_mlp_params(
        key,
        hidden_dim=hidden_dim,
        num_components=num_components,
        component_mean_init_span=component_mean_init_span,
    )


def _run_filter(
    cell_type: str,
    params: dict[str, jax.Array],
    batch,
    state_params: LinearGaussianParams,
    *,
    num_components: int,
    component_mean_init_span: float,
    min_var: float,
):
    if cell_type == "component_mixture":
        return run_component_mixture_mlp_filter(
            params,
            batch,
            state_params,
            num_components=num_components,
            min_var=min_var,
            component_mean_init_span=component_mean_init_span,
        )
    return run_direct_mixture_mlp_filter(
        params,
        batch,
        state_params,
        num_components=num_components,
        min_var=min_var,
    )


def _rollout_distillation_loss(
    cell_type: str,
    params: dict[str, jax.Array],
    batch,
    state_params: LinearGaussianParams,
    target_weights: jax.Array,
    target_mean: jax.Array,
    target_var: jax.Array,
    target_predictive_weights: jax.Array,
    target_predictive_mean: jax.Array,
    target_predictive_var: jax.Array,
    density_nodes: jax.Array,
    density_log_weights: jax.Array,
    *,
    num_components: int,
    horizon: int,
    stride: int,
    min_var: float,
) -> jax.Array:
    """Short self-fed rollouts initialized from teacher filtering beliefs."""

    starts = range(0, batch.x.shape[1] - horizon + 1, stride)
    losses = []
    for start in starts:
        init = _teacher_previous_carry(
            target_weights,
            target_mean,
            target_var,
            target_predictive_weights,
            target_predictive_mean,
            target_predictive_var,
            state_params,
            start=start,
        )
        x_window = batch.x[:, start : start + horizon].T
        y_window = batch.y[:, start : start + horizon].T

        def step(carry, obs):
            prev_weights, prev_mean, prev_var = carry
            x_t, y_t = obs
            if cell_type == "component_mixture":
                outputs = component_mixture_mlp_step(
                    params,
                    prev_weights,
                    prev_mean,
                    prev_var,
                    x_t,
                    y_t,
                    state_params,
                    min_var=min_var,
                )
            else:
                outputs = direct_mixture_mlp_step(
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
            next_carry = (outputs.filter_weights, outputs.component_mean, outputs.component_var)
            return next_carry, outputs

        _, window_outputs = jax.lax.scan(step, init, (x_window, y_window))
        pred_weights = _time_major_to_batch_major(window_outputs.filter_weights)
        pred_mean = _time_major_to_batch_major(window_outputs.component_mean)
        pred_var = _time_major_to_batch_major(window_outputs.component_var)
        losses.append(
            _mixture_density_projection_loss(
                target_weights[:, start : start + horizon],
                target_mean[:, start : start + horizon],
                target_var[:, start : start + horizon],
                pred_weights,
                pred_mean,
                pred_var,
                density_nodes,
                density_log_weights,
                min_var=min_var,
            )
        )
    if not losses:
        raise ValueError("rollout distillation produced no windows")
    return jnp.mean(jnp.stack(losses))


def _teacher_previous_carry(
    target_weights: jax.Array,
    target_mean: jax.Array,
    target_var: jax.Array,
    target_predictive_weights: jax.Array,
    target_predictive_mean: jax.Array,
    target_predictive_var: jax.Array,
    state_params: LinearGaussianParams,
    *,
    start: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if start > 0:
        return (
            target_weights[:, start - 1],
            target_mean[:, start - 1],
            target_var[:, start - 1],
        )
    return (
        target_predictive_weights[:, 0],
        target_predictive_mean[:, 0],
        jnp.maximum(target_predictive_var[:, 0] - state_params.q, 1e-12),
    )


def _time_major_to_batch_major(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, 0, 1)


def _component_stability_loss(
    outputs,
    state_params: LinearGaussianParams,
    *,
    num_components: int,
    component_mean_init_span: float,
    mean_weight: float,
    logvar_weight: float,
    weight_weight: float,
    min_var: float,
) -> jax.Array:
    """Penalize component churn without assuming a specific alias geometry."""

    prev_weights, prev_mean, prev_var = _previous_mixture_filter_beliefs(
        outputs,
        state_params,
        num_components=num_components,
        component_mean_init_span=component_mean_init_span,
    )
    pred_var = jnp.maximum(prev_var + state_params.q, min_var)
    current_var = jnp.maximum(outputs.component_var, min_var)
    component_weights = jax.lax.stop_gradient(
        0.5 * (jnp.clip(prev_weights, min_var, 1.0) + jnp.clip(outputs.filter_weights, min_var, 1.0))
    )
    component_weights = component_weights / jnp.sum(component_weights, axis=-1, keepdims=True)
    mean_loss = jnp.sum(component_weights * (outputs.component_mean - prev_mean) ** 2 / pred_var, axis=-1)
    logvar_loss = jnp.sum(
        component_weights * (jnp.log(current_var) - jnp.log(pred_var)) ** 2,
        axis=-1,
    )
    weight_loss = jnp.sum(
        jnp.clip(prev_weights, min_var, 1.0)
        * (jnp.log(jnp.clip(prev_weights, min_var, 1.0)) - jnp.log(jnp.clip(outputs.filter_weights, min_var, 1.0))),
        axis=-1,
    )
    return jnp.mean(
        mean_weight * mean_loss
        + logvar_weight * logvar_loss
        + weight_weight * weight_loss
    )


def _run_hybrid_refined_filter(
    params: dict[str, jax.Array],
    cell_type: str,
    batch,
    state_params: LinearGaussianParams,
    *,
    num_components: int,
    component_mean_init_span: float,
    likelihood_power: float,
    num_points: int,
    refinement_steps: int,
    min_var: float,
) -> dict[str, np.ndarray]:
    x = np.asarray(batch.x)
    y = np.asarray(batch.y)
    batch_size, time_steps = x.shape
    offsets = np.asarray(_component_offsets(num_components, component_mean_init_span))
    weights = np.full((batch_size, num_components), 1.0 / num_components)
    means = np.full((batch_size, num_components), float(state_params.m0)) + offsets[None, :]
    vars_ = np.full((batch_size, num_components), float(state_params.p0))
    nodes, gh_weights = np.polynomial.hermite.hermgauss(num_points)
    log_gh_weights = np.log(gh_weights) - 0.5 * np.log(np.pi)

    weights_hist = np.zeros((batch_size, time_steps, num_components))
    means_hist = np.zeros((batch_size, time_steps, num_components))
    vars_hist = np.zeros((batch_size, time_steps, num_components))
    filter_mean = np.zeros((batch_size, time_steps))
    filter_var = np.zeros((batch_size, time_steps))
    predictive_y_log_prob = np.zeros((batch_size, time_steps))

    for t in range(time_steps):
        pred_weights = weights
        pred_means = means
        pred_vars = vars_ + float(state_params.q)
        z_support = pred_means[..., None] + np.sqrt(2.0 * pred_vars[..., None]) * nodes
        obs_mean = x[:, t, None, None] * np.sin(z_support)
        base_log_mass = (
            np.log(np.clip(pred_weights, min_var, None))[..., None] + log_gh_weights
        )
        log_likelihood = _np_normal_log_prob(y[:, t, None, None], obs_mean, float(state_params.r))
        predictive_y_log_prob[:, t] = _np_logsumexp(
            (base_log_mass + log_likelihood).reshape(batch_size, -1),
            axis=1,
        )
        target_log_mass = base_log_mass + likelihood_power * log_likelihood
        target_log_mass = target_log_mass.reshape(batch_size, -1)
        target_log_mass = target_log_mass - _np_logsumexp(target_log_mass, axis=1)[:, None]
        target_weights = np.exp(target_log_mass)
        target_z = z_support.reshape(batch_size, -1)

        proposal = _hybrid_proposal_step(
            params,
            cell_type,
            weights,
            means,
            vars_,
            x[:, t],
            y[:, t],
            state_params,
            num_components=num_components,
            min_var=min_var,
        )
        weights, means, vars_ = _project_mixture_em(
            target_z,
            target_weights,
            components=num_components,
            init_span=component_mean_init_span,
            em_steps=refinement_steps,
            min_var=min_var,
            init_weights=proposal[0],
            init_mean=proposal[1],
            init_var=proposal[2],
        )

        weights_hist[:, t] = weights
        means_hist[:, t] = means
        vars_hist[:, t] = vars_
        filter_mean[:, t], filter_var[:, t] = _np_mixture_mean_var(weights, means, vars_)

    return {
        "weights": weights_hist,
        "component_mean": means_hist,
        "component_var": vars_hist,
        "filter_mean": filter_mean,
        "filter_var": filter_var,
        "predictive_y_log_prob": predictive_y_log_prob,
    }


def _hybrid_proposal_step(
    params: dict[str, jax.Array],
    cell_type: str,
    prev_weights: np.ndarray,
    prev_mean: np.ndarray,
    prev_var: np.ndarray,
    x_t: np.ndarray,
    y_t: np.ndarray,
    state_params: LinearGaussianParams,
    *,
    num_components: int,
    min_var: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev_weights_j = jnp.asarray(prev_weights)
    prev_mean_j = jnp.asarray(prev_mean)
    prev_var_j = jnp.asarray(prev_var)
    x_t_j = jnp.asarray(x_t)
    y_t_j = jnp.asarray(y_t)
    if cell_type == "component_mixture":
        outputs = component_mixture_mlp_step(
            params,
            prev_weights_j,
            prev_mean_j,
            prev_var_j,
            x_t_j,
            y_t_j,
            state_params,
            min_var=min_var,
        )
    else:
        outputs = direct_mixture_mlp_step(
            params,
            prev_weights_j,
            prev_mean_j,
            prev_var_j,
            x_t_j,
            y_t_j,
            state_params,
            num_components=num_components,
            min_var=min_var,
        )
    return (
        np.asarray(outputs.filter_weights),
        np.asarray(outputs.component_mean),
        np.asarray(outputs.component_var),
    )


def _run_predictive_carry_head(
    params: dict[str, jax.Array],
    outputs,
    x: jax.Array,
    state_params: LinearGaussianParams,
    *,
    num_components: int,
    component_mean_init_span: float,
    min_var: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    prev_weights, prev_mean, prev_var = _previous_mixture_filter_beliefs(
        outputs,
        state_params,
        num_components=num_components,
        component_mean_init_span=component_mean_init_span,
    )
    q = jnp.asarray(state_params.q, dtype=x.dtype)
    r = jnp.asarray(state_params.r, dtype=x.dtype)
    features = jnp.stack(
        (
            jnp.log(jnp.clip(prev_weights, min_var)),
            prev_mean,
            jnp.log(prev_var),
            jnp.broadcast_to(x[..., None], prev_mean.shape),
            jnp.broadcast_to(jnp.log(q), prev_mean.shape),
            jnp.broadcast_to(jnp.log(r), prev_mean.shape),
        ),
        axis=-1,
    )
    hidden = jnp.tanh(features @ params["w1"] + params["b1"])
    raw = hidden @ params["w2"] + params["b2"]
    predictive_weights = jax.nn.softmax(jnp.log(jnp.clip(prev_weights, min_var)) + raw[..., 0])
    predictive_mean = prev_mean + raw[..., 1]
    predictive_var = (prev_var + q) * jnp.exp(jnp.clip(raw[..., 2], -5.0, 5.0)) + min_var
    return predictive_weights, predictive_mean, predictive_var


def _previous_mixture_filter_beliefs(
    outputs,
    state_params: LinearGaussianParams,
    *,
    num_components: int,
    component_mean_init_span: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_size = outputs.component_mean.shape[0]
    offsets = _component_offsets(num_components, component_mean_init_span)
    initial_weights = jnp.full(
        (batch_size, 1, num_components),
        1.0 / num_components,
        dtype=outputs.filter_weights.dtype,
    )
    initial_mean = (
        jnp.full(
            (batch_size, 1, num_components),
            state_params.m0,
            dtype=outputs.component_mean.dtype,
        )
        + offsets[None, None, :]
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


def _component_offsets(num_components: int, component_mean_init_span: float) -> jax.Array:
    if component_mean_init_span == 0.0:
        return jnp.zeros((num_components,), dtype=DEFAULT_DTYPE)
    return jnp.linspace(
        -0.5 * component_mean_init_span,
        0.5 * component_mean_init_span,
        num_components,
        dtype=DEFAULT_DTYPE,
    )


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


def _mixture_density_projection_loss(
    target_weights: jax.Array,
    target_mean: jax.Array,
    target_var: jax.Array,
    pred_weights: jax.Array,
    pred_mean: jax.Array,
    pred_var: jax.Array,
    density_nodes: jax.Array,
    density_log_weights: jax.Array,
    *,
    min_var: float,
) -> jax.Array:
    safe_target_var = jnp.maximum(target_var, min_var)
    target_z = target_mean[..., None] + jnp.sqrt(2.0 * safe_target_var[..., None]) * density_nodes
    target_log_mass = (
        jnp.log(jnp.clip(target_weights, min_var, 1.0))[..., None] + density_log_weights
    )
    log_q_at_target = _mixture_log_prob(
        target_z,
        jnp.clip(pred_weights, min_var, 1.0)[..., None, None, :],
        pred_mean[..., None, None, :],
        jnp.maximum(pred_var, min_var)[..., None, None, :],
    )
    return -jnp.mean(jnp.sum(jnp.exp(target_log_mass) * log_q_at_target, axis=(-2, -1)))


def _mixture_preassimilation_log_prob_y_from_components(
    weights: jax.Array,
    mean: jax.Array,
    var: jax.Array,
    x: jax.Array,
    y: jax.Array,
    state_params: LinearGaussianParams,
    *,
    num_points: int,
    min_var: float,
) -> jax.Array:
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(num_points)
    nodes = jnp.asarray(nodes_np, dtype=mean.dtype)
    log_quadrature_weights = jnp.log(jnp.asarray(weights_np, dtype=mean.dtype))
    safe_var = jnp.maximum(var, min_var)
    z = mean[..., None] + jnp.sqrt(2.0 * safe_var[..., None]) * nodes
    obs_mean = x[..., None, None] * jnp.sin(z)
    log_likelihood = _normal_log_prob(y[..., None, None], obs_mean, state_params.r)
    component_log_prob = (
        jnp.log(jnp.clip(weights, min_var, 1.0))
        + jax.nn.logsumexp(log_quadrature_weights + log_likelihood, axis=-1)
        - 0.5 * jnp.log(jnp.pi)
    )
    return jax.nn.logsumexp(component_log_prob, axis=-1)


def _flatten_params_for_npz(params: dict[str, Any]) -> dict[str, np.ndarray]:
    flattened = {}
    for group_name, group in params.items():
        if isinstance(group, dict):
            for name, value in group.items():
                flattened[f"{group_name}/{name}"] = np.asarray(value)
        else:
            flattened[group_name] = np.asarray(group)
    return flattened


def _normal_log_prob(
    value: jax.Array,
    mean: jax.Array,
    var: jax.Array,
) -> jax.Array:
    return -0.5 * (jnp.log(2.0 * jnp.pi) + jnp.log(var) + (value - mean) ** 2 / var)


def _render_summary(name: str, metrics: dict[str, Any], history: list[tuple[int, float]]) -> str:
    lines = [f"# {name}", "", "| Metric | Value |", "|---|---:|"]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Loss History", "", "| Step | Loss |", "|---:|---:|"])
    for step, loss in history:
        lines.append(f"| {step} | {loss:.6f} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
