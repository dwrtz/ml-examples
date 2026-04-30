"""Run deterministic quadrature ADF / Power-EP nonlinear baselines."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vbf.data import LinearGaussianParams
from vbf.nonlinear import GridReferenceConfig, NonlinearDataConfig
from vbf.nonlinear_cache import load_or_compute_nonlinear_reference


LOG_2PI = np.log(2.0 * np.pi)
DEFAULT_CONFIGS = (
    "experiments/nonlinear/01_sine_observation.yaml",
    "experiments/nonlinear/03_weak_sine_observation.yaml",
    "experiments/nonlinear/04_intermittent_sine_observation.yaml",
    "experiments/nonlinear/05_zero_sine_observation.yaml",
    "experiments/nonlinear/06_random_normal_sine_observation.yaml",
)
DEFAULT_MODELS = (
    "quadrature_adf_gaussian",
    "quadrature_adf_k2",
    "quadrature_adf_k4_2pi",
    "quadrature_power_ep_k4_alpha_0p5",
    "quadrature_alias_k5_2pi",
    "quadrature_alias_power_ep_k5_alpha_0p5",
    "quadrature_alias_prior_k5_2pi",
    "quadrature_alias_prior_power_ep_k5_alpha_0p5",
    "quadrature_alias_prior_power_ep_k5_top3_alpha_0p5",
    "quadrature_alias_prior_power_ep_k5_top2_alpha_0p5",
    "quadrature_alias_prior_power_ep_k5_shrink_0p85_alpha_0p5",
    "quadrature_alias_prior_power_ep_k5_shrink_0p70_alpha_0p5",
)
METRICS = (
    "state_rmse",
    "reference_state_rmse",
    "state_nll",
    "reference_state_nll",
    "predictive_nll",
    "predictive_y_nll",
    "reference_predictive_nll",
    "coverage_90",
    "reference_coverage_90",
    "variance_ratio",
)


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    label: str
    components: int
    likelihood_power: float = 1.0
    alpha: float | None = None
    init_span: float = 0.0
    projection: str = "em"
    alias_spacing: float = 0.0
    initial_weighting: str = "uniform"
    max_active_aliases: int = 0
    alias_mean_shrink: float = 1.0


@dataclass(frozen=True)
class QuadratureAdfOutputs:
    predictive_weights: np.ndarray
    predictive_component_mean: np.ndarray
    predictive_component_var: np.ndarray
    weights: np.ndarray
    component_mean: np.ndarray
    component_var: np.ndarray
    filter_mean: np.ndarray
    filter_var: np.ndarray
    predictive_mean: np.ndarray
    predictive_var: np.ndarray
    predictive_y_log_prob: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--em-steps", type=int, default=30)
    parser.add_argument("--output-dir", default="outputs/nonlinear_quadrature_adf_suite")
    parser.add_argument("--cache-dir", default="outputs/cache/nonlinear_reference")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _selected_specs(args.models)
    config_paths = _parse_config_paths(args.configs)
    seeds = None if args.seeds is None else _parse_ints(args.seeds, name="--seeds")

    rows = []
    for config_path in config_paths:
        base_config = _read_config(config_path)
        for seed in seeds or [int(base_config["seed"])]:
            data_config = NonlinearDataConfig(**base_config["data"])
            state_params = LinearGaussianParams(**base_config["state_space"])
            reference_config = GridReferenceConfig(**base_config.get("reference", {}))
            cached = load_or_compute_nonlinear_reference(
                data_config,
                state_params,
                seed=seed,
                grid_config=reference_config,
                cache_dir=Path(args.cache_dir),
                use_cache=not args.no_cache,
            )
            for spec in specs:
                outputs = run_quadrature_adf_filter(
                    np.asarray(cached.batch.x),
                    np.asarray(cached.batch.y),
                    state_params,
                    components=spec.components,
                    likelihood_power=spec.likelihood_power,
                    init_span=spec.init_span,
                    projection=spec.projection,
                    alias_spacing=spec.alias_spacing,
                    initial_weighting=spec.initial_weighting,
                    max_active_aliases=spec.max_active_aliases,
                    alias_mean_shrink=spec.alias_mean_shrink,
                    num_points=args.num_points,
                    em_steps=args.em_steps,
                )
                run_name = _run_name(str(base_config["name"]), spec.key, seed=seed, seeds=seeds)
                run_dir = output_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                metrics = _metrics(
                    outputs,
                    z=np.asarray(cached.batch.z),
                    y=np.asarray(cached.batch.y),
                    reference=cached.reference,
                    data_config=data_config,
                    state_params=state_params,
                    spec=spec,
                    seed=seed,
                    num_points=args.num_points,
                    em_steps=args.em_steps,
                    reference_cache_hit=cached.cache_hit,
                    reference_cache_path=str(cached.cache_path),
                )
                (run_dir / "metrics.json").write_text(
                    json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                np.savez(
                    run_dir / "diagnostics.npz",
                    predictive_weights=outputs.predictive_weights,
                    predictive_component_mean=outputs.predictive_component_mean,
                    predictive_component_var=outputs.predictive_component_var,
                    weights=outputs.weights,
                    component_mean=outputs.component_mean,
                    component_var=outputs.component_var,
                    filter_mean=outputs.filter_mean,
                    filter_var=outputs.filter_var,
                    predictive_mean=outputs.predictive_mean,
                    predictive_var=outputs.predictive_var,
                    predictive_y_log_prob=outputs.predictive_y_log_prob,
                )
                (run_dir / "evaluation_summary.md").write_text(
                    _render_run_summary(run_name, metrics),
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "name": run_name,
                        "config": str(config_path),
                        "model": spec.label,
                        **{key: metrics[key] for key in _csv_metric_keys()},
                    }
                )

    _write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(_render_report(rows), encoding="utf-8")
    print(f"Wrote {report_path}")


def run_quadrature_adf_filter(
    x: np.ndarray,
    y: np.ndarray,
    params: LinearGaussianParams,
    *,
    components: int,
    likelihood_power: float,
    init_span: float,
    projection: str = "em",
    alias_spacing: float = 0.0,
    initial_weighting: str = "uniform",
    max_active_aliases: int = 0,
    alias_mean_shrink: float = 1.0,
    num_points: int,
    em_steps: int,
    min_var: float = 1e-6,
) -> QuadratureAdfOutputs:
    """Run deterministic local Bayes projection using only model and observations."""

    if components <= 0:
        raise ValueError("components must be positive")
    if likelihood_power <= 0.0:
        raise ValueError("likelihood_power must be positive")
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if em_steps <= 0:
        raise ValueError("em_steps must be positive")
    if projection not in {"em", "mode_preserving"}:
        raise ValueError("projection must be one of: em, mode_preserving")
    if projection == "mode_preserving" and components < 2:
        raise ValueError("mode_preserving projection requires components > 1")
    if initial_weighting not in {"uniform", "prior_alias"}:
        raise ValueError("initial_weighting must be one of: uniform, prior_alias")
    if max_active_aliases < 0:
        raise ValueError("max_active_aliases must be nonnegative")
    if max_active_aliases > components:
        raise ValueError("max_active_aliases cannot exceed components")
    if not 0.0 < alias_mean_shrink <= 1.0:
        raise ValueError("alias_mean_shrink must be in (0, 1]")

    batch_size, time_steps = x.shape
    dtype = np.float64
    offsets = _initial_component_offsets(
        components,
        init_span=init_span,
        projection=projection,
        alias_spacing=alias_spacing,
        dtype=dtype,
    )
    means = np.full((batch_size, components), float(params.m0), dtype=dtype) + offsets
    vars_ = np.full((batch_size, components), float(params.p0), dtype=dtype)
    weights = _initial_component_weights(
        offsets,
        params=params,
        batch_size=batch_size,
        initial_weighting=initial_weighting,
        min_var=min_var,
    )

    nodes, gh_weights = np.polynomial.hermite.hermgauss(num_points)
    log_gh_weights = np.log(gh_weights) - 0.5 * np.log(np.pi)

    weights_hist = np.zeros((batch_size, time_steps, components), dtype=dtype)
    means_hist = np.zeros((batch_size, time_steps, components), dtype=dtype)
    vars_hist = np.zeros((batch_size, time_steps, components), dtype=dtype)
    pred_weights_hist = np.zeros((batch_size, time_steps, components), dtype=dtype)
    pred_means_hist = np.zeros((batch_size, time_steps, components), dtype=dtype)
    pred_vars_hist = np.zeros((batch_size, time_steps, components), dtype=dtype)
    filter_mean = np.zeros((batch_size, time_steps), dtype=dtype)
    filter_var = np.zeros((batch_size, time_steps), dtype=dtype)
    predictive_mean = np.zeros((batch_size, time_steps), dtype=dtype)
    predictive_var = np.zeros((batch_size, time_steps), dtype=dtype)
    predictive_y_log_prob = np.zeros((batch_size, time_steps), dtype=dtype)

    for t in range(time_steps):
        pred_weights = weights
        pred_means = means
        pred_vars = vars_ + float(params.q)
        pred_weights_hist[:, t] = pred_weights
        pred_means_hist[:, t] = pred_means
        pred_vars_hist[:, t] = pred_vars
        z_support = pred_means[..., None] + np.sqrt(2.0 * pred_vars[..., None]) * nodes
        obs_mean = x[:, t, None, None] * np.sin(z_support)
        base_log_mass = np.log(np.clip(pred_weights, min_var, None))[..., None] + log_gh_weights
        log_likelihood = _normal_log_prob(y[:, t, None, None], obs_mean, float(params.r))
        predictive_y_log_prob[:, t] = _logsumexp(
            (base_log_mass + log_likelihood).reshape(batch_size, -1),
            axis=1,
        )
        base_mass = np.exp(base_log_mass)
        predictive_mean[:, t] = np.sum(base_mass * obs_mean, axis=(1, 2))
        predictive_var[:, t] = np.sum(
            base_mass * ((obs_mean - predictive_mean[:, t, None, None]) ** 2 + float(params.r)),
            axis=(1, 2),
        )

        target_log_mass = base_log_mass + likelihood_power * log_likelihood
        target_log_mass = target_log_mass.reshape(batch_size, -1)
        target_log_mass = target_log_mass - _logsumexp(target_log_mass, axis=1)[:, None]
        target_weights = np.exp(target_log_mass)
        target_z = z_support.reshape(batch_size, -1)

        if components == 1:
            weights, means, vars_ = _project_gaussian(target_z, target_weights, min_var=min_var)
        elif projection == "mode_preserving":
            weights, means, vars_ = _project_mode_preserving(
                z_support,
                target_log_mass.reshape(batch_size, components, num_points),
                max_active_aliases=max_active_aliases,
                alias_mean_shrink=alias_mean_shrink,
                min_var=min_var,
            )
        else:
            weights, means, vars_ = _project_mixture_em(
                target_z,
                target_weights,
                components=components,
                init_span=init_span,
                em_steps=em_steps,
                min_var=min_var,
            )

        weights_hist[:, t] = weights
        means_hist[:, t] = means
        vars_hist[:, t] = vars_
        filter_mean[:, t], filter_var[:, t] = _mixture_mean_var(weights, means, vars_)

    return QuadratureAdfOutputs(
        predictive_weights=pred_weights_hist,
        predictive_component_mean=pred_means_hist,
        predictive_component_var=pred_vars_hist,
        weights=weights_hist,
        component_mean=means_hist,
        component_var=vars_hist,
        filter_mean=filter_mean,
        filter_var=filter_var,
        predictive_mean=predictive_mean,
        predictive_var=predictive_var,
        predictive_y_log_prob=predictive_y_log_prob,
    )


def _initial_component_offsets(
    components: int,
    *,
    init_span: float,
    projection: str,
    alias_spacing: float,
    dtype: type[np.float64],
) -> np.ndarray:
    if projection == "mode_preserving":
        spacing = alias_spacing if alias_spacing != 0.0 else init_span
        if spacing <= 0.0:
            raise ValueError("mode_preserving projection requires positive alias_spacing or init_span")
        return (np.arange(components, dtype=dtype) - 0.5 * (components - 1)) * spacing
    if components > 1:
        return np.linspace(-0.5 * init_span, 0.5 * init_span, components, dtype=dtype)
    return np.zeros((1,), dtype=dtype)


def _initial_component_weights(
    offsets: np.ndarray,
    *,
    params: LinearGaussianParams,
    batch_size: int,
    initial_weighting: str,
    min_var: float,
) -> np.ndarray:
    if initial_weighting == "uniform":
        weights = np.full((offsets.shape[0],), 1.0 / offsets.shape[0], dtype=offsets.dtype)
    else:
        log_weights = _normal_log_prob(offsets, 0.0, float(params.p0))
        log_weights = log_weights - _logsumexp(log_weights, axis=0)
        weights = np.exp(log_weights)
    weights = np.maximum(weights, min_var)
    weights = weights / np.sum(weights)
    return np.broadcast_to(weights[None, :], (batch_size, offsets.shape[0])).copy()


def _project_mode_preserving(
    z_support: np.ndarray,
    target_log_mass: np.ndarray,
    *,
    max_active_aliases: int = 0,
    alias_mean_shrink: float = 1.0,
    min_var: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project each alias component independently without EM component relabeling."""

    component_log_mass = _logsumexp(target_log_mass, axis=2)
    weights = np.exp(component_log_mass)
    weights = _prune_alias_weights(weights, max_active_aliases=max_active_aliases)
    weights = np.maximum(weights, min_var)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    node_weights = np.exp(target_log_mass - component_log_mass[..., None])
    means = np.sum(node_weights * z_support, axis=2)
    vars_ = np.sum(node_weights * (z_support - means[..., None]) ** 2, axis=2)
    if alias_mean_shrink != 1.0:
        mixture_mean = np.sum(weights * means, axis=1, keepdims=True)
        means = mixture_mean + alias_mean_shrink * (means - mixture_mean)
    vars_ = np.maximum(vars_, min_var)
    return weights, means, vars_


def _prune_alias_weights(weights: np.ndarray, *, max_active_aliases: int) -> np.ndarray:
    if max_active_aliases == 0 or max_active_aliases == weights.shape[1]:
        return weights
    threshold_indices = np.argpartition(weights, -max_active_aliases, axis=1)[
        :, -max_active_aliases:
    ]
    mask = np.zeros_like(weights, dtype=bool)
    np.put_along_axis(mask, threshold_indices, True, axis=1)
    return np.where(mask, weights, 0.0)


def _project_gaussian(
    z: np.ndarray,
    weights: np.ndarray,
    *,
    min_var: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.sum(weights * z, axis=1, keepdims=True)
    var = np.maximum(np.sum(weights * (z - mean) ** 2, axis=1, keepdims=True), min_var)
    return np.ones_like(mean), mean, var


def _project_mixture_em(
    z: np.ndarray,
    target_weights: np.ndarray,
    *,
    components: int,
    init_span: float,
    em_steps: int,
    min_var: float,
    init_weights: np.ndarray | None = None,
    init_mean: np.ndarray | None = None,
    init_var: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_size = z.shape[0]
    target_mean = np.sum(target_weights * z, axis=1, keepdims=True)
    target_var = np.maximum(np.sum(target_weights * (z - target_mean) ** 2, axis=1), min_var)
    if init_weights is None or init_mean is None or init_var is None:
        offsets = np.linspace(-0.5 * init_span, 0.5 * init_span, components, dtype=z.dtype)
        if init_span == 0.0:
            quantiles = (np.arange(components, dtype=z.dtype) + 0.5) / components
            offsets = np.sqrt(target_var.mean()) * (2.0 * quantiles - 1.0)
        means = target_mean + offsets[None, :]
        vars_ = np.broadcast_to(target_var[:, None], (batch_size, components)).copy()
        weights = np.full((batch_size, components), 1.0 / components, dtype=z.dtype)
    else:
        weights = np.asarray(init_weights, dtype=z.dtype)
        weights = np.maximum(weights, min_var)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        means = np.asarray(init_mean, dtype=z.dtype)
        vars_ = np.maximum(np.asarray(init_var, dtype=z.dtype), min_var)

    for _ in range(em_steps):
        log_resp = (
            np.log(np.clip(weights[:, None, :], min_var, None))
            + _normal_log_prob(z[..., None], means[:, None, :], vars_[:, None, :])
        )
        log_resp = log_resp - _logsumexp(log_resp, axis=2)[..., None]
        resp = np.exp(log_resp)
        effective = np.sum(target_weights[..., None] * resp, axis=1)
        effective = np.maximum(effective, min_var)
        weights = effective / np.sum(effective, axis=1, keepdims=True)
        means = np.sum(target_weights[..., None] * resp * z[..., None], axis=1) / effective
        vars_ = (
            np.sum(target_weights[..., None] * resp * (z[..., None] - means[:, None, :]) ** 2, axis=1)
            / effective
        )
        vars_ = np.maximum(vars_, min_var)

    return weights, means, vars_


def _metrics(
    outputs: QuadratureAdfOutputs,
    *,
    z: np.ndarray,
    y: np.ndarray,
    reference: Any,
    data_config: NonlinearDataConfig,
    state_params: LinearGaussianParams,
    spec: BaselineSpec,
    seed: int,
    num_points: int,
    em_steps: int,
    reference_cache_hit: bool,
    reference_cache_path: str,
) -> dict[str, Any]:
    reference_filter_mean = np.asarray(reference.filter_mean)
    reference_filter_var = np.asarray(reference.filter_var)
    reference_predictive_mean = np.asarray(reference.predictive_mean)
    reference_predictive_var = np.asarray(reference.predictive_var)
    state_nll = -_mixture_log_prob(
        z,
        outputs.weights,
        outputs.component_mean,
        outputs.component_var,
    )
    reference_state_nll = _gaussian_nll(z, reference_filter_mean, reference_filter_var)
    predictive_nll = _gaussian_nll(y, outputs.predictive_mean, outputs.predictive_var)
    reference_predictive_nll = _gaussian_nll(y, reference_predictive_mean, reference_predictive_var)
    return {
        "benchmark": "nonlinear",
        "objective": "quadrature_adf",
        "seed": seed,
        "observation": data_config.observation,
        "x_pattern": data_config.x_pattern,
        "time_steps": data_config.time_steps,
        "batch_size": data_config.batch_size,
        "model_key": spec.key,
        "model": spec.label,
        "components": spec.components,
        "likelihood_power": spec.likelihood_power,
        "alpha": spec.alpha,
        "init_span": spec.init_span,
        "projection": spec.projection,
        "alias_spacing": spec.alias_spacing,
        "initial_weighting": spec.initial_weighting,
        "max_active_aliases": spec.max_active_aliases,
        "alias_mean_shrink": spec.alias_mean_shrink,
        "num_points": num_points,
        "em_steps": em_steps,
        "q": float(state_params.q),
        "r": float(state_params.r),
        "m0": float(state_params.m0),
        "p0": float(state_params.p0),
        "state_nll_estimator": "mixture_density",
        "coverage_estimator": "moment_gaussian",
        "reference_cache_hit": reference_cache_hit,
        "reference_cache_path": reference_cache_path,
        "state_rmse": float(np.sqrt(np.mean((outputs.filter_mean - z) ** 2))),
        "reference_state_rmse": float(np.sqrt(np.mean((reference_filter_mean - z) ** 2))),
        "state_nll": float(np.mean(state_nll)),
        "reference_state_nll": float(np.mean(reference_state_nll)),
        "predictive_nll": float(np.mean(predictive_nll)),
        "predictive_y_nll": float(np.mean(-outputs.predictive_y_log_prob)),
        "reference_predictive_nll": float(np.mean(reference_predictive_nll)),
        "coverage_90": float(_gaussian_interval_coverage(z, outputs.filter_mean, outputs.filter_var)),
        "reference_coverage_90": float(
            _gaussian_interval_coverage(z, reference_filter_mean, reference_filter_var)
        ),
        "mean_filter_variance": float(np.mean(outputs.filter_var)),
        "reference_mean_filter_variance": float(np.mean(reference_filter_var)),
        "variance_ratio": float(np.mean(outputs.filter_var) / np.mean(reference_filter_var)),
    }


def _selected_specs(value: str) -> list[BaselineSpec]:
    all_specs = {
        "quadrature_adf_gaussian": BaselineSpec(
            key="quadrature_adf_gaussian",
            label="reference-free quadrature ADF Gaussian",
            components=1,
        ),
        "quadrature_adf_k2": BaselineSpec(
            key="quadrature_adf_k2",
            label="reference-free quadrature ADF K2",
            components=2,
        ),
        "quadrature_adf_k4_2pi": BaselineSpec(
            key="quadrature_adf_k4_2pi",
            label="reference-free quadrature ADF K4 spread 2pi",
            components=4,
            init_span=6.283185307179586,
        ),
        "quadrature_power_ep_k4_alpha_0p5": BaselineSpec(
            key="quadrature_power_ep_k4_alpha_0p5",
            label="reference-free quadrature Power-EP K4 alpha 0.5 spread 2pi",
            components=4,
            likelihood_power=0.5,
            alpha=0.5,
            init_span=6.283185307179586,
        ),
        "quadrature_alias_k5_2pi": BaselineSpec(
            key="quadrature_alias_k5_2pi",
            label="reference-free quadrature alias-indexed K5 spacing 2pi",
            components=5,
            init_span=12.566370614359172,
            projection="mode_preserving",
            alias_spacing=6.283185307179586,
        ),
        "quadrature_alias_power_ep_k5_alpha_0p5": BaselineSpec(
            key="quadrature_alias_power_ep_k5_alpha_0p5",
            label="reference-free quadrature alias-indexed Power-EP K5 alpha 0.5 spacing 2pi",
            components=5,
            likelihood_power=0.5,
            alpha=0.5,
            init_span=12.566370614359172,
            projection="mode_preserving",
            alias_spacing=6.283185307179586,
        ),
        "quadrature_alias_prior_k5_2pi": BaselineSpec(
            key="quadrature_alias_prior_k5_2pi",
            label="reference-free quadrature prior-weighted alias-indexed K5 spacing 2pi",
            components=5,
            init_span=12.566370614359172,
            projection="mode_preserving",
            alias_spacing=6.283185307179586,
            initial_weighting="prior_alias",
        ),
        "quadrature_alias_prior_power_ep_k5_alpha_0p5": BaselineSpec(
            key="quadrature_alias_prior_power_ep_k5_alpha_0p5",
            label=(
                "reference-free quadrature prior-weighted alias-indexed "
                "Power-EP K5 alpha 0.5 spacing 2pi"
            ),
            components=5,
            likelihood_power=0.5,
            alpha=0.5,
            init_span=12.566370614359172,
            projection="mode_preserving",
            alias_spacing=6.283185307179586,
            initial_weighting="prior_alias",
        ),
        "quadrature_alias_prior_power_ep_k5_top3_alpha_0p5": BaselineSpec(
            key="quadrature_alias_prior_power_ep_k5_top3_alpha_0p5",
            label=(
                "reference-free quadrature prior-weighted alias-indexed "
                "Power-EP K5 top3 alpha 0.5 spacing 2pi"
            ),
            components=5,
            likelihood_power=0.5,
            alpha=0.5,
            init_span=12.566370614359172,
            projection="mode_preserving",
            alias_spacing=6.283185307179586,
            initial_weighting="prior_alias",
            max_active_aliases=3,
        ),
        "quadrature_alias_prior_power_ep_k5_top2_alpha_0p5": BaselineSpec(
            key="quadrature_alias_prior_power_ep_k5_top2_alpha_0p5",
            label=(
                "reference-free quadrature prior-weighted alias-indexed "
                "Power-EP K5 top2 alpha 0.5 spacing 2pi"
            ),
            components=5,
            likelihood_power=0.5,
            alpha=0.5,
            init_span=12.566370614359172,
            projection="mode_preserving",
            alias_spacing=6.283185307179586,
            initial_weighting="prior_alias",
            max_active_aliases=2,
        ),
        "quadrature_alias_prior_power_ep_k5_shrink_0p85_alpha_0p5": BaselineSpec(
            key="quadrature_alias_prior_power_ep_k5_shrink_0p85_alpha_0p5",
            label=(
                "reference-free quadrature prior-weighted alias-indexed "
                "Power-EP K5 shrink 0.85 alpha 0.5 spacing 2pi"
            ),
            components=5,
            likelihood_power=0.5,
            alpha=0.5,
            init_span=12.566370614359172,
            projection="mode_preserving",
            alias_spacing=6.283185307179586,
            initial_weighting="prior_alias",
            alias_mean_shrink=0.85,
        ),
        "quadrature_alias_prior_power_ep_k5_shrink_0p70_alpha_0p5": BaselineSpec(
            key="quadrature_alias_prior_power_ep_k5_shrink_0p70_alpha_0p5",
            label=(
                "reference-free quadrature prior-weighted alias-indexed "
                "Power-EP K5 shrink 0.70 alpha 0.5 spacing 2pi"
            ),
            components=5,
            likelihood_power=0.5,
            alpha=0.5,
            init_span=12.566370614359172,
            projection="mode_preserving",
            alias_spacing=6.283185307179586,
            initial_weighting="prior_alias",
            alias_mean_shrink=0.70,
        ),
    }
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(keys) - set(all_specs))
    if unknown:
        raise ValueError(f"Unknown quadrature ADF model keys: {unknown}")
    return [all_specs[key] for key in keys]


def _parse_config_paths(value: str) -> list[Path]:
    paths = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not paths:
        raise ValueError("--configs must include at least one config path")
    return paths


def _parse_ints(value: str, *, name: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must include at least one integer")
    return values


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _run_name(config_name: str, spec_key: str, *, seed: int, seeds: list[int] | None) -> str:
    if seeds is None:
        return f"{config_name}_{spec_key}"
    return f"{config_name}_seed_{seed}_{spec_key}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["name", "config", "model", *_csv_metric_keys()]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_metric_keys() -> list[str]:
    return [
        "seed",
        "observation",
        "x_pattern",
        "time_steps",
        "batch_size",
        "components",
        "likelihood_power",
        "alpha",
        "init_span",
        "projection",
        "alias_spacing",
        "initial_weighting",
        "max_active_aliases",
        "alias_mean_shrink",
        "num_points",
        "em_steps",
        "state_nll_estimator",
        "coverage_estimator",
        *METRICS,
    ]


def _render_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Nonlinear Quadrature ADF / Power-EP Suite",
        "",
        "These rows are deterministic reference-free baselines: they use only the known",
        "transition, known observation model, and observed `x,y`, then project the",
        "local tilted distribution back to a strict Gaussian or Gaussian mixture.",
        "",
        "| x pattern | Model | components | power | state NLL | ref state NLL | cov 90 | var ratio | pred-y NLL | ref pred NLL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        alpha_suffix = "" if row["alpha"] is None else f" alpha={row['alpha']:.2f}"
        lines.append(
            "| {x_pattern} | {model}{alpha_suffix} | {components} | {likelihood_power:.2f} | "
            "{state_nll:.6f} | {reference_state_nll:.6f} | {coverage_90:.6f} | "
            "{variance_ratio:.6f} | {predictive_y_nll:.6f} | "
            "{reference_predictive_nll:.6f} |".format(alpha_suffix=alpha_suffix, **row)
        )
    lines.append("")
    return "\n".join(lines)


def _render_run_summary(name: str, metrics: dict[str, Any]) -> str:
    lines = [f"# {name}", "", "| Metric | Value |", "|---|---:|"]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.append("")
    return "\n".join(lines)


def _mixture_mean_var(
    weights: np.ndarray,
    means: np.ndarray,
    vars_: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(weights * means, axis=-1)
    second = np.sum(weights * (vars_ + means**2), axis=-1)
    return mean, np.maximum(second - mean**2, 0.0)


def _mixture_log_prob(
    value: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    vars_: np.ndarray,
) -> np.ndarray:
    return _logsumexp(
        np.log(np.clip(weights, 1e-12, None)) + _normal_log_prob(value[..., None], means, vars_),
        axis=-1,
    )


def _normal_log_prob(value: np.ndarray, mean: np.ndarray | float, var: np.ndarray | float) -> np.ndarray:
    return -0.5 * (LOG_2PI + np.log(var) + (value - mean) ** 2 / var)


def _gaussian_nll(value: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -_normal_log_prob(value, mean, var)


def _gaussian_interval_coverage(value: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    half_width = 1.6448536269514722 * np.sqrt(var)
    return np.mean((value >= mean - half_width) & (value <= mean + half_width))


def _logsumexp(value: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    max_value = np.max(value, axis=axis, keepdims=True)
    result = np.log(np.sum(np.exp(value - max_value), axis=axis, keepdims=True)) + max_value
    return np.squeeze(result, axis=axis)


if __name__ == "__main__":
    main()
