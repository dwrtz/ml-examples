"""Run deterministic quadrature Pareto combinations across nonlinear stress settings."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sweep_quadrature_adf import (
    DEFAULT_CONFIGS,
    _gaussian_interval_coverage,
    _gaussian_nll,
    _mixture_log_prob,
    run_quadrature_adf_filter,
)
from vbf.data import LinearGaussianParams
from vbf.nonlinear import GridReferenceConfig, NonlinearDataConfig
from vbf.nonlinear_cache import load_or_compute_nonlinear_reference


@dataclass(frozen=True)
class FilterSpec:
    key: str
    label: str
    components: int
    likelihood_power: float
    init_span: float
    projection: str = "em"
    alias_spacing: float = 0.0
    initial_weighting: str = "uniform"


STATE_SPECS = {
    "alias_prior_power_ep_k5": FilterSpec(
        key="alias_prior_power_ep_k5",
        label="prior-weighted alias-indexed Power-EP K5 alpha 0.5 spacing 2pi",
        components=5,
        likelihood_power=0.5,
        init_span=12.566370614359172,
        projection="mode_preserving",
        alias_spacing=6.283185307179586,
        initial_weighting="prior_alias",
    ),
    "alias_prior_k5": FilterSpec(
        key="alias_prior_k5",
        label="prior-weighted alias-indexed K5 spacing 2pi",
        components=5,
        likelihood_power=1.0,
        init_span=12.566370614359172,
        projection="mode_preserving",
        alias_spacing=6.283185307179586,
        initial_weighting="prior_alias",
    ),
}
PREDICTIVE_SPECS = {
    "power_ep_k4": FilterSpec(
        key="power_ep_k4",
        label="quadrature Power-EP K4 alpha 0.5 spread 2pi",
        components=4,
        likelihood_power=0.5,
        init_span=6.283185307179586,
    ),
    "adf_k4": FilterSpec(
        key="adf_k4",
        label="quadrature ADF K4 spread 2pi",
        components=4,
        likelihood_power=1.0,
        init_span=6.283185307179586,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--state-model", default="alias_prior_power_ep_k5")
    parser.add_argument("--predictive-model", default="power_ep_k4")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--num-points", type=int, default=64)
    parser.add_argument("--em-steps", type=int, default=30)
    parser.add_argument("--output-dir", default="outputs/nonlinear_quadrature_pareto_suite")
    parser.add_argument("--cache-dir", default="outputs/cache/nonlinear_reference")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    state_spec = _selected_spec(args.state_model, STATE_SPECS, name="--state-model")
    predictive_spec = _selected_spec(
        args.predictive_model,
        PREDICTIVE_SPECS,
        name="--predictive-model",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
            state_outputs = _run_filter(
                state_spec,
                cached,
                state_params,
                num_points=args.num_points,
                em_steps=args.em_steps,
            )
            predictive_outputs = _run_filter(
                predictive_spec,
                cached,
                state_params,
                num_points=args.num_points,
                em_steps=args.em_steps,
            )
            run_name = _run_name(
                str(base_config["name"]),
                state_spec.key,
                predictive_spec.key,
                seed=seed,
                seeds=seeds,
            )
            row = _metrics(
                state_outputs,
                predictive_outputs,
                z=np.asarray(cached.batch.z),
                y=np.asarray(cached.batch.y),
                reference=cached.reference,
                data_config=data_config,
                state_params=state_params,
                state_spec=state_spec,
                predictive_spec=predictive_spec,
                seed=seed,
                num_points=args.num_points,
                em_steps=args.em_steps,
            )
            run_dir = output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics.json").write_text(
                json.dumps(row, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            np.savez(
                run_dir / "diagnostics.npz",
                state_weights=state_outputs.weights,
                state_component_mean=state_outputs.component_mean,
                state_component_var=state_outputs.component_var,
                state_filter_mean=state_outputs.filter_mean,
                state_filter_var=state_outputs.filter_var,
                predictive_weights=predictive_outputs.weights,
                predictive_component_mean=predictive_outputs.component_mean,
                predictive_component_var=predictive_outputs.component_var,
                predictive_y_log_prob=predictive_outputs.predictive_y_log_prob,
            )
            rows.append(
                {
                    "name": run_name,
                    "config": str(config_path),
                    **row,
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


def _run_filter(
    spec: FilterSpec,
    cached: Any,
    state_params: LinearGaussianParams,
    *,
    num_points: int,
    em_steps: int,
):
    return run_quadrature_adf_filter(
        np.asarray(cached.batch.x),
        np.asarray(cached.batch.y),
        state_params,
        components=spec.components,
        likelihood_power=spec.likelihood_power,
        init_span=spec.init_span,
        projection=spec.projection,
        alias_spacing=spec.alias_spacing,
        initial_weighting=spec.initial_weighting,
        num_points=num_points,
        em_steps=em_steps,
    )


def _metrics(
    state_outputs,
    predictive_outputs,
    *,
    z: np.ndarray,
    y: np.ndarray,
    reference: Any,
    data_config: NonlinearDataConfig,
    state_params: LinearGaussianParams,
    state_spec: FilterSpec,
    predictive_spec: FilterSpec,
    seed: int,
    num_points: int,
    em_steps: int,
) -> dict[str, Any]:
    reference_filter_mean = np.asarray(reference.filter_mean)
    reference_filter_var = np.asarray(reference.filter_var)
    reference_predictive_mean = np.asarray(reference.predictive_mean)
    reference_predictive_var = np.asarray(reference.predictive_var)
    state_nll = -_mixture_log_prob(
        z,
        state_outputs.weights,
        state_outputs.component_mean,
        state_outputs.component_var,
    )
    predictive_state_nll = -_mixture_log_prob(
        z,
        predictive_outputs.weights,
        predictive_outputs.component_mean,
        predictive_outputs.component_var,
    )
    return {
        "benchmark": "nonlinear",
        "objective": "quadrature_pareto",
        "seed": seed,
        "observation": data_config.observation,
        "x_pattern": data_config.x_pattern,
        "time_steps": data_config.time_steps,
        "batch_size": data_config.batch_size,
        "state_model_key": state_spec.key,
        "state_model": state_spec.label,
        "predictive_model_key": predictive_spec.key,
        "predictive_model": predictive_spec.label,
        "state_components": state_spec.components,
        "predictive_components": predictive_spec.components,
        "state_likelihood_power": state_spec.likelihood_power,
        "predictive_likelihood_power": predictive_spec.likelihood_power,
        "num_points": num_points,
        "em_steps": em_steps,
        "q": float(state_params.q),
        "r": float(state_params.r),
        "m0": float(state_params.m0),
        "p0": float(state_params.p0),
        "pareto_state_nll": float(np.mean(state_nll)),
        "pareto_predictive_y_nll": float(np.mean(-predictive_outputs.predictive_y_log_prob)),
        "pareto_coverage_90": float(
            _gaussian_interval_coverage(z, state_outputs.filter_mean, state_outputs.filter_var)
        ),
        "pareto_variance_ratio": float(
            np.mean(state_outputs.filter_var) / np.mean(reference_filter_var)
        ),
        "state_leg_predictive_y_nll": float(np.mean(-state_outputs.predictive_y_log_prob)),
        "predictive_leg_state_nll": float(np.mean(predictive_state_nll)),
        "reference_state_nll": float(
            np.mean(_gaussian_nll(z, reference_filter_mean, reference_filter_var))
        ),
        "reference_predictive_nll": float(
            np.mean(_gaussian_nll(y, reference_predictive_mean, reference_predictive_var))
        ),
        "reference_coverage_90": float(
            _gaussian_interval_coverage(z, reference_filter_mean, reference_filter_var)
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "config",
        "benchmark",
        "objective",
        "seed",
        "observation",
        "x_pattern",
        "time_steps",
        "batch_size",
        "state_model_key",
        "state_model",
        "predictive_model_key",
        "predictive_model",
        "state_components",
        "predictive_components",
        "state_likelihood_power",
        "predictive_likelihood_power",
        "num_points",
        "em_steps",
        "q",
        "r",
        "m0",
        "p0",
        "pareto_state_nll",
        "pareto_predictive_y_nll",
        "pareto_coverage_90",
        "pareto_variance_ratio",
        "state_leg_predictive_y_nll",
        "predictive_leg_state_nll",
        "reference_state_nll",
        "reference_predictive_nll",
        "reference_coverage_90",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Nonlinear Quadrature Pareto Suite",
        "",
        "| x pattern | state leg | predictive leg | pareto state NLL | pareto pred-y NLL | cov 90 | var ratio | state-leg pred-y | pred-leg state NLL |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {x_pattern} | {state_model} | {predictive_model} | "
            "{pareto_state_nll:.6f} | {pareto_predictive_y_nll:.6f} | "
            "{pareto_coverage_90:.6f} | {pareto_variance_ratio:.6f} | "
            "{state_leg_predictive_y_nll:.6f} | {predictive_leg_state_nll:.6f} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def _selected_spec(
    key: str,
    specs: dict[str, FilterSpec],
    *,
    name: str,
) -> FilterSpec:
    if key not in specs:
        raise ValueError(f"Unknown {name} value {key!r}; options: {sorted(specs)}")
    return specs[key]


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


def _run_name(
    config_name: str,
    state_key: str,
    predictive_key: str,
    *,
    seed: int,
    seeds: list[int] | None,
) -> str:
    suffix = f"{state_key}_state_{predictive_key}_predictive"
    if seeds is None:
        return f"{config_name}_{suffix}"
    return f"{config_name}_seed_{seed}_{suffix}"


if __name__ == "__main__":
    main()
