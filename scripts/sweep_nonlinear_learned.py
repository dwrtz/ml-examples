"""Run and aggregate learned nonlinear filter stress evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


METRICS = (
    "state_rmse",
    "reference_state_rmse",
    "state_nll",
    "reference_state_nll",
    "predictive_nll",
    "reference_predictive_nll",
    "coverage_90",
    "reference_coverage_90",
    "variance_ratio",
)

DEFAULT_CONFIGS = (
    "experiments/nonlinear/01_sine_observation.yaml",
    "experiments/nonlinear/03_weak_sine_observation.yaml",
    "experiments/nonlinear/04_intermittent_sine_observation.yaml",
    "experiments/nonlinear/05_zero_sine_observation.yaml",
    "experiments/nonlinear/06_random_normal_sine_observation.yaml",
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    objective: str
    reference_variance_ratio_weight: float
    reference_time_variance_ratio_weight: float = 0.0
    reference_log_variance_weight: float = 0.0
    reference_low_observation_variance_ratio_weight: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--models", default="direct_elbo")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--num-elbo-samples", type=int, default=16)
    parser.add_argument("--reference-variance-ratio-weight", type=float, default=1.0)
    parser.add_argument("--reference-time-variance-ratio-weight", type=float, default=1.0)
    parser.add_argument("--reference-log-variance-weight", type=float, default=1.0)
    parser.add_argument("--reference-low-observation-variance-ratio-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/nonlinear_learned_suite")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = _parse_config_paths(args.configs)
    model_specs = _selected_model_specs(
        args.models,
        reference_variance_ratio_weight=args.reference_variance_ratio_weight,
        reference_time_variance_ratio_weight=args.reference_time_variance_ratio_weight,
        reference_log_variance_weight=args.reference_log_variance_weight,
        reference_low_observation_variance_ratio_weight=(
            args.reference_low_observation_variance_ratio_weight
        ),
    )
    base_train_config = _read_config(Path("experiments/nonlinear/08_direct_elbo_sine_mlp.yaml"))

    rows = []
    for config_path in config_paths:
        reference_config = _read_config(config_path)
        for spec in model_specs:
            run_name = f"{reference_config['name']}_{spec.key}"
            run_dir = output_dir / run_name
            run_config_path = output_dir / "configs" / f"{run_name}.yaml"
            config = _make_train_config(
                base_train_config,
                reference_config,
                spec=spec,
                steps=args.steps,
                num_elbo_samples=args.num_elbo_samples,
                output_dir=run_dir,
            )
            _write_config(run_config_path, config)
            if not args.skip_run:
                _run_training(run_config_path)
            rows.append(_load_row(run_dir, config_path=run_config_path, config=config, spec=spec))

    _write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(_render_report(rows), encoding="utf-8")
    print(f"Wrote {report_path}")


def _parse_config_paths(value: str) -> list[Path]:
    paths = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not paths:
        raise ValueError("--configs must include at least one config path")
    return paths


def _selected_model_specs(
    value: str,
    *,
    reference_variance_ratio_weight: float,
    reference_time_variance_ratio_weight: float,
    reference_log_variance_weight: float,
    reference_low_observation_variance_ratio_weight: float,
) -> list[ModelSpec]:
    all_specs = {
        "direct_elbo": ModelSpec(
            key="direct_elbo",
            label="direct nonlinear MC ELBO",
            objective="direct_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
        ),
        "structured_elbo": ModelSpec(
            key="structured_elbo",
            label="EKF-residualized nonlinear MC ELBO",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
        ),
        "direct_elbo_ref_calibrated": ModelSpec(
            key="direct_elbo_ref_calibrated",
            label="direct nonlinear MC ELBO + reference variance calibration",
            objective="direct_elbo_sine_mlp",
            reference_variance_ratio_weight=reference_variance_ratio_weight,
        ),
        "structured_elbo_ref_calibrated": ModelSpec(
            key="structured_elbo_ref_calibrated",
            label="EKF-residualized nonlinear MC ELBO + reference variance calibration",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=reference_variance_ratio_weight,
        ),
        "structured_elbo_ref_time_calibrated": ModelSpec(
            key="structured_elbo_ref_time_calibrated",
            label="EKF-residualized nonlinear MC ELBO + reference time variance calibration",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            reference_time_variance_ratio_weight=reference_time_variance_ratio_weight,
        ),
        "structured_elbo_ref_logvar_calibrated": ModelSpec(
            key="structured_elbo_ref_logvar_calibrated",
            label="EKF-residualized nonlinear MC ELBO + reference log-variance calibration",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            reference_log_variance_weight=reference_log_variance_weight,
        ),
        "structured_elbo_ref_low_obs_calibrated": ModelSpec(
            key="structured_elbo_ref_low_obs_calibrated",
            label="EKF-residualized nonlinear MC ELBO + reference low-observation calibration",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            reference_low_observation_variance_ratio_weight=(
                reference_low_observation_variance_ratio_weight
            ),
        ),
    }
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(keys) - set(all_specs))
    if unknown:
        raise ValueError(f"Unknown nonlinear learned model keys: {unknown}")
    return [all_specs[key] for key in keys]


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _make_train_config(
    base_train_config: dict[str, Any],
    reference_config: dict[str, Any],
    *,
    spec: ModelSpec,
    steps: int,
    num_elbo_samples: int,
    output_dir: Path,
) -> dict[str, Any]:
    config = {
        **base_train_config,
        "name": f"{reference_config['name']}_{spec.key}",
        "model": spec.objective,
        "output_dir": str(output_dir),
        "data": dict(reference_config["data"]),
        "state_space": dict(reference_config["state_space"]),
        "reference": dict(reference_config.get("reference", {})),
    }
    config["evaluation"] = {
        **base_train_config.get("evaluation", {}),
        "data": {"batch_size": reference_config["data"]["batch_size"]},
    }
    config["training"] = {
        **base_train_config["training"],
        "steps": steps,
        "num_elbo_samples": num_elbo_samples,
        "reference_variance_ratio_weight": spec.reference_variance_ratio_weight,
        "reference_time_variance_ratio_weight": spec.reference_time_variance_ratio_weight,
        "reference_log_variance_weight": spec.reference_log_variance_weight,
        "reference_low_observation_variance_ratio_weight": (
            spec.reference_low_observation_variance_ratio_weight
        ),
    }
    return config


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run_training(config_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/train_nonlinear.py", "--config", str(config_path)],
        check=True,
    )


def _load_row(
    run_dir: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    spec: ModelSpec,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing nonlinear learned metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "name": config["name"],
        "config": str(config_path),
        "model": spec.label,
        "objective": metrics["objective"],
        "x_pattern": metrics["x_pattern"],
        "time_steps": config["data"]["time_steps"],
        "steps": metrics["training_steps"],
        "num_elbo_samples": metrics["num_elbo_samples"],
        "reference_variance_ratio_weight": metrics["reference_variance_ratio_weight"],
        "reference_time_variance_ratio_weight": metrics["reference_time_variance_ratio_weight"],
        "reference_log_variance_weight": metrics["reference_log_variance_weight"],
        "reference_low_observation_variance_ratio_weight": metrics[
            "reference_low_observation_variance_ratio_weight"
        ],
        **{metric: metrics[metric] for metric in METRICS},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "config",
        "model",
        "objective",
        "x_pattern",
        "time_steps",
        "steps",
        "num_elbo_samples",
        "reference_variance_ratio_weight",
        "reference_time_variance_ratio_weight",
        "reference_log_variance_weight",
        "reference_low_observation_variance_ratio_weight",
        *METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Nonlinear Learned Filter Suite",
        "",
        "| x pattern | Model | Steps | state NLL | ref state NLL | cov 90 | ref cov 90 | var ratio | pred NLL | ref pred NLL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {x_pattern} | {model} | {steps} | {state_nll:.6f} | "
            "{reference_state_nll:.6f} | {coverage_90:.6f} | "
            "{reference_coverage_90:.6f} | {variance_ratio:.6f} | "
            "{predictive_nll:.6f} | {reference_predictive_nll:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Rows with nonzero reference calibration weights use grid-reference filtering",
            "variances as calibration targets and should be reported as reference-calibrated",
            "diagnostics, not fully unsupervised ELBO baselines.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
