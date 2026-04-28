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
    "predictive_y_nll",
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
    elbo_weight: float = 1.0
    joint_elbo_weight: float = 0.0
    joint_elbo_horizon: int = 1
    joint_elbo_num_samples: int = 16
    joint_elbo_num_windows: int = 8
    predictive_y_weight: float = 0.0
    predictive_y_num_samples: int = 32
    predictive_y_estimator: str = "quadrature"
    mask_y_probability: float = 0.0
    mask_y_span_probability: float = 0.0
    mask_y_span_length: int = 1
    reference_mean_weight: float = 0.0
    reference_rollout_weight: float = 0.0
    reference_rollout_horizon: int = 1
    teacher_forced: bool = False
    resample_batch: bool = False
    reference_time_variance_ratio_weight: float = 0.0
    reference_log_variance_weight: float = 0.0
    reference_low_observation_variance_ratio_weight: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--models", default="direct_elbo")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--num-elbo-samples", type=int, default=16)
    parser.add_argument("--batch-seed-stride", type=int, default=1)
    parser.add_argument("--reference-variance-ratio-weight", type=float, default=1.0)
    parser.add_argument("--reference-time-variance-ratio-weight", type=float, default=1.0)
    parser.add_argument("--reference-log-variance-weight", type=float, default=1.0)
    parser.add_argument(
        "--reference-low-observation-variance-ratio-weight", type=float, default=1.0
    )
    parser.add_argument("--output-dir", default="outputs/nonlinear_learned_suite")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = _parse_config_paths(args.configs)
    seeds = None if args.seeds is None else _parse_ints(args.seeds, name="--seeds")
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
        for seed in seeds or [int(reference_config["seed"])]:
            for spec in model_specs:
                run_name = _run_name(reference_config["name"], spec.key, seed=seed, seeds=seeds)
                run_dir = output_dir / run_name
                run_config_path = output_dir / "configs" / f"{run_name}.yaml"
                config = _make_train_config(
                    base_train_config,
                    reference_config,
                    spec=spec,
                    run_name=run_name,
                    seed=seed,
                    steps=args.steps,
                    num_elbo_samples=args.num_elbo_samples,
                    batch_seed_stride=args.batch_seed_stride,
                    output_dir=run_dir,
                )
                _write_config(run_config_path, config)
                if not args.skip_run:
                    _run_training(run_config_path)
                rows.append(
                    _load_row(
                        run_dir,
                        config_path=run_config_path,
                        config=config,
                        spec=spec,
                    )
                )

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


def _parse_ints(value: str, *, name: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must include at least one integer")
    return values


def _run_name(config_name: str, spec_key: str, *, seed: int, seeds: list[int] | None) -> str:
    if seeds is None:
        return f"{config_name}_{spec_key}"
    return f"{config_name}_seed_{seed}_{spec_key}"


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
        "direct_elbo_predictive_y": ModelSpec(
            key="direct_elbo_predictive_y",
            label="direct nonlinear MC ELBO + predictive-y auxiliary",
            objective="direct_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            predictive_y_weight=1.0,
        ),
        "direct_moment_distilled": ModelSpec(
            key="direct_moment_distilled",
            label="direct nonlinear MLP + reference moment distillation",
            objective="direct_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            reference_mean_weight=1.0,
            reference_log_variance_weight=1.0,
        ),
        "direct_moment_calibrated": ModelSpec(
            key="direct_moment_calibrated",
            label="direct nonlinear MLP + reference moment and variance-ratio calibration",
            objective="direct_elbo_sine_mlp",
            reference_variance_ratio_weight=reference_variance_ratio_weight,
            elbo_weight=0.0,
            reference_mean_weight=1.0,
            reference_log_variance_weight=1.0,
        ),
        "direct_moment_time_calibrated": ModelSpec(
            key="direct_moment_time_calibrated",
            label="direct nonlinear MLP + reference moment and time variance calibration",
            objective="direct_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            reference_mean_weight=1.0,
            reference_log_variance_weight=1.0,
            reference_time_variance_ratio_weight=reference_time_variance_ratio_weight,
        ),
        "direct_moment_low_obs_calibrated": ModelSpec(
            key="direct_moment_low_obs_calibrated",
            label="direct nonlinear MLP + reference moment and low-observation calibration",
            objective="direct_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            reference_mean_weight=1.0,
            reference_log_variance_weight=1.0,
            reference_low_observation_variance_ratio_weight=(
                reference_low_observation_variance_ratio_weight
            ),
        ),
        "direct_moment_teacher_forced": ModelSpec(
            key="direct_moment_teacher_forced",
            label="direct nonlinear MLP + teacher-forced reference moment distillation",
            objective="direct_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            reference_mean_weight=1.0,
            reference_log_variance_weight=1.0,
            teacher_forced=True,
        ),
        "structured_elbo": ModelSpec(
            key="structured_elbo",
            label="EKF-residualized nonlinear MC ELBO",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
        ),
        "structured_elbo_predictive_y": ModelSpec(
            key="structured_elbo_predictive_y",
            label="EKF-residualized nonlinear MC ELBO + predictive-y auxiliary",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            predictive_y_weight=1.0,
        ),
        "structured_elbo_resampled": ModelSpec(
            key="structured_elbo_resampled",
            label="EKF-residualized nonlinear MC ELBO (resampled batches)",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            resample_batch=True,
        ),
        "structured_joint_elbo_h1": ModelSpec(
            key="structured_joint_elbo_h1",
            label="EKF-residualized nonlinear windowed ELBO h1",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            joint_elbo_weight=1.0,
            joint_elbo_horizon=1,
        ),
        "structured_joint_elbo_h2": ModelSpec(
            key="structured_joint_elbo_h2",
            label="EKF-residualized nonlinear windowed ELBO h2",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            joint_elbo_weight=1.0,
            joint_elbo_horizon=2,
        ),
        "structured_joint_elbo_h4": ModelSpec(
            key="structured_joint_elbo_h4",
            label="EKF-residualized nonlinear windowed ELBO h4",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            joint_elbo_weight=1.0,
            joint_elbo_horizon=4,
        ),
        "structured_joint_elbo_h4_predictive_y": ModelSpec(
            key="structured_joint_elbo_h4_predictive_y",
            label="EKF-residualized nonlinear MC ELBO + joint h4 and predictive-y",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            joint_elbo_weight=0.3,
            joint_elbo_horizon=4,
            predictive_y_weight=1.0,
        ),
        "structured_joint_elbo_h8": ModelSpec(
            key="structured_joint_elbo_h8",
            label="EKF-residualized nonlinear windowed ELBO h8",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            joint_elbo_weight=1.0,
            joint_elbo_horizon=8,
        ),
        "direct_joint_elbo_h4": ModelSpec(
            key="direct_joint_elbo_h4",
            label="direct nonlinear windowed ELBO h4",
            objective="direct_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            joint_elbo_weight=1.0,
            joint_elbo_horizon=4,
        ),
        "structured_elbo_masked_y": ModelSpec(
            key="structured_elbo_masked_y",
            label="EKF-residualized nonlinear MC ELBO + masked-y updates",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            mask_y_probability=0.15,
        ),
        "structured_elbo_masked_y_spans_h2": ModelSpec(
            key="structured_elbo_masked_y_spans_h2",
            label="EKF-residualized nonlinear MC ELBO + masked-y spans h2",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            mask_y_span_probability=0.15,
            mask_y_span_length=2,
        ),
        "structured_elbo_masked_y_spans_h4": ModelSpec(
            key="structured_elbo_masked_y_spans_h4",
            label="EKF-residualized nonlinear MC ELBO + masked-y spans h4",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            mask_y_span_probability=0.15,
            mask_y_span_length=4,
        ),
        "structured_elbo_predictive_y_masked_y_spans_h4": ModelSpec(
            key="structured_elbo_predictive_y_masked_y_spans_h4",
            label="EKF-residualized nonlinear MC ELBO + predictive-y and masked-y spans h4",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            predictive_y_weight=1.0,
            mask_y_span_probability=0.15,
            mask_y_span_length=4,
        ),
        "structured_joint_elbo_h4_predictive_y_masked_y_spans_h4": ModelSpec(
            key="structured_joint_elbo_h4_predictive_y_masked_y_spans_h4",
            label=(
                "EKF-residualized nonlinear MC ELBO + joint h4, predictive-y, "
                "and masked-y spans h4"
            ),
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            joint_elbo_weight=0.3,
            joint_elbo_horizon=4,
            predictive_y_weight=1.0,
            mask_y_span_probability=0.15,
            mask_y_span_length=4,
        ),
        "structured_elbo_masked_y_spans_h8": ModelSpec(
            key="structured_elbo_masked_y_spans_h8",
            label="EKF-residualized nonlinear MC ELBO + masked-y spans h8",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            mask_y_span_probability=0.15,
            mask_y_span_length=8,
        ),
        "structured_moment_distilled": ModelSpec(
            key="structured_moment_distilled",
            label="EKF-residualized nonlinear MLP + reference moment distillation",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            reference_mean_weight=1.0,
            reference_log_variance_weight=1.0,
        ),
        "structured_moment_teacher_forced": ModelSpec(
            key="structured_moment_teacher_forced",
            label="EKF-residualized nonlinear MLP + teacher-forced reference moment distillation",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            reference_mean_weight=1.0,
            reference_log_variance_weight=1.0,
            teacher_forced=True,
        ),
        "structured_moment_rollout_h2": ModelSpec(
            key="structured_moment_rollout_h2",
            label="EKF-residualized nonlinear MLP + h2 reference rollout distillation",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            reference_rollout_weight=1.0,
            reference_rollout_horizon=2,
        ),
        "structured_moment_rollout_h4": ModelSpec(
            key="structured_moment_rollout_h4",
            label="EKF-residualized nonlinear MLP + h4 reference rollout distillation",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            reference_rollout_weight=1.0,
            reference_rollout_horizon=4,
        ),
        "structured_moment_rollout_h8": ModelSpec(
            key="structured_moment_rollout_h8",
            label="EKF-residualized nonlinear MLP + h8 reference rollout distillation",
            objective="structured_elbo_sine_mlp",
            reference_variance_ratio_weight=0.0,
            elbo_weight=0.0,
            reference_rollout_weight=1.0,
            reference_rollout_horizon=8,
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
    run_name: str,
    seed: int,
    steps: int,
    num_elbo_samples: int,
    batch_seed_stride: int,
    output_dir: Path,
) -> dict[str, Any]:
    config = {
        **base_train_config,
        "name": run_name,
        "seed": seed,
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
        "elbo_weight": spec.elbo_weight,
        "predictive_y_weight": spec.predictive_y_weight,
        "joint_elbo_weight": spec.joint_elbo_weight,
        "joint_elbo_horizon": spec.joint_elbo_horizon,
        "joint_elbo_num_samples": spec.joint_elbo_num_samples,
        "joint_elbo_num_windows": spec.joint_elbo_num_windows,
        "predictive_y_num_samples": spec.predictive_y_num_samples,
        "predictive_y_estimator": spec.predictive_y_estimator,
        "mask_y_probability": spec.mask_y_probability,
        "mask_y_span_probability": spec.mask_y_span_probability,
        "mask_y_span_length": spec.mask_y_span_length,
        "reference_mean_weight": spec.reference_mean_weight,
        "reference_rollout_weight": spec.reference_rollout_weight,
        "reference_rollout_horizon": spec.reference_rollout_horizon,
        "teacher_forced": spec.teacher_forced,
        "resample_batch": spec.resample_batch,
        "batch_seed_stride": batch_seed_stride,
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
        "seed": metrics["seed"],
        "objective": metrics["objective"],
        "x_pattern": metrics["x_pattern"],
        "time_steps": config["data"]["time_steps"],
        "steps": metrics["training_steps"],
        "num_elbo_samples": metrics["num_elbo_samples"],
        "elbo_weight": metrics["elbo_weight"],
        "predictive_y_weight": metrics["predictive_y_weight"],
        "joint_elbo_weight": metrics["joint_elbo_weight"],
        "joint_elbo_horizon": metrics["joint_elbo_horizon"],
        "joint_elbo_num_samples": metrics["joint_elbo_num_samples"],
        "joint_elbo_num_windows": metrics["joint_elbo_num_windows"],
        "predictive_y_num_samples": metrics["predictive_y_num_samples"],
        "predictive_y_estimator": metrics["predictive_y_estimator"],
        "mask_y_probability": metrics["mask_y_probability"],
        "mask_y_span_probability": metrics["mask_y_span_probability"],
        "mask_y_span_length": metrics["mask_y_span_length"],
        "training_signal": _training_signal(metrics),
        "reference_mean_weight": metrics["reference_mean_weight"],
        "reference_rollout_weight": metrics["reference_rollout_weight"],
        "reference_rollout_horizon": metrics["reference_rollout_horizon"],
        "teacher_forced": metrics["teacher_forced"],
        "resample_batch": metrics["resample_batch"],
        "batch_seed_stride": metrics["batch_seed_stride"],
        "reference_variance_ratio_weight": metrics["reference_variance_ratio_weight"],
        "reference_time_variance_ratio_weight": metrics["reference_time_variance_ratio_weight"],
        "reference_log_variance_weight": metrics["reference_log_variance_weight"],
        "reference_low_observation_variance_ratio_weight": metrics[
            "reference_low_observation_variance_ratio_weight"
        ],
        **{metric: metrics[metric] for metric in METRICS},
    }


def _training_signal(metrics: dict[str, Any]) -> str:
    if (
        metrics["reference_variance_ratio_weight"] != 0.0
        or metrics["reference_time_variance_ratio_weight"] != 0.0
        or metrics["reference_low_observation_variance_ratio_weight"] != 0.0
    ):
        return "oracle_calibrated"
    if (
        metrics["reference_mean_weight"] != 0.0
        or metrics["reference_rollout_weight"] != 0.0
        or metrics["reference_log_variance_weight"] != 0.0
    ):
        return "reference_distilled"
    return "unsupervised"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "config",
        "model",
        "seed",
        "objective",
        "x_pattern",
        "time_steps",
        "steps",
        "num_elbo_samples",
        "elbo_weight",
        "predictive_y_weight",
        "joint_elbo_weight",
        "joint_elbo_horizon",
        "joint_elbo_num_samples",
        "joint_elbo_num_windows",
        "predictive_y_num_samples",
        "predictive_y_estimator",
        "mask_y_probability",
        "mask_y_span_probability",
        "mask_y_span_length",
        "training_signal",
        "reference_mean_weight",
        "reference_rollout_weight",
        "reference_rollout_horizon",
        "teacher_forced",
        "resample_batch",
        "batch_seed_stride",
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
        "| x pattern | Model | signal | Steps | state NLL | ref state NLL | cov 90 | ref cov 90 | var ratio | pred-y NLL | pred NLL | ref pred NLL |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {x_pattern} | {model} | {training_signal} | {steps} | {state_nll:.6f} | "
            "{reference_state_nll:.6f} | {coverage_90:.6f} | "
            "{reference_coverage_90:.6f} | {variance_ratio:.6f} | "
            "{predictive_y_nll:.6f} | {predictive_nll:.6f} | "
            "{reference_predictive_nll:.6f} |".format(**row)
        )
    aggregate_rows = _aggregate_rows(rows)
    if len(aggregate_rows) != len(rows):
        lines.extend(
            [
                "",
                "## Aggregate By Seed",
                "",
                "| x pattern | Model | Steps | seeds | state NLL | coverage 90 | variance ratio |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in aggregate_rows:
            lines.append(
                "| {x_pattern} | {model} | {steps} | {num_seeds} | "
                "{state_nll_mean:.6f} +/- {state_nll_std:.6f} | "
                "{coverage_90_mean:.6f} +/- {coverage_90_std:.6f} | "
                "{variance_ratio_mean:.6f} +/- {variance_ratio_std:.6f} |".format(**row)
            )
    lines.extend(
        [
            "",
            "The signal column classifies each training row as unsupervised,",
            "reference-distilled, or oracle-calibrated according to the objective",
            "weights used during training.",
            "",
        ]
    )
    return "\n".join(lines)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["x_pattern"], row["model"], int(row["steps"]))
        groups.setdefault(key, []).append(row)
    aggregate = []
    for (x_pattern, model, steps), group in groups.items():
        item: dict[str, Any] = {
            "x_pattern": x_pattern,
            "model": model,
            "steps": steps,
            "num_seeds": len(group),
        }
        for metric in ("state_nll", "coverage_90", "variance_ratio"):
            values = [float(row[metric]) for row in group]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = variance**0.5
        aggregate.append(item)
    return sorted(aggregate, key=lambda row: (row["x_pattern"], row["model"], row["steps"]))


if __name__ == "__main__":
    main()
