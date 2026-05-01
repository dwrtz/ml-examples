"""Run quadrature ADF distillation configs across nonlinear stress settings."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIGS = (
    "experiments/nonlinear/01_sine_observation.yaml",
    "experiments/nonlinear/03_weak_sine_observation.yaml",
    "experiments/nonlinear/04_intermittent_sine_observation.yaml",
    "experiments/nonlinear/05_zero_sine_observation.yaml",
    "experiments/nonlinear/06_random_normal_sine_observation.yaml",
)

METRICS = (
    "state_nll",
    "predictive_y_nll",
    "hybrid_state_nll",
    "hybrid_predictive_y_nll",
    "pareto_state_nll",
    "pareto_predictive_y_nll",
    "coverage_90",
    "variance_ratio",
    "hybrid_coverage_90",
    "hybrid_variance_ratio",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-dir", default="outputs/nonlinear_quadrature_adf_distilled_suite")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = _read_config(Path(args.base_config))
    stress_configs = _parse_config_paths(args.configs)
    seeds = None if args.seeds is None else _parse_ints(args.seeds, name="--seeds")

    rows = []
    for stress_path in stress_configs:
        stress_config = _read_config(stress_path)
        for seed in seeds or [int(stress_config["seed"])]:
            run_name = _run_name(
                stress_config["name"],
                str(base_config["name"]),
                seed=seed,
                seeds=seeds,
            )
            run_dir = output_dir / run_name
            run_config_path = output_dir / "configs" / f"{run_name}.yaml"
            config = _make_config(base_config, stress_config, run_name=run_name, seed=seed)
            _write_config(run_config_path, config)
            if not args.skip_run:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/train_quadrature_adf_distilled.py",
                        "--config",
                        str(run_config_path),
                        "--output-dir",
                        str(run_dir),
                    ],
                    check=True,
                )
            rows.append(_load_row(run_dir, config_path=run_config_path, config=config))

    _write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(_render_report(rows), encoding="utf-8")
    print(f"Wrote {report_path}")


def _make_config(
    base_config: dict[str, Any],
    stress_config: dict[str, Any],
    *,
    run_name: str,
    seed: int,
) -> dict[str, Any]:
    return {
        **base_config,
        "name": run_name,
        "seed": seed,
        "output_dir": run_name,
        "data": dict(stress_config["data"]),
        "state_space": dict(stress_config["state_space"]),
        "reference": dict(stress_config.get("reference", {})),
        "evaluation": {
            **base_config.get("evaluation", {}),
            "data": {"batch_size": stress_config["data"]["batch_size"]},
        },
    }


def _load_row(run_dir: Path, *, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "name": config["name"],
        "config": str(config_path),
        "model": config["model"],
        "seed": metrics["seed"],
        "x_pattern": metrics["x_pattern"],
        "time_steps": config["data"]["time_steps"],
        "steps": metrics["training_steps"],
        "cell_type": metrics["cell_type"],
        "mixture_components": metrics["mixture_components"],
        "target_likelihood_power": metrics["target_likelihood_power"],
        "hybrid_refinement_steps": metrics["hybrid_refinement_steps"],
        **{metric: metrics.get(metric) for metric in METRICS},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "config",
        "model",
        "seed",
        "x_pattern",
        "time_steps",
        "steps",
        "cell_type",
        "mixture_components",
        "target_likelihood_power",
        "hybrid_refinement_steps",
        *METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Quadrature ADF Distillation Suite",
        "",
        "| x pattern | state NLL | pred-y NLL | hybrid state NLL | hybrid pred-y NLL | pareto state NLL | pareto pred-y NLL | cov 90 | var ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['x_pattern']} | {_fmt_metric(row['state_nll'])} | "
            f"{_fmt_metric(row['predictive_y_nll'])} | "
            f"{_fmt_metric(row['hybrid_state_nll'])} | "
            f"{_fmt_metric(row['hybrid_predictive_y_nll'])} | "
            f"{_fmt_metric(row['pareto_state_nll'])} | "
            f"{_fmt_metric(row['pareto_predictive_y_nll'])} | "
            f"{_fmt_metric(row['coverage_90'])} | {_fmt_metric(row['variance_ratio'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6f}"


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


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run_name(
    stress_name: str,
    base_name: str,
    *,
    seed: int,
    seeds: list[int] | None,
) -> str:
    suffix = base_name.replace("nonlinear_", "")
    if seeds is None:
        return f"{stress_name}_{suffix}"
    return f"{stress_name}_seed_{seed}_{suffix}"


if __name__ == "__main__":
    main()
