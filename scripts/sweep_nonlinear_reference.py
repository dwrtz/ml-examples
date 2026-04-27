"""Run and aggregate nonlinear grid-reference stress evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


METRICS = (
    "state_rmse",
    "state_nll",
    "predictive_nll",
    "coverage_90",
    "mean_filter_variance",
    "mean_predictive_variance",
)

DEFAULT_CONFIGS = (
    "experiments/nonlinear/01_sine_observation.yaml",
    "experiments/nonlinear/03_weak_sine_observation.yaml",
    "experiments/nonlinear/04_intermittent_sine_observation.yaml",
    "experiments/nonlinear/05_zero_sine_observation.yaml",
    "experiments/nonlinear/06_random_normal_sine_observation.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--output-dir", default="outputs/nonlinear_reference_suite")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = _parse_config_paths(args.configs)
    rows = []
    for config_path in config_paths:
        base_config = _read_config(config_path)
        run_dir = output_dir / str(base_config["name"])
        run_config_path = output_dir / "configs" / f"{base_config['name']}.yaml"
        config = {**base_config, "output_dir": str(run_dir)}
        _write_config(run_config_path, config)
        if not args.skip_run:
            _run_evaluation(run_config_path)
        rows.append(_load_row(run_dir, config_path=config_path, config=config))

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


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run_evaluation(config_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/evaluate_nonlinear.py", "--config", str(config_path)],
        check=True,
    )


def _load_row(run_dir: Path, *, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing nonlinear metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    data = config["data"]
    return {
        "name": config["name"],
        "config": str(config_path),
        "observation": metrics["observation"],
        "x_pattern": metrics["x_pattern"],
        "time_steps": metrics["time_steps"],
        "batch_size": metrics["batch_size"],
        "x_amplitude": data.get("x_amplitude"),
        "x_missing_period": data.get("x_missing_period"),
        **{metric: metrics[metric] for metric in METRICS},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "config",
        "observation",
        "x_pattern",
        "time_steps",
        "batch_size",
        "x_amplitude",
        "x_missing_period",
        *METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Nonlinear Grid Reference Suite",
        "",
        "| Name | x pattern | T | state NLL | cov 90 | pred NLL | mean filter var |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {x_pattern} | {time_steps} | {state_nll:.6f} | "
            "{coverage_90:.6f} | {predictive_nll:.6f} | "
            "{mean_filter_variance:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "These rows are deterministic grid-reference diagnostics for the nonlinear",
            "`y_t = x_t sin(z_t) + v_t` benchmark. They are references for learned",
            "nonlinear filters, not trained models. The longer-sequence config is",
            "available separately as `experiments/nonlinear/07_long_sine_observation.yaml`",
            "because its wider grid is too slow for the default smoke suite.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
