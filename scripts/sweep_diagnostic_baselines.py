"""Run scalar linear-Gaussian diagnostic baseline sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class Row:
    seed: int
    steps: int
    model: str
    objective: str
    filter_kl: float
    edge_kl: float
    state_rmse: float
    state_nll: float
    coverage_90: float
    variance_ratio: float
    predictive_nll: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="321,322,323,324,325")
    parser.add_argument("--steps", default="250")
    parser.add_argument("--output-dir", default="outputs/linear_gaussian_diagnostic_baselines")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    steps_values = _parse_steps(args.steps)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [
        ("zero-init no training", Path("experiments/linear_gaussian/09_zero_init_edge_mlp.yaml")),
        (
            "frozen marginal backward MLP",
            Path("experiments/linear_gaussian/10_frozen_marginal_backward_mlp.yaml"),
        ),
        (
            "split-head supervised MLP",
            Path("experiments/linear_gaussian/11_supervised_edge_split_mlp.yaml"),
        ),
    ]

    rows: list[Row] = []
    for label, config_path in run_specs:
        base_config = _read_config(config_path)
        objective = str(base_config["model"])
        model_steps = [0] if objective == "zero_init_edge_mlp" else steps_values
        for steps in model_steps:
            for seed in seeds:
                run_dir = output_dir / objective / f"steps_{steps}" / f"seed_{seed}"
                run_config_path = (
                    output_dir / "configs" / objective / f"steps_{steps}" / f"seed_{seed}.yaml"
                )
                config = {**base_config, "seed": seed, "output_dir": str(run_dir)}
                config["training"] = {**config["training"], "steps": steps}
                _write_config(run_config_path, config)
                if not args.skip_train:
                    _run_training(run_config_path)
                rows.append(_load_run(run_dir, seed=seed, steps=steps, model=label))

    _write_csv(output_dir / "metrics.csv", rows)
    summary = _aggregate(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(_render_report(summary, rows), encoding="utf-8")
    print(f"Wrote {report_path}")


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must include at least one integer seed")
    return seeds


def _parse_steps(value: str) -> list[int]:
    steps = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not steps:
        raise ValueError("--steps must include at least one integer step count")
    if any(step < 0 for step in steps):
        raise ValueError("--steps values must be nonnegative")
    return steps


def _read_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return yaml.safe_load(stream)


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run_training(config_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/train_linear_gaussian.py", "--config", str(config_path)],
        check=True,
    )


def _load_run(run_dir: Path, *, seed: int, steps: int, model: str) -> Row:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return Row(
        seed=seed,
        steps=steps,
        model=model,
        objective=str(metrics["objective"]),
        filter_kl=float(metrics["filter_kl"]),
        edge_kl=float(metrics["edge_kl"]),
        state_rmse=float(metrics["state_rmse_global"]),
        state_nll=float(metrics["state_nll"]),
        coverage_90=float(metrics["coverage_90"]),
        variance_ratio=float(metrics["variance_ratio"]),
        predictive_nll=float(metrics["predictive_nll"]),
    )


def _write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Row.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _aggregate(rows: list[Row]) -> dict[str, dict[str, float | str | int]]:
    summary: dict[str, dict[str, float | str | int]] = {}
    for objective, steps in sorted({(row.objective, row.steps) for row in rows}):
        grouped = [row for row in rows if row.objective == objective and row.steps == steps]
        key = f"{objective}_steps_{steps}"
        summary[key] = {
            "model": grouped[0].model,
            "objective": objective,
            "steps": steps,
            "num_seeds": len(grouped),
        }
        for metric in (
            "filter_kl",
            "edge_kl",
            "state_rmse",
            "state_nll",
            "coverage_90",
            "variance_ratio",
            "predictive_nll",
        ):
            values = np.asarray([getattr(row, metric) for row in grouped], dtype=np.float64)
            summary[key][f"{metric}_mean"] = float(np.mean(values))
            summary[key][f"{metric}_std"] = float(np.std(values, ddof=0))
    return summary


def _render_report(
    summary: dict[str, dict[str, float | str | int]],
    rows: list[Row],
) -> str:
    lines = [
        "# Linear-Gaussian Diagnostic Baselines",
        "",
        "| Model | Objective | Steps | Seeds | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, metrics in summary.items():
        lines.append(
            "| {model} | {objective} | {steps} | {num_seeds} | {filter_kl_mean:.6f} +/- "
            "{filter_kl_std:.6f} | {edge_kl_mean:.6f} +/- {edge_kl_std:.6f} | "
            "{state_rmse_mean:.6f} +/- {state_rmse_std:.6f} | "
            "{state_nll_mean:.6f} +/- {state_nll_std:.6f} | "
            "{coverage_90_mean:.6f} +/- {coverage_90_std:.6f} | "
            "{variance_ratio_mean:.6f} +/- {variance_ratio_std:.6f} | "
            "{predictive_nll_mean:.6f} +/- {predictive_nll_std:.6f} |".format(
                **metrics,
            )
        )
    lines.extend(
        [
            "",
            "## Per-Seed Rows",
            "",
            "| Seed | Steps | Model | Objective | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL |",
            "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.seed} | {row.steps} | {row.model} | {row.objective} | "
            f"{row.filter_kl:.6f} | "
            f"{row.edge_kl:.6f} | {row.state_rmse:.6f} | {row.state_nll:.6f} | "
            f"{row.coverage_90:.6f} | {row.variance_ratio:.6f} | "
            f"{row.predictive_nll:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
