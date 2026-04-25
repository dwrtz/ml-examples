"""Sweep self-fed supervised filtering variance-ratio regularization."""

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
    variance_ratio_weight: float
    filter_kl: float
    edge_kl: float
    state_rmse: float
    state_nll: float
    coverage_90: float
    variance_ratio: float
    predictive_nll: float
    closed_form_elbo: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/linear_gaussian/12_self_fed_supervised_edge_mlp.yaml",
    )
    parser.add_argument("--seeds", default="321,322,323,324,325")
    parser.add_argument("--weights", default="0,0.001,0.01,0.05,0.1")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument(
        "--output-dir",
        default="outputs/linear_gaussian_self_fed_variance_regularizer",
    )
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    seeds = _parse_ints(args.seeds, name="--seeds")
    weights = _parse_floats(args.weights, name="--weights")
    base_config = _read_config(Path(args.config))
    if base_config["model"] != "self_fed_supervised_edge_mlp":
        raise ValueError("variance regularizer sweep requires self_fed_supervised_edge_mlp")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    for weight in weights:
        weight_label = _weight_label(weight)
        for seed in seeds:
            run_dir = output_dir / f"variance_weight_{weight_label}" / f"seed_{seed}"
            run_config_path = (
                output_dir / "configs" / f"variance_weight_{weight_label}" / f"seed_{seed}.yaml"
            )
            config = _make_config(
                base_config,
                seed=seed,
                steps=args.steps,
                weight=weight,
                output_dir=run_dir,
            )
            _write_config(run_config_path, config)
            if not args.skip_train:
                _run_training(run_config_path)
            rows.append(_load_run(run_dir, seed=seed, weight=weight))

    _write_csv(output_dir / "metrics.csv", rows)
    summary = _aggregate(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(_render_report(summary, rows), encoding="utf-8")
    print(f"Wrote {report_path}")


def _parse_ints(value: str, *, name: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must include at least one integer")
    return values


def _parse_floats(value: str, *, name: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must include at least one number")
    return values


def _read_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return yaml.safe_load(stream)


def _make_config(
    base_config: dict[str, Any],
    *,
    seed: int,
    steps: int,
    weight: float,
    output_dir: Path,
) -> dict[str, Any]:
    training = {
        **base_config["training"],
        "steps": steps,
        "variance_ratio_weight": weight,
    }
    return {
        **base_config,
        "name": f"self_fed_supervised_variance_{weight:g}_seed_{seed}",
        "seed": seed,
        "output_dir": str(output_dir),
        "training": training,
    }


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run_training(config_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/train_linear_gaussian.py", "--config", str(config_path)],
        check=True,
    )


def _load_run(run_dir: Path, *, seed: int, weight: float) -> Row:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return Row(
        seed=seed,
        variance_ratio_weight=weight,
        filter_kl=float(metrics["filter_kl"]),
        edge_kl=float(metrics["edge_kl"]),
        state_rmse=float(metrics["state_rmse_global"]),
        state_nll=float(metrics["state_nll"]),
        coverage_90=float(metrics["coverage_90"]),
        variance_ratio=float(metrics["variance_ratio"]),
        predictive_nll=float(metrics["predictive_nll"]),
        closed_form_elbo=float(metrics["closed_form_elbo"]),
    )


def _write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Row.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _aggregate(rows: list[Row]) -> list[dict[str, float | int]]:
    summary: list[dict[str, float | int]] = []
    for weight in sorted({row.variance_ratio_weight for row in rows}):
        grouped = [row for row in rows if row.variance_ratio_weight == weight]
        item: dict[str, float | int] = {
            "variance_ratio_weight": weight,
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
            "closed_form_elbo",
        ):
            values = np.asarray([getattr(row, metric) for row in grouped], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary.append(item)
    return summary


def _render_report(summary: list[dict[str, float | int]], rows: list[Row]) -> str:
    lines = [
        "# Self-Fed Supervised Variance Regularizer Sweep",
        "",
        "| variance weight | Seeds | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {variance_ratio_weight:g} | {num_seeds} | {filter_kl_mean:.6f} +/- "
            "{filter_kl_std:.6f} | {edge_kl_mean:.6f} +/- {edge_kl_std:.6f} | "
            "{state_rmse_mean:.6f} +/- {state_rmse_std:.6f} | "
            "{state_nll_mean:.6f} +/- {state_nll_std:.6f} | "
            "{coverage_90_mean:.6f} +/- {coverage_90_std:.6f} | "
            "{variance_ratio_mean:.6f} +/- {variance_ratio_std:.6f} | "
            "{predictive_nll_mean:.6f} +/- {predictive_nll_std:.6f} | "
            "{closed_form_elbo_mean:.6f} +/- {closed_form_elbo_std:.6f} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Per-Seed Rows",
            "",
            "| Seed | variance weight | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.seed} | {row.variance_ratio_weight:g} | {row.filter_kl:.6f} | "
            f"{row.edge_kl:.6f} | {row.state_rmse:.6f} | {row.state_nll:.6f} | "
            f"{row.coverage_90:.6f} | {row.variance_ratio:.6f} | "
            f"{row.predictive_nll:.6f} | {row.closed_form_elbo:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _weight_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


if __name__ == "__main__":
    main()
