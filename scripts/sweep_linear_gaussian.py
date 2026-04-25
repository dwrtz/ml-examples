"""Run and aggregate scalar linear-Gaussian training sweeps."""

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


LOG_2PI = np.log(2.0 * np.pi)


@dataclass(frozen=True)
class Row:
    seed: int
    model: str
    objective: str
    filter_kl: float
    edge_kl: float
    state_rmse: float
    state_rmse_time_mean: float
    state_nll: float
    coverage_90: float
    variance_ratio: float
    predictive_nll: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--supervised-config",
        default="experiments/linear_gaussian/01_supervised_edge_mlp.yaml",
    )
    parser.add_argument(
        "--elbo-config",
        default="experiments/linear_gaussian/02_elbo_edge_mlp.yaml",
    )
    parser.add_argument("--seeds", default="321,322,323,324,325")
    parser.add_argument("--output-dir", default="outputs/linear_gaussian_sweep")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [
        ("MLP supervised edge KL", Path(args.supervised_config)),
        ("MLP edge ELBO", Path(args.elbo_config)),
    ]

    rows: list[Row] = []
    oracle_rows: list[Row] = []
    for label, config_path in run_specs:
        base_config = _read_config(config_path)
        objective = str(base_config["model"])
        for seed in seeds:
            run_dir = output_dir / objective / f"seed_{seed}"
            run_config_path = output_dir / "configs" / objective / f"seed_{seed}.yaml"
            config = {**base_config, "seed": seed, "output_dir": str(run_dir)}
            _write_config(run_config_path, config)
            if not args.skip_train:
                _run_training(run_config_path)
            rows.append(_load_run(run_dir, seed=seed, model=label))
            if objective == "supervised_edge_mlp":
                oracle_rows.append(_load_oracle_reference(run_dir, seed=seed))

    all_rows = [*rows, *oracle_rows]
    _write_csv(output_dir / "metrics.csv", all_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(_aggregate(all_rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(_render_report(all_rows), encoding="utf-8")
    print(f"Wrote {report_path}")


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must include at least one integer seed")
    return seeds


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


def _load_run(run_dir: Path, *, seed: int, model: str) -> Row:
    metrics_path = run_dir / "metrics.json"
    diagnostics_path = run_dir / "diagnostics.npz"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing diagnostics file: {diagnostics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(diagnostics_path) as diagnostics:
        state_rmse = _global_rmse(diagnostics["learned_filter_mean"], diagnostics["z"])
        state_rmse_time_mean = _time_mean_rmse(diagnostics["learned_filter_mean"], diagnostics["z"])
        state_nll = _scalar_gaussian_nll(
            diagnostics["z"],
            diagnostics["learned_filter_mean"],
            diagnostics["learned_filter_var"],
        )
        coverage = _coverage_90(
            diagnostics["z"],
            diagnostics["learned_filter_mean"],
            diagnostics["learned_filter_var"],
        )

    return Row(
        seed=seed,
        model=model,
        objective=str(metrics["objective"]),
        filter_kl=float(metrics["filter_kl"]),
        edge_kl=float(metrics["edge_kl"]),
        state_rmse=float(metrics.get("state_rmse_global", state_rmse)),
        state_rmse_time_mean=float(metrics.get("state_rmse_time_mean", state_rmse_time_mean)),
        state_nll=float(np.mean(state_nll)),
        coverage_90=float(metrics.get("coverage_90", coverage)),
        variance_ratio=float(metrics.get("variance_ratio", np.nan)),
        predictive_nll=float(metrics["predictive_nll"]),
    )


def _load_oracle_reference(run_dir: Path, *, seed: int) -> Row:
    diagnostics_path = run_dir / "diagnostics.npz"
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing diagnostics file: {diagnostics_path}")

    with np.load(diagnostics_path) as diagnostics:
        state_rmse = _global_rmse(diagnostics["oracle_filter_mean"], diagnostics["z"])
        state_rmse_time_mean = _time_mean_rmse(diagnostics["oracle_filter_mean"], diagnostics["z"])
        state_nll = _scalar_gaussian_nll(
            diagnostics["z"],
            diagnostics["oracle_filter_mean"],
            diagnostics["oracle_filter_var"],
        )
        coverage = _coverage_90(
            diagnostics["z"],
            diagnostics["oracle_filter_mean"],
            diagnostics["oracle_filter_var"],
        )
        predictive_nll = _scalar_gaussian_nll(
            diagnostics["y"],
            diagnostics["oracle_predictive_mean"],
            diagnostics["oracle_predictive_var"],
        )

    return Row(
        seed=seed,
        model="exact Kalman",
        objective="oracle",
        filter_kl=0.0,
        edge_kl=0.0,
        state_rmse=float(state_rmse),
        state_rmse_time_mean=float(state_rmse_time_mean),
        state_nll=float(np.mean(state_nll)),
        coverage_90=float(coverage),
        variance_ratio=1.0,
        predictive_nll=float(np.mean(predictive_nll)),
    )


def _scalar_gaussian_nll(value: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return 0.5 * (LOG_2PI + np.log(var) + (value - mean) ** 2 / var)


def _global_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _time_mean_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((pred - target) ** 2, axis=0))))


def _coverage_90(value: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    z_score = 1.6448536269514722
    half_width = z_score * np.sqrt(var)
    return float(np.mean((value >= mean - half_width) & (value <= mean + half_width)))


def _write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "seed",
                "model",
                "objective",
                "filter_kl",
                "edge_kl",
                "state_rmse",
                "state_rmse_time_mean",
                "state_nll",
                "coverage_90",
                "variance_ratio",
                "predictive_nll",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _aggregate(rows: list[Row]) -> dict[str, dict[str, float | str | int]]:
    summary: dict[str, dict[str, float | str | int]] = {}
    for key in sorted({row.objective for row in rows}):
        grouped = [row for row in rows if row.objective == key]
        summary[key] = {"model": grouped[0].model, "num_seeds": len(grouped)}
        for metric in (
            "filter_kl",
            "edge_kl",
            "state_rmse",
            "state_rmse_time_mean",
            "state_nll",
            "coverage_90",
            "variance_ratio",
            "predictive_nll",
        ):
            values = np.asarray([getattr(row, metric) for row in grouped], dtype=np.float64)
            summary[key][f"{metric}_mean"] = float(np.mean(values))
            summary[key][f"{metric}_std"] = float(np.std(values, ddof=0))
    return summary


def _render_report(rows: list[Row]) -> str:
    summary = _aggregate(rows)
    lines = [
        "# Linear-Gaussian Seed Sweep",
        "",
        "| Model | Objective | Seeds | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for objective, metrics in summary.items():
        lines.append(
            "| {model} | {objective} | {num_seeds} | {filter_kl_mean:.6f} +/- "
            "{filter_kl_std:.6f} | {edge_kl_mean:.6f} +/- {edge_kl_std:.6f} | "
            "{state_rmse_mean:.6f} +/- {state_rmse_std:.6f} | "
            "{state_nll_mean:.6f} +/- {state_nll_std:.6f} | "
            "{coverage_90_mean:.6f} +/- {coverage_90_std:.6f} | "
            "{variance_ratio_mean:.6f} +/- {variance_ratio_std:.6f} | "
            "{predictive_nll_mean:.6f} +/- {predictive_nll_std:.6f} |".format(
                objective=objective,
                **metrics,
            )
        )
    lines.extend(
        [
            "",
            "## Per-Seed Rows",
            "",
            "| Seed | Model | Objective | filter KL | edge KL | state RMSE global | state RMSE time mean | state NLL | cov 90 | var ratio | pred NLL |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.seed} | {row.model} | {row.objective} | {row.filter_kl:.6f} | "
            f"{row.edge_kl:.6f} | {row.state_rmse:.6f} | "
            f"{row.state_rmse_time_mean:.6f} | {row.state_nll:.6f} | "
            f"{row.coverage_90:.6f} | {row.variance_ratio:.6f} | "
            f"{row.predictive_nll:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
