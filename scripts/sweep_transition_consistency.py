"""Run unsupervised transition-consistency regularizer sweeps for ELBO training."""

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
    transition_consistency_weight: float
    filter_kl: float
    edge_kl: float
    state_rmse: float
    state_rmse_time_mean: float
    state_nll: float
    coverage_90: float
    variance_ratio: float
    predictive_nll: float
    elbo: float
    oracle_elbo: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/linear_gaussian/02_elbo_edge_mlp.yaml")
    parser.add_argument("--seeds", default="321,322,323,324,325")
    parser.add_argument("--weights", default="0,0.01,0.05,0.1")
    parser.add_argument("--output-dir", default="outputs/linear_gaussian_transition_consistency")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    seeds = _parse_ints(args.seeds, name="--seeds")
    weights = _parse_floats(args.weights, name="--weights")
    base_config = _read_config(Path(args.config))
    if base_config["model"] != "elbo_edge_mlp":
        raise ValueError("transition consistency sweep requires model: elbo_edge_mlp")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    for weight in weights:
        for seed in seeds:
            weight_label = _weight_label(weight)
            run_dir = output_dir / f"transition_weight_{weight_label}" / f"seed_{seed}"
            run_config_path = (
                output_dir / "configs" / f"transition_weight_{weight_label}" / f"seed_{seed}.yaml"
            )
            config = _make_config(
                base_config,
                seed=seed,
                transition_consistency_weight=weight,
                output_dir=run_dir,
            )
            _write_config(run_config_path, config)
            if not args.skip_train:
                _run_training(run_config_path)
            rows.append(_load_run(run_dir, seed=seed, transition_consistency_weight=weight))

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
    transition_consistency_weight: float,
    output_dir: Path,
) -> dict[str, Any]:
    training = {
        **base_config["training"],
        "edge_kl_weight": 0.0,
        "transition_consistency_weight": transition_consistency_weight,
    }
    return {
        **base_config,
        "name": (
            "linear_gaussian_elbo_edge_mlp_transition_"
            f"{transition_consistency_weight:g}_seed_{seed}"
        ),
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


def _load_run(run_dir: Path, *, seed: int, transition_consistency_weight: float) -> Row:
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
        transition_consistency_weight=transition_consistency_weight,
        filter_kl=float(metrics["filter_kl"]),
        edge_kl=float(metrics["edge_kl"]),
        state_rmse=float(metrics.get("state_rmse_global", state_rmse)),
        state_rmse_time_mean=float(metrics.get("state_rmse_time_mean", state_rmse_time_mean)),
        state_nll=float(np.mean(state_nll)),
        coverage_90=float(metrics.get("coverage_90", coverage)),
        variance_ratio=float(metrics.get("variance_ratio", np.nan)),
        predictive_nll=float(metrics["predictive_nll"]),
        elbo=float(metrics["elbo"]),
        oracle_elbo=float(metrics["oracle_elbo"]),
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
        writer = csv.DictWriter(stream, fieldnames=list(Row.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _aggregate(rows: list[Row]) -> list[dict[str, float | int]]:
    summary: list[dict[str, float | int]] = []
    for weight in sorted({row.transition_consistency_weight for row in rows}):
        grouped = [row for row in rows if row.transition_consistency_weight == weight]
        item: dict[str, float | int] = {
            "transition_consistency_weight": weight,
            "num_seeds": len(grouped),
        }
        for metric in (
            "filter_kl",
            "edge_kl",
            "state_rmse",
            "state_rmse_time_mean",
            "state_nll",
            "coverage_90",
            "variance_ratio",
            "predictive_nll",
            "elbo",
            "oracle_elbo",
        ):
            values = np.asarray([getattr(row, metric) for row in grouped], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary.append(item)
    return summary


def _render_report(summary: list[dict[str, float | int]], rows: list[Row]) -> str:
    lines = [
        "# Linear-Gaussian Transition-Consistency Sweep",
        "",
        "| transition weight | Seeds | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL | ELBO | oracle ELBO |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {transition_consistency_weight:g} | {num_seeds} | {filter_kl_mean:.6f} +/- "
            "{filter_kl_std:.6f} | {edge_kl_mean:.6f} +/- {edge_kl_std:.6f} | "
            "{state_rmse_mean:.6f} +/- {state_rmse_std:.6f} | "
            "{state_nll_mean:.6f} +/- {state_nll_std:.6f} | "
            "{coverage_90_mean:.6f} +/- {coverage_90_std:.6f} | "
            "{variance_ratio_mean:.6f} +/- {variance_ratio_std:.6f} | "
            "{predictive_nll_mean:.6f} +/- {predictive_nll_std:.6f} | "
            "{elbo_mean:.6f} +/- {elbo_std:.6f} | "
            "{oracle_elbo_mean:.6f} +/- {oracle_elbo_std:.6f} |".format(**item)
        )
    lines.extend(
        [
            "",
            "This sweep is unsupervised with respect to posterior targets; it uses only the known transition scale.",
            "",
            "## Per-Seed Rows",
            "",
            "| Seed | transition weight | filter KL | edge KL | state RMSE global | state RMSE time mean | state NLL | cov 90 | var ratio | pred NLL | ELBO | oracle ELBO |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.seed} | {row.transition_consistency_weight:g} | {row.filter_kl:.6f} | "
            f"{row.edge_kl:.6f} | {row.state_rmse:.6f} | "
            f"{row.state_rmse_time_mean:.6f} | {row.state_nll:.6f} | "
            f"{row.coverage_90:.6f} | {row.variance_ratio:.6f} | "
            f"{row.predictive_nll:.6f} | {row.elbo:.6f} | {row.oracle_elbo:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _weight_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


if __name__ == "__main__":
    main()
