"""Run ELBO sample-count and training-budget ablations."""

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
    num_elbo_samples: int
    steps: int
    filter_kl: float
    edge_kl: float
    state_rmse: float
    state_rmse_time_mean: float
    state_nll: float
    coverage_90: float
    variance_ratio: float
    predictive_nll: float


@dataclass(frozen=True)
class OracleRow:
    seed: int
    state_rmse: float
    state_rmse_time_mean: float
    state_nll: float
    coverage_90: float
    predictive_nll: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/linear_gaussian/02_elbo_edge_mlp.yaml")
    parser.add_argument("--seeds", default="321,322,323,324,325")
    parser.add_argument("--samples", default="1,4,8,16,32")
    parser.add_argument("--steps", default="250,1000")
    parser.add_argument("--output-dir", default="outputs/linear_gaussian_elbo_ablation")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    seeds = _parse_ints(args.seeds, name="--seeds")
    sample_counts = _parse_ints(args.samples, name="--samples")
    step_counts = _parse_ints(args.steps, name="--steps")
    base_config = _read_config(Path(args.config))
    if base_config["model"] != "elbo_edge_mlp":
        raise ValueError("ELBO ablation config must use model: elbo_edge_mlp")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    oracle_rows: dict[int, OracleRow] = {}
    for steps in step_counts:
        for num_samples in sample_counts:
            for seed in seeds:
                run_dir = output_dir / f"steps_{steps}" / f"samples_{num_samples}" / f"seed_{seed}"
                run_config_path = (
                    output_dir / "configs" / f"steps_{steps}" / f"samples_{num_samples}" / f"seed_{seed}.yaml"
                )
                config = _make_config(
                    base_config,
                    seed=seed,
                    steps=steps,
                    num_samples=num_samples,
                    output_dir=run_dir,
                )
                _write_config(run_config_path, config)
                if not args.skip_train:
                    _run_training(run_config_path)
                rows.append(
                    _load_run(
                        run_dir,
                        seed=seed,
                        steps=steps,
                        num_samples=num_samples,
                    )
                )
                if seed not in oracle_rows:
                    oracle_rows[seed] = _load_oracle_reference(run_dir, seed=seed)

    _write_rows_csv(output_dir / "metrics.csv", rows)
    _write_oracle_csv(output_dir / "oracle_metrics.csv", list(oracle_rows.values()))
    summary = _aggregate(rows)
    oracle_summary = _aggregate_oracle(list(oracle_rows.values()))
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "elbo": summary,
                "oracle": oracle_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(_render_report(summary, oracle_summary, rows), encoding="utf-8")
    print(f"Wrote {report_path}")


def _parse_ints(value: str, *, name: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must include at least one integer")
    return values


def _read_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return yaml.safe_load(stream)


def _make_config(
    base_config: dict[str, Any],
    *,
    seed: int,
    steps: int,
    num_samples: int,
    output_dir: Path,
) -> dict[str, Any]:
    training = {
        **base_config["training"],
        "steps": steps,
        "num_elbo_samples": num_samples,
    }
    return {
        **base_config,
        "name": f"linear_gaussian_elbo_edge_mlp_samples_{num_samples}_steps_{steps}_seed_{seed}",
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


def _load_run(run_dir: Path, *, seed: int, steps: int, num_samples: int) -> Row:
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
        num_elbo_samples=num_samples,
        steps=steps,
        filter_kl=float(metrics["filter_kl"]),
        edge_kl=float(metrics["edge_kl"]),
        state_rmse=float(metrics.get("state_rmse_global", state_rmse)),
        state_rmse_time_mean=float(metrics.get("state_rmse_time_mean", state_rmse_time_mean)),
        state_nll=float(np.mean(state_nll)),
        coverage_90=float(metrics.get("coverage_90", coverage)),
        variance_ratio=float(metrics.get("variance_ratio", np.nan)),
        predictive_nll=float(metrics["predictive_nll"]),
    )


def _load_oracle_reference(run_dir: Path, *, seed: int) -> OracleRow:
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
    return OracleRow(
        seed=seed,
        state_rmse=float(state_rmse),
        state_rmse_time_mean=float(state_rmse_time_mean),
        state_nll=float(np.mean(state_nll)),
        coverage_90=float(coverage),
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


def _write_rows_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Row.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _write_oracle_csv(path: Path, rows: list[OracleRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(OracleRow.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _aggregate(rows: list[Row]) -> list[dict[str, float | int]]:
    summary: list[dict[str, float | int]] = []
    keys = sorted({(row.steps, row.num_elbo_samples) for row in rows})
    for steps, num_samples in keys:
        grouped = [
            row for row in rows if row.steps == steps and row.num_elbo_samples == num_samples
        ]
        item: dict[str, float | int] = {
            "steps": steps,
            "num_elbo_samples": num_samples,
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
        ):
            values = np.asarray([getattr(row, metric) for row in grouped], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary.append(item)
    return summary


def _aggregate_oracle(rows: list[OracleRow]) -> dict[str, float | int]:
    state_rmse = np.asarray([row.state_rmse for row in rows], dtype=np.float64)
    state_rmse_time_mean = np.asarray(
        [row.state_rmse_time_mean for row in rows],
        dtype=np.float64,
    )
    state_nll = np.asarray([row.state_nll for row in rows], dtype=np.float64)
    coverage_90 = np.asarray([row.coverage_90 for row in rows], dtype=np.float64)
    predictive_nll = np.asarray([row.predictive_nll for row in rows], dtype=np.float64)
    return {
        "num_seeds": len(rows),
        "state_rmse_mean": float(np.mean(state_rmse)),
        "state_rmse_std": float(np.std(state_rmse, ddof=0)),
        "state_rmse_time_mean_mean": float(np.mean(state_rmse_time_mean)),
        "state_rmse_time_mean_std": float(np.std(state_rmse_time_mean, ddof=0)),
        "state_nll_mean": float(np.mean(state_nll)),
        "state_nll_std": float(np.std(state_nll, ddof=0)),
        "coverage_90_mean": float(np.mean(coverage_90)),
        "coverage_90_std": float(np.std(coverage_90, ddof=0)),
        "predictive_nll_mean": float(np.mean(predictive_nll)),
        "predictive_nll_std": float(np.std(predictive_nll, ddof=0)),
    }


def _render_report(
    summary: list[dict[str, float | int]],
    oracle: dict[str, float | int],
    rows: list[Row],
) -> str:
    lines = [
        "# Linear-Gaussian ELBO Ablation",
        "",
        "| Steps | MC samples | Seeds | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {steps} | {num_elbo_samples} | {num_seeds} | {filter_kl_mean:.6f} +/- "
            "{filter_kl_std:.6f} | {edge_kl_mean:.6f} +/- {edge_kl_std:.6f} | "
            "{state_rmse_mean:.6f} +/- {state_rmse_std:.6f} | "
            "{state_nll_mean:.6f} +/- {state_nll_std:.6f} | "
            "{coverage_90_mean:.6f} +/- {coverage_90_std:.6f} | "
            "{variance_ratio_mean:.6f} +/- {variance_ratio_std:.6f} | "
            "{predictive_nll_mean:.6f} +/- {predictive_nll_std:.6f} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Oracle Reference",
            "",
            "| Model | Seeds | state RMSE global | state RMSE time mean | state NLL | cov 90 | pred NLL |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                "| exact Kalman | {num_seeds} | {state_rmse_mean:.6f} +/- "
                "{state_rmse_std:.6f} | {state_rmse_time_mean_mean:.6f} +/- "
                "{state_rmse_time_mean_std:.6f} | {state_nll_mean:.6f} +/- "
                "{state_nll_std:.6f} | {coverage_90_mean:.6f} +/- {coverage_90_std:.6f} | "
                "{predictive_nll_mean:.6f} +/- {predictive_nll_std:.6f} |"
            ).format(**oracle),
            "",
            "## Per-Seed Rows",
            "",
            "| Seed | Steps | MC samples | filter KL | edge KL | state RMSE global | state RMSE time mean | state NLL | cov 90 | var ratio | pred NLL |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.seed} | {row.steps} | {row.num_elbo_samples} | {row.filter_kl:.6f} | "
            f"{row.edge_kl:.6f} | {row.state_rmse:.6f} | "
            f"{row.state_rmse_time_mean:.6f} | {row.state_nll:.6f} | "
            f"{row.coverage_90:.6f} | {row.variance_ratio:.6f} | "
            f"{row.predictive_nll:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
