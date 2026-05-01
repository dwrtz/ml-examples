"""Run and aggregate scalar linear-Gaussian predictive-head sweeps."""

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
    predictive_nll: float
    exact_predictive_nll: float
    predictive_rmse: float
    exact_predictive_rmse: float
    variance_ratio: float
    learned_filter_head_predictive_nll: float | None
    learned_filter_analytic_predictive_nll: float | None
    learned_filter_exact_predictive_nll: float | None
    learned_filter_head_variance_ratio: float | None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/linear_gaussian/06_predictive_head.yaml")
    parser.add_argument("--seeds", default="321,322,323,324,325")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/linear_gaussian_predictive_head_sweep")
    parser.add_argument(
        "--learned-filter-root",
        default="outputs/linear_gaussian_sweep_corrected_metrics/elbo_edge_mlp",
        help="Root containing seed_<seed>/diagnostics.npz for optional learned-filter evaluation.",
    )
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    base_config = _read_config(Path(args.config))
    supported_models = {
        "predictive_head",
        "analytic_residual_predictive_head",
        "direct_predictive_head",
    }
    if base_config["model"] not in supported_models:
        raise ValueError(f"predictive sweep config must use one of: {sorted(supported_models)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    learned_filter_root = Path(args.learned_filter_root)

    rows: list[Row] = []
    for seed in seeds:
        run_dir = output_dir / f"seed_{seed}"
        run_config_path = output_dir / "configs" / f"seed_{seed}.yaml"
        config = {**base_config, "seed": seed, "output_dir": str(run_dir)}
        if args.steps is not None:
            config = {
                **config,
                "training": {**config["training"], "steps": args.steps},
            }
        diagnostics_path = learned_filter_root / f"seed_{seed}" / "diagnostics.npz"
        if diagnostics_path.exists():
            config = {
                **config,
                "evaluation": {
                    **config.get("evaluation", {}),
                    "learned_filter_diagnostics": str(diagnostics_path),
                },
            }
        _write_config(run_config_path, config)
        if not args.skip_train:
            _run_training(run_config_path)
        rows.append(_load_run(run_dir, seed=seed))

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


def _read_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return yaml.safe_load(stream)


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run_training(config_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/train_predictive_head.py", "--config", str(config_path)],
        check=True,
    )


def _load_run(run_dir: Path, *, seed: int) -> Row:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return Row(
        seed=seed,
        predictive_nll=float(metrics["predictive_nll"]),
        exact_predictive_nll=float(metrics["exact_predictive_nll"]),
        predictive_rmse=float(metrics["predictive_rmse"]),
        exact_predictive_rmse=float(metrics["exact_predictive_rmse"]),
        variance_ratio=float(metrics["variance_ratio"]),
        learned_filter_head_predictive_nll=_optional_float(
            metrics,
            "learned_filter_head_predictive_nll",
        ),
        learned_filter_analytic_predictive_nll=_optional_float(
            metrics,
            "learned_filter_analytic_predictive_nll",
        ),
        learned_filter_exact_predictive_nll=_optional_float(
            metrics,
            "learned_filter_exact_predictive_nll",
        ),
        learned_filter_head_variance_ratio=_optional_float(
            metrics,
            "learned_filter_head_variance_ratio",
        ),
    )


def _optional_float(metrics: dict, key: str) -> float | None:
    if key not in metrics:
        return None
    return float(metrics[key])


def _write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Row.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _aggregate(rows: list[Row]) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {"num_seeds": len(rows)}
    for metric in (
        "predictive_nll",
        "exact_predictive_nll",
        "predictive_rmse",
        "exact_predictive_rmse",
        "variance_ratio",
        "learned_filter_head_predictive_nll",
        "learned_filter_analytic_predictive_nll",
        "learned_filter_exact_predictive_nll",
        "learned_filter_head_variance_ratio",
    ):
        values = np.asarray(
            [getattr(row, metric) for row in rows if getattr(row, metric) is not None],
            dtype=np.float64,
        )
        if values.size == 0:
            summary[f"{metric}_mean"] = None
            summary[f"{metric}_std"] = None
            continue
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=0))
    return summary


def _render_report(summary: dict[str, float | int | None], rows: list[Row]) -> str:
    lines = [
        "# Linear-Gaussian Predictive Head Sweep",
        "",
        "| Predictor | Seeds | predictive NLL | predictive RMSE | variance ratio |",
        "|---|---:|---:|---:|---:|",
        _summary_row(
            "learned head on oracle belief",
            int(summary["num_seeds"]),
            summary["predictive_nll_mean"],
            summary["predictive_nll_std"],
            summary["predictive_rmse_mean"],
            summary["predictive_rmse_std"],
            summary["variance_ratio_mean"],
            summary["variance_ratio_std"],
        ),
        _summary_row(
            "exact Kalman predictive",
            int(summary["num_seeds"]),
            summary["exact_predictive_nll_mean"],
            summary["exact_predictive_nll_std"],
            summary["exact_predictive_rmse_mean"],
            summary["exact_predictive_rmse_std"],
            1.0,
            0.0,
        ),
    ]
    if summary["learned_filter_head_predictive_nll_mean"] is not None:
        lines.extend(
            [
                _summary_row(
                    "learned head on ELBO belief",
                    int(summary["num_seeds"]),
                    summary["learned_filter_head_predictive_nll_mean"],
                    summary["learned_filter_head_predictive_nll_std"],
                    None,
                    None,
                    summary["learned_filter_head_variance_ratio_mean"],
                    summary["learned_filter_head_variance_ratio_std"],
                ),
                _summary_row(
                    "analytic predictive from ELBO belief",
                    int(summary["num_seeds"]),
                    summary["learned_filter_analytic_predictive_nll_mean"],
                    summary["learned_filter_analytic_predictive_nll_std"],
                    None,
                    None,
                    None,
                    None,
                ),
                _summary_row(
                    "exact predictive on same ELBO eval",
                    int(summary["num_seeds"]),
                    summary["learned_filter_exact_predictive_nll_mean"],
                    summary["learned_filter_exact_predictive_nll_std"],
                    None,
                    None,
                    1.0,
                    0.0,
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Per-Seed Rows",
            "",
            "| Seed | head/oracle NLL | exact NLL | head/ELBO NLL | analytic/ELBO NLL | exact same-eval NLL |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.seed} | {row.predictive_nll:.6f} | {row.exact_predictive_nll:.6f} | "
            f"{_fmt(row.learned_filter_head_predictive_nll)} | "
            f"{_fmt(row.learned_filter_analytic_predictive_nll)} | "
            f"{_fmt(row.learned_filter_exact_predictive_nll)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _summary_row(
    label: str,
    num_seeds: int,
    nll_mean: float | int | None,
    nll_std: float | int | None,
    rmse_mean: float | int | None,
    rmse_std: float | int | None,
    variance_mean: float | int | None,
    variance_std: float | int | None,
) -> str:
    return (
        f"| {label} | {num_seeds} | {_fmt_pm(nll_mean, nll_std)} | "
        f"{_fmt_pm(rmse_mean, rmse_std)} | {_fmt_pm(variance_mean, variance_std)} |"
    )


def _fmt_pm(mean: float | int | None, std: float | int | None) -> str:
    if mean is None or std is None:
        return "n/a"
    return f"{float(mean):.6f} +/- {float(std):.6f}"


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
