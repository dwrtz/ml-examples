"""Run matched-budget supervised-vs-ELBO linear-Gaussian sweeps."""

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
    model: str
    objective: str
    steps: int
    num_elbo_samples: int
    filter_kl: float
    edge_kl: float
    state_rmse: float
    state_nll: float
    coverage_90: float
    variance_ratio: float
    predictive_nll: float
    closed_form_elbo: float
    oracle_closed_form_elbo: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated config paths. Overrides --supervised-config/--elbo-config.",
    )
    parser.add_argument(
        "--supervised-config",
        default="experiments/linear_gaussian/01_supervised_edge_mlp.yaml",
    )
    parser.add_argument(
        "--elbo-config",
        default="experiments/linear_gaussian/02_elbo_edge_mlp.yaml",
    )
    parser.add_argument("--seeds", default="321,322,323,324,325")
    parser.add_argument("--steps", default="250,1000,3000")
    parser.add_argument("--num-elbo-samples", type=int, default=32)
    parser.add_argument("--output-dir", default="outputs/linear_gaussian_objective_budget")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    seeds = _parse_ints(args.seeds, name="--seeds")
    step_counts = _parse_ints(args.steps, name="--steps")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = _run_specs_from_args(args)

    rows: list[Row] = []
    for label, config_path in run_specs:
        base_config = _read_config(config_path)
        objective = str(base_config["model"])
        for steps in step_counts:
            for seed in seeds:
                run_dir = output_dir / objective / f"steps_{steps}" / f"seed_{seed}"
                run_config_path = (
                    output_dir / "configs" / objective / f"steps_{steps}" / f"seed_{seed}.yaml"
                )
                config = _make_config(
                    base_config,
                    seed=seed,
                    steps=steps,
                    num_elbo_samples=args.num_elbo_samples,
                    output_dir=run_dir,
                )
                _write_config(run_config_path, config)
                if not args.skip_train:
                    _run_training(run_config_path)
                rows.append(
                    _load_run(
                        run_dir,
                        seed=seed,
                        model=label,
                        steps=steps,
                    )
                )

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


def _run_specs_from_args(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.configs is None:
        return [
            ("MLP supervised edge KL", Path(args.supervised_config)),
            ("MLP edge ELBO", Path(args.elbo_config)),
        ]
    paths = [Path(item.strip()) for item in args.configs.split(",") if item.strip()]
    if not paths:
        raise ValueError("--configs must include at least one config path")
    specs = []
    for path in paths:
        config = _read_config(path)
        specs.append((str(config.get("name", config["model"])), path))
    return specs


def _read_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return yaml.safe_load(stream)


def _make_config(
    base_config: dict[str, Any],
    *,
    seed: int,
    steps: int,
    num_elbo_samples: int,
    output_dir: Path,
) -> dict[str, Any]:
    training = {**base_config["training"], "steps": steps}
    if base_config["model"] in {"elbo_edge_mlp", "direct_elbo_edge_mlp"}:
        training["num_elbo_samples"] = num_elbo_samples
    return {
        **base_config,
        "name": f"{base_config['model']}_steps_{steps}_seed_{seed}",
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


def _load_run(run_dir: Path, *, seed: int, model: str, steps: int) -> Row:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return Row(
        seed=seed,
        model=model,
        objective=str(metrics["objective"]),
        steps=steps,
        num_elbo_samples=int(metrics.get("num_elbo_samples", 0)),
        filter_kl=float(metrics["filter_kl"]),
        edge_kl=float(metrics["edge_kl"]),
        state_rmse=float(metrics["state_rmse_global"]),
        state_nll=float(metrics["state_nll"]),
        coverage_90=float(metrics["coverage_90"]),
        variance_ratio=float(metrics["variance_ratio"]),
        predictive_nll=float(metrics["predictive_nll"]),
        closed_form_elbo=float(metrics["closed_form_elbo"]),
        oracle_closed_form_elbo=float(metrics["oracle_closed_form_elbo"]),
    )


def _write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Row.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _aggregate(rows: list[Row]) -> list[dict[str, float | int | str]]:
    summary: list[dict[str, float | int | str]] = []
    keys = sorted({(row.objective, row.steps) for row in rows})
    for objective, steps in keys:
        grouped = [row for row in rows if row.objective == objective and row.steps == steps]
        item: dict[str, float | int | str] = {
            "model": grouped[0].model,
            "objective": objective,
            "steps": steps,
            "num_seeds": len(grouped),
            "num_elbo_samples": grouped[0].num_elbo_samples,
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
            "oracle_closed_form_elbo",
        ):
            values = np.asarray([getattr(row, metric) for row in grouped], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary.append(item)
    return summary


def _render_report(summary: list[dict[str, float | int | str]], rows: list[Row]) -> str:
    lines = [
        "# Linear-Gaussian Objective Budget Sweep",
        "",
        "| Model | Objective | Steps | Seeds | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {model} | {objective} | {steps} | {num_seeds} | {filter_kl_mean:.6f} +/- "
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
            "| Seed | Model | Objective | Steps | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.seed} | {row.model} | {row.objective} | {row.steps} | "
            f"{row.filter_kl:.6f} | {row.edge_kl:.6f} | {row.state_rmse:.6f} | "
            f"{row.state_nll:.6f} | {row.coverage_90:.6f} | {row.variance_ratio:.6f} | "
            f"{row.predictive_nll:.6f} | {row.closed_form_elbo:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
