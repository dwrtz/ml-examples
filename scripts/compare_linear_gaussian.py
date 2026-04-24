"""Compare scalar linear-Gaussian experiment runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOG_2PI = np.log(2.0 * np.pi)


@dataclass(frozen=True)
class RunSummary:
    label: str
    objective: str
    filter_kl: float
    edge_kl: float
    state_rmse: float
    state_nll: float
    predictive_nll: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--supervised-run-dir",
        default="outputs/linear_gaussian_supervised_edge_mlp",
    )
    parser.add_argument(
        "--elbo-run-dir",
        default="outputs/linear_gaussian_elbo_edge_mlp",
    )
    parser.add_argument(
        "--output",
        default="outputs/linear_gaussian_comparison.md",
    )
    args = parser.parse_args()

    supervised = _load_run(Path(args.supervised_run_dir), label="MLP supervised edge KL")
    elbo = _load_run(Path(args.elbo_run_dir), label="MLP edge ELBO")
    oracle = _load_oracle_reference(Path(args.supervised_run_dir))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_report([supervised, elbo, oracle]), encoding="utf-8")
    print(f"Wrote {output_path}")


def _load_run(run_dir: Path, *, label: str) -> RunSummary:
    metrics_path = run_dir / "metrics.json"
    diagnostics_path = run_dir / "diagnostics.npz"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing diagnostics file: {diagnostics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(diagnostics_path) as diagnostics:
        state_nll = _scalar_gaussian_nll(
            diagnostics["z"],
            diagnostics["learned_filter_mean"],
            diagnostics["learned_filter_var"],
        )

    return RunSummary(
        label=label,
        objective=str(metrics["objective"]),
        filter_kl=float(metrics["filter_kl"]),
        edge_kl=float(metrics["edge_kl"]),
        state_rmse=float(metrics["state_rmse"]),
        state_nll=float(np.mean(state_nll)),
        predictive_nll=float(metrics["predictive_nll"]),
    )


def _load_oracle_reference(run_dir: Path) -> RunSummary:
    diagnostics_path = run_dir / "diagnostics.npz"
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing diagnostics file: {diagnostics_path}")

    with np.load(diagnostics_path) as diagnostics:
        state_rmse = np.sqrt(np.mean((diagnostics["oracle_filter_mean"] - diagnostics["z"]) ** 2))
        state_nll = _scalar_gaussian_nll(
            diagnostics["z"],
            diagnostics["oracle_filter_mean"],
            diagnostics["oracle_filter_var"],
        )
        predictive_nll = _scalar_gaussian_nll(
            diagnostics["y"],
            diagnostics["oracle_predictive_mean"],
            diagnostics["oracle_predictive_var"],
        )

    return RunSummary(
        label="exact Kalman",
        objective="oracle",
        filter_kl=0.0,
        edge_kl=0.0,
        state_rmse=float(state_rmse),
        state_nll=float(np.mean(state_nll)),
        predictive_nll=float(np.mean(predictive_nll)),
    )


def _scalar_gaussian_nll(value: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return 0.5 * (LOG_2PI + np.log(var) + (value - mean) ** 2 / var)


def _render_report(rows: list[RunSummary]) -> str:
    lines = [
        "# Linear-Gaussian Comparison",
        "",
        "| Model | Objective | filter KL | edge KL | state RMSE | state NLL | predictive NLL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(_render_row(row) for row in rows)
    lines.append("")
    return "\n".join(lines)


def _render_row(row: RunSummary) -> str:
    return (
        f"| {row.label} | {row.objective} | {row.filter_kl:.6f} | {row.edge_kl:.6f} | "
        f"{row.state_rmse:.6f} | {row.state_nll:.6f} | {row.predictive_nll:.6f} |"
    )


if __name__ == "__main__":
    main()
