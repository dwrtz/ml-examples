"""Decompose nonlinear learned predictive-y diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_METRICS = (
    "outputs/nonlinear_k4_projection_beta_0p3_spread_2pi_1000/metrics.csv",
    "outputs/nonlinear_k4_spread_predictive_y_1000/metrics.csv",
    "outputs/nonlinear_particle_filter_reference_1000/metrics.csv",
)

STATE_MODEL = "direct nonlinear K4 mixture local ADF projection beta 0.3 spread 2pi"
PREDICTIVE_MODEL = (
    "direct nonlinear K4 mixture local ADF projection beta 0.3 spread 2pi "
    "+ late predictive-y w0.1"
)
PF_MODEL = "bootstrap particle filter n512"

MODEL_LABELS = {
    STATE_MODEL: "K4 spread state-density",
    PREDICTIVE_MODEL: "K4 spread late pred-y",
    PF_MODEL: "PF n512 reference",
}

METRICS = (
    "state_nll",
    "coverage_90",
    "variance_ratio",
    "predictive_y_nll",
    "predictive_nll",
    "reference_predictive_nll",
)


@dataclass(frozen=True)
class AggregateRow:
    x_pattern: str
    model: str
    seeds: int
    metrics: dict[str, float]
    metric_stds: dict[str, float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument(
        "--output-dir",
        default="outputs/nonlinear_predictive_decomposition_report",
    )
    args = parser.parse_args()

    paths = [Path(item.strip()) for item in args.metrics.split(",") if item.strip()]
    rows = _aggregate(_load_rows(paths))
    report_rows = _decomposition_rows(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", report_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(report_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(_render_report(report_rows), encoding="utf-8")
    print(f"Wrote {report_path}")


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics CSV: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["model"] in MODEL_LABELS:
                    rows.append(row)
    if not rows:
        raise ValueError("No matching rows found")
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[AggregateRow]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["x_pattern"], row["model"])].append(row)

    aggregates = []
    for (x_pattern, model), group_rows in sorted(grouped.items()):
        metrics = {}
        metric_stds = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in group_rows]
            metrics[metric] = mean(values)
            metric_stds[metric] = pstdev(values) if len(values) > 1 else 0.0
        aggregates.append(
            AggregateRow(
                x_pattern=x_pattern,
                model=model,
                seeds=len(group_rows),
                metrics=metrics,
                metric_stds=metric_stds,
            )
        )
    return aggregates


def _decomposition_rows(rows: list[AggregateRow]) -> list[dict[str, Any]]:
    report_rows = []
    for pattern in sorted({row.x_pattern for row in rows}):
        pf = _find(rows, pattern, PF_MODEL)
        if pf is None:
            continue
        for model in (STATE_MODEL, PREDICTIVE_MODEL):
            learned = _find(rows, pattern, model)
            if learned is None:
                continue
            exact_pred = learned.metrics["predictive_y_nll"]
            moment_pred = learned.metrics["predictive_nll"]
            pf_pred = pf.metrics["predictive_y_nll"]
            reference_pred = learned.metrics["reference_predictive_nll"]
            report_rows.append(
                {
                    "x_pattern": pattern,
                    "model": MODEL_LABELS[model],
                    "seeds": learned.seeds,
                    "state_nll": learned.metrics["state_nll"],
                    "coverage_90": learned.metrics["coverage_90"],
                    "variance_ratio": learned.metrics["variance_ratio"],
                    "exact_mixture_pred_y_nll": exact_pred,
                    "gaussian_moment_pred_y_nll": moment_pred,
                    "moment_minus_exact": moment_pred - exact_pred,
                    "pf_pred_y_nll": pf_pred,
                    "grid_reference_pred_y_nll": reference_pred,
                    "exact_minus_pf": exact_pred - pf_pred,
                    "exact_minus_grid": exact_pred - reference_pred,
                }
            )
    return report_rows


def _find(rows: list[AggregateRow], pattern: str, model: str) -> AggregateRow | None:
    for row in rows:
        if row.x_pattern == pattern and row.model == model:
            return row
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Nonlinear Predictive-Y Decomposition",
        "",
        "`exact mixture pred-y` is the Gauss-Hermite mixture predictive likelihood used by the learned metrics. `Gaussian moment pred-y` is the older moment-Gaussian approximation.",
        "",
        "| Pattern | Row | state NLL | cov 90 | var ratio | exact mixture pred-y | Gaussian moment pred-y | moment - exact | PF pred-y | grid pred-y | exact - PF |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {x_pattern} | {model} | {state_nll:.3f} | {coverage_90:.3f} | "
            "{variance_ratio:.3f} | {exact_mixture_pred_y_nll:.3f} | "
            "{gaussian_moment_pred_y_nll:.3f} | {moment_minus_exact:.3f} | "
            "{pf_pred_y_nll:.3f} | {grid_reference_pred_y_nll:.3f} | {exact_minus_pf:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The evaluator is already using exact mixture quadrature for `predictive_y_nll`; the large `moment - exact` gap shows the Gaussian moment approximation is not the right selection metric for multimodal K4 rows.",
            "- Any remaining `exact - PF` gap is a true pre-update predictive-belief gap, not an artifact of the Gaussian moment approximation.",
            "- The next training objective should target the transition-predictive belief or its normalizer directly.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
