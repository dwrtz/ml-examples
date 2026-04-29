"""Compare nonlinear K4 candidate rows against the particle-filter reference."""

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
    parser.add_argument("--output-dir", default="outputs/nonlinear_k4_pareto_report")
    parser.add_argument("--pred-y-tolerance", type=float, default=0.03)
    args = parser.parse_args()

    paths = [Path(item.strip()) for item in args.metrics.split(",") if item.strip()]
    rows = _aggregate(_load_rows(paths))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_rows = _comparison_rows(rows, pred_y_tolerance=args.pred_y_tolerance)
    _write_csv(output_dir / "summary.csv", table_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(table_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(
        _render_report(rows, table_rows, pred_y_tolerance=args.pred_y_tolerance),
        encoding="utf-8",
    )
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
        raise ValueError("No candidate rows found in metrics CSVs")
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


def _comparison_rows(
    rows: list[AggregateRow],
    *,
    pred_y_tolerance: float,
) -> list[dict[str, Any]]:
    table_rows = []
    patterns = sorted({row.x_pattern for row in rows})
    for pattern in patterns:
        state = _find(rows, pattern, STATE_MODEL)
        predictive = _find(rows, pattern, PREDICTIVE_MODEL)
        pf = _find(rows, pattern, PF_MODEL)
        if state is None or predictive is None or pf is None:
            continue
        pred_gain = state.metrics["predictive_y_nll"] - predictive.metrics["predictive_y_nll"]
        state_cost = predictive.metrics["state_nll"] - state.metrics["state_nll"]
        pf_pred_gap = predictive.metrics["predictive_y_nll"] - pf.metrics["predictive_y_nll"]
        pf_state_delta = predictive.metrics["state_nll"] - pf.metrics["state_nll"]
        promote = state_cost <= pred_y_tolerance and pred_gain > 0.0
        table_rows.append(
            {
                "x_pattern": pattern,
                "state_candidate_state_nll": state.metrics["state_nll"],
                "state_candidate_pred_y_nll": state.metrics["predictive_y_nll"],
                "predictive_candidate_state_nll": predictive.metrics["state_nll"],
                "predictive_candidate_pred_y_nll": predictive.metrics["predictive_y_nll"],
                "pf_state_nll": pf.metrics["state_nll"],
                "pf_pred_y_nll": pf.metrics["predictive_y_nll"],
                "pred_y_gain": pred_gain,
                "state_nll_cost": state_cost,
                "pf_pred_y_gap": pf_pred_gap,
                "pf_state_nll_delta": pf_state_delta,
                "promote_predictive_candidate": promote,
                "recommendation": (
                    "late pred-y secondary"
                    if promote
                    else "keep state candidate"
                ),
            }
        )
    return table_rows


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


def _render_report(
    rows: list[AggregateRow],
    table_rows: list[dict[str, Any]],
    *,
    pred_y_tolerance: float,
) -> str:
    lines = [
        "# Nonlinear K4 Pareto Promotion Report",
        "",
        "This report compares the K4 spread state-density candidate, the K4 spread late predictive-y candidate, and the bootstrap particle-filter reference.",
        "",
        "## Candidate Metrics",
        "",
        "| Pattern | Row | seeds | state NLL | cov 90 | var ratio | pred-y NLL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {pattern} | {model} | {seeds} | {state:.3f} | {cov:.3f} | {var:.3f} | {pred:.3f} |".format(
                pattern=row.x_pattern,
                model=MODEL_LABELS[row.model],
                seeds=row.seeds,
                state=row.metrics["state_nll"],
                cov=row.metrics["coverage_90"],
                var=row.metrics["variance_ratio"],
                pred=row.metrics["predictive_y_nll"],
            )
        )
    lines.extend(
        [
            "",
            "## Promotion Decision",
            "",
            f"`late pred-y` is marked as a secondary candidate when it improves pred-y and costs no more than {pred_y_tolerance:.2f} state NLL.",
            "",
            "| Pattern | pred-y gain | state NLL cost | PF pred-y gap | recommendation |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in table_rows:
        lines.append(
            "| {x_pattern} | {pred_y_gain:.3f} | {state_nll_cost:.3f} | {pf_pred_y_gap:.3f} | {recommendation} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Bottom Line",
            "",
            "- The K4 spread state-density row remains the main promotion candidate.",
            "- The late predictive-y row is useful as a secondary/Pareto row, but its pred-y gains are small relative to the PF reference gap.",
            "- The next research step should target the predictive normalizer directly rather than adding another scalar predictive-y weight.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
