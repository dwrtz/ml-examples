"""Report current nonlinear state-density and predictive-y Pareto candidates."""

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
    "outputs/nonlinear_quadrature_adf_suite_2026_04_30/metrics.csv",
    "outputs/nonlinear_quadrature_alias_prior_suite_250/metrics.csv",
    "outputs/nonlinear_quadrature_alias_pruned_suite_250/metrics.csv",
    "outputs/nonlinear_quadrature_alias_shrink_suite_250/metrics.csv",
    "outputs/nonlinear_fivo_bridge_resampling_suite_250/metrics.csv",
    "outputs/nonlinear_auxiliary_fivo_bridge_suite_250/metrics.csv",
    "outputs/nonlinear_fivo_twist_suite_250/metrics.csv",
)

METRICS = (
    "state_nll",
    "predictive_y_nll",
    "coverage_90",
    "variance_ratio",
    "eval_fivo_mean_ess",
)


@dataclass(frozen=True)
class AggregateRow:
    suite: str
    x_pattern: str
    model: str
    seeds: int
    metrics: dict[str, float | None]
    metric_stds: dict[str, float | None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--output-dir", default="outputs/nonlinear_current_pareto_report")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    paths = [Path(item.strip()) for item in args.metrics.split(",") if item.strip()]
    rows = _aggregate(_load_rows(paths))
    report_rows = _candidate_rows(rows)
    frontier_rows = _frontier_rows(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "candidates.csv", report_rows)
    _write_csv(output_dir / "frontier.csv", frontier_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "candidate_rows": report_rows,
                "frontier_rows": frontier_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(
        _render_report(rows, report_rows, frontier_rows, top_k=args.top_k),
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
                normalized = _normalize_row(row)
                normalized["suite"] = path.parent.name
                rows.append(normalized)
    if not rows:
        raise ValueError("No rows found")
    return rows


def _normalize_row(row: dict[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "x_pattern": row["x_pattern"],
        "model": row.get("model") or row.get("state_model") or row.get("name", "unknown"),
    }
    for metric in METRICS:
        normalized[metric] = _optional_float(row.get(metric))
    return normalized


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _aggregate(rows: list[dict[str, Any]]) -> list[AggregateRow]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["x_pattern"], row["model"])].append(row)

    aggregates = []
    for (suite, x_pattern, model), group_rows in sorted(grouped.items()):
        metrics: dict[str, float | None] = {}
        metric_stds: dict[str, float | None] = {}
        for metric in METRICS:
            values = [row[metric] for row in group_rows if row[metric] is not None]
            metrics[metric] = mean(values) if values else None
            metric_stds[metric] = pstdev(values) if len(values) > 1 else None
        aggregates.append(
            AggregateRow(
                suite=suite,
                x_pattern=x_pattern,
                model=model,
                seeds=len(group_rows),
                metrics=metrics,
                metric_stds=metric_stds,
            )
        )
    return aggregates


def _candidate_rows(rows: list[AggregateRow]) -> list[dict[str, Any]]:
    report_rows = []
    for pattern in sorted({row.x_pattern for row in rows}):
        pattern_rows = [row for row in rows if row.x_pattern == pattern]
        state = _best(pattern_rows, "state_nll")
        predictive = _best(pattern_rows, "predictive_y_nll")
        if state is not None:
            report_rows.append(_candidate_row("state-density", state))
        if predictive is not None and predictive != state:
            report_rows.append(_candidate_row("predictive-y", predictive))
    return report_rows


def _frontier_rows(rows: list[AggregateRow]) -> list[dict[str, Any]]:
    report_rows = []
    for pattern in sorted({row.x_pattern for row in rows}):
        pattern_rows = [
            row
            for row in rows
            if row.x_pattern == pattern
            and row.metrics["state_nll"] is not None
            and row.metrics["predictive_y_nll"] is not None
        ]
        for row in pattern_rows:
            if not _is_dominated(row, pattern_rows):
                report_rows.append(_candidate_row("pareto", row))
    return sorted(
        report_rows,
        key=lambda row: (
            row["x_pattern"],
            _sort_value(row["predictive_y_nll"]),
            _sort_value(row["state_nll"]),
        ),
    )


def _best(rows: list[AggregateRow], metric: str) -> AggregateRow | None:
    candidates = [row for row in rows if row.metrics[metric] is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row.metrics[metric] or float("inf"))


def _is_dominated(row: AggregateRow, rows: list[AggregateRow]) -> bool:
    state = row.metrics["state_nll"]
    pred = row.metrics["predictive_y_nll"]
    if state is None or pred is None:
        return True
    for other in rows:
        other_state = other.metrics["state_nll"]
        other_pred = other.metrics["predictive_y_nll"]
        if other_state is None or other_pred is None:
            continue
        at_least_as_good = other_state <= state and other_pred <= pred
        strictly_better = other_state < state or other_pred < pred
        if at_least_as_good and strictly_better:
            return True
    return False


def _candidate_row(role: str, row: AggregateRow) -> dict[str, Any]:
    return {
        "role": role,
        "x_pattern": row.x_pattern,
        "suite": row.suite,
        "model": row.model,
        "seeds": row.seeds,
        "state_nll": row.metrics["state_nll"],
        "predictive_y_nll": row.metrics["predictive_y_nll"],
        "coverage_90": row.metrics["coverage_90"],
        "variance_ratio": row.metrics["variance_ratio"],
        "eval_fivo_mean_ess": row.metrics["eval_fivo_mean_ess"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(
    rows: list[AggregateRow],
    candidate_rows: list[dict[str, Any]],
    frontier_rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> str:
    lines = [
        "# Nonlinear Current Pareto Report",
        "",
        "This report compares current deterministic and learned nonlinear suites on state density and pre-update predictive-y likelihood.",
        "",
        "## Candidate Rows",
        "",
        "| Pattern | Role | suite | model | state NLL | pred-y NLL | cov 90 | var ratio | ESS |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_rows:
        lines.append(_format_row(row))

    lines.extend(
        [
            "",
            "## Pareto Frontier",
            "",
            "| Pattern | Role | suite | model | state NLL | pred-y NLL | cov 90 | var ratio | ESS |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in frontier_rows:
        lines.append(_format_row(row))

    lines.extend(
        [
            "",
            f"## Top {top_k} By Predictive-Y",
            "",
        ]
    )
    for pattern in sorted({row.x_pattern for row in rows}):
        ranked = _ranked(rows, pattern, "predictive_y_nll")[:top_k]
        lines.extend(
            [
                f"### {pattern}",
                "",
                "| rank | suite | model | pred-y NLL | state NLL | cov 90 | var ratio | ESS |",
                "|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for index, row in enumerate(ranked, start=1):
            lines.append(
                "| {rank} | {suite} | {model} | {pred} | {state} | {cov} | {var} | {ess} |".format(
                    rank=index,
                    suite=row.suite,
                    model=row.model,
                    pred=_fmt(row.metrics["predictive_y_nll"]),
                    state=_fmt(row.metrics["state_nll"]),
                    cov=_fmt(row.metrics["coverage_90"]),
                    var=_fmt(row.metrics["variance_ratio"]),
                    ess=_fmt(row.metrics["eval_fivo_mean_ess"]),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Deterministic quadrature ADF/Power-EP remains the strongest predictive-y family across nonzero observation regimes.",
            "- Learned FIVO, auxiliary FIVO, and the fixed-lag twist rows do not currently dominate the quadrature rows on predictive-y NLL.",
            "- The next implementation branch should prioritize distilling the quadrature/alias filter into the learned strict filter rather than adding another FIVO/twist variant.",
            "",
        ]
    )
    return "\n".join(lines)


def _ranked(rows: list[AggregateRow], pattern: str, metric: str) -> list[AggregateRow]:
    candidates = [
        row for row in rows if row.x_pattern == pattern and row.metrics[metric] is not None
    ]
    return sorted(candidates, key=lambda row: _sort_value(row.metrics[metric]))


def _format_row(row: dict[str, Any]) -> str:
    return (
        "| {x_pattern} | {role} | {suite} | {model} | {state} | {pred} | {cov} | {var} | {ess} |"
    ).format(
        x_pattern=row["x_pattern"],
        role=row["role"],
        suite=row["suite"],
        model=row["model"],
        state=_fmt(row["state_nll"]),
        pred=_fmt(row["predictive_y_nll"]),
        cov=_fmt(row["coverage_90"]),
        var=_fmt(row["variance_ratio"]),
        ess=_fmt(row["eval_fivo_mean_ess"]),
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _sort_value(value: float | None) -> float:
    return value if value is not None else float("inf")


if __name__ == "__main__":
    main()
