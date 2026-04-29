"""Build the nonlinear unsupervised objective final aggregation report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_INPUTS = [
    "outputs/nonlinear_unsupervised_predictive_y_pilot_1000/metrics.csv",
    "outputs/nonlinear_unsupervised_masked_y_pilot_1000/metrics.csv",
    "outputs/nonlinear_unsupervised_joint_elbo_pilot_1000/metrics.csv",
    "outputs/nonlinear_unsupervised_joint_predictive_masked_y_pilot_1000/metrics.csv",
    "outputs/nonlinear_unsupervised_joint_weight_sweep_1000/metrics.csv",
    "outputs/nonlinear_unsupervised_objective_robustness_full_1000/metrics.csv",
    "outputs/nonlinear_head_seed_sweep_1000/metrics.csv",
]

METRICS = [
    "state_nll",
    "coverage_90",
    "variance_ratio",
    "predictive_nll",
    "predictive_y_nll",
]

PROMOTED_MODEL = (
    "EKF-residualized nonlinear MC ELBO + joint h4 w0.05, predictive-y, and masked-y spans h4"
)
STRUCTURED_BASELINE = "EKF-residualized nonlinear MC ELBO"
DIRECT_BASELINE = "direct nonlinear MC ELBO"
DIRECT_DISTILL = "direct nonlinear MLP + reference moment distillation"
ROLLOUT_DISTILL = "EKF-residualized nonlinear MLP + h4 reference rollout distillation"
PREDICTIVE_Y_REGRESSION_TOLERANCE = 0.03

MODEL_LABELS = {
    STRUCTURED_BASELINE: "structured ELBO",
    DIRECT_BASELINE: "direct ELBO",
    PROMOTED_MODEL: "joint h4 w0.05 + predictive-y + masked-y h4",
    DIRECT_DISTILL: "direct reference moment distillation",
    ROLLOUT_DISTILL: "structured h4 reference rollout distillation",
    "direct nonlinear K2 mixture local ADF projection": "direct K2 local ADF projection",
    "direct nonlinear K2 mixture local ADF projection w0.1": "direct K2 local ADF w0.1",
    "direct nonlinear K2 mixture local ADF projection w0.3": "direct K2 local ADF w0.3",
    "direct nonlinear K2 mixture local ADF projection beta 0.3": "direct K2 local ADF beta 0.3",
    "direct nonlinear K2 mixture local ADF projection beta 0.5": "direct K2 local ADF beta 0.5",
    "direct nonlinear K2 mixture local ADF projection beta 0.7": "direct K2 local ADF beta 0.7",
    "direct nonlinear K2 mixture FIVO n16": "direct K2 FIVO n16",
    "direct nonlinear K2 mixture FIVO n32": "direct K2 FIVO n32",
    "direct nonlinear K2 mixture IWAE h4 k16 + local ADF projection": (
        "direct K2 IWAE h4 k16 + local ADF projection"
    ),
    "direct nonlinear K2 mixture IWAE h4 k16 + local ADF projection w0.1": (
        "direct K2 IWAE h4 k16 + local ADF w0.1"
    ),
    "direct nonlinear K2 mixture IWAE h4 k16 + local ADF projection w0.3": (
        "direct K2 IWAE h4 k16 + local ADF w0.3"
    ),
    "direct nonlinear local ADF projection": "direct local ADF projection",
}


@dataclass(frozen=True)
class AggregateRow:
    suite: str
    x_pattern: str
    training_signal: str
    model: str
    steps: int
    seeds: int
    metrics: dict[str, float]
    metric_stds: dict[str, float]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_INPUTS),
        help="Comma-separated metrics.csv paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/nonlinear_unsupervised_objective_final_report",
    )
    args = parser.parse_args()

    paths = [Path(item) for item in args.metrics.split(",") if item]
    rows = _load_rows(paths)
    aggregate_rows = _aggregate(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    md_path = output_dir / "summary.md"
    _write_summary_csv(csv_path, aggregate_rows)
    json_path.write_text(
        json.dumps(_json_rows(aggregate_rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_render_report(aggregate_rows, paths), encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        suite = path.parent.name
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                normalized = dict(row)
                normalized["suite"] = suite
                normalized.setdefault("training_signal", _infer_training_signal(normalized))
                if "predictive_y_nll" not in normalized:
                    normalized["predictive_y_nll"] = normalized.get("predictive_nll", "")
                rows.append(normalized)
    if not rows:
        formatted = ", ".join(str(path) for path in paths)
        raise FileNotFoundError(f"No metrics rows found in: {formatted}")
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[AggregateRow]:
    groups: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["suite"]),
            str(row["x_pattern"]),
            str(row["training_signal"]),
            str(row["model"]),
            int(float(row["steps"])),
        )
        groups[key].append(row)

    aggregates = []
    for (suite, x_pattern, signal, model, steps), group in groups.items():
        values = {
            metric: [float(row[metric]) for row in group if row.get(metric) not in {"", None}]
            for metric in METRICS
        }
        aggregates.append(
            AggregateRow(
                suite=suite,
                x_pattern=x_pattern,
                training_signal=signal,
                model=model,
                steps=steps,
                seeds=len({str(row["seed"]) for row in group}),
                metrics={metric: mean(vals) for metric, vals in values.items() if vals},
                metric_stds={metric: pstdev(vals) for metric, vals in values.items() if vals},
            )
        )
    aggregates.sort(
        key=lambda row: (
            row.suite,
            row.x_pattern,
            row.training_signal,
            _model_label(row.model),
            row.steps,
        )
    )
    return aggregates


def _infer_training_signal(row: dict[str, Any]) -> str:
    model = str(row.get("model", ""))
    if "reference" in model:
        return "reference_distilled"
    reference_weights = [
        "reference_mean_weight",
        "reference_rollout_weight",
        "reference_variance_ratio_weight",
        "reference_time_variance_ratio_weight",
        "reference_log_variance_weight",
        "reference_low_observation_variance_ratio_weight",
    ]
    if any(float(row.get(key, 0.0) or 0.0) > 0.0 for key in reference_weights):
        return "reference_distilled"
    return "unsupervised"


def _write_summary_csv(path: Path, rows: list[AggregateRow]) -> None:
    fieldnames = [
        "suite",
        "x_pattern",
        "training_signal",
        "model",
        "model_label",
        "steps",
        "seeds",
    ]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output: dict[str, Any] = {
                "suite": row.suite,
                "x_pattern": row.x_pattern,
                "training_signal": row.training_signal,
                "model": row.model,
                "model_label": _model_label(row.model),
                "steps": row.steps,
                "seeds": row.seeds,
            }
            for metric in METRICS:
                output[f"{metric}_mean"] = row.metrics.get(metric)
                output[f"{metric}_std"] = row.metric_stds.get(metric)
            writer.writerow(output)


def _json_rows(rows: list[AggregateRow]) -> list[dict[str, Any]]:
    return [
        {
            "suite": row.suite,
            "x_pattern": row.x_pattern,
            "training_signal": row.training_signal,
            "model": row.model,
            "model_label": _model_label(row.model),
            "steps": row.steps,
            "seeds": row.seeds,
            "metrics": row.metrics,
            "metric_stds": row.metric_stds,
        }
        for row in rows
    ]


def _render_report(rows: list[AggregateRow], input_paths: list[Path]) -> str:
    robustness_rows = [
        row for row in rows if row.suite == "nonlinear_unsupervised_objective_robustness_full_1000"
    ]
    pilot_rows = [
        row
        for row in rows
        if row.training_signal == "unsupervised"
        and row.suite != "nonlinear_unsupervised_objective_robustness_full_1000"
    ]
    diagnostic_rows = [row for row in robustness_rows if row.training_signal != "unsupervised"]

    lines = [
        "# Nonlinear Unsupervised Objective Final Report",
        "",
        "## Executive Summary",
        "",
        "- The best fully unsupervised nonlinear row is `structured_joint_elbo_h4_w005_predictive_y_masked_y_spans_h4`.",
        "- It improves degraded-observation robustness relative to vanilla structured ELBO on weak, intermittent, zero, and random-normal stressors.",
        "- It is not a solved nonlinear filter: clean sinusoidal performance is slightly worse than structured ELBO, and absolute coverage/variance calibration remains weak.",
        "- Reference-distilled rows remain much stronger and should be reported only as upper-bound diagnostics, not unsupervised results.",
        "",
        "## Final Claim",
        "",
        "A combined short-horizon joint ELBO, causal predictive-y objective, and masked-y span training objective materially reduces the nonlinear strict-filter failure under weak, intermittent, and non-informative observations. The remaining gap to reference-distilled controls indicates that the next research step should focus on objective/divergence design or posterior expressivity, not more local ELBO tuning.",
        "",
        "## Robustness Suite",
        "",
        "| Pattern | Row | signal | state NLL | cov 90 | var ratio | pred-y NLL | pred NLL |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in _selected_robustness_rows(robustness_rows):
        lines.append(_metric_table_row(row))

    lines.extend(
        [
            "",
            "## Fully Unsupervised Delta",
            "",
            "| Pattern | Candidate NLL delta | Candidate cov delta | Candidate var-ratio delta | Interpretation |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for pattern in sorted({row.x_pattern for row in robustness_rows}):
        baseline = _find_row(robustness_rows, pattern, STRUCTURED_BASELINE)
        candidate = _find_row(robustness_rows, pattern, PROMOTED_MODEL)
        if baseline is None or candidate is None:
            continue
        nll_delta = candidate.metrics["state_nll"] - baseline.metrics["state_nll"]
        cov_delta = candidate.metrics["coverage_90"] - baseline.metrics["coverage_90"]
        var_delta = candidate.metrics["variance_ratio"] - baseline.metrics["variance_ratio"]
        lines.append(
            "| {pattern} | {nll_delta:.3f} | {cov_delta:.3f} | {var_delta:.3f} | {interp} |".format(
                pattern=pattern,
                nll_delta=nll_delta,
                cov_delta=cov_delta,
                var_delta=var_delta,
                interp=_delta_interpretation(nll_delta, cov_delta, var_delta),
            )
        )

    lines.extend(
        [
            "",
            "## Objective Variants Tested",
            "",
            "| Suite | Pattern | Row | state NLL | cov 90 | var ratio | pred-y NLL |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in _selected_pilot_rows(pilot_rows):
        lines.append(
            "| {suite} | {pattern} | {model} | {nll:.3f} | {cov:.3f} | {var:.3f} | {pred_y:.3f} |".format(
                suite=row.suite,
                pattern=row.x_pattern,
                model=_model_label(row.model),
                nll=row.metrics["state_nll"],
                cov=row.metrics["coverage_90"],
                var=row.metrics["variance_ratio"],
                pred_y=row.metrics.get("predictive_y_nll", row.metrics.get("predictive_nll")),
            )
        )

    lines.extend(
        [
            "",
            "## Predictive-Y Promotion Gate",
            "",
            f"Promotable rows must keep predictive-y NLL within {PREDICTIVE_Y_REGRESSION_TOLERANCE:.2f} of the current promoted baseline for the same pattern.",
            "",
            "| Pattern | state-density candidate | predictive-y candidate | promotable candidate | baseline pred-y NLL |",
            "|---|---|---|---|---:|",
        ]
    )
    for pattern in sorted({row.x_pattern for row in pilot_rows + robustness_rows}):
        baseline = _find_row(rows, pattern, PROMOTED_MODEL)
        baseline_pred_y = (
            baseline.metrics.get("predictive_y_nll", baseline.metrics.get("predictive_nll"))
            if baseline is not None
            else None
        )
        candidates = [
            row
            for row in rows
            if row.x_pattern == pattern and row.training_signal == "unsupervised"
        ]
        state_candidate = _best_row(candidates, "state_nll")
        predictive_candidate = _best_row(candidates, "predictive_y_nll")
        promotable = (
            _best_row(
                [
                    row
                    for row in candidates
                    if row.metrics.get("predictive_y_nll", row.metrics.get("predictive_nll"))
                    <= baseline_pred_y + PREDICTIVE_Y_REGRESSION_TOLERANCE
                ],
                "state_nll",
            )
            if baseline_pred_y is not None
            else None
        )
        lines.append(
            "| {pattern} | {state} | {pred_y} | {promotable} | {baseline} |".format(
                pattern=pattern,
                state=_candidate_label(state_candidate),
                pred_y=_candidate_label(predictive_candidate),
                promotable=_candidate_label(promotable),
                baseline="" if baseline_pred_y is None else f"{baseline_pred_y:.3f}",
            )
        )

    lines.extend(
        [
            "",
            "## Reference-Distilled Diagnostics",
            "",
            "| Pattern | Diagnostic | state NLL | cov 90 | var ratio |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in _selected_diagnostic_rows(diagnostic_rows):
        lines.append(
            "| {pattern} | {model} | {nll:.3f} | {cov:.3f} | {var:.3f} |".format(
                pattern=row.x_pattern,
                model=_model_label(row.model),
                nll=row.metrics["state_nll"],
                cov=row.metrics["coverage_90"],
                var=row.metrics["variance_ratio"],
            )
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Continue the unsupervised program only as a targeted objective/divergence branch. The current combined objective is the best fully unsupervised robustness baseline, but it does not meet the original calibration gate. The strongest next candidates are multi-sample/IWAE-style objectives, alpha/Renyi objectives, or entropy/calibration terms that remain fully unsupervised.",
            "",
            "## Source Artifacts",
            "",
        ]
    )
    for path in input_paths:
        marker = "" if path.exists() else " (missing when report was generated)"
        lines.append(f"- `{path}`{marker}")
    lines.append("")
    return "\n".join(lines)


def _selected_robustness_rows(rows: list[AggregateRow]) -> list[AggregateRow]:
    selected_models = {STRUCTURED_BASELINE, DIRECT_BASELINE, PROMOTED_MODEL, DIRECT_DISTILL}
    return [
        row
        for row in rows
        if row.model in selected_models
        and row.x_pattern
        in {"sinusoidal", "weak_sinusoidal", "intermittent_sinusoidal", "zero", "random_normal"}
    ]


def _selected_pilot_rows(rows: list[AggregateRow]) -> list[AggregateRow]:
    candidate_rows = [
        row
        for row in rows
        if row.x_pattern in {"weak_sinusoidal", "intermittent_sinusoidal"}
        and row.model
        in {
            STRUCTURED_BASELINE,
            "EKF-residualized nonlinear MC ELBO + predictive-y auxiliary",
            "EKF-residualized nonlinear MC ELBO + masked-y spans h4",
            "EKF-residualized nonlinear MC ELBO + predictive-y and masked-y spans h4",
            "EKF-residualized nonlinear windowed ELBO h4",
            "EKF-residualized nonlinear MC ELBO + joint h4 and predictive-y",
            "EKF-residualized nonlinear MC ELBO + joint h4, predictive-y, and masked-y spans h4",
            "direct nonlinear K2 mixture local ADF projection",
            "direct nonlinear K2 mixture local ADF projection w0.1",
            "direct nonlinear K2 mixture local ADF projection w0.3",
            "direct nonlinear K2 mixture local ADF projection beta 0.3",
            "direct nonlinear K2 mixture local ADF projection beta 0.5",
            "direct nonlinear K2 mixture local ADF projection beta 0.7",
            "direct nonlinear K2 mixture FIVO n16",
            "direct nonlinear K2 mixture FIVO n32",
            "direct nonlinear K2 mixture IWAE h4 k16 + local ADF projection",
            "direct nonlinear K2 mixture IWAE h4 k16 + local ADF projection w0.1",
            "direct nonlinear K2 mixture IWAE h4 k16 + local ADF projection w0.3",
            "direct nonlinear local ADF projection",
            PROMOTED_MODEL,
        }
    ]
    return sorted(
        candidate_rows,
        key=lambda row: (row.suite, row.x_pattern, _model_label(row.model)),
    )


def _selected_diagnostic_rows(rows: list[AggregateRow]) -> list[AggregateRow]:
    return [
        row
        for row in rows
        if row.model in {DIRECT_DISTILL, ROLLOUT_DISTILL}
        and row.x_pattern
        in {"sinusoidal", "weak_sinusoidal", "intermittent_sinusoidal", "zero", "random_normal"}
    ]


def _find_row(rows: list[AggregateRow], pattern: str, model: str) -> AggregateRow | None:
    for row in rows:
        if row.x_pattern == pattern and row.model == model:
            return row
    return None


def _metric_table_row(row: AggregateRow) -> str:
    return (
        "| {pattern} | {model} | {signal} | {nll:.3f} | {cov:.3f} | {var:.3f} | "
        "{pred_y:.3f} | {pred:.3f} |"
    ).format(
        pattern=row.x_pattern,
        model=_model_label(row.model),
        signal=row.training_signal,
        nll=row.metrics["state_nll"],
        cov=row.metrics["coverage_90"],
        var=row.metrics["variance_ratio"],
        pred_y=row.metrics.get("predictive_y_nll", row.metrics.get("predictive_nll")),
        pred=row.metrics["predictive_nll"],
    )


def _best_row(rows: list[AggregateRow], metric: str) -> AggregateRow | None:
    available = [row for row in rows if metric in row.metrics]
    if not available:
        return None
    return min(available, key=lambda row: row.metrics[metric])


def _candidate_label(row: AggregateRow | None) -> str:
    if row is None:
        return ""
    pred_y = row.metrics.get("predictive_y_nll", row.metrics.get("predictive_nll"))
    return "{model} (state {state:.3f}, pred-y {pred_y:.3f})".format(
        model=_model_label(row.model),
        state=row.metrics["state_nll"],
        pred_y=pred_y,
    )


def _delta_interpretation(nll_delta: float, cov_delta: float, var_delta: float) -> str:
    if nll_delta < 0 and cov_delta > 0 and var_delta > 0:
        return "improves robustness"
    if nll_delta <= 0 and (cov_delta > 0 or var_delta > 0):
        return "mixed but useful"
    if nll_delta > 0 and cov_delta < 0:
        return "regresses"
    return "mixed"


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


if __name__ == "__main__":
    main()
