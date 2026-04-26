"""Aggregate split weak-observability summaries into one report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = (
    "filter_kl",
    "edge_kl",
    "state_rmse",
    "state_nll",
    "coverage_90",
    "variance_ratio",
    "predictive_nll",
    "closed_form_elbo",
)

PATTERN_ORDER = {
    "sinusoidal_reference": 0,
    "weak_sinusoidal": 1,
    "intermittent_sinusoidal": 2,
    "zero_unobservable": 3,
    "random_normal": 4,
}

MODEL_ORDER = {
    "exact Kalman": 0,
    "zero-init no training": 1,
    "frozen marginal backward MLP": 2,
    "self-fed supervised": 3,
    "self-fed supervised var 0.1": 4,
    "MC ELBO structured": 5,
    "calibrated MC ELBO": 6,
    "direct closed-form ELBO": 7,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weak-root",
        default="outputs/linear_gaussian_weak_observability_split",
        help="Directory containing per-model weak-observability split summaries.",
    )
    parser.add_argument(
        "--calibrated-elbo-summary",
        action="append",
        default=[
            "outputs/linear_gaussian_elbo_calibration_3000_low_observation_w1/summary.json",
            "outputs/linear_gaussian_elbo_calibration_3000_low_observation_w1_full_remaining/summary.json",
        ],
        help="Calibrated ELBO summary.json path. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/linear_gaussian_weak_observability_canonical",
    )
    args = parser.parse_args()

    rows = _load_weak_rows(Path(args.weak_root))
    for path in args.calibrated_elbo_summary:
        rows.extend(_load_calibrated_elbo_rows(Path(path)))

    rows = _dedupe_rows(rows)
    rows.sort(key=_sort_key)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(_render_report(rows), encoding="utf-8")
    print(f"Wrote {report_path}")


def _load_weak_rows(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        raise FileNotFoundError(f"Missing weak-observability root: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/summary.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _load_calibrated_elbo_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing calibrated ELBO summary: {path}")
    rows = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        normalized = dict(row)
        normalized["model"] = "calibrated MC ELBO"
        normalized["objective"] = "elbo_edge_mlp_low_observation_var_1"
        normalized.pop("variant", None)
        rows.append(normalized)
    return rows


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["pattern"]),
            str(row["model"]),
            str(row["objective"]),
            int(row["steps"]),
        )
        deduped[key] = row
    return list(deduped.values())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pattern",
        "model",
        "objective",
        "steps",
        "num_seeds",
        *[f"{metric}_mean" for metric in METRICS],
        *[f"{metric}_std" for metric in METRICS],
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Linear-Gaussian Weak Observability Canonical Summary",
        "",
        "| Pattern | Model | Steps | Seeds | filter KL | edge KL | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {pattern} | {model} | {steps} | {num_seeds} | "
            "{filter_kl_mean:.6f} +/- {filter_kl_std:.6f} | "
            "{edge_kl_mean:.6f} +/- {edge_kl_std:.6f} | "
            "{state_nll_mean:.6f} +/- {state_nll_std:.6f} | "
            "{coverage_90_mean:.6f} +/- {coverage_90_std:.6f} | "
            "{variance_ratio_mean:.6f} +/- {variance_ratio_std:.6f} | "
            "{predictive_nll_mean:.6f} +/- {predictive_nll_std:.6f} | "
            "{closed_form_elbo_mean:.6f} +/- {closed_form_elbo_std:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Compact Comparison",
            "",
            "| Pattern | Model | state NLL | cov 90 | var ratio | pred NLL |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    compact_models = {
        "exact Kalman",
        "frozen marginal backward MLP",
        "self-fed supervised var 0.1",
        "MC ELBO structured",
        "calibrated MC ELBO",
    }
    for row in rows:
        if row["model"] not in compact_models:
            continue
        lines.append(
            "| {pattern} | {model} | {state_nll_mean:.6f} | {coverage_90_mean:.6f} | "
            "{variance_ratio_mean:.6f} | {predictive_nll_mean:.6f} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def _sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        PATTERN_ORDER.get(str(row["pattern"]), 999),
        MODEL_ORDER.get(str(row["model"]), 999),
        str(row["model"]),
    )


if __name__ == "__main__":
    main()
