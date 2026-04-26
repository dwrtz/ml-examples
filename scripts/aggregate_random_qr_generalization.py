"""Aggregate canonical randomized-Q/R summaries into one report."""

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
    "oracle_predictive_nll",
)

MODEL_ORDER = {
    "frozen marginal backward MLP": 0,
    "regime-local self-fed supervised": 1,
    "regime-local calibrated MC ELBO": 2,
}

EVAL_ORDER = {
    (0.03, 0.03): 0,
    (0.03, 0.3): 1,
    (0.1, 0.1): 2,
    (0.3, 0.03): 3,
    (0.3, 0.3): 4,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-summary",
        default="outputs/linear_gaussian_random_qr_generalization_full/frozen/summary.json",
    )
    parser.add_argument(
        "--self-fed-summary",
        default="outputs/linear_gaussian_random_qr_calibration_3000_regime_w1/self_fed/summary.json",
    )
    parser.add_argument(
        "--elbo-summary",
        default="outputs/linear_gaussian_random_qr_calibration_3000_regime_w1/elbo/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/linear_gaussian_random_qr_generalization_canonical",
    )
    args = parser.parse_args()

    rows = [
        *_load_rows(Path(args.frozen_summary), model="frozen marginal backward MLP"),
        *_load_rows(Path(args.self_fed_summary), model="regime-local self-fed supervised"),
        *_load_rows(Path(args.elbo_summary), model="regime-local calibrated MC ELBO"),
    ]
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


def _load_rows(path: Path, *, model: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing randomized-Q/R summary: {path}")
    rows = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        normalized = dict(row)
        normalized["model"] = model
        rows.append(normalized)
    return rows


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, float, float, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["model"]),
            float(row["eval_q"]),
            float(row["eval_r"]),
            int(row["steps"]),
        )
        deduped[key] = row
    return list(deduped.values())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "objective",
        "steps",
        "num_seeds",
        "train_q",
        "train_r",
        "eval_q",
        "eval_r",
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
        "# Linear-Gaussian Randomized Q/R Canonical Summary",
        "",
        "| Model | eval Q | eval R | Steps | Seeds | filter KL | edge KL | state NLL | cov 90 | var ratio | pred NLL | oracle pred NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {eval_q:g} | {eval_r:g} | {steps} | {num_seeds} | "
            "{filter_kl_mean:.6f} +/- {filter_kl_std:.6f} | "
            "{edge_kl_mean:.6f} +/- {edge_kl_std:.6f} | "
            "{state_nll_mean:.6f} +/- {state_nll_std:.6f} | "
            "{coverage_90_mean:.6f} +/- {coverage_90_std:.6f} | "
            "{variance_ratio_mean:.6f} +/- {variance_ratio_std:.6f} | "
            "{predictive_nll_mean:.6f} +/- {predictive_nll_std:.6f} | "
            "{oracle_predictive_nll_mean:.6f} +/- {oracle_predictive_nll_std:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Compact Comparison",
            "",
            "| eval Q | eval R | Model | state NLL | cov 90 | var ratio | pred NLL |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {eval_q:g} | {eval_r:g} | {model} | {state_nll_mean:.6f} | "
            "{coverage_90_mean:.6f} | {variance_ratio_mean:.6f} | "
            "{predictive_nll_mean:.6f} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def _sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    eval_key = (round(float(row["eval_q"]), 10), round(float(row["eval_r"]), 10))
    return (
        EVAL_ORDER.get(eval_key, 999),
        MODEL_ORDER.get(str(row["model"]), 999),
        str(row["model"]),
    )


if __name__ == "__main__":
    main()
