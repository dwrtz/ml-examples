"""Aggregate and plot nonlinear learned-filter sweep metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PLOT_METRICS = (
    ("variance_ratio", "variance ratio", 1.0),
    ("coverage_90", "90% coverage", None),
    ("state_nll", "state NLL", None),
)

CALIBRATION_LABELS = {
    "EKF-residualized nonlinear MC ELBO": "baseline",
    "EKF-residualized nonlinear MC ELBO (resampled batches)": "resampled",
    "EKF-residualized nonlinear windowed ELBO h1": "joint-h1",
    "EKF-residualized nonlinear windowed ELBO h2": "joint-h2",
    "EKF-residualized nonlinear windowed ELBO h4": "joint-h4",
    "EKF-residualized nonlinear MC ELBO + joint h4 and predictive-y": "joint-h4+pred-y",
    "EKF-residualized nonlinear windowed ELBO h8": "joint-h8",
    "direct nonlinear windowed ELBO h4": "direct-joint-h4",
    "EKF-residualized nonlinear MLP + reference moment distillation": "moment-distill",
    "direct nonlinear MLP + reference moment distillation": "direct-distill",
    "EKF-residualized nonlinear MC ELBO + predictive-y auxiliary": "pred-y",
    "direct nonlinear MC ELBO + predictive-y auxiliary": "direct-pred-y",
    "EKF-residualized nonlinear MC ELBO + masked-y updates": "masked-y",
    "EKF-residualized nonlinear MC ELBO + masked-y spans h2": "masked-h2",
    "EKF-residualized nonlinear MC ELBO + masked-y spans h4": "masked-h4",
    "EKF-residualized nonlinear MC ELBO + predictive-y and masked-y spans h4": (
        "pred-y+masked-h4"
    ),
    "EKF-residualized nonlinear MC ELBO + joint h4, predictive-y, and masked-y spans h4": (
        "joint-h4+pred-y+masked-h4"
    ),
    "EKF-residualized nonlinear MC ELBO + joint h4 w0.05, predictive-y, and masked-y spans h4": (
        "joint-h4-w0.05+pred-y+masked-h4"
    ),
    "EKF-residualized nonlinear MC ELBO + joint h4 w0.1, predictive-y, and masked-y spans h4": (
        "joint-h4-w0.1+pred-y+masked-h4"
    ),
    "EKF-residualized nonlinear MC ELBO + joint h4 w0.3, predictive-y, and masked-y spans h4": (
        "joint-h4-w0.3+pred-y+masked-h4"
    ),
    "EKF-residualized nonlinear MC ELBO + masked-y spans h8": "masked-h8",
    "direct nonlinear MLP + reference moment and variance-ratio calibration": "direct-cal",
    "direct nonlinear MLP + reference moment and time variance calibration": "direct-time",
    "direct nonlinear MLP + reference moment and low-observation calibration": "direct-low",
    "EKF-residualized nonlinear MLP + teacher-forced reference moment distillation": ("moment-tf"),
    "direct nonlinear MLP + teacher-forced reference moment distillation": "direct-tf",
    "EKF-residualized nonlinear MLP + h2 reference rollout distillation": "rollout-h2",
    "EKF-residualized nonlinear MLP + h4 reference rollout distillation": "rollout-h4",
    "EKF-residualized nonlinear MLP + h8 reference rollout distillation": "rollout-h8",
    "EKF-residualized nonlinear MC ELBO + reference variance calibration": "global",
    "EKF-residualized nonlinear MC ELBO + reference time variance calibration": "time",
    "EKF-residualized nonlinear MC ELBO + reference log-variance calibration": "log-var",
    "EKF-residualized nonlinear MC ELBO + reference low-observation calibration": "low-obs",
    "direct nonlinear MC ELBO": "direct",
    "direct nonlinear MC ELBO + reference variance calibration": "direct-global",
    "direct nonlinear K2 mixture local ADF projection": "direct-k2-adf",
    "direct nonlinear K2 mixture local ADF projection w0.1": "direct-k2-adf-w0.1",
    "direct nonlinear K2 mixture local ADF projection w0.3": "direct-k2-adf-w0.3",
    "direct nonlinear K2 mixture local ADF projection beta 0.3": "direct-k2-adf-b0.3",
    "direct nonlinear K4 mixture local ADF projection beta 0.3": "direct-k4-adf-b0.3",
    "direct nonlinear K4 mixture local ADF projection beta 0.3 spread 2pi": (
        "direct-k4-adf-b0.3-spread-2pi"
    ),
    "direct nonlinear K4 mixture local ADF projection beta 0.3 spread 2pi + predictive-y w0.05": (
        "direct-k4-adf-b0.3-spread-2pi-predy-w0.05"
    ),
    "direct nonlinear K4 mixture local ADF projection beta 0.3 spread 2pi + late predictive-y w0.1": (
        "direct-k4-adf-b0.3-spread-2pi-predy-w0.1-late"
    ),
    "direct nonlinear K4 mixture local ADF projection beta 0.3 spread 2pi + late pre-update predictive w0.1": (
        "direct-k4-adf-b0.3-spread-2pi-preupdate-w0.1-late"
    ),
    "direct nonlinear K2 mixture local ADF projection beta 0.5": "direct-k2-adf-b0.5",
    "direct nonlinear K2 mixture local ADF projection beta 0.7": "direct-k2-adf-b0.7",
    "direct nonlinear K2 mixture local alpha 0.5": "direct-k2-alpha-0.5",
    "direct nonlinear K2 mixture local alpha 0.7": "direct-k2-alpha-0.7",
    "direct nonlinear K2 mixture FIVO n16": "direct-k2-fivo-n16",
    "direct nonlinear K2 mixture FIVO n32": "direct-k2-fivo-n32",
    "direct nonlinear K2 mixture FIVO bridge n16": "direct-k2-fivo-bridge-n16",
    "direct nonlinear K2 mixture FIVO bridge n32": "direct-k2-fivo-bridge-n32",
    "direct nonlinear K2 mixture IWAE h4 k16 + local ADF projection": (
        "direct-k2-iwae-k16+adf"
    ),
    "direct nonlinear K2 mixture IWAE h4 k16 + local ADF projection w0.1": (
        "direct-k2-iwae-k16+adf-w0.1"
    ),
    "direct nonlinear K2 mixture IWAE h4 k16 + local ADF projection w0.3": (
        "direct-k2-iwae-k16+adf-w0.3"
    ),
    "direct nonlinear local ADF projection": "direct-adf",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Comma-separated metrics.csv paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-metrics", default=None)
    parser.add_argument(
        "--patterns", default=None, help="Optional comma-separated x_pattern filter"
    )
    parser.add_argument(
        "--weights", default=None, help="Optional comma-separated labels for --metrics"
    )
    args = parser.parse_args()

    metric_paths = _parse_paths(args.metrics)
    weights = _parse_labels(args.weights, expected=len(metric_paths))
    patterns = set(_parse_values(args.patterns)) if args.patterns else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    has_baseline_metrics = args.baseline_metrics is not None
    if has_baseline_metrics:
        rows.extend(
            _load_rows(
                Path(args.baseline_metrics),
                weight="0",
                patterns=patterns,
                baseline_only=True,
                include_baseline=True,
            )
        )
    for path, weight in zip(metric_paths, weights):
        rows.extend(
            _load_rows(
                path,
                weight=weight,
                patterns=patterns,
                baseline_only=False,
                include_baseline=not has_baseline_metrics,
            )
        )
    if not rows:
        raise ValueError("No rows matched the requested nonlinear sweep inputs")

    _write_csv(output_dir / "aggregate_metrics.csv", rows)
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_render_summary(rows), encoding="utf-8")
    _plot(rows, output_dir / "sweep_comparison.png")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'sweep_comparison.png'}")


def _parse_paths(value: str) -> list[Path]:
    paths = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not paths:
        raise ValueError("--metrics must include at least one path")
    return paths


def _parse_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_labels(value: str | None, *, expected: int) -> list[str]:
    if value is None:
        return [str(index + 1) for index in range(expected)]
    labels = _parse_values(value)
    if len(labels) != expected:
        raise ValueError("--weights must have the same number of entries as --metrics")
    return labels


def _load_rows(
    path: Path,
    *,
    weight: str,
    patterns: set[str] | None,
    baseline_only: bool,
    include_baseline: bool,
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            if patterns is not None and row["x_pattern"] not in patterns:
                continue
            calibration = CALIBRATION_LABELS.get(row["model"], row["model"])
            if baseline_only and calibration != "baseline":
                continue
            if not include_baseline and calibration == "baseline":
                continue
            rows.append({**row, "calibration": calibration, "weight": weight})
        return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "x_pattern",
        "calibration",
        "weight",
        "state_nll",
        "coverage_90",
        "variance_ratio",
        "predictive_nll",
        "reference_state_nll",
        "reference_coverage_90",
        "reference_predictive_nll",
        "name",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def _render_summary(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Nonlinear Sweep Comparison",
        "",
        "| pattern | calibration | weight | state NLL | coverage 90 | variance ratio | pred NLL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in _sorted_rows(rows):
        lines.append(
            "| {x_pattern} | {calibration} | {weight} | {state_nll:.6f} | "
            "{coverage_90:.6f} | {variance_ratio:.6f} | {predictive_nll:.6f} |".format(
                **_typed_row(row)
            )
        )
    lines.append("")
    return "\n".join(lines)


def _plot(rows: list[dict[str, str]], path: Path) -> None:
    patterns = sorted({row["x_pattern"] for row in rows})
    series = _series(rows)
    x = np.arange(len(patterns))
    width = min(0.12, 0.8 / max(len(series), 1))

    fig, axes = plt.subplots(len(PLOT_METRICS), 1, figsize=(12, 11), constrained_layout=True)
    for axis, (metric, title, target) in zip(axes, PLOT_METRICS):
        for index, key in enumerate(series):
            values = [
                _metric_value(rows, pattern=pattern, series_key=key, metric=metric)
                for pattern in patterns
            ]
            axis.bar(
                x + (index - (len(series) - 1) / 2) * width,
                values,
                width=width,
                label=_series_label(key),
                alpha=0.86,
            )
        if target is not None:
            axis.axhline(target, color="black", linestyle="--", linewidth=1)
        if metric == "variance_ratio":
            axis.set_yscale("log")
        if metric == "coverage_90":
            reference = [_reference_coverage(rows, pattern) for pattern in patterns]
            axis.plot(x, reference, color="black", marker="o", linewidth=1.2, label="grid ref")
        axis.set_title(title)
        axis.set_xticks(x, [_short_pattern(pattern) for pattern in patterns])
        axis.grid(True, axis="y", alpha=0.3)
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.45))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _series(rows: Iterable[dict[str, str]]) -> list[tuple[str, str]]:
    keys = {(row["calibration"], row["weight"]) for row in rows}
    return sorted(keys, key=lambda key: (_series_rank(key[0]), _weight_rank(key[1])))


def _series_rank(calibration: str) -> int:
    order = {
        "baseline": 0,
        "resampled": 1,
        "joint-h1": 2,
        "joint-h2": 3,
        "joint-h4": 4,
        "joint-h4+pred-y": 5,
        "joint-h8": 6,
        "direct-joint-h4": 7,
        "moment-distill": 8,
        "direct-distill": 9,
        "pred-y": 10,
        "direct-pred-y": 11,
        "masked-y": 12,
        "masked-h2": 13,
        "masked-h4": 14,
        "pred-y+masked-h4": 15,
        "joint-h4+pred-y+masked-h4": 16,
        "joint-h4-w0.05+pred-y+masked-h4": 17,
        "joint-h4-w0.1+pred-y+masked-h4": 18,
        "joint-h4-w0.3+pred-y+masked-h4": 19,
        "masked-h8": 20,
        "direct-cal": 21,
        "direct-time": 22,
        "direct-low": 23,
        "moment-tf": 24,
        "direct-tf": 25,
        "rollout-h2": 26,
        "rollout-h4": 27,
        "rollout-h8": 28,
        "global": 29,
        "time": 30,
        "log-var": 31,
        "low-obs": 32,
    }
    return order.get(calibration, 100)


def _weight_rank(weight: str) -> float:
    named_order = {
        "self-fed": 0.0,
        "teacher": 1.0,
        "rollout": 2.0,
        "rollout250": 2.5,
        "rollout1000": 3.0,
    }
    if weight in named_order:
        return named_order[weight]
    try:
        return float(weight)
    except ValueError:
        return float("inf")


def _series_label(key: tuple[str, str]) -> str:
    calibration, weight = key
    if calibration == "baseline":
        return "baseline"
    return f"{calibration} {weight}"


def _metric_value(
    rows: list[dict[str, str]],
    *,
    pattern: str,
    series_key: tuple[str, str],
    metric: str,
) -> float:
    calibration, weight = series_key
    values = [
        float(item[metric])
        for item in rows
        if item["x_pattern"] == pattern
        and item["calibration"] == calibration
        and item["weight"] == weight
    ]
    return float(np.mean(values))


def _reference_coverage(rows: list[dict[str, str]], pattern: str) -> float:
    values = [
        float(item["reference_coverage_90"]) for item in rows if item["x_pattern"] == pattern
    ]
    return float(np.mean(values))


def _short_pattern(pattern: str) -> str:
    return pattern.removesuffix("_sinusoidal")


def _sorted_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            row["x_pattern"],
            _series_rank(row["calibration"]),
            _weight_rank(row["weight"]),
        ),
    )


def _typed_row(row: dict[str, str]) -> dict[str, str | float]:
    typed: dict[str, str | float] = dict(row)
    for key in ("state_nll", "coverage_90", "variance_ratio", "predictive_nll"):
        typed[key] = float(row[key])
    return typed


if __name__ == "__main__":
    main()
