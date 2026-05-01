"""Summarize nonlinear grid-reference posterior shape by stressor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vbf.data import LinearGaussianParams
from vbf.nonlinear import (
    GridReferenceConfig,
    NonlinearDataConfig,
    make_nonlinear_batch,
    nonlinear_grid_filter_shape_diagnostics,
)


DEFAULT_CONFIGS = (
    "experiments/nonlinear/01_sine_observation.yaml",
    "experiments/nonlinear/03_weak_sine_observation.yaml",
    "experiments/nonlinear/04_intermittent_sine_observation.yaml",
    "experiments/nonlinear/05_zero_sine_observation.yaml",
    "experiments/nonlinear/06_random_normal_sine_observation.yaml",
)

METRICS = (
    "mean_entropy",
    "mean_normalized_entropy",
    "mean_peak_count",
    "p90_peak_count",
    "max_peak_count",
    "frac_peak_gt_1",
    "frac_peak_gt_2",
    "frac_peak_gt_4",
    "mean_max_mass",
    "mean_credible_width_90",
    "p90_credible_width_90",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--seeds", default="321,322,323")
    parser.add_argument("--peak-fraction", type=float, default=0.1)
    parser.add_argument("--output-dir", default="outputs/nonlinear_reference_shape_report")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = _parse_config_paths(args.configs)
    seeds = _parse_ints(args.seeds, name="--seeds")

    rows = []
    for config_path in config_paths:
        config = _read_config(config_path)
        for seed in seeds:
            rows.append(
                _evaluate_shape(
                    config,
                    config_path=config_path,
                    seed=seed,
                    peak_fraction=args.peak_fraction,
                )
            )

    aggregate_rows = _aggregate_rows(rows)
    _write_csv(output_dir / "metrics.csv", rows)
    _write_csv(output_dir / "summary.csv", aggregate_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(_render_report(aggregate_rows, peak_fraction=args.peak_fraction))
    print(f"Wrote {report_path}")


def _parse_config_paths(value: str) -> list[Path]:
    paths = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not paths:
        raise ValueError("--configs must include at least one path")
    return paths


def _parse_ints(value: str, *, name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc
    if not parsed:
        raise ValueError(f"{name} must include at least one integer")
    return parsed


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _evaluate_shape(
    config: dict[str, Any],
    *,
    config_path: Path,
    seed: int,
    peak_fraction: float,
) -> dict[str, Any]:
    data_config = NonlinearDataConfig(**config["data"])
    state_params = LinearGaussianParams(**config["state_space"])
    grid_config = GridReferenceConfig(**config.get("reference", {}))
    batch = make_nonlinear_batch(data_config, state_params, seed=seed)
    shape = nonlinear_grid_filter_shape_diagnostics(
        batch,
        state_params,
        data_config=data_config,
        grid_config=grid_config,
        peak_fraction=peak_fraction,
    )
    peak_count = np.asarray(shape.peak_count)
    credible_width = np.asarray(shape.credible_width_90)
    entropy = np.asarray(shape.entropy)
    normalized_entropy = np.asarray(shape.normalized_entropy)
    max_mass = np.asarray(shape.max_mass)
    return {
        "config": str(config_path),
        "name": config["name"],
        "seed": seed,
        "x_pattern": data_config.x_pattern,
        "batch_size": data_config.batch_size,
        "time_steps": data_config.time_steps,
        "peak_fraction": peak_fraction,
        "mean_entropy": float(np.mean(entropy)),
        "mean_normalized_entropy": float(np.mean(normalized_entropy)),
        "mean_peak_count": float(np.mean(peak_count)),
        "p90_peak_count": float(np.quantile(peak_count, 0.9)),
        "max_peak_count": float(np.max(peak_count)),
        "frac_peak_gt_1": float(np.mean(peak_count > 1)),
        "frac_peak_gt_2": float(np.mean(peak_count > 2)),
        "frac_peak_gt_4": float(np.mean(peak_count > 4)),
        "mean_max_mass": float(np.mean(max_mass)),
        "mean_credible_width_90": float(np.mean(credible_width)),
        "p90_credible_width_90": float(np.quantile(credible_width, 0.9)),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["x_pattern"]), []).append(row)
    aggregates = []
    for x_pattern, pattern_rows in sorted(grouped.items()):
        aggregate = {
            "x_pattern": x_pattern,
            "seeds": len(pattern_rows),
            "batch_size": pattern_rows[0]["batch_size"],
            "time_steps": pattern_rows[0]["time_steps"],
            "peak_fraction": pattern_rows[0]["peak_fraction"],
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in pattern_rows])
            aggregate[metric] = float(np.mean(values))
            aggregate[f"{metric}_std"] = float(np.std(values))
        aggregates.append(aggregate)
    return aggregates


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(rows: list[dict[str, Any]], *, peak_fraction: float) -> str:
    lines = [
        "# Nonlinear Reference Posterior Shape",
        "",
        f"Peak counts use local maxima with mass at least {peak_fraction:.2f} of the posterior max mass.",
        "",
        "| x pattern | seeds | mean peaks | p90 peaks | max peaks | frac >2 peaks | norm entropy | max mass | width 90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {x_pattern} | {seeds} | {mean_peak_count:.3f} | {p90_peak_count:.3f} | "
            "{max_peak_count:.0f} | {frac_peak_gt_2:.3f} | {mean_normalized_entropy:.3f} | "
            "{mean_max_mass:.4f} | {mean_credible_width_90:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `frac >2 peaks` is the main K2 stress indicator: high values mean the exact grid posterior often has more modes than a two-component Gaussian mixture can represent.",
            "- `width 90` and normalized entropy separate broad uncertainty from genuinely multi-peaked posteriors.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
