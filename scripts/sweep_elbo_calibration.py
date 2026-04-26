"""Sweep ELBO variance-calibration penalties on weak-observability regimes."""

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
    pattern: str
    seed: int
    variant: str
    steps: int
    filter_kl: float
    edge_kl: float
    state_rmse: float
    state_nll: float
    coverage_90: float
    variance_ratio: float
    predictive_nll: float
    closed_form_elbo: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/linear_gaussian/02_elbo_edge_mlp.yaml")
    parser.add_argument(
        "--suite-config",
        default="experiments/linear_gaussian/08_weak_observability.yaml",
    )
    parser.add_argument(
        "--patterns",
        default="weak_sinusoidal,intermittent_sinusoidal,zero_unobservable",
    )
    parser.add_argument("--seeds", default="321,322,323,324,325")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--num-elbo-samples", type=int, default=32)
    parser.add_argument("--weights", default="0,0.1,1")
    parser.add_argument("--penalties", default="time,low_observation")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--output-dir", default="outputs/linear_gaussian_elbo_calibration")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    base_config = _read_config(Path(args.config))
    if base_config["model"] != "elbo_edge_mlp":
        raise ValueError("ELBO calibration sweep requires model: elbo_edge_mlp")

    suite_config = _read_config(Path(args.suite_config))
    selected_patterns = set(_parse_strings(args.patterns, name="--patterns"))
    patterns = [
        pattern
        for pattern in suite_config["patterns"]
        if str(pattern["name"]) in selected_patterns
    ]
    if len(patterns) != len(selected_patterns):
        available = sorted(str(pattern["name"]) for pattern in suite_config["patterns"])
        raise ValueError(f"Selected patterns must be among: {', '.join(available)}")

    seeds = _parse_ints(args.seeds, name="--seeds")
    weights = _parse_floats(args.weights, name="--weights")
    penalties = _parse_strings(args.penalties, name="--penalties")
    unknown = sorted(set(penalties) - {"global", "time", "low_observation"})
    if unknown:
        raise ValueError(f"Unknown penalties: {', '.join(unknown)}")

    variants = [] if args.no_baseline else [("baseline", {})]
    for penalty in penalties:
        for weight in weights:
            if weight == 0.0:
                continue
            variants.append((_variant_name(penalty, weight), _penalty_overrides(penalty, weight)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    for pattern in patterns:
        pattern_name = str(pattern["name"])
        pattern_data = dict(pattern.get("data", {}))
        for variant_name, overrides in variants:
            for seed in seeds:
                run_dir = output_dir / pattern_name / variant_name / f"seed_{seed}"
                run_config_path = (
                    output_dir / "configs" / pattern_name / variant_name / f"seed_{seed}.yaml"
                )
                config = _make_config(
                    base_config,
                    pattern_name=pattern_name,
                    pattern_data=pattern_data,
                    variant_name=variant_name,
                    overrides=overrides,
                    seed=seed,
                    steps=args.steps,
                    num_elbo_samples=args.num_elbo_samples,
                    output_dir=run_dir,
                )
                _write_config(run_config_path, config)
                if not args.skip_train:
                    _run_training(run_config_path)
                rows.append(
                    _load_run(
                        run_dir,
                        pattern=pattern_name,
                        seed=seed,
                        variant=variant_name,
                        steps=args.steps,
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


def _parse_strings(value: str, *, name: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must include at least one item")
    return values


def _parse_ints(value: str, *, name: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must include at least one integer")
    return values


def _parse_floats(value: str, *, name: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must include at least one number")
    return values


def _variant_name(penalty: str, weight: float) -> str:
    return f"{penalty}_var_{weight:g}".replace(".", "p")


def _penalty_overrides(penalty: str, weight: float) -> dict[str, float]:
    if penalty == "global":
        return {"elbo_variance_ratio_weight": weight}
    if penalty == "time":
        return {"elbo_time_variance_ratio_weight": weight}
    if penalty == "low_observation":
        return {"elbo_low_observation_variance_ratio_weight": weight}
    raise ValueError(f"Unknown penalty: {penalty}")


def _read_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return yaml.safe_load(stream)


def _make_config(
    base_config: dict[str, Any],
    *,
    pattern_name: str,
    pattern_data: dict[str, Any],
    variant_name: str,
    overrides: dict[str, float],
    seed: int,
    steps: int,
    num_elbo_samples: int,
    output_dir: Path,
) -> dict[str, Any]:
    training = {
        **base_config["training"],
        "steps": steps,
        "num_elbo_samples": num_elbo_samples,
        **overrides,
    }
    return {
        **base_config,
        "name": f"elbo_calibration_{pattern_name}_{variant_name}_seed_{seed}",
        "seed": seed,
        "output_dir": str(output_dir),
        "data": {**base_config["data"], **pattern_data},
        "evaluation": {
            **base_config.get("evaluation", {}),
            "data": {**base_config.get("evaluation", {}).get("data", {}), **pattern_data},
        },
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


def _load_run(run_dir: Path, *, pattern: str, seed: int, variant: str, steps: int) -> Row:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return Row(
        pattern=pattern,
        seed=seed,
        variant=variant,
        steps=steps,
        filter_kl=float(metrics["filter_kl"]),
        edge_kl=float(metrics["edge_kl"]),
        state_rmse=float(metrics["state_rmse_global"]),
        state_nll=float(metrics["state_nll"]),
        coverage_90=float(metrics["coverage_90"]),
        variance_ratio=float(metrics["variance_ratio"]),
        predictive_nll=float(metrics["predictive_nll"]),
        closed_form_elbo=float(metrics["closed_form_elbo"]),
    )


def _write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Row.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _aggregate(rows: list[Row]) -> list[dict[str, float | int | str]]:
    summary: list[dict[str, float | int | str]] = []
    keys = sorted({(row.pattern, row.variant, row.steps) for row in rows})
    for pattern, variant, steps in keys:
        grouped = [
            row
            for row in rows
            if row.pattern == pattern and row.variant == variant and row.steps == steps
        ]
        item: dict[str, float | int | str] = {
            "pattern": pattern,
            "variant": variant,
            "steps": steps,
            "num_seeds": len(grouped),
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
        ):
            values = np.asarray([getattr(row, metric) for row in grouped], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary.append(item)
    return summary


def _render_report(summary: list[dict[str, float | int | str]], rows: list[Row]) -> str:
    lines = [
        "# ELBO Calibration Sweep",
        "",
        "| Pattern | Variant | Steps | Seeds | filter KL | edge KL | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {pattern} | {variant} | {steps} | {num_seeds} | "
            "{filter_kl_mean:.6f} +/- {filter_kl_std:.6f} | "
            "{edge_kl_mean:.6f} +/- {edge_kl_std:.6f} | "
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
            "| Pattern | Seed | Variant | Steps | filter KL | edge KL | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.pattern} | {row.seed} | {row.variant} | {row.steps} | "
            f"{row.filter_kl:.6f} | {row.edge_kl:.6f} | {row.state_nll:.6f} | "
            f"{row.coverage_90:.6f} | {row.variance_ratio:.6f} | "
            f"{row.predictive_nll:.6f} | {row.closed_form_elbo:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
