"""Run and aggregate nonlinear particle-filter reference evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


METRICS = (
    "state_rmse",
    "reference_state_rmse",
    "state_nll",
    "reference_state_nll",
    "predictive_y_nll",
    "predictive_nll",
    "reference_predictive_nll",
    "coverage_90",
    "reference_coverage_90",
    "variance_ratio",
    "mean_ess",
)

DEFAULT_CONFIGS = (
    "experiments/nonlinear/01_sine_observation.yaml",
    "experiments/nonlinear/03_weak_sine_observation.yaml",
    "experiments/nonlinear/04_intermittent_sine_observation.yaml",
    "experiments/nonlinear/05_zero_sine_observation.yaml",
    "experiments/nonlinear/06_random_normal_sine_observation.yaml",
)


@dataclass(frozen=True)
class ParticleFilterSpec:
    key: str
    label: str
    num_particles: int
    kde_bandwidth_scale: float = 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--models", default="bootstrap_particle_filter_n128")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-dir", default="outputs/nonlinear_particle_filter_suite")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = _parse_config_paths(args.configs)
    seeds = None if args.seeds is None else _parse_ints(args.seeds, name="--seeds")
    specs = _selected_specs(args.models)

    rows = []
    for config_path in config_paths:
        reference_config = _read_config(config_path)
        for seed in seeds or [int(reference_config["seed"])]:
            for spec in specs:
                run_name = _run_name(reference_config["name"], spec.key, seed=seed, seeds=seeds)
                run_dir = output_dir / run_name
                run_config_path = output_dir / "configs" / f"{run_name}.yaml"
                config = _make_config(
                    reference_config,
                    spec=spec,
                    run_name=run_name,
                    seed=seed,
                    output_dir=run_dir,
                )
                _write_config(run_config_path, config)
                if not args.skip_run:
                    _run_evaluation(run_config_path)
                rows.append(_load_row(run_dir, config_path=run_config_path, spec=spec))

    _write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(_render_report(rows), encoding="utf-8")
    print(f"Wrote {report_path}")


def _parse_config_paths(value: str) -> list[Path]:
    paths = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if not paths:
        raise ValueError("--configs must include at least one config path")
    return paths


def _parse_ints(value: str, *, name: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc


def _selected_specs(value: str) -> list[ParticleFilterSpec]:
    specs = {
        "bootstrap_particle_filter_n128": ParticleFilterSpec(
            key="bootstrap_particle_filter_n128",
            label="bootstrap particle filter n128",
            num_particles=128,
        ),
        "bootstrap_particle_filter_n512": ParticleFilterSpec(
            key="bootstrap_particle_filter_n512",
            label="bootstrap particle filter n512",
            num_particles=512,
        ),
    }
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(keys) - set(specs))
    if unknown:
        raise ValueError(f"Unknown particle filter model keys: {unknown}")
    return [specs[key] for key in keys]


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _make_config(
    reference_config: dict[str, Any],
    *,
    spec: ParticleFilterSpec,
    run_name: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        **reference_config,
        "name": run_name,
        "seed": seed,
        "output_dir": str(output_dir),
        "evaluation": {
            "seed_offset": 10_000,
            "data": {"batch_size": reference_config["data"]["batch_size"]},
        },
        "particle_filter": {
            "filter_name": spec.label,
            "num_particles": spec.num_particles,
            "kde_bandwidth_scale": spec.kde_bandwidth_scale,
        },
    }


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run_evaluation(config_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/evaluate_nonlinear_particle_filter.py", "--config", str(config_path)],
        check=True,
    )


def _load_row(
    run_dir: Path,
    *,
    config_path: Path,
    spec: ParticleFilterSpec,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing particle filter metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "name": run_dir.name,
        "config": str(config_path),
        "model": spec.label,
        "seed": metrics["seed"],
        "objective": metrics["model"],
        "x_pattern": metrics["x_pattern"],
        "time_steps": metrics["time_steps"],
        "batch_size": metrics["batch_size"],
        "num_particles": metrics["num_particles"],
        "kde_bandwidth_scale": metrics["kde_bandwidth_scale"],
        "state_nll_estimator": metrics["state_nll_estimator"],
        "coverage_estimator": metrics["coverage_estimator"],
        "training_signal": "particle_reference",
        **{metric: metrics[metric] for metric in METRICS},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "config",
        "model",
        "seed",
        "objective",
        "x_pattern",
        "time_steps",
        "batch_size",
        "num_particles",
        "kde_bandwidth_scale",
        "state_nll_estimator",
        "coverage_estimator",
        "training_signal",
        *METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Nonlinear Particle Filter Suite",
        "",
        "| x pattern | Model | seeds | state NLL | cov 90 | var ratio | pred-y NLL | mean ESS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _aggregate_rows(rows):
        lines.append(
            "| {x_pattern} | {model} | {seeds} | {state_nll:.6f} | {coverage_90:.6f} | "
            "{variance_ratio:.6f} | {predictive_y_nll:.6f} | {mean_ess:.2f} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["x_pattern"], row["model"]), []).append(row)
    aggregates = []
    for (x_pattern, model), group in grouped.items():
        item = {
            "x_pattern": x_pattern,
            "model": model,
            "seeds": len({row["seed"] for row in group}),
        }
        for metric in METRICS:
            item[metric] = sum(float(row[metric]) for row in group) / len(group)
        aggregates.append(item)
    return sorted(aggregates, key=lambda row: (row["x_pattern"], row["model"]))


def _run_name(base_name: str, model_key: str, *, seed: int, seeds: list[int] | None) -> str:
    seed_part = f"_seed_{seed}" if seeds is not None else ""
    return f"{base_name}{seed_part}_{model_key}"


if __name__ == "__main__":
    main()
