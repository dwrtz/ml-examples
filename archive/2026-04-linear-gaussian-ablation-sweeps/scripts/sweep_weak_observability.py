"""Run weak-observability linear-Gaussian comparison sweeps."""

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
    model: str
    objective: str
    steps: int
    filter_kl: float
    edge_kl: float
    state_rmse: float
    state_nll: float
    coverage_90: float
    variance_ratio: float
    predictive_nll: float
    closed_form_elbo: float | None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite-config",
        default="experiments/linear_gaussian/08_weak_observability.yaml",
    )
    parser.add_argument("--seeds", default="321,322,323,324,325")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--num-elbo-samples", type=int, default=32)
    parser.add_argument("--variance-ratio-weight", type=float, default=0.1)
    parser.add_argument("--elbo-low-observation-weight", type=float, default=1.0)
    parser.add_argument(
        "--models",
        default="zero,frozen,self_fed,self_fed_calibrated,elbo,elbo_calibrated,direct_closed_form",
    )
    parser.add_argument("--output-dir", default="outputs/linear_gaussian_weak_observability")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    seeds = _parse_ints(args.seeds, name="--seeds")
    model_keys = _parse_model_keys(args.models)
    suite_config = _read_config(Path(args.suite_config))
    patterns = suite_config["patterns"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _model_specs(
        args.steps,
        args.num_elbo_samples,
        args.variance_ratio_weight,
        args.elbo_low_observation_weight,
    )
    rows: list[Row] = []
    oracle_seen: set[tuple[str, int]] = set()
    for pattern in patterns:
        pattern_name = str(pattern["name"])
        pattern_data = dict(pattern.get("data", {}))
        for model_key in model_keys:
            spec = specs[model_key]
            base_config = _read_config(spec.config_path)
            for seed in seeds:
                run_dir = output_dir / pattern_name / spec.objective_label / f"seed_{seed}"
                run_config_path = (
                    output_dir
                    / "configs"
                    / pattern_name
                    / spec.objective_label
                    / f"seed_{seed}.yaml"
                )
                config = _make_config(
                    base_config,
                    spec=spec,
                    pattern_name=pattern_name,
                    pattern_data=pattern_data,
                    seed=seed,
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
                        model=spec.model_label,
                        steps=spec.steps,
                    )
                )
                oracle_key = (pattern_name, seed)
                if oracle_key not in oracle_seen:
                    rows.append(_load_oracle_row(run_dir, pattern=pattern_name, seed=seed))
                    oracle_seen.add(oracle_key)

    _write_csv(output_dir / "metrics.csv", rows)
    summary = _aggregate(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "summary.md"
    report_path.write_text(_render_report(summary, rows), encoding="utf-8")
    print(f"Wrote {report_path}")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_label: str
    objective_label: str
    config_path: Path
    steps: int
    training_overrides: dict[str, Any]


def _model_specs(
    steps: int,
    num_elbo_samples: int,
    variance_ratio_weight: float,
    elbo_low_observation_weight: float,
) -> dict[str, ModelSpec]:
    return {
        "zero": ModelSpec(
            key="zero",
            model_label="zero-init no training",
            objective_label="zero_init_edge_mlp",
            config_path=Path("experiments/linear_gaussian/09_zero_init_edge_mlp.yaml"),
            steps=0,
            training_overrides={"steps": 0},
        ),
        "frozen": ModelSpec(
            key="frozen",
            model_label="frozen marginal backward MLP",
            objective_label="frozen_marginal_backward_mlp",
            config_path=Path("experiments/linear_gaussian/10_frozen_marginal_backward_mlp.yaml"),
            steps=steps,
            training_overrides={"steps": steps},
        ),
        "self_fed": ModelSpec(
            key="self_fed",
            model_label="self-fed supervised",
            objective_label="self_fed_supervised_edge_mlp",
            config_path=Path("experiments/linear_gaussian/12_self_fed_supervised_edge_mlp.yaml"),
            steps=steps,
            training_overrides={"steps": steps, "variance_ratio_weight": 0.0},
        ),
        "self_fed_calibrated": ModelSpec(
            key="self_fed_calibrated",
            model_label=f"self-fed supervised var {variance_ratio_weight:g}",
            objective_label=f"self_fed_supervised_edge_mlp_var_{variance_ratio_weight:g}".replace(
                ".", "p"
            ),
            config_path=Path("experiments/linear_gaussian/12_self_fed_supervised_edge_mlp.yaml"),
            steps=steps,
            training_overrides={
                "steps": steps,
                "variance_ratio_weight": variance_ratio_weight,
            },
        ),
        "elbo": ModelSpec(
            key="elbo",
            model_label="MC ELBO structured",
            objective_label="elbo_edge_mlp",
            config_path=Path("experiments/linear_gaussian/02_elbo_edge_mlp.yaml"),
            steps=steps,
            training_overrides={"steps": steps, "num_elbo_samples": num_elbo_samples},
        ),
        "elbo_calibrated": ModelSpec(
            key="elbo_calibrated",
            model_label=f"MC ELBO low-observation var {elbo_low_observation_weight:g}",
            objective_label=f"elbo_edge_mlp_low_obs_{elbo_low_observation_weight:g}".replace(
                ".", "p"
            ),
            config_path=Path("experiments/linear_gaussian/02_elbo_edge_mlp.yaml"),
            steps=steps,
            training_overrides={
                "steps": steps,
                "num_elbo_samples": num_elbo_samples,
                "elbo_low_observation_variance_ratio_weight": elbo_low_observation_weight,
            },
        ),
        "direct_closed_form": ModelSpec(
            key="direct_closed_form",
            model_label="direct closed-form ELBO",
            objective_label="direct_closed_form_elbo_edge_mlp",
            config_path=Path(
                "experiments/linear_gaussian/15_direct_closed_form_elbo_edge_mlp.yaml"
            ),
            steps=steps,
            training_overrides={"steps": steps},
        ),
    }


def _parse_ints(value: str, *, name: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must include at least one integer")
    return values


def _parse_model_keys(value: str) -> list[str]:
    keys = [item.strip() for item in value.split(",") if item.strip()]
    specs = _model_specs(1, 1, 0.1, 1.0)
    unknown = sorted(set(keys) - set(specs))
    if unknown:
        raise ValueError(f"Unknown --models entries: {', '.join(unknown)}")
    if not keys:
        raise ValueError("--models must include at least one model key")
    return keys


def _read_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return yaml.safe_load(stream)


def _make_config(
    base_config: dict[str, Any],
    *,
    spec: ModelSpec,
    pattern_name: str,
    pattern_data: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    training = {**base_config["training"], **spec.training_overrides}
    return {
        **base_config,
        "name": f"{base_config['model']}_{pattern_name}_seed_{seed}",
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


def _load_run(
    run_dir: Path,
    *,
    pattern: str,
    seed: int,
    model: str,
    steps: int,
) -> Row:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return Row(
        pattern=pattern,
        seed=seed,
        model=model,
        objective=str(metrics["objective"]),
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


def _load_oracle_row(run_dir: Path, *, pattern: str, seed: int) -> Row:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return Row(
        pattern=pattern,
        seed=seed,
        model="exact Kalman",
        objective="oracle",
        steps=0,
        filter_kl=0.0,
        edge_kl=0.0,
        state_rmse=float(metrics["oracle_state_rmse_global"]),
        state_nll=float(metrics["oracle_state_nll"]),
        coverage_90=float(metrics["oracle_coverage_90"]),
        variance_ratio=1.0,
        predictive_nll=float(metrics["oracle_predictive_nll"]),
        closed_form_elbo=float(metrics["oracle_closed_form_elbo"]),
    )


def _write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Row.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _aggregate(rows: list[Row]) -> list[dict[str, float | int | str | None]]:
    summary: list[dict[str, float | int | str | None]] = []
    keys = sorted({(row.pattern, row.objective, row.model, row.steps) for row in rows})
    for pattern, objective, model, steps in keys:
        grouped = [
            row
            for row in rows
            if (
                row.pattern == pattern
                and row.objective == objective
                and row.model == model
                and row.steps == steps
            )
        ]
        item: dict[str, float | int | str | None] = {
            "pattern": pattern,
            "model": model,
            "objective": objective,
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
            values = np.asarray(
                [getattr(row, metric) for row in grouped if getattr(row, metric) is not None],
                dtype=np.float64,
            )
            if values.size == 0:
                item[f"{metric}_mean"] = None
                item[f"{metric}_std"] = None
            else:
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary.append(item)
    return summary


def _render_report(
    summary: list[dict[str, float | int | str | None]],
    rows: list[Row],
) -> str:
    lines = [
        "# Linear-Gaussian Weak Observability Sweep",
        "",
        "| Pattern | Model | Objective | Steps | Seeds | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {pattern} | {model} | {objective} | {steps} | {num_seeds} | "
            "{filter_kl_mean:.6f} +/- {filter_kl_std:.6f} | "
            "{edge_kl_mean:.6f} +/- {edge_kl_std:.6f} | "
            "{state_rmse_mean:.6f} +/- {state_rmse_std:.6f} | "
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
            "| Pattern | Seed | Model | Objective | Steps | filter KL | edge KL | state RMSE global | state NLL | cov 90 | var ratio | pred NLL | closed-form ELBO |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.pattern} | {row.seed} | {row.model} | {row.objective} | {row.steps} | "
            f"{row.filter_kl:.6f} | {row.edge_kl:.6f} | {row.state_rmse:.6f} | "
            f"{row.state_nll:.6f} | {row.coverage_90:.6f} | {row.variance_ratio:.6f} | "
            f"{row.predictive_nll:.6f} | {_fmt(row.closed_form_elbo)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
