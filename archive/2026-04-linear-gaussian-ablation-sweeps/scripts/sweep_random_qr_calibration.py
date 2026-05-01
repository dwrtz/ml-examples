"""Sweep randomized-Q/R regime-local variance calibration weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from sweep_qr_generalization import (
    Row,
    _aggregate,
    _evaluate_params,
    _load_params,
    _model_specs,
    _num_label,
    _pairs_from_grid,
    _parse_ints,
    _parse_model_keys,
    _parse_pairs,
    _read_config,
    _render_report,
    _run_training,
    _write_config,
    _write_csv,
)
from sweep_random_qr_generalization import _parse_floats, _regime_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite-config",
        default="experiments/linear_gaussian/07_random_qr_generalization.yaml",
    )
    parser.add_argument("--seeds", default="321,322,323")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-elbo-samples", type=int, default=32)
    parser.add_argument("--weights", default="0,0.1,1")
    parser.add_argument("--train-q-values", default=None)
    parser.add_argument("--train-r-values", default=None)
    parser.add_argument("--eval-pairs", default="0.03:0.03,0.1:0.1,0.3:0.3,0.03:0.3,0.3:0.03")
    parser.add_argument("--models", default="self_fed_calibrated,elbo_calibrated")
    parser.add_argument("--variance-ratio-weight", type=float, default=0.1)
    parser.add_argument("--elbo-low-observation-weight", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="outputs/linear_gaussian_random_qr_calibration",
    )
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    suite_config = _read_config(Path(args.suite_config))
    seeds = _parse_ints(args.seeds, name="--seeds")
    weights = _parse_floats(args.weights)
    train_q_values = (
        _parse_floats(args.train_q_values)
        if args.train_q_values is not None
        else [float(value) for value in suite_config["train_q_values"]]
    )
    train_r_values = (
        _parse_floats(args.train_r_values)
        if args.train_r_values is not None
        else [float(value) for value in suite_config["train_r_values"]]
    )
    eval_pairs = (
        _parse_pairs(args.eval_pairs)
        if args.eval_pairs is not None
        else _pairs_from_grid(suite_config["eval_q_values"], suite_config["eval_r_values"])
    )
    model_keys = _parse_model_keys(args.models)
    specs = _model_specs(
        args.steps,
        args.num_elbo_samples,
        args.variance_ratio_weight,
        args.elbo_low_observation_weight,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    train_q_mean = float(np.mean(train_q_values))
    train_r_mean = float(np.mean(train_r_values))
    regime_label = _regime_label(train_q_values, train_r_values)
    for model_key in model_keys:
        base_spec = specs[model_key]
        base_config = _read_config(base_spec.config_path)
        for weight in weights:
            spec = _calibrated_spec(base_spec, model_key=model_key, weight=weight)
            for seed in seeds:
                run_dir = output_dir / spec.objective_label / regime_label / f"seed_{seed}"
                run_config_path = (
                    output_dir
                    / "configs"
                    / spec.objective_label
                    / regime_label
                    / f"seed_{seed}.yaml"
                )
                config = _make_random_train_config(
                    base_config,
                    spec=spec,
                    seed=seed,
                    train_q_values=train_q_values,
                    train_r_values=train_r_values,
                    output_dir=run_dir,
                )
                _write_config(run_config_path, config)
                if not args.skip_train:
                    _run_training(run_config_path)

                params = _load_params(run_dir / "params.npz")
                for eval_q, eval_r in eval_pairs:
                    rows.append(
                        _evaluate_params(
                            params,
                            config,
                            model=spec.model_label,
                            objective=spec.objective_label,
                            seed=seed,
                            train_q=train_q_mean,
                            train_r=train_r_mean,
                            eval_q=eval_q,
                            eval_r=eval_r,
                            steps=spec.steps,
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


def _calibrated_spec(base_spec, *, model_key: str, weight: float):
    label = _num_label(weight)
    if model_key == "self_fed_calibrated":
        return replace(
            base_spec,
            model_label=f"self-fed supervised regime var {weight:g}",
            objective_label=f"{base_spec.objective_label}_regime_{label}",
            training_overrides={
                **base_spec.training_overrides,
                "variance_ratio_weight": 0.0,
                "regime_variance_ratio_weight": weight,
            },
        )
    if model_key == "elbo_calibrated":
        return replace(
            base_spec,
            model_label=f"MC ELBO low-observation + regime var {weight:g}",
            objective_label=f"{base_spec.objective_label}_regime_{label}",
            training_overrides={
                **base_spec.training_overrides,
                "elbo_regime_variance_ratio_weight": weight,
            },
        )
    raise ValueError(f"Unsupported calibration model: {model_key}")


def _make_random_train_config(
    base_config: dict[str, Any],
    *,
    spec,
    seed: int,
    train_q_values: list[float],
    train_r_values: list[float],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        **base_config,
        "name": f"{base_config['model']}_random_qr_calibration_seed_{seed}",
        "seed": seed,
        "output_dir": str(output_dir),
        "state_space": {
            **base_config["state_space"],
            "q": float(np.mean(train_q_values)),
            "r": float(np.mean(train_r_values)),
            "random_q_values": train_q_values,
            "random_r_values": train_r_values,
        },
        "training": {**base_config["training"], **spec.training_overrides},
        "evaluation": {
            **base_config.get("evaluation", {}),
            "state_space": {
                "q": float(np.mean(train_q_values)),
                "r": float(np.mean(train_r_values)),
            },
        },
    }


if __name__ == "__main__":
    main()
