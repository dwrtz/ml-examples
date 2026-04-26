"""Train fixed-Q/R filters and evaluate them across held-out Q/R regimes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from vbf.data import EpisodeBatch, LinearGaussianDataConfig, LinearGaussianParams
from vbf.data import make_linear_gaussian_batch
from vbf.kalman import kalman_edge_posterior_scalar
from vbf.losses import gaussian_kl
from vbf.metrics import (
    gaussian_interval_coverage,
    rmse_global,
    scalar_gaussian_kl,
    scalar_gaussian_nll,
)
from vbf.models.cells import (
    edge_mean_cov_from_outputs,
    run_structured_mlp_filter,
)
from vbf.predictive import linear_gaussian_predictive_from_filter


@dataclass(frozen=True)
class Row:
    seed: int
    model: str
    objective: str
    train_q: float
    train_r: float
    eval_q: float
    eval_r: float
    steps: int
    filter_kl: float
    edge_kl: float
    state_rmse: float
    state_nll: float
    coverage_90: float
    variance_ratio: float
    predictive_nll: float
    oracle_state_nll: float
    oracle_predictive_nll: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite-config",
        default="experiments/linear_gaussian/07_random_qr_generalization.yaml",
    )
    parser.add_argument("--seeds", default="321,322,323,324,325")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--num-elbo-samples", type=int, default=32)
    parser.add_argument("--train-pairs", default=None, help="Comma list like q:r,q:r")
    parser.add_argument("--eval-pairs", default=None, help="Comma list like q:r,q:r")
    parser.add_argument(
        "--models",
        default="frozen,self_fed_calibrated,elbo_calibrated",
    )
    parser.add_argument("--variance-ratio-weight", type=float, default=0.1)
    parser.add_argument("--elbo-low-observation-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/linear_gaussian_qr_generalization")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    suite_config = _read_config(Path(args.suite_config))
    seeds = _parse_ints(args.seeds, name="--seeds")
    train_pairs = (
        _parse_pairs(args.train_pairs)
        if args.train_pairs is not None
        else _pairs_from_grid(suite_config["train_q_values"], suite_config["train_r_values"])
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
    for model_key in model_keys:
        spec = specs[model_key]
        base_config = _read_config(spec.config_path)
        for train_q, train_r in train_pairs:
            for seed in seeds:
                run_dir = (
                    output_dir
                    / spec.objective_label
                    / f"train_q_{_num_label(train_q)}_r_{_num_label(train_r)}"
                    / f"seed_{seed}"
                )
                run_config_path = (
                    output_dir
                    / "configs"
                    / spec.objective_label
                    / f"train_q_{_num_label(train_q)}_r_{_num_label(train_r)}"
                    / f"seed_{seed}.yaml"
                )
                config = _make_train_config(
                    base_config,
                    spec=spec,
                    seed=seed,
                    train_q=train_q,
                    train_r=train_r,
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
                            train_q=train_q,
                            train_r=train_r,
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
        "frozen": ModelSpec(
            key="frozen",
            model_label="frozen marginal backward MLP",
            objective_label="frozen_marginal_backward_mlp",
            config_path=Path("experiments/linear_gaussian/10_frozen_marginal_backward_mlp.yaml"),
            steps=steps,
            training_overrides={"steps": steps},
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
    }


def _evaluate_params(
    params: dict[str, jax.Array],
    train_config: dict[str, Any],
    *,
    model: str,
    objective: str,
    seed: int,
    train_q: float,
    train_r: float,
    eval_q: float,
    eval_r: float,
    steps: int,
) -> Row:
    data_config = LinearGaussianDataConfig(**train_config["evaluation"]["data"])
    state_params = LinearGaussianParams(**{**train_config["state_space"], "q": eval_q, "r": eval_r})
    eval_seed = int(seed) + int(train_config.get("evaluation", {}).get("seed_offset", 10_000))
    eval_num_batches = int(train_config.get("evaluation", {}).get("num_batches", 1))
    batch = _make_eval_batch(data_config, state_params, eval_seed, eval_num_batches)
    oracle = kalman_edge_posterior_scalar(batch, state_params)
    outputs = run_structured_mlp_filter(
        params,
        batch,
        state_params,
        min_var=float(train_config["training"].get("min_var", 1e-6)),
    )
    pred_edge_mean, pred_edge_cov = edge_mean_cov_from_outputs(outputs)
    filter_kl = scalar_gaussian_kl(
        oracle.filter_mean,
        oracle.filter_var,
        outputs.filter_mean,
        outputs.filter_var,
    )
    edge_kl = gaussian_kl(oracle.edge_mean, oracle.edge_cov, pred_edge_mean, pred_edge_cov)
    state_nll = scalar_gaussian_nll(batch.z, outputs.filter_mean, outputs.filter_var)
    oracle_state_nll = scalar_gaussian_nll(batch.z, oracle.filter_mean, oracle.filter_var)
    learned_predictive = linear_gaussian_predictive_from_filter(
        outputs.filter_mean,
        outputs.filter_var,
        batch,
        state_params,
    )
    oracle_predictive = linear_gaussian_predictive_from_filter(
        oracle.filter_mean,
        oracle.filter_var,
        batch,
        state_params,
    )
    predictive_nll = scalar_gaussian_nll(batch.y, learned_predictive.mean, learned_predictive.var)
    oracle_predictive_nll = scalar_gaussian_nll(
        batch.y,
        oracle_predictive.mean,
        oracle_predictive.var,
    )
    return Row(
        seed=seed,
        model=model,
        objective=objective,
        train_q=train_q,
        train_r=train_r,
        eval_q=eval_q,
        eval_r=eval_r,
        steps=steps,
        filter_kl=float(jnp.mean(filter_kl)),
        edge_kl=float(jnp.mean(edge_kl)),
        state_rmse=float(rmse_global(outputs.filter_mean, batch.z)),
        state_nll=float(jnp.mean(state_nll)),
        coverage_90=float(
            gaussian_interval_coverage(
                batch.z,
                outputs.filter_mean,
                outputs.filter_var,
                z_score=1.6448536269514722,
            )
        ),
        variance_ratio=float(jnp.mean(outputs.filter_var) / jnp.mean(oracle.filter_var)),
        predictive_nll=float(jnp.mean(predictive_nll)),
        oracle_state_nll=float(jnp.mean(oracle_state_nll)),
        oracle_predictive_nll=float(jnp.mean(oracle_predictive_nll)),
    )


def _make_eval_batch(
    data_config: LinearGaussianDataConfig,
    state_params: LinearGaussianParams,
    seed_start: int,
    num_batches: int,
) -> EpisodeBatch:
    batches = [
        make_linear_gaussian_batch(data_config, state_params, seed=seed_start + index)
        for index in range(num_batches)
    ]
    return EpisodeBatch(
        x=jnp.concatenate([batch.x for batch in batches], axis=0),
        y=jnp.concatenate([batch.y for batch in batches], axis=0),
        z=jnp.concatenate([batch.z for batch in batches], axis=0),
    )


def _make_train_config(
    base_config: dict[str, Any],
    *,
    spec: ModelSpec,
    seed: int,
    train_q: float,
    train_r: float,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        **base_config,
        "name": f"{base_config['model']}_q_{train_q:g}_r_{train_r:g}_seed_{seed}",
        "seed": seed,
        "output_dir": str(output_dir),
        "state_space": {**base_config["state_space"], "q": train_q, "r": train_r},
        "training": {**base_config["training"], **spec.training_overrides},
    }


def _load_params(path: Path) -> dict[str, jax.Array]:
    if not path.exists():
        raise FileNotFoundError(f"Missing params file: {path}")
    with np.load(path) as data:
        return {key: jnp.asarray(data[key]) for key in data.files}


def _parse_model_keys(value: str) -> list[str]:
    keys = [item.strip() for item in value.split(",") if item.strip()]
    specs = _model_specs(1, 1, 0.1, 1.0)
    unknown = sorted(set(keys) - set(specs))
    if unknown:
        raise ValueError(f"Unknown --models entries: {', '.join(unknown)}")
    if not keys:
        raise ValueError("--models must include at least one model key")
    return keys


def _parse_ints(value: str, *, name: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must include at least one integer")
    return values


def _parse_pairs(value: str) -> list[tuple[float, float]]:
    pairs = []
    for item in value.split(","):
        if not item.strip():
            continue
        q_text, r_text = item.split(":")
        pairs.append((float(q_text), float(r_text)))
    if not pairs:
        raise ValueError("Q/R pair list must include at least one pair")
    return pairs


def _pairs_from_grid(q_values: list[float], r_values: list[float]) -> list[tuple[float, float]]:
    return [(float(q), float(r)) for q in q_values for r in r_values]


def _num_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _read_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return yaml.safe_load(stream)


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run_training(config_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/train_linear_gaussian.py", "--config", str(config_path)],
        check=True,
    )


def _write_csv(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(Row.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _aggregate(rows: list[Row]) -> list[dict[str, float | int | str]]:
    summary: list[dict[str, float | int | str]] = []
    keys = sorted(
        {
            (row.model, row.objective, row.train_q, row.train_r, row.eval_q, row.eval_r)
            for row in rows
        }
    )
    for model, objective, train_q, train_r, eval_q, eval_r in keys:
        grouped = [
            row
            for row in rows
            if (
                row.model == model
                and row.objective == objective
                and row.train_q == train_q
                and row.train_r == train_r
                and row.eval_q == eval_q
                and row.eval_r == eval_r
            )
        ]
        item: dict[str, float | int | str] = {
            "model": model,
            "objective": objective,
            "train_q": train_q,
            "train_r": train_r,
            "eval_q": eval_q,
            "eval_r": eval_r,
            "steps": grouped[0].steps,
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
            "oracle_state_nll",
            "oracle_predictive_nll",
        ):
            values = np.asarray([getattr(row, metric) for row in grouped], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=0))
        summary.append(item)
    return summary


def _render_report(summary: list[dict[str, float | int | str]], rows: list[Row]) -> str:
    lines = [
        "# Linear-Gaussian Fixed Q/R Generalization",
        "",
        "| Model | train Q | train R | eval Q | eval R | Seeds | filter KL | edge KL | state NLL | cov 90 | var ratio | pred NLL | oracle pred NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {model} | {train_q:g} | {train_r:g} | {eval_q:g} | {eval_r:g} | {num_seeds} | "
            "{filter_kl_mean:.6f} +/- {filter_kl_std:.6f} | "
            "{edge_kl_mean:.6f} +/- {edge_kl_std:.6f} | "
            "{state_nll_mean:.6f} +/- {state_nll_std:.6f} | "
            "{coverage_90_mean:.6f} +/- {coverage_90_std:.6f} | "
            "{variance_ratio_mean:.6f} +/- {variance_ratio_std:.6f} | "
            "{predictive_nll_mean:.6f} +/- {predictive_nll_std:.6f} | "
            "{oracle_predictive_nll_mean:.6f} +/- {oracle_predictive_nll_std:.6f} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Per-Seed Rows",
            "",
            "| Seed | Model | train Q | train R | eval Q | eval R | filter KL | edge KL | state NLL | cov 90 | var ratio | pred NLL |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.seed} | {row.model} | {row.train_q:g} | {row.train_r:g} | "
            f"{row.eval_q:g} | {row.eval_r:g} | {row.filter_kl:.6f} | "
            f"{row.edge_kl:.6f} | {row.state_nll:.6f} | {row.coverage_90:.6f} | "
            f"{row.variance_ratio:.6f} | {row.predictive_nll:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
