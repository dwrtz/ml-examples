"""Evaluate filters on nonlinear benchmark variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

from vbf.metrics import gaussian_interval_coverage, rmse_global, scalar_gaussian_nll
from vbf.nonlinear import (
    GridReferenceConfig,
    NonlinearDataConfig,
    make_nonlinear_batch,
    nonlinear_grid_filter,
)
from vbf.data import LinearGaussianParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)

    data_config = NonlinearDataConfig(**config["data"])
    state_params = LinearGaussianParams(**config["state_space"])
    reference_config = GridReferenceConfig(**config.get("reference", {}))
    batch = make_nonlinear_batch(data_config, state_params, seed=int(config["seed"]))
    reference = nonlinear_grid_filter(
        batch,
        state_params,
        data_config=data_config,
        grid_config=reference_config,
    )
    state_nll = scalar_gaussian_nll(batch.z, reference.filter_mean, reference.filter_var)
    predictive_nll = scalar_gaussian_nll(
        batch.y,
        reference.predictive_mean,
        reference.predictive_var,
    )
    metrics = {
        "benchmark": "nonlinear",
        "observation": data_config.observation,
        "batch_size": data_config.batch_size,
        "time_steps": data_config.time_steps,
        "x_pattern": data_config.x_pattern,
        "state_rmse": float(rmse_global(reference.filter_mean, batch.z)),
        "state_nll": float(jnp.mean(state_nll)),
        "predictive_nll": float(jnp.mean(predictive_nll)),
        "coverage_90": float(
            gaussian_interval_coverage(
                batch.z,
                reference.filter_mean,
                reference.filter_var,
                z_score=1.6448536269514722,
            )
        ),
        "mean_filter_variance": float(jnp.mean(reference.filter_var)),
        "mean_predictive_variance": float(jnp.mean(reference.predictive_var)),
    }

    output_dir = Path(config.get("output_dir", "outputs/nonlinear_sine_observation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(
        output_dir / "diagnostics.npz",
        x=np.asarray(batch.x),
        y=np.asarray(batch.y),
        z=np.asarray(batch.z),
        reference_filter_mean=np.asarray(reference.filter_mean),
        reference_filter_var=np.asarray(reference.filter_var),
        reference_predictive_mean=np.asarray(reference.predictive_mean),
        reference_predictive_var=np.asarray(reference.predictive_var),
    )
    summary_path = output_dir / "evaluation_summary.md"
    summary_path.write_text(_render_summary(config["name"], metrics), encoding="utf-8")
    print(f"Wrote {summary_path}")


def _render_summary(name: str, metrics: dict[str, float | int | str]) -> str:
    lines = [
        f"# {name}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics.json`",
            "- `diagnostics.npz`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
