"""Evaluate particle filters on nonlinear benchmark variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

from vbf.data import LinearGaussianParams
from vbf.metrics import gaussian_interval_coverage, rmse_global, scalar_gaussian_nll
from vbf.nonlinear import (
    GridReferenceConfig,
    NonlinearDataConfig,
    make_nonlinear_batch,
    nonlinear_bootstrap_particle_filter,
)
from vbf.nonlinear_cache import load_or_compute_nonlinear_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", default="outputs/cache/nonlinear_reference")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    state_params = LinearGaussianParams(**config["state_space"])
    evaluation_config = config.get("evaluation", {})
    pf_config = config.get("particle_filter", {})
    reference_config = GridReferenceConfig(**config.get("reference", {}))

    eval_data_config = NonlinearDataConfig(
        **{**config["data"], **evaluation_config.get("data", {})}
    )
    eval_seed = int(config["seed"]) + int(evaluation_config.get("seed_offset", 0))
    eval_batch = make_nonlinear_batch(eval_data_config, state_params, seed=eval_seed)
    reference_cached = load_or_compute_nonlinear_reference(
        eval_data_config,
        state_params,
        seed=eval_seed,
        grid_config=reference_config,
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
    )

    num_particles = int(pf_config.get("num_particles", 128))
    filter_name = str(pf_config.get("filter_name", f"bootstrap_particle_filter_n{num_particles}"))
    kde_bandwidth_scale = float(pf_config.get("kde_bandwidth_scale", 1.0))
    outputs = nonlinear_bootstrap_particle_filter(
        eval_batch,
        state_params,
        data_config=eval_data_config,
        num_particles=num_particles,
        seed=int(config["seed"]) + int(pf_config.get("seed_offset", 20_000)),
        kde_bandwidth_scale=kde_bandwidth_scale,
    )

    reference = reference_cached.reference
    reference_state_nll = scalar_gaussian_nll(
        eval_batch.z,
        reference.filter_mean,
        reference.filter_var,
    )
    learned_predictive_nll = scalar_gaussian_nll(
        eval_batch.y,
        outputs.predictive_mean,
        outputs.predictive_var,
    )
    reference_predictive_nll = scalar_gaussian_nll(
        eval_batch.y,
        reference.predictive_mean,
        reference.predictive_var,
    )
    metrics = {
        "benchmark": "nonlinear",
        "model": filter_name,
        "seed": int(config["seed"]),
        "observation": eval_data_config.observation,
        "x_pattern": eval_data_config.x_pattern,
        "batch_size": eval_data_config.batch_size,
        "time_steps": eval_data_config.time_steps,
        "num_particles": num_particles,
        "kde_bandwidth_scale": kde_bandwidth_scale,
        "state_nll_estimator": "particle_kde",
        "coverage_estimator": "moment_gaussian",
        "reference_state_nll_estimator": "grid_moment_gaussian",
        "state_rmse": float(rmse_global(outputs.filter_mean, eval_batch.z)),
        "reference_state_rmse": float(rmse_global(reference.filter_mean, eval_batch.z)),
        "state_nll": float(jnp.mean(-outputs.filter_log_prob_z)),
        "reference_state_nll": float(jnp.mean(reference_state_nll)),
        "predictive_y_nll": float(jnp.mean(-outputs.predictive_log_prob_y)),
        "predictive_nll": float(jnp.mean(learned_predictive_nll)),
        "reference_predictive_nll": float(jnp.mean(reference_predictive_nll)),
        "coverage_90": float(
            gaussian_interval_coverage(
                eval_batch.z,
                outputs.filter_mean,
                outputs.filter_var,
                z_score=1.6448536269514722,
            )
        ),
        "reference_coverage_90": float(
            gaussian_interval_coverage(
                eval_batch.z,
                reference.filter_mean,
                reference.filter_var,
                z_score=1.6448536269514722,
            )
        ),
        "mean_filter_variance": float(jnp.mean(outputs.filter_var)),
        "reference_mean_filter_variance": float(jnp.mean(reference.filter_var)),
        "variance_ratio": float(jnp.mean(outputs.filter_var) / jnp.mean(reference.filter_var)),
        "mean_ess": float(outputs.mean_ess),
        "eval_reference_cache_hit": reference_cached.cache_hit,
        "eval_reference_cache_path": str(reference_cached.cache_path),
    }

    output_dir = Path(config.get("output_dir", "outputs/nonlinear_particle_filter"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(
        output_dir / "diagnostics.npz",
        x=np.asarray(eval_batch.x),
        y=np.asarray(eval_batch.y),
        z=np.asarray(eval_batch.z),
        filter_mean=np.asarray(outputs.filter_mean),
        filter_var=np.asarray(outputs.filter_var),
        predictive_mean=np.asarray(outputs.predictive_mean),
        predictive_var=np.asarray(outputs.predictive_var),
        predictive_log_prob_y=np.asarray(outputs.predictive_log_prob_y),
        filter_log_prob_z=np.asarray(outputs.filter_log_prob_z),
        reference_filter_mean=np.asarray(reference.filter_mean),
        reference_filter_var=np.asarray(reference.filter_var),
        reference_predictive_mean=np.asarray(reference.predictive_mean),
        reference_predictive_var=np.asarray(reference.predictive_var),
    )
    summary_path = output_dir / "evaluation_summary.md"
    summary_path.write_text(_render_summary(config["name"], metrics), encoding="utf-8")
    print(f"Wrote {summary_path}")
    cache_status = "hit" if reference_cached.cache_hit else "miss"
    print(f"Eval reference cache {cache_status}: {reference_cached.cache_path}")


def _render_summary(name: str, metrics: dict[str, float | int | str | bool]) -> str:
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
