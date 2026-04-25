"""Evaluate filters on the scalar linear-Gaussian benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import yaml

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.kalman import kalman_edge_posterior_scalar, kalman_filter_scalar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)

    data_config = LinearGaussianDataConfig(**config["data"])
    params_config = config.get("state_space", config.get("model"))
    if params_config is None:
        raise ValueError("Config must include state_space or legacy model parameters")
    params = LinearGaussianParams(**params_config)
    batch = make_linear_gaussian_batch(data_config, params, seed=config["seed"])

    kalman = kalman_filter_scalar(batch, params)
    edge = kalman_edge_posterior_scalar(batch, params)
    max_marginal_mean_error = float(jnp.max(jnp.abs(edge.filter_mean - kalman.filter_mean)))
    max_marginal_var_error = float(jnp.max(jnp.abs(edge.filter_var - kalman.filter_var)))
    state_rmse = float(jnp.sqrt(jnp.mean((kalman.filter_mean - batch.z) ** 2)))
    mean_predictive_var = float(jnp.mean(kalman.predictive_var))

    output_dir = Path(config.get("output_dir", "outputs/linear_gaussian_oracle_check"))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# {config['name']}",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| batch size | {data_config.batch_size} |",
                f"| time steps | {data_config.time_steps} |",
                f"| state RMSE | {state_rmse:.6f} |",
                f"| mean predictive variance | {mean_predictive_var:.6f} |",
                f"| max edge/filter mean error | {max_marginal_mean_error:.3e} |",
                f"| max edge/filter variance error | {max_marginal_var_error:.3e} |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
