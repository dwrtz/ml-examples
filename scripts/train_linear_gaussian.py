"""Train learned filters on the scalar linear-Gaussian benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.kalman import kalman_edge_posterior_scalar
from vbf.losses import supervised_edge_kl_loss
from vbf.metrics import scalar_gaussian_kl
from vbf.models.cells import edge_mean_cov_from_outputs, init_structured_mlp_params, run_structured_mlp_filter
from vbf.train import adam_update, init_adam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)

    if config["model"] != "supervised_edge_mlp":
        raise ValueError(f"Unsupported linear-Gaussian training model: {config['model']}")

    data_config = LinearGaussianDataConfig(**config["data"])
    state_params = LinearGaussianParams(**config["state_space"])
    training_config = config["training"]
    min_var = float(training_config.get("min_var", 1e-6))

    batch = make_linear_gaussian_batch(data_config, state_params, seed=config["seed"])
    oracle = kalman_edge_posterior_scalar(batch, state_params)
    params = init_structured_mlp_params(
        jax.random.PRNGKey(config["seed"] + 1),
        hidden_dim=int(training_config["hidden_dim"]),
    )
    opt_state = init_adam(params)

    def loss_fn(current_params: dict[str, jax.Array]) -> jax.Array:
        return supervised_edge_kl_loss(
            current_params,
            batch,
            state_params,
            oracle,
            min_var=min_var,
        )

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    history: list[tuple[int, float]] = []
    for step in range(1, int(training_config["steps"]) + 1):
        loss_value, grads = value_and_grad(params)
        params, opt_state = adam_update(
            params,
            grads,
            opt_state,
            learning_rate=float(training_config["learning_rate"]),
        )
        if step == 1 or step % int(training_config["log_every"]) == 0:
            history.append((step, float(loss_value)))

    final_loss = float(loss_fn(params))
    outputs = run_structured_mlp_filter(params, batch, state_params, min_var=min_var)
    pred_edge_mean, pred_edge_cov = edge_mean_cov_from_outputs(outputs)
    filter_kl = float(
        jnp.mean(
            scalar_gaussian_kl(
                oracle.filter_mean,
                oracle.filter_var,
                outputs.filter_mean,
                outputs.filter_var,
            )
        )
    )
    state_rmse = float(jnp.sqrt(jnp.mean((outputs.filter_mean - batch.z) ** 2)))
    mean_backward_var = float(jnp.mean(outputs.backward_var))
    mean_edge_var_trace = float(jnp.mean(jnp.trace(pred_edge_cov, axis1=-2, axis2=-1)))
    max_abs_edge_mean = float(jnp.max(jnp.abs(pred_edge_mean)))

    output_dir = Path(config.get("output_dir", "outputs/linear_gaussian_supervised_edge_mlp"))
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
                f"| training steps | {training_config['steps']} |",
                f"| final edge KL | {final_loss:.6f} |",
                f"| filter KL | {filter_kl:.6f} |",
                f"| state RMSE | {state_rmse:.6f} |",
                f"| mean backward variance | {mean_backward_var:.6f} |",
                f"| mean edge covariance trace | {mean_edge_var_trace:.6f} |",
                f"| max abs edge mean | {max_abs_edge_mean:.6f} |",
                "",
                "## Loss History",
                "",
                "| Step | Edge KL |",
                "|---:|---:|",
                *[f"| {step} | {loss:.6f} |" for step, loss in history],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
