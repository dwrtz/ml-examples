"""Train learned filters on the scalar linear-Gaussian benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from vbf.data import EpisodeBatch, LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.kalman import kalman_edge_posterior_scalar
from vbf.losses import edge_elbo_loss, gaussian_kl, supervised_edge_kl_loss
from vbf.metrics import mean_over_batch, rmse_over_batch, scalar_gaussian_kl
from vbf.models.cells import edge_mean_cov_from_outputs, init_structured_mlp_params, run_structured_mlp_filter
from vbf.train import adam_update, init_adam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)

    if config["model"] not in {"supervised_edge_mlp", "elbo_edge_mlp"}:
        raise ValueError(f"Unsupported linear-Gaussian training model: {config['model']}")

    data_config = LinearGaussianDataConfig(**config["data"])
    state_params = LinearGaussianParams(**config["state_space"])
    training_config = config["training"]
    evaluation_config = config.get("evaluation", {})
    min_var = float(training_config.get("min_var", 1e-6))
    objective = config["model"]

    train_batch = make_linear_gaussian_batch(data_config, state_params, seed=config["seed"])
    train_oracle = kalman_edge_posterior_scalar(train_batch, state_params)
    eval_data_config = LinearGaussianDataConfig(
        **{**config["data"], **evaluation_config.get("data", {})}
    )
    eval_num_batches = int(evaluation_config.get("num_batches", 1))
    eval_seed_start = int(config["seed"]) + int(evaluation_config.get("seed_offset", 10_000))
    eval_batch = _make_eval_batch(eval_data_config, state_params, eval_seed_start, eval_num_batches)
    eval_oracle = kalman_edge_posterior_scalar(eval_batch, state_params)
    params = init_structured_mlp_params(
        jax.random.PRNGKey(config["seed"] + 1),
        hidden_dim=int(training_config["hidden_dim"]),
    )
    opt_state = init_adam(params)

    def loss_fn(current_params: dict[str, jax.Array], key: jax.Array) -> jax.Array:
        if objective == "supervised_edge_mlp":
            return supervised_edge_kl_loss(
                current_params,
                train_batch,
                state_params,
                train_oracle,
                min_var=min_var,
            )
        return edge_elbo_loss(
            current_params,
            train_batch,
            state_params,
            key,
            num_samples=int(training_config.get("num_elbo_samples", 8)),
            min_var=min_var,
        )

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    history: list[tuple[int, float]] = []
    train_key = jax.random.PRNGKey(config["seed"] + 2)
    for step in range(1, int(training_config["steps"]) + 1):
        train_key, step_key = jax.random.split(train_key)
        loss_value, grads = value_and_grad(params, step_key)
        params, opt_state = adam_update(
            params,
            grads,
            opt_state,
            learning_rate=float(training_config["learning_rate"]),
        )
        if step == 1 or step % int(training_config["log_every"]) == 0:
            history.append((step, float(loss_value)))

    final_loss = float(loss_fn(params, jax.random.PRNGKey(config["seed"] + 3)))
    outputs = run_structured_mlp_filter(params, eval_batch, state_params, min_var=min_var)
    pred_edge_mean, pred_edge_cov = edge_mean_cov_from_outputs(outputs)
    filter_kl_bt = scalar_gaussian_kl(
        eval_oracle.filter_mean,
        eval_oracle.filter_var,
        outputs.filter_mean,
        outputs.filter_var,
    )
    edge_kl_bt = gaussian_kl(eval_oracle.edge_mean, eval_oracle.edge_cov, pred_edge_mean, pred_edge_cov)
    state_rmse_t = rmse_over_batch(outputs.filter_mean, eval_batch.z)
    filter_kl_t = mean_over_batch(filter_kl_bt)
    edge_kl_t = mean_over_batch(edge_kl_bt)
    filter_kl = float(jnp.mean(filter_kl_bt))
    state_rmse = float(jnp.mean(state_rmse_t))
    mean_backward_var = float(jnp.mean(outputs.backward_var))
    mean_edge_var_trace = float(jnp.mean(jnp.trace(pred_edge_cov, axis1=-2, axis2=-1)))
    max_abs_edge_mean = float(jnp.max(jnp.abs(pred_edge_mean)))

    output_dir = Path(config.get("output_dir", "outputs/linear_gaussian_supervised_edge_mlp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_loss_history(output_dir / "loss_history.csv", history)
    metrics = {
        "objective": objective,
        "train_batch_size": data_config.batch_size,
        "eval_batch_size": eval_data_config.batch_size,
        "eval_num_batches": eval_num_batches,
        "eval_total_batch_size": eval_batch.x.shape[0],
        "eval_seed_start": eval_seed_start,
        "time_steps": data_config.time_steps,
        "training_steps": int(training_config["steps"]),
        "final_loss": final_loss,
        "final_edge_kl": float(jnp.mean(edge_kl_bt)),
        "filter_kl": filter_kl,
        "edge_kl": float(jnp.mean(edge_kl_bt)),
        "state_rmse": state_rmse,
        "mean_backward_variance": mean_backward_var,
        "mean_edge_covariance_trace": mean_edge_var_trace,
        "max_abs_edge_mean": max_abs_edge_mean,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(
        output_dir / "diagnostics.npz",
        x=np.asarray(eval_batch.x),
        y=np.asarray(eval_batch.y),
        z=np.asarray(eval_batch.z),
        oracle_filter_mean=np.asarray(eval_oracle.filter_mean),
        oracle_filter_var=np.asarray(eval_oracle.filter_var),
        learned_filter_mean=np.asarray(outputs.filter_mean),
        learned_filter_var=np.asarray(outputs.filter_var),
        edge_kl_over_time=np.asarray(edge_kl_t),
        filter_kl_over_time=np.asarray(filter_kl_t),
        state_rmse_over_time=np.asarray(state_rmse_t),
        loss_history_step=np.asarray([step for step, _ in history], dtype=np.int64),
        loss_history_loss=np.asarray([loss for _, loss in history], dtype=np.float64),
    )
    summary_path = output_dir / "evaluation_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# {config['name']}",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| train batch size | {data_config.batch_size} |",
                f"| eval batch size | {eval_batch.x.shape[0]} |",
                f"| eval seed start | {eval_seed_start} |",
                f"| time steps | {data_config.time_steps} |",
                f"| training steps | {training_config['steps']} |",
                f"| objective | {objective} |",
                f"| final loss | {final_loss:.6f} |",
                f"| edge KL | {float(jnp.mean(edge_kl_bt)):.6f} |",
                f"| filter KL | {filter_kl:.6f} |",
                f"| state RMSE | {state_rmse:.6f} |",
                f"| mean backward variance | {mean_backward_var:.6f} |",
                f"| mean edge covariance trace | {mean_edge_var_trace:.6f} |",
                f"| max abs edge mean | {max_abs_edge_mean:.6f} |",
                "",
                "## Loss History",
                "",
                "| Step | Loss |",
                "|---:|---:|",
                *[f"| {step} | {loss:.6f} |" for step, loss in history],
                "",
                "## Artifacts",
                "",
                "- `loss_history.csv`",
                "- `metrics.json`",
                "- `diagnostics.npz`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


def _make_eval_batch(
    data_config: LinearGaussianDataConfig,
    state_params: LinearGaussianParams,
    seed_start: int,
    num_batches: int,
) -> EpisodeBatch:
    if num_batches <= 0:
        raise ValueError("evaluation.num_batches must be positive")

    batches = [
        make_linear_gaussian_batch(data_config, state_params, seed=seed_start + index)
        for index in range(num_batches)
    ]
    return EpisodeBatch(
        x=jnp.concatenate([batch.x for batch in batches], axis=0),
        y=jnp.concatenate([batch.y for batch in batches], axis=0),
        z=jnp.concatenate([batch.z for batch in batches], axis=0),
    )


def _write_loss_history(path: Path, history: list[tuple[int, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "loss"))
        writer.writerows(history)


if __name__ == "__main__":
    main()
