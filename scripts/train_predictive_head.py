"""Train a learned one-step predictive head for the scalar linear-Gaussian benchmark."""

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
from vbf.metrics import rmse_global, rmse_over_batch, scalar_gaussian_nll
from vbf.models.heads import init_predictive_mlp_params, run_predictive_mlp_head
from vbf.predictive import linear_gaussian_predictive_from_filter, previous_filter_beliefs
from vbf.train import adam_update, init_adam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)

    if config["model"] != "predictive_head":
        raise ValueError("predictive head training requires model: predictive_head")

    data_config = LinearGaussianDataConfig(**config["data"])
    state_params = LinearGaussianParams(**config["state_space"])
    training_config = config["training"]
    evaluation_config = config.get("evaluation", {})
    min_var = float(training_config.get("min_var", 1e-6))

    train_batch = make_linear_gaussian_batch(data_config, state_params, seed=config["seed"])
    train_oracle = kalman_edge_posterior_scalar(train_batch, state_params)
    train_prev_mean, train_prev_var = previous_filter_beliefs(
        train_oracle.filter_mean,
        train_oracle.filter_var,
        state_params,
    )

    eval_data_config = LinearGaussianDataConfig(
        **{**config["data"], **evaluation_config.get("data", {})}
    )
    eval_num_batches = int(evaluation_config.get("num_batches", 1))
    eval_seed_start = int(config["seed"]) + int(evaluation_config.get("seed_offset", 10_000))
    eval_batch = _make_eval_batch(eval_data_config, state_params, eval_seed_start, eval_num_batches)
    eval_oracle = kalman_edge_posterior_scalar(eval_batch, state_params)
    eval_prev_mean, eval_prev_var = previous_filter_beliefs(
        eval_oracle.filter_mean,
        eval_oracle.filter_var,
        state_params,
    )

    params = init_predictive_mlp_params(
        jax.random.PRNGKey(config["seed"] + 1),
        hidden_dim=int(training_config["hidden_dim"]),
    )
    opt_state = init_adam(params)

    def loss_fn(current_params: dict[str, jax.Array]) -> jax.Array:
        outputs = run_predictive_mlp_head(
            current_params,
            train_prev_mean,
            train_prev_var,
            train_batch.x,
            state_params,
            min_var=min_var,
        )
        return jnp.mean(scalar_gaussian_nll(train_batch.y, outputs.mean, outputs.var))

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
    outputs = run_predictive_mlp_head(
        params,
        eval_prev_mean,
        eval_prev_var,
        eval_batch.x,
        state_params,
        min_var=min_var,
    )
    exact_predictive = linear_gaussian_predictive_from_filter(
        eval_oracle.filter_mean,
        eval_oracle.filter_var,
        eval_batch,
        state_params,
    )
    learned_nll_bt = scalar_gaussian_nll(eval_batch.y, outputs.mean, outputs.var)
    exact_nll_bt = scalar_gaussian_nll(eval_batch.y, exact_predictive.mean, exact_predictive.var)
    learned_rmse_t = rmse_over_batch(outputs.mean, eval_batch.y)
    exact_rmse_t = rmse_over_batch(exact_predictive.mean, eval_batch.y)

    output_dir = Path(config.get("output_dir", "outputs/linear_gaussian_predictive_head"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_loss_history(output_dir / "loss_history.csv", history)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    np.savez(output_dir / "params.npz", **{name: np.asarray(value) for name, value in params.items()})
    metrics = {
        "objective": "predictive_head",
        "train_batch_size": data_config.batch_size,
        "eval_batch_size": eval_data_config.batch_size,
        "eval_num_batches": eval_num_batches,
        "eval_total_batch_size": eval_batch.x.shape[0],
        "eval_seed_start": eval_seed_start,
        "time_steps": data_config.time_steps,
        "training_steps": int(training_config["steps"]),
        "final_loss": final_loss,
        "predictive_nll": float(jnp.mean(learned_nll_bt)),
        "predictive_rmse": float(rmse_global(outputs.mean, eval_batch.y)),
        "exact_predictive_nll": float(jnp.mean(exact_nll_bt)),
        "exact_predictive_rmse": float(rmse_global(exact_predictive.mean, eval_batch.y)),
        "mean_predictive_variance": float(jnp.mean(outputs.var)),
        "exact_mean_predictive_variance": float(jnp.mean(exact_predictive.var)),
        "variance_ratio": float(jnp.mean(outputs.var) / jnp.mean(exact_predictive.var)),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(
        output_dir / "diagnostics.npz",
        x=np.asarray(eval_batch.x),
        y=np.asarray(eval_batch.y),
        learned_predictive_mean=np.asarray(outputs.mean),
        learned_predictive_var=np.asarray(outputs.var),
        exact_predictive_mean=np.asarray(exact_predictive.mean),
        exact_predictive_var=np.asarray(exact_predictive.var),
        predictive_nll_over_time=np.asarray(jnp.mean(learned_nll_bt, axis=0)),
        exact_predictive_nll_over_time=np.asarray(jnp.mean(exact_nll_bt, axis=0)),
        predictive_rmse_over_time=np.asarray(learned_rmse_t),
        exact_predictive_rmse_over_time=np.asarray(exact_rmse_t),
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
                f"| eval batch size | {eval_batch.x.shape[0]} |",
                f"| training steps | {training_config['steps']} |",
                f"| final loss | {final_loss:.6f} |",
                f"| predictive NLL | {metrics['predictive_nll']:.6f} |",
                f"| exact predictive NLL | {metrics['exact_predictive_nll']:.6f} |",
                f"| predictive RMSE | {metrics['predictive_rmse']:.6f} |",
                f"| exact predictive RMSE | {metrics['exact_predictive_rmse']:.6f} |",
                f"| variance ratio | {metrics['variance_ratio']:.6f} |",
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
        writer.writerow(["step", "loss"])
        writer.writerows(history)


if __name__ == "__main__":
    main()
