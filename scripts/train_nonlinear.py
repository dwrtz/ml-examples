"""Train learned filters on nonlinear benchmark variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from vbf.data import LinearGaussianParams
from vbf.metrics import gaussian_interval_coverage, rmse_global, scalar_gaussian_nll
from vbf.models.cells import (
    edge_mean_cov_from_outputs,
    init_direct_mlp_params,
    run_direct_mlp_filter,
)
from vbf.nonlinear import (
    GridReferenceConfig,
    NonlinearDataConfig,
    make_nonlinear_batch,
    nonlinear_grid_filter,
    nonlinear_observation_mean,
    nonlinear_predictive_moments_from_filter,
)
from vbf.train import adam_update, init_adam


LOG_2PI = jnp.log(2.0 * jnp.pi)
SUPPORTED_MODELS = {"direct_elbo_sine_mlp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)

    if config["model"] not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported nonlinear training model: {config['model']}")

    data_config = NonlinearDataConfig(**config["data"])
    state_params = LinearGaussianParams(**config["state_space"])
    training_config = config["training"]
    evaluation_config = config.get("evaluation", {})
    reference_config = GridReferenceConfig(**config.get("reference", {}))
    min_var = float(training_config.get("min_var", 1e-6))

    train_batch = make_nonlinear_batch(data_config, state_params, seed=int(config["seed"]))
    eval_data_config = NonlinearDataConfig(**{**config["data"], **evaluation_config.get("data", {})})
    eval_batch = make_nonlinear_batch(
        eval_data_config,
        state_params,
        seed=int(config["seed"]) + int(evaluation_config.get("seed_offset", 10_000)),
    )

    params = init_direct_mlp_params(
        jax.random.PRNGKey(int(config["seed"]) + 1),
        hidden_dim=int(training_config["hidden_dim"]),
    )
    opt_state = init_adam(params)

    def loss_fn(current_params: dict[str, jax.Array], key: jax.Array) -> jax.Array:
        outputs = run_direct_mlp_filter(
            current_params,
            train_batch,
            state_params,
            min_var=min_var,
        )
        return -jnp.mean(
            _nonlinear_edge_elbo(
                outputs,
                train_batch,
                state_params,
                key,
                observation=data_config.observation,
                num_samples=int(training_config.get("num_elbo_samples", 8)),
            )
        )

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    history: list[tuple[int, float]] = []
    train_key = jax.random.PRNGKey(int(config["seed"]) + 2)
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

    final_loss = float(loss_fn(params, jax.random.PRNGKey(int(config["seed"]) + 3)))
    outputs = run_direct_mlp_filter(params, eval_batch, state_params, min_var=min_var)
    reference = nonlinear_grid_filter(
        eval_batch,
        state_params,
        data_config=eval_data_config,
        grid_config=reference_config,
    )
    learned_predictive_mean, learned_predictive_var = nonlinear_predictive_moments_from_filter(
        outputs.filter_mean,
        outputs.filter_var,
        eval_batch.x,
        state_params,
        observation=eval_data_config.observation,
    )
    learned_state_nll = scalar_gaussian_nll(eval_batch.z, outputs.filter_mean, outputs.filter_var)
    reference_state_nll = scalar_gaussian_nll(
        eval_batch.z,
        reference.filter_mean,
        reference.filter_var,
    )
    learned_predictive_nll = scalar_gaussian_nll(
        eval_batch.y,
        learned_predictive_mean,
        learned_predictive_var,
    )
    reference_predictive_nll = scalar_gaussian_nll(
        eval_batch.y,
        reference.predictive_mean,
        reference.predictive_var,
    )
    _, edge_cov = edge_mean_cov_from_outputs(outputs)
    metrics = {
        "benchmark": "nonlinear",
        "objective": config["model"],
        "observation": eval_data_config.observation,
        "x_pattern": eval_data_config.x_pattern,
        "training_steps": int(training_config["steps"]),
        "num_elbo_samples": int(training_config.get("num_elbo_samples", 8)),
        "final_loss": final_loss,
        "state_rmse": float(rmse_global(outputs.filter_mean, eval_batch.z)),
        "reference_state_rmse": float(rmse_global(reference.filter_mean, eval_batch.z)),
        "state_nll": float(jnp.mean(learned_state_nll)),
        "reference_state_nll": float(jnp.mean(reference_state_nll)),
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
        "mean_edge_covariance_trace": float(jnp.mean(jnp.trace(edge_cov, axis1=-2, axis2=-1))),
    }

    output_dir = Path(config.get("output_dir", "outputs/nonlinear_direct_elbo_sine_mlp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_loss_history(output_dir / "loss_history.csv", history)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez(output_dir / "params.npz", **{name: np.asarray(value) for name, value in params.items()})
    np.savez(
        output_dir / "diagnostics.npz",
        x=np.asarray(eval_batch.x),
        y=np.asarray(eval_batch.y),
        z=np.asarray(eval_batch.z),
        learned_filter_mean=np.asarray(outputs.filter_mean),
        learned_filter_var=np.asarray(outputs.filter_var),
        reference_filter_mean=np.asarray(reference.filter_mean),
        reference_filter_var=np.asarray(reference.filter_var),
        learned_predictive_mean=np.asarray(learned_predictive_mean),
        learned_predictive_var=np.asarray(learned_predictive_var),
        reference_predictive_mean=np.asarray(reference.predictive_mean),
        reference_predictive_var=np.asarray(reference.predictive_var),
        loss_history_step=np.asarray([step for step, _ in history], dtype=np.int64),
        loss_history_loss=np.asarray([loss for _, loss in history], dtype=np.float64),
    )
    summary_path = output_dir / "evaluation_summary.md"
    summary_path.write_text(_render_summary(config["name"], metrics, history), encoding="utf-8")
    print(f"Wrote {summary_path}")


def _nonlinear_edge_elbo(
    outputs,
    batch,
    state_params: LinearGaussianParams,
    key: jax.Array,
    *,
    observation: str,
    num_samples: int,
) -> jax.Array:
    eps_t_key, eps_tm1_key = jax.random.split(key)
    sample_shape = (num_samples,) + outputs.filter_mean.shape
    eps_t = jax.random.normal(eps_t_key, shape=sample_shape, dtype=outputs.filter_mean.dtype)
    eps_tm1 = jax.random.normal(eps_tm1_key, shape=sample_shape, dtype=outputs.filter_mean.dtype)
    z_t = outputs.filter_mean[None, ...] + jnp.sqrt(outputs.filter_var)[None, ...] * eps_t
    backward_mean = outputs.backward_a[None, ...] * z_t + outputs.backward_b[None, ...]
    z_tm1 = backward_mean + jnp.sqrt(outputs.backward_var)[None, ...] * eps_tm1
    prev_mean, prev_var = _previous_filter_beliefs(outputs.filter_mean, outputs.filter_var, state_params)
    observation_mean = nonlinear_observation_mean(z_t, batch.x[None, ...], observation)
    elbo = (
        _normal_log_prob(batch.y[None, ...], observation_mean, state_params.r)
        + _normal_log_prob(z_t, z_tm1, state_params.q)
        + _normal_log_prob(z_tm1, prev_mean[None, ...], prev_var[None, ...])
        - _normal_log_prob(z_t, outputs.filter_mean[None, ...], outputs.filter_var[None, ...])
        - _normal_log_prob(z_tm1, backward_mean, outputs.backward_var[None, ...])
    )
    return jnp.mean(elbo, axis=0)


def _previous_filter_beliefs(
    filter_mean: jax.Array,
    filter_var: jax.Array,
    state_params: LinearGaussianParams,
) -> tuple[jax.Array, jax.Array]:
    initial_mean = jnp.full((filter_mean.shape[0], 1), state_params.m0, dtype=filter_mean.dtype)
    initial_var = jnp.full((filter_var.shape[0], 1), state_params.p0, dtype=filter_var.dtype)
    return (
        jnp.concatenate((initial_mean, filter_mean[:, :-1]), axis=1),
        jnp.concatenate((initial_var, filter_var[:, :-1]), axis=1),
    )


def _normal_log_prob(value: jax.Array, mean: jax.Array | float, var: jax.Array | float) -> jax.Array:
    return -0.5 * (LOG_2PI + jnp.log(var) + (value - mean) ** 2 / var)


def _write_loss_history(path: Path, history: list[tuple[int, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "loss"))
        writer.writerows(history)


def _render_summary(
    name: str,
    metrics: dict[str, float | int | str],
    history: list[tuple[int, float]],
) -> str:
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
    lines.extend(["", "## Loss History", "", "| Step | Loss |", "|---:|---:|"])
    lines.extend(f"| {step} | {loss:.6f} |" for step, loss in history)
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `loss_history.csv`",
            "- `metrics.json`",
            "- `config.yaml`",
            "- `params.npz`",
            "- `diagnostics.npz`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
