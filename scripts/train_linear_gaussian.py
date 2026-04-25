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
from vbf.losses import (
    EdgeElboTerms,
    edge_elbo_closed_form_terms_from_outputs,
    edge_elbo_loss,
    edge_elbo_terms,
    gaussian_kl,
    oracle_edge_elbo_closed_form_terms,
    oracle_edge_elbo_terms,
    self_fed_supervised_edge_kl_loss,
    supervised_edge_kl_loss,
)
from vbf.metrics import (
    gaussian_interval_coverage,
    mean_over_batch,
    rmse_global,
    rmse_over_batch,
    rmse_time_mean,
    scalar_gaussian_kl,
    scalar_gaussian_nll,
)
from vbf.models.cells import (
    StructuredMLPOutputs,
    edge_mean_cov_from_outputs,
    init_split_head_mlp_params,
    init_structured_mlp_params,
    run_split_head_mlp_filter,
    run_split_head_mlp_teacher_forced,
    run_structured_mlp_filter,
)
from vbf.predictive import linear_gaussian_predictive_from_filter
from vbf.train import adam_update, init_adam


SUPPORTED_MODELS = {
    "supervised_edge_mlp",
    "self_fed_supervised_edge_mlp",
    "elbo_edge_mlp",
    "zero_init_edge_mlp",
    "frozen_marginal_backward_mlp",
    "supervised_edge_split_mlp",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)

    if config["model"] not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported linear-Gaussian training model: {config['model']}")

    data_config = LinearGaussianDataConfig(**config["data"])
    state_params = LinearGaussianParams(**config["state_space"])
    training_config = config["training"]
    evaluation_config = config.get("evaluation", {})
    min_var = float(training_config.get("min_var", 1e-6))
    edge_kl_weight = float(training_config.get("edge_kl_weight", 0.0))
    transition_consistency_weight = float(training_config.get("transition_consistency_weight", 0.0))
    variance_ratio_weight = float(training_config.get("variance_ratio_weight", 0.0))
    objective = config["model"]
    if objective == "zero_init_edge_mlp" and int(training_config["steps"]) != 0:
        raise ValueError("zero_init_edge_mlp must use training.steps: 0")

    train_batch = make_linear_gaussian_batch(data_config, state_params, seed=config["seed"])
    train_oracle = kalman_edge_posterior_scalar(train_batch, state_params)
    eval_data_config = LinearGaussianDataConfig(
        **{**config["data"], **evaluation_config.get("data", {})}
    )
    eval_num_batches = int(evaluation_config.get("num_batches", 1))
    eval_seed_start = int(config["seed"]) + int(evaluation_config.get("seed_offset", 10_000))
    eval_batch = _make_eval_batch(eval_data_config, state_params, eval_seed_start, eval_num_batches)
    eval_oracle = kalman_edge_posterior_scalar(eval_batch, state_params)
    params = _init_model_params(objective, config, training_config)
    opt_state = init_adam(params)

    def loss_fn(current_params: dict[str, jax.Array], key: jax.Array) -> jax.Array:
        if objective in {"supervised_edge_mlp", "zero_init_edge_mlp", "frozen_marginal_backward_mlp"}:
            return supervised_edge_kl_loss(
                current_params,
                train_batch,
                state_params,
                train_oracle,
                min_var=min_var,
            )
        if objective == "self_fed_supervised_edge_mlp":
            return self_fed_supervised_edge_kl_loss(
                current_params,
                train_batch,
                state_params,
                train_oracle,
                min_var=min_var,
                variance_ratio_weight=variance_ratio_weight,
            )
        if objective == "supervised_edge_split_mlp":
            return _supervised_split_head_edge_kl_loss(
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
            oracle=train_oracle,
            edge_kl_weight=edge_kl_weight,
            transition_consistency_weight=transition_consistency_weight,
        )

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    history: list[tuple[int, float]] = []
    train_key = jax.random.PRNGKey(config["seed"] + 2)
    for step in range(1, int(training_config["steps"]) + 1):
        train_key, step_key = jax.random.split(train_key)
        loss_value, grads = value_and_grad(params, step_key)
        if objective == "frozen_marginal_backward_mlp":
            grads = _freeze_structured_filter_head_grads(grads)
        params, opt_state = adam_update(
            params,
            grads,
            opt_state,
            learning_rate=float(training_config["learning_rate"]),
        )
        if step == 1 or step % int(training_config["log_every"]) == 0:
            history.append((step, float(loss_value)))

    final_loss = float(loss_fn(params, jax.random.PRNGKey(config["seed"] + 3)))
    outputs = _run_model_filter(params, objective, eval_batch, state_params, min_var=min_var)
    pred_edge_mean, pred_edge_cov = edge_mean_cov_from_outputs(outputs)
    filter_kl_bt = scalar_gaussian_kl(
        eval_oracle.filter_mean,
        eval_oracle.filter_var,
        outputs.filter_mean,
        outputs.filter_var,
    )
    edge_kl_bt = gaussian_kl(eval_oracle.edge_mean, eval_oracle.edge_cov, pred_edge_mean, pred_edge_cov)
    state_rmse_t = rmse_over_batch(outputs.filter_mean, eval_batch.z)
    oracle_state_rmse_t = rmse_over_batch(eval_oracle.filter_mean, eval_batch.z)
    filter_kl_t = mean_over_batch(filter_kl_bt)
    edge_kl_t = mean_over_batch(edge_kl_bt)
    filter_kl = float(jnp.mean(filter_kl_bt))
    state_rmse = float(rmse_global(outputs.filter_mean, eval_batch.z))
    state_rmse_time_mean = float(rmse_time_mean(outputs.filter_mean, eval_batch.z))
    oracle_state_rmse = float(rmse_global(eval_oracle.filter_mean, eval_batch.z))
    oracle_state_rmse_time_mean = float(rmse_time_mean(eval_oracle.filter_mean, eval_batch.z))
    state_nll_bt = scalar_gaussian_nll(eval_batch.z, outputs.filter_mean, outputs.filter_var)
    oracle_state_nll_bt = scalar_gaussian_nll(
        eval_batch.z,
        eval_oracle.filter_mean,
        eval_oracle.filter_var,
    )
    state_nll = float(jnp.mean(state_nll_bt))
    oracle_state_nll = float(jnp.mean(oracle_state_nll_bt))
    learned_predictive = linear_gaussian_predictive_from_filter(
        outputs.filter_mean,
        outputs.filter_var,
        eval_batch,
        state_params,
    )
    oracle_predictive = linear_gaussian_predictive_from_filter(
        eval_oracle.filter_mean,
        eval_oracle.filter_var,
        eval_batch,
        state_params,
    )
    predictive_nll_bt = scalar_gaussian_nll(
        eval_batch.y,
        learned_predictive.mean,
        learned_predictive.var,
    )
    oracle_predictive_nll_bt = scalar_gaussian_nll(
        eval_batch.y,
        oracle_predictive.mean,
        oracle_predictive.var,
    )
    predictive_rmse_t = rmse_over_batch(learned_predictive.mean, eval_batch.y)
    oracle_predictive_rmse_t = rmse_over_batch(oracle_predictive.mean, eval_batch.y)
    predictive_nll = float(jnp.mean(predictive_nll_bt))
    predictive_rmse = float(jnp.mean(predictive_rmse_t))
    oracle_predictive_nll = float(jnp.mean(oracle_predictive_nll_bt))
    oracle_predictive_rmse = float(jnp.mean(oracle_predictive_rmse_t))
    coverage_metrics = _coverage_metrics(
        eval_batch.z,
        outputs.filter_mean,
        outputs.filter_var,
        prefix="coverage",
    )
    oracle_coverage_metrics = _coverage_metrics(
        eval_batch.z,
        eval_oracle.filter_mean,
        eval_oracle.filter_var,
        prefix="oracle_coverage",
    )
    mean_filter_var = float(jnp.mean(outputs.filter_var))
    oracle_mean_filter_var = float(jnp.mean(eval_oracle.filter_var))
    variance_ratio = mean_filter_var / oracle_mean_filter_var
    mean_backward_var = float(jnp.mean(outputs.backward_var))
    mean_edge_var_trace = float(jnp.mean(jnp.trace(pred_edge_cov, axis1=-2, axis2=-1)))
    max_abs_edge_mean = float(jnp.max(jnp.abs(pred_edge_mean)))
    elbo_terms = None
    oracle_elbo_terms = None
    if objective == "elbo_edge_mlp":
        elbo_terms = edge_elbo_terms(
            params,
            eval_batch,
            state_params,
            jax.random.PRNGKey(config["seed"] + 4),
            num_samples=int(training_config.get("num_elbo_samples", 8)),
            min_var=min_var,
        )
        oracle_elbo_terms = oracle_edge_elbo_terms(
            eval_oracle,
            eval_batch,
            state_params,
            jax.random.PRNGKey(config["seed"] + 5),
            num_samples=int(training_config.get("num_elbo_samples", 8)),
        )
    closed_form_elbo_terms = edge_elbo_closed_form_terms_from_outputs(
        outputs,
        eval_batch,
        state_params,
    )
    oracle_closed_form_elbo_terms = oracle_edge_elbo_closed_form_terms(
        eval_oracle,
        eval_batch,
        state_params,
    )

    output_dir = Path(config.get("output_dir", "outputs/linear_gaussian_supervised_edge_mlp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_loss_history(output_dir / "loss_history.csv", history)
    _write_config(output_dir / "config.yaml", config)
    np.savez(output_dir / "params.npz", **{name: np.asarray(value) for name, value in params.items()})
    metrics = {
        "objective": objective,
        "train_batch_size": data_config.batch_size,
        "eval_batch_size": eval_data_config.batch_size,
        "eval_num_batches": eval_num_batches,
        "eval_total_batch_size": eval_batch.x.shape[0],
        "eval_seed_start": eval_seed_start,
        "time_steps": data_config.time_steps,
        "training_steps": int(training_config["steps"]),
        "num_elbo_samples": int(training_config.get("num_elbo_samples", 0)),
        "edge_kl_weight": edge_kl_weight,
        "transition_consistency_weight": transition_consistency_weight,
        "variance_ratio_weight": variance_ratio_weight,
        "final_loss": final_loss,
        "final_edge_kl": float(jnp.mean(edge_kl_bt)),
        "filter_kl": filter_kl,
        "edge_kl": float(jnp.mean(edge_kl_bt)),
        "state_rmse": state_rmse,
        "state_rmse_global": state_rmse,
        "state_rmse_time_mean": state_rmse_time_mean,
        "state_nll": state_nll,
        "oracle_state_rmse": oracle_state_rmse,
        "oracle_state_rmse_global": oracle_state_rmse,
        "oracle_state_rmse_time_mean": oracle_state_rmse_time_mean,
        "oracle_state_nll": oracle_state_nll,
        "predictive_nll": predictive_nll,
        "predictive_rmse": predictive_rmse,
        "oracle_predictive_nll": oracle_predictive_nll,
        "oracle_predictive_rmse": oracle_predictive_rmse,
        "mean_filter_variance": mean_filter_var,
        "oracle_mean_filter_variance": oracle_mean_filter_var,
        "variance_ratio": variance_ratio,
        "mean_backward_variance": mean_backward_var,
        "mean_edge_covariance_trace": mean_edge_var_trace,
        "max_abs_edge_mean": max_abs_edge_mean,
        **coverage_metrics,
        **oracle_coverage_metrics,
    }
    if elbo_terms is not None:
        metrics.update(_mean_elbo_term_metrics(elbo_terms, prefix="elbo"))
    if oracle_elbo_terms is not None:
        metrics.update(_mean_elbo_term_metrics(oracle_elbo_terms, prefix="oracle_elbo"))
    metrics.update(_mean_elbo_term_metrics(closed_form_elbo_terms, prefix="closed_form_elbo"))
    metrics.update(
        _mean_elbo_term_metrics(
            oracle_closed_form_elbo_terms,
            prefix="oracle_closed_form_elbo",
        )
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics = {
        "x": np.asarray(eval_batch.x),
        "y": np.asarray(eval_batch.y),
        "z": np.asarray(eval_batch.z),
        "oracle_filter_mean": np.asarray(eval_oracle.filter_mean),
        "oracle_filter_var": np.asarray(eval_oracle.filter_var),
        "learned_filter_mean": np.asarray(outputs.filter_mean),
        "learned_filter_var": np.asarray(outputs.filter_var),
        "learned_predictive_mean": np.asarray(learned_predictive.mean),
        "learned_predictive_var": np.asarray(learned_predictive.var),
        "oracle_predictive_mean": np.asarray(oracle_predictive.mean),
        "oracle_predictive_var": np.asarray(oracle_predictive.var),
        "edge_kl_over_time": np.asarray(edge_kl_t),
        "filter_kl_over_time": np.asarray(filter_kl_t),
        "state_rmse_over_time": np.asarray(state_rmse_t),
        "oracle_state_rmse_over_time": np.asarray(oracle_state_rmse_t),
        "state_nll_over_time": np.asarray(mean_over_batch(state_nll_bt)),
        "oracle_state_nll_over_time": np.asarray(mean_over_batch(oracle_state_nll_bt)),
        "predictive_nll_over_time": np.asarray(mean_over_batch(predictive_nll_bt)),
        "predictive_rmse_over_time": np.asarray(predictive_rmse_t),
        "oracle_predictive_nll_over_time": np.asarray(mean_over_batch(oracle_predictive_nll_bt)),
        "oracle_predictive_rmse_over_time": np.asarray(oracle_predictive_rmse_t),
        "mean_filter_variance_over_time": np.asarray(mean_over_batch(outputs.filter_var)),
        "oracle_mean_filter_variance_over_time": np.asarray(mean_over_batch(eval_oracle.filter_var)),
        "variance_ratio_over_time": np.asarray(
            mean_over_batch(outputs.filter_var) / mean_over_batch(eval_oracle.filter_var)
        ),
        **_coverage_time_series(eval_batch.z, outputs.filter_mean, outputs.filter_var, "coverage"),
        **_coverage_time_series(
            eval_batch.z,
            eval_oracle.filter_mean,
            eval_oracle.filter_var,
            "oracle_coverage",
        ),
        "loss_history_step": np.asarray([step for step, _ in history], dtype=np.int64),
        "loss_history_loss": np.asarray([loss for _, loss in history], dtype=np.float64),
    }
    if elbo_terms is not None:
        diagnostics.update(_elbo_term_time_series(elbo_terms, prefix="elbo"))
    if oracle_elbo_terms is not None:
        diagnostics.update(_elbo_term_time_series(oracle_elbo_terms, prefix="oracle_elbo"))
    diagnostics.update(_elbo_term_time_series(closed_form_elbo_terms, prefix="closed_form_elbo"))
    diagnostics.update(
        _elbo_term_time_series(
            oracle_closed_form_elbo_terms,
            prefix="oracle_closed_form_elbo",
        )
    )
    np.savez(output_dir / "diagnostics.npz", **diagnostics)
    elbo_summary_lines = [
        *_elbo_summary_lines(elbo_terms, label="ELBO"),
        *_elbo_summary_lines(oracle_elbo_terms, label="oracle ELBO"),
        *_elbo_summary_lines(closed_form_elbo_terms, label="closed-form ELBO"),
        *_elbo_summary_lines(
            oracle_closed_form_elbo_terms,
            label="oracle closed-form ELBO",
        ),
    ]
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
                f"| state RMSE global | {state_rmse:.6f} |",
                f"| state RMSE time mean | {state_rmse_time_mean:.6f} |",
                f"| state NLL | {state_nll:.6f} |",
                f"| oracle state RMSE global | {oracle_state_rmse:.6f} |",
                f"| oracle state RMSE time mean | {oracle_state_rmse_time_mean:.6f} |",
                f"| oracle state NLL | {oracle_state_nll:.6f} |",
                f"| predictive NLL | {predictive_nll:.6f} |",
                f"| predictive RMSE | {predictive_rmse:.6f} |",
                f"| oracle predictive NLL | {oracle_predictive_nll:.6f} |",
                f"| oracle predictive RMSE | {oracle_predictive_rmse:.6f} |",
                f"| coverage 50 | {coverage_metrics['coverage_50']:.6f} |",
                f"| coverage 90 | {coverage_metrics['coverage_90']:.6f} |",
                f"| coverage 95 | {coverage_metrics['coverage_95']:.6f} |",
                f"| oracle coverage 50 | {oracle_coverage_metrics['oracle_coverage_50']:.6f} |",
                f"| oracle coverage 90 | {oracle_coverage_metrics['oracle_coverage_90']:.6f} |",
                f"| oracle coverage 95 | {oracle_coverage_metrics['oracle_coverage_95']:.6f} |",
                f"| mean filter variance | {mean_filter_var:.6f} |",
                f"| oracle mean filter variance | {oracle_mean_filter_var:.6f} |",
                f"| variance ratio | {variance_ratio:.6f} |",
                f"| mean backward variance | {mean_backward_var:.6f} |",
                f"| mean edge covariance trace | {mean_edge_var_trace:.6f} |",
                f"| max abs edge mean | {max_abs_edge_mean:.6f} |",
                *elbo_summary_lines,
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
                "- `config.yaml`",
                "- `params.npz`",
                "- `diagnostics.npz`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


def _mean_elbo_term_metrics(terms: EdgeElboTerms, *, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_log_likelihood": float(jnp.mean(terms.log_likelihood)),
        f"{prefix}_log_transition": float(jnp.mean(terms.log_transition)),
        f"{prefix}_log_prev_filter": float(jnp.mean(terms.log_prev_filter)),
        f"{prefix}_neg_log_current_filter": float(jnp.mean(terms.neg_log_current_filter)),
        f"{prefix}_neg_log_backward": float(jnp.mean(terms.neg_log_backward)),
        prefix: float(jnp.mean(terms.elbo)),
    }


def _elbo_term_time_series(terms: EdgeElboTerms, *, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_log_likelihood_over_time": np.asarray(mean_over_batch(terms.log_likelihood)),
        f"{prefix}_log_transition_over_time": np.asarray(mean_over_batch(terms.log_transition)),
        f"{prefix}_log_prev_filter_over_time": np.asarray(mean_over_batch(terms.log_prev_filter)),
        f"{prefix}_neg_log_current_filter_over_time": np.asarray(
            mean_over_batch(terms.neg_log_current_filter)
        ),
        f"{prefix}_neg_log_backward_over_time": np.asarray(mean_over_batch(terms.neg_log_backward)),
        f"{prefix}_over_time": np.asarray(mean_over_batch(terms.elbo)),
    }


def _elbo_summary_lines(terms: EdgeElboTerms | None, *, label: str) -> list[str]:
    if terms is None:
        return []
    metrics = _mean_elbo_term_metrics(terms, prefix="term")
    return [
        f"| {label} | {metrics['term']:.6f} |",
        f"| {label} log likelihood | {metrics['term_log_likelihood']:.6f} |",
        f"| {label} log transition | {metrics['term_log_transition']:.6f} |",
        f"| {label} log previous filter | {metrics['term_log_prev_filter']:.6f} |",
        f"| {label} negative log current filter | {metrics['term_neg_log_current_filter']:.6f} |",
        f"| {label} negative log backward | {metrics['term_neg_log_backward']:.6f} |",
    ]


def _coverage_metrics(
    value: jax.Array,
    mean: jax.Array,
    var: jax.Array,
    *,
    prefix: str,
) -> dict[str, float]:
    z_scores = {
        "50": 0.6744897501960817,
        "90": 1.6448536269514722,
        "95": 1.959963984540054,
    }
    return {
        f"{prefix}_{level}": float(
            gaussian_interval_coverage(value, mean, var, z_score=z_score)
        )
        for level, z_score in z_scores.items()
    }


def _coverage_time_series(
    value: jax.Array,
    mean: jax.Array,
    var: jax.Array,
    prefix: str,
) -> dict[str, np.ndarray]:
    z_scores = {
        "50": 0.6744897501960817,
        "90": 1.6448536269514722,
        "95": 1.959963984540054,
    }
    metrics = {}
    for level, z_score in z_scores.items():
        half_width = z_score * jnp.sqrt(var)
        covered = (value >= mean - half_width) & (value <= mean + half_width)
        metrics[f"{prefix}_{level}_over_time"] = np.asarray(jnp.mean(covered, axis=0))
    return metrics


def _init_model_params(
    objective: str,
    config: dict,
    training_config: dict,
) -> dict[str, jax.Array]:
    key = jax.random.PRNGKey(config["seed"] + 1)
    hidden_dim = int(training_config["hidden_dim"])
    if objective == "supervised_edge_split_mlp":
        return init_split_head_mlp_params(key, hidden_dim=hidden_dim)
    return init_structured_mlp_params(key, hidden_dim=hidden_dim)


def _run_model_filter(
    params: dict[str, jax.Array],
    objective: str,
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    *,
    min_var: float,
) -> StructuredMLPOutputs:
    if objective == "supervised_edge_split_mlp":
        return run_split_head_mlp_filter(params, batch, state_params, min_var=min_var)
    return run_structured_mlp_filter(params, batch, state_params, min_var=min_var)


def _supervised_split_head_edge_kl_loss(
    params: dict[str, jax.Array],
    batch: EpisodeBatch,
    state_params: LinearGaussianParams,
    oracle,
    *,
    min_var: float,
) -> jax.Array:
    outputs = run_split_head_mlp_teacher_forced(
        params,
        batch,
        state_params,
        oracle.filter_mean,
        oracle.filter_var,
        min_var=min_var,
    )
    pred_mean, pred_cov = edge_mean_cov_from_outputs(outputs)
    return jnp.mean(gaussian_kl(oracle.edge_mean, oracle.edge_cov, pred_mean, pred_cov))


def _freeze_structured_filter_head_grads(
    grads: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    """Freeze raw gain/variance outputs while learning the backward conditional."""

    return {
        **grads,
        "w2": grads["w2"].at[:, :2].set(0.0),
        "b2": grads["b2"].at[:2].set(0.0),
    }


def _write_config(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


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
