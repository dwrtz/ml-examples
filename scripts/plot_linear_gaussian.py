"""Plot scalar linear-Gaussian benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vbf.plotting import load_linear_gaussian_diagnostics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    diagnostics = load_linear_gaussian_diagnostics(run_dir)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    episode = args.episode
    if episode < 0 or episode >= diagnostics["z"].shape[0]:
        raise ValueError(f"episode must be in [0, {diagnostics['z'].shape[0] - 1}]")

    posterior_path = plot_dir / f"posterior_episode_{episode}.png"
    metrics_path = plot_dir / "metrics_over_time.png"
    _plot_posterior_episode(diagnostics, episode, posterior_path)
    _plot_metrics_over_time(diagnostics, metrics_path)
    print(f"Wrote {posterior_path}")
    print(f"Wrote {metrics_path}")
    if "elbo_over_time" in diagnostics:
        elbo_terms_path = plot_dir / "elbo_terms_over_time.png"
        _plot_elbo_terms_over_time(diagnostics, elbo_terms_path)
        print(f"Wrote {elbo_terms_path}")


def _plot_posterior_episode(data: dict[str, np.ndarray], episode: int, path: Path) -> None:
    time = np.arange(data["z"].shape[1])
    z = data["z"][episode]
    oracle_mean = data["oracle_filter_mean"][episode]
    oracle_std = np.sqrt(data["oracle_filter_var"][episode])
    learned_mean = data["learned_filter_mean"][episode]
    learned_std = np.sqrt(data["learned_filter_var"][episode])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, data["x"][episode], label="x", color="tab:blue")
    axes[0].plot(time, data["y"][episode], label="y", color="tab:green", alpha=0.8)
    if "learned_predictive_mean" in data:
        learned_pred_mean = data["learned_predictive_mean"][episode]
        learned_pred_std = np.sqrt(data["learned_predictive_var"][episode])
        oracle_pred_mean = data["oracle_predictive_mean"][episode]
        oracle_pred_std = np.sqrt(data["oracle_predictive_var"][episode])
        axes[0].plot(time, oracle_pred_mean, label="oracle pred", color="tab:red", alpha=0.8)
        axes[0].fill_between(
            time,
            oracle_pred_mean - 2.0 * oracle_pred_std,
            oracle_pred_mean + 2.0 * oracle_pred_std,
            color="tab:red",
            alpha=0.10,
        )
        axes[0].plot(time, learned_pred_mean, label="learned pred", color="tab:purple", alpha=0.8)
        axes[0].fill_between(
            time,
            learned_pred_mean - 2.0 * learned_pred_std,
            learned_pred_mean + 2.0 * learned_pred_std,
            color="tab:purple",
            alpha=0.08,
        )
    axes[0].set_title("Observations")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(time, z, label="true z", color="black", linestyle=":")
    axes[1].plot(time, oracle_mean, label="oracle", color="tab:red")
    axes[1].fill_between(
        time,
        oracle_mean - 2.0 * oracle_std,
        oracle_mean + 2.0 * oracle_std,
        color="tab:red",
        alpha=0.18,
    )
    axes[1].plot(time, learned_mean, label="learned", color="tab:purple")
    axes[1].fill_between(
        time,
        learned_mean - 2.0 * learned_std,
        learned_mean + 2.0 * learned_std,
        color="tab:purple",
        alpha=0.14,
    )
    axes[1].set_title("Filtering posterior")
    axes[1].set_xlabel("time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_metrics_over_time(data: dict[str, np.ndarray], path: Path) -> None:
    time = np.arange(data["filter_kl_over_time"].shape[0])
    has_predictive = "predictive_nll_over_time" in data
    num_axes = 4 if has_predictive else 3
    fig, axes = plt.subplots(
        num_axes, 1, figsize=(10, 10 if has_predictive else 8), sharex=True, constrained_layout=True
    )
    axes[0].plot(time, data["edge_kl_over_time"], color="tab:orange")
    axes[0].set_ylabel("edge KL")
    axes[1].plot(time, data["filter_kl_over_time"], color="tab:purple")
    axes[1].set_ylabel("filter KL")
    axes[2].plot(time, data["state_rmse_over_time"], color="tab:blue")
    axes[2].set_ylabel("state RMSE")
    if has_predictive:
        axes[3].plot(time, data["predictive_nll_over_time"], color="tab:green", label="learned")
        axes[3].plot(
            time,
            data["oracle_predictive_nll_over_time"],
            color="tab:red",
            linestyle="--",
            label="oracle",
        )
        axes[3].set_ylabel("pred NLL")
        axes[3].legend(loc="best")
    axes[-1].set_xlabel("time")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_elbo_terms_over_time(data: dict[str, np.ndarray], path: Path) -> None:
    time = np.arange(data["elbo_over_time"].shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)

    axes[0].plot(time, data["elbo_over_time"], color="black", label="learned")
    if "oracle_elbo_over_time" in data:
        axes[0].plot(time, data["oracle_elbo_over_time"], color="tab:red", label="oracle")
    axes[0].set_ylabel("ELBO")
    axes[0].legend(loc="best")

    term_specs = [
        ("log_likelihood", "log likelihood"),
        ("log_transition", "log transition"),
        ("log_prev_filter", "log previous filter"),
        ("neg_log_current_filter", "-log current filter"),
        ("neg_log_backward", "-log backward"),
    ]
    for term, label in term_specs:
        axes[1].plot(time, data[f"elbo_{term}_over_time"], label=f"learned {label}")
        oracle_key = f"oracle_elbo_{term}_over_time"
        if oracle_key in data:
            axes[1].plot(time, data[oracle_key], linestyle="--", label=f"oracle {label}")
    axes[1].set_ylabel("term value")
    axes[1].set_xlabel("time")
    axes[1].legend(loc="best", ncols=2)

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
