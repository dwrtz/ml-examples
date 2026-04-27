"""Plot nonlinear learned-filter and grid-reference diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    diagnostics_path = run_dir / "diagnostics.npz"
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing diagnostics: {diagnostics_path}")
    diagnostics = dict(np.load(diagnostics_path))
    episode = int(args.episode)
    if episode < 0 or episode >= diagnostics["z"].shape[0]:
        raise ValueError(f"episode must be in [0, {diagnostics['z'].shape[0] - 1}]")

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = plot_dir / f"posterior_episode_{episode}.png"
    variance_path = plot_dir / "variance_over_time.png"
    predictive_path = plot_dir / f"predictive_episode_{episode}.png"
    _plot_posterior_episode(diagnostics, episode, posterior_path)
    _plot_variance_over_time(diagnostics, variance_path)
    _plot_predictive_episode(diagnostics, episode, predictive_path)
    print(f"Wrote {posterior_path}")
    print(f"Wrote {variance_path}")
    print(f"Wrote {predictive_path}")


def _plot_posterior_episode(data: dict[str, np.ndarray], episode: int, path: Path) -> None:
    time = np.arange(data["z"].shape[1])
    z = data["z"][episode]
    reference_mean = data["reference_filter_mean"][episode]
    reference_std = np.sqrt(data["reference_filter_var"][episode])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, data["x"][episode], label="x", color="tab:blue")
    axes[0].plot(time, data["y"][episode], label="y", color="tab:green", alpha=0.85)
    axes[0].set_title("Observed covariate and measurement")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(time, z, label="true z", color="black", linestyle=":")
    axes[1].plot(time, reference_mean, label="grid reference", color="tab:red")
    axes[1].fill_between(
        time,
        reference_mean - 2.0 * reference_std,
        reference_mean + 2.0 * reference_std,
        color="tab:red",
        alpha=0.16,
    )
    if "learned_filter_mean" in data:
        learned_mean = data["learned_filter_mean"][episode]
        learned_std = np.sqrt(data["learned_filter_var"][episode])
        axes[1].plot(time, learned_mean, label="learned", color="tab:purple")
        axes[1].fill_between(
            time,
            learned_mean - 2.0 * learned_std,
            learned_mean + 2.0 * learned_std,
            color="tab:purple",
            alpha=0.16,
        )
    axes[1].set_title("Filtering posterior")
    axes[1].set_xlabel("time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_variance_over_time(data: dict[str, np.ndarray], path: Path) -> None:
    time = np.arange(data["reference_filter_var"].shape[1])
    reference_var_t = np.mean(data["reference_filter_var"], axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, reference_var_t, label="grid reference", color="tab:red")
    if "learned_filter_var" in data:
        learned_var_t = np.mean(data["learned_filter_var"], axis=0)
        axes[0].plot(time, learned_var_t, label="learned", color="tab:purple")
        ratio = learned_var_t / np.maximum(reference_var_t, 1e-12)
        axes[1].plot(time, ratio, color="tab:purple")
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    else:
        axes[1].axis("off")
    axes[0].set_ylabel("mean filter variance")
    axes[0].set_title("Variance over time")
    axes[0].legend(loc="best")
    axes[1].set_ylabel("learned / reference")
    axes[1].set_xlabel("time")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_predictive_episode(data: dict[str, np.ndarray], episode: int, path: Path) -> None:
    time = np.arange(data["y"].shape[1])
    fig, axis = plt.subplots(1, 1, figsize=(11, 4), constrained_layout=True)
    axis.plot(time, data["y"][episode], label="y", color="black", linestyle=":")
    reference_mean = data["reference_predictive_mean"][episode]
    reference_std = np.sqrt(data["reference_predictive_var"][episode])
    axis.plot(time, reference_mean, label="grid reference pred", color="tab:red")
    axis.fill_between(
        time,
        reference_mean - 2.0 * reference_std,
        reference_mean + 2.0 * reference_std,
        color="tab:red",
        alpha=0.16,
    )
    if "learned_predictive_mean" in data:
        learned_mean = data["learned_predictive_mean"][episode]
        learned_std = np.sqrt(data["learned_predictive_var"][episode])
        axis.plot(time, learned_mean, label="learned pred", color="tab:purple")
        axis.fill_between(
            time,
            learned_mean - 2.0 * learned_std,
            learned_mean + 2.0 * learned_std,
            color="tab:purple",
            alpha=0.16,
        )
    axis.set_title("One-step predictive measurement distribution")
    axis.set_xlabel("time")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
