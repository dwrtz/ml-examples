"""Plot nonlinear learned-filter and grid-reference diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from vbf.data import EpisodeBatch, LinearGaussianParams  # noqa: E402
from vbf.nonlinear import (  # noqa: E402
    GridReferenceConfig,
    NonlinearDataConfig,
    nonlinear_grid_filter_shape_diagnostics,
)


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
    time_metrics_path = plot_dir / "time_metrics.csv"
    time_calibration_path = plot_dir / "time_calibration.png"
    time_metrics = _time_metrics(diagnostics)
    _plot_posterior_episode(diagnostics, episode, posterior_path)
    _plot_variance_over_time(diagnostics, variance_path)
    _plot_predictive_episode(diagnostics, episode, predictive_path)
    _write_time_metrics(time_metrics_path, time_metrics)
    _plot_time_calibration(time_metrics, time_calibration_path)
    shape_metrics_path = None
    shape_plot_path = None
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        shape_metrics = _reference_shape_metrics(diagnostics, config_path)
        shape_metrics_path = plot_dir / "reference_shape_metrics.csv"
        shape_plot_path = plot_dir / "reference_shape.png"
        _write_time_metrics(shape_metrics_path, shape_metrics)
        _plot_reference_shape(shape_metrics, time_metrics, shape_plot_path)
    print(f"Wrote {posterior_path}")
    print(f"Wrote {variance_path}")
    print(f"Wrote {predictive_path}")
    print(f"Wrote {time_metrics_path}")
    print(f"Wrote {time_calibration_path}")
    if shape_metrics_path is not None and shape_plot_path is not None:
        print(f"Wrote {shape_metrics_path}")
        print(f"Wrote {shape_plot_path}")


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


def _time_metrics(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    reference_var_t = np.mean(data["reference_filter_var"], axis=0)
    learned_var_t = np.mean(data["learned_filter_var"], axis=0)
    reference_nll_t = _scalar_gaussian_nll(
        data["z"],
        data["reference_filter_mean"],
        data["reference_filter_var"],
    ).mean(axis=0)
    learned_nll_t = _scalar_gaussian_nll(
        data["z"],
        data["learned_filter_mean"],
        data["learned_filter_var"],
    ).mean(axis=0)
    return {
        "time": np.arange(data["z"].shape[1]),
        "mean_x2": np.mean(data["x"] ** 2, axis=0),
        "reference_filter_var": reference_var_t,
        "learned_filter_var": learned_var_t,
        "variance_ratio": learned_var_t / np.maximum(reference_var_t, 1e-12),
        "reference_coverage_90": _coverage(
            data["z"],
            data["reference_filter_mean"],
            data["reference_filter_var"],
            z_score=1.6448536269514722,
        ),
        "learned_coverage_90": _coverage(
            data["z"],
            data["learned_filter_mean"],
            data["learned_filter_var"],
            z_score=1.6448536269514722,
        ),
        "reference_state_nll": reference_nll_t,
        "learned_state_nll": learned_nll_t,
    }


def _write_time_metrics(path: Path, metrics: dict[str, np.ndarray]) -> None:
    fieldnames = list(metrics)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(metrics["time"])):
            writer.writerow({field: float(metrics[field][index]) for field in fieldnames})


def _plot_time_calibration(metrics: dict[str, np.ndarray], path: Path) -> None:
    time = metrics["time"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)

    axes[0].plot(time, metrics["mean_x2"], color="tab:blue")
    axes[0].set_ylabel("mean x^2")
    axes[0].set_title("Observation strength")

    axes[1].plot(time, metrics["variance_ratio"], color="tab:purple")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("var ratio")
    axes[1].set_title("Learned / reference variance")

    axes[2].plot(time, metrics["reference_coverage_90"], label="grid reference", color="tab:red")
    axes[2].plot(time, metrics["learned_coverage_90"], label="learned", color="tab:purple")
    axes[2].axhline(0.9, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("coverage 90")
    axes[2].legend(loc="best")

    axes[3].plot(time, metrics["reference_state_nll"], label="grid reference", color="tab:red")
    axes[3].plot(time, metrics["learned_state_nll"], label="learned", color="tab:purple")
    axes[3].set_ylabel("state NLL")
    axes[3].set_xlabel("time")
    axes[3].legend(loc="best")

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _reference_shape_metrics(
    data: dict[str, np.ndarray],
    config_path: Path,
) -> dict[str, np.ndarray]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = NonlinearDataConfig(**config["data"])
    state_params = LinearGaussianParams(**config["state_space"])
    grid_config = GridReferenceConfig(**config.get("reference", {}))
    batch = EpisodeBatch(
        x=data["x"],
        y=data["y"],
        z=data["z"],
    )
    shape = nonlinear_grid_filter_shape_diagnostics(
        batch,
        state_params,
        data_config=data_config,
        grid_config=grid_config,
    )
    return {
        "time": np.arange(data["z"].shape[1]),
        "mean_entropy": np.mean(np.asarray(shape.entropy), axis=0),
        "mean_normalized_entropy": np.mean(np.asarray(shape.normalized_entropy), axis=0),
        "mean_peak_count": np.mean(np.asarray(shape.peak_count), axis=0),
        "max_peak_count": np.max(np.asarray(shape.peak_count), axis=0),
        "mean_max_mass": np.mean(np.asarray(shape.max_mass), axis=0),
        "mean_credible_width_90": np.mean(np.asarray(shape.credible_width_90), axis=0),
    }


def _plot_reference_shape(
    shape_metrics: dict[str, np.ndarray],
    time_metrics: dict[str, np.ndarray],
    path: Path,
) -> None:
    time = shape_metrics["time"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)

    axes[0].plot(time, time_metrics["mean_x2"], color="tab:blue")
    axes[0].set_ylabel("mean x^2")
    axes[0].set_title("Observation strength")

    axes[1].plot(time, shape_metrics["mean_peak_count"], label="mean", color="tab:orange")
    axes[1].plot(time, shape_metrics["max_peak_count"], label="max", color="tab:red", alpha=0.75)
    axes[1].set_ylabel("peak count")
    axes[1].set_title("Reference posterior local peaks")
    axes[1].legend(loc="best")

    axes[2].plot(time, shape_metrics["mean_normalized_entropy"], color="tab:green")
    axes[2].set_ylabel("norm entropy")

    axes[3].plot(time, time_metrics["variance_ratio"], label="variance ratio", color="tab:purple")
    axes[3].plot(
        time,
        time_metrics["learned_coverage_90"],
        label="learned coverage",
        color="tab:brown",
    )
    axes[3].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[3].axhline(0.9, color="black", linestyle=":", linewidth=1.0)
    axes[3].set_ylabel("learned")
    axes[3].set_xlabel("time")
    axes[3].legend(loc="best")

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _scalar_gaussian_nll(value: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return 0.5 * (np.log(2.0 * np.pi * var) + (value - mean) ** 2 / var)


def _coverage(
    value: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    *,
    z_score: float,
) -> np.ndarray:
    radius = z_score * np.sqrt(var)
    return np.mean(np.abs(value - mean) <= radius, axis=0)


if __name__ == "__main__":
    main()
