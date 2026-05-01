"""Compare Gaussian and top-mode mixture projections of nonlinear grid posteriors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from vbf.data import EpisodeBatch, LinearGaussianParams  # noqa: E402
from vbf.metrics import gaussian_interval_coverage, scalar_gaussian_nll  # noqa: E402
from vbf.nonlinear import (  # noqa: E402
    GridReferenceConfig,
    NonlinearDataConfig,
    nonlinear_grid_filter_masses,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--components", default="1,2,3")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    component_counts = [int(item) for item in args.components.split(",") if item.strip()]
    if any(count <= 0 for count in component_counts):
        raise ValueError("--components must contain positive integers")

    diagnostics = dict(np.load(run_dir / "diagnostics.npz"))
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    batch = EpisodeBatch(
        x=diagnostics["x"],
        y=diagnostics["y"],
        z=diagnostics["z"],
    )
    masses = nonlinear_grid_filter_masses(
        batch,
        LinearGaussianParams(**config["state_space"]),
        data_config=NonlinearDataConfig(**config["data"]),
        grid_config=GridReferenceConfig(**config.get("reference", {})),
    )
    grid = np.asarray(masses.grid)
    mass = np.asarray(masses.filter_mass)
    rows = _projection_rows(
        grid,
        mass,
        diagnostics["z"],
        diagnostics=diagnostics,
        component_counts=component_counts,
        window=args.window,
    )

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = plot_dir / "mixture_projection_metrics.csv"
    json_path = plot_dir / "mixture_projection_metrics.json"
    plot_path = plot_dir / "mixture_projection.png"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(rows, plot_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")


def _projection_rows(
    grid: np.ndarray,
    mass: np.ndarray,
    z: np.ndarray,
    *,
    diagnostics: dict[str, np.ndarray],
    component_counts: list[int],
    window: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    grid_density = mass / np.maximum(_grid_spacing(grid), 1e-12)
    rows.append(
        {
            "projection": "grid",
            "components": 0,
            "state_nll": float(np.mean(-np.log(_lookup_grid_density(grid, grid_density, z)))),
            "coverage_90": np.nan,
            "mean_variance": np.nan,
        }
    )
    mean = np.sum(mass * grid[None, None, :], axis=2)
    var = np.sum(mass * (grid[None, None, :] - mean[:, :, None]) ** 2, axis=2)
    rows.append(
        {
            "projection": "moment_gaussian",
            "components": 1,
            "state_nll": float(np.mean(scalar_gaussian_nll(z, mean, var))),
            "coverage_90": float(
                gaussian_interval_coverage(z, mean, var, z_score=1.6448536269514722)
            ),
            "mean_variance": float(np.mean(var)),
        }
    )
    if "learned_filter_mean" in diagnostics and "learned_filter_var" in diagnostics:
        learned_var = diagnostics["learned_filter_var"]
        rows.append(
            {
                "projection": "learned_gaussian",
                "components": 1,
                "state_nll": float(
                    np.mean(
                        scalar_gaussian_nll(
                            z,
                            diagnostics["learned_filter_mean"],
                            learned_var,
                        )
                    )
                ),
                "coverage_90": float(
                    gaussian_interval_coverage(
                        z,
                        diagnostics["learned_filter_mean"],
                        learned_var,
                        z_score=1.6448536269514722,
                    )
                ),
                "mean_variance": float(np.mean(learned_var)),
            }
        )
    for count in component_counts:
        mixture = _quantile_mixture(grid, mass, components=count)
        density = _mixture_density(z, mixture)
        rows.append(
            {
                "projection": "quantile_mixture",
                "components": count,
                "state_nll": float(np.mean(-np.log(np.maximum(density, 1e-300)))),
                "coverage_90": np.nan,
                "mean_variance": float(np.mean(_mixture_variance(mixture))),
            }
        )
        mixture = _top_mode_mixture(grid, mass, components=count, window=window)
        density = _mixture_density(z, mixture)
        rows.append(
            {
                "projection": "top_mode_mixture",
                "components": count,
                "state_nll": float(np.mean(-np.log(np.maximum(density, 1e-300)))),
                "coverage_90": np.nan,
                "mean_variance": float(np.mean(_mixture_variance(mixture))),
            }
        )
    return rows


def _grid_spacing(grid: np.ndarray) -> float:
    return float((grid[-1] - grid[0]) / (len(grid) - 1))


def _lookup_grid_density(grid: np.ndarray, density: np.ndarray, z: np.ndarray) -> np.ndarray:
    flat_z = z.reshape(-1)
    flat_density = density.reshape((-1, density.shape[-1]))
    values = np.empty(flat_z.shape, dtype=np.float64)
    for index, value in enumerate(flat_z):
        values[index] = np.interp(value, grid, flat_density[index], left=1e-300, right=1e-300)
    return np.maximum(values.reshape(z.shape), 1e-300)


def _top_mode_mixture(
    grid: np.ndarray,
    mass: np.ndarray,
    *,
    components: int,
    window: int,
) -> dict[str, np.ndarray]:
    batch_size, time_steps, _ = mass.shape
    weights = np.zeros((batch_size, time_steps, components), dtype=np.float64)
    means = np.zeros_like(weights)
    variances = np.zeros_like(weights)
    fallback_mean = np.sum(mass * grid[None, None, :], axis=2)
    fallback_var = np.sum(mass * (grid[None, None, :] - fallback_mean[:, :, None]) ** 2, axis=2)
    for batch_index in range(batch_size):
        for time_index in range(time_steps):
            selected = _top_peak_indices(mass[batch_index, time_index], components)
            for component_index, peak_index in enumerate(selected):
                lo = max(0, peak_index - window)
                hi = min(len(grid), peak_index + window + 1)
                local_mass = mass[batch_index, time_index, lo:hi]
                weight = float(np.sum(local_mass))
                if weight <= 0.0:
                    continue
                local_grid = grid[lo:hi]
                mean = float(np.sum(local_mass * local_grid) / weight)
                var = float(np.sum(local_mass * (local_grid - mean) ** 2) / weight)
                weights[batch_index, time_index, component_index] = weight
                means[batch_index, time_index, component_index] = mean
                variances[batch_index, time_index, component_index] = max(var, 1e-6)
            total = np.sum(weights[batch_index, time_index])
            if total <= 0.0:
                weights[batch_index, time_index, 0] = 1.0
                means[batch_index, time_index, 0] = fallback_mean[batch_index, time_index]
                variances[batch_index, time_index, 0] = fallback_var[batch_index, time_index]
            else:
                weights[batch_index, time_index] /= total
                empty = weights[batch_index, time_index] == 0.0
                means[batch_index, time_index, empty] = fallback_mean[batch_index, time_index]
                variances[batch_index, time_index, empty] = fallback_var[batch_index, time_index]
    return {"weights": weights, "means": means, "variances": variances}


def _quantile_mixture(
    grid: np.ndarray,
    mass: np.ndarray,
    *,
    components: int,
) -> dict[str, np.ndarray]:
    batch_size, time_steps, _ = mass.shape
    weights = np.zeros((batch_size, time_steps, components), dtype=np.float64)
    means = np.zeros_like(weights)
    variances = np.zeros_like(weights)
    cdf = np.cumsum(mass, axis=2)
    for batch_index in range(batch_size):
        for time_index in range(time_steps):
            start = 0
            for component_index in range(components):
                if component_index == components - 1:
                    stop = len(grid)
                else:
                    threshold = (component_index + 1) / components
                    stop = int(np.searchsorted(cdf[batch_index, time_index], threshold)) + 1
                local_mass = mass[batch_index, time_index, start:stop]
                local_grid = grid[start:stop]
                weight = float(np.sum(local_mass))
                if weight <= 0.0:
                    weights[batch_index, time_index, component_index] = 0.0
                    means[batch_index, time_index, component_index] = grid[start]
                    variances[batch_index, time_index, component_index] = 1e-6
                else:
                    mean = float(np.sum(local_mass * local_grid) / weight)
                    var = float(np.sum(local_mass * (local_grid - mean) ** 2) / weight)
                    weights[batch_index, time_index, component_index] = weight
                    means[batch_index, time_index, component_index] = mean
                    variances[batch_index, time_index, component_index] = max(var, 1e-6)
                start = stop
            total = np.sum(weights[batch_index, time_index])
            if total > 0.0:
                weights[batch_index, time_index] /= total
    return {"weights": weights, "means": means, "variances": variances}


def _top_peak_indices(values: np.ndarray, count: int) -> list[int]:
    local = (
        np.flatnonzero(
            (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]) & (values[1:-1] > 0.0)
        )
        + 1
    )
    if local.size == 0:
        local = np.asarray([int(np.argmax(values))])
    order = local[np.argsort(values[local])[::-1]]
    selected = list(order[:count])
    if len(selected) < count:
        ranked = list(np.argsort(values)[::-1])
        for index in ranked:
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == count:
                break
    return selected


def _mixture_density(z: np.ndarray, mixture: dict[str, np.ndarray]) -> np.ndarray:
    value = z[:, :, None]
    weights = mixture["weights"]
    means = mixture["means"]
    variances = mixture["variances"]
    component_density = np.exp(
        -0.5 * (np.log(2.0 * np.pi * variances) + (value - means) ** 2 / variances)
    )
    return np.sum(weights * component_density, axis=2)


def _mixture_variance(mixture: dict[str, np.ndarray]) -> np.ndarray:
    weights = mixture["weights"]
    means = mixture["means"]
    variances = mixture["variances"]
    mixture_mean = np.sum(weights * means, axis=2)
    return np.sum(weights * (variances + (means - mixture_mean[:, :, None]) ** 2), axis=2)


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, float | int | str]], path: Path) -> None:
    labels = [
        "grid" if row["projection"] == "grid" else f"{row['projection']} {row['components']}"
        for row in rows
    ]
    state_nll = [float(row["state_nll"]) for row in rows]
    fig, axis = plt.subplots(1, 1, figsize=(9, 4), constrained_layout=True)
    axis.bar(labels, state_nll, color="tab:purple", alpha=0.85)
    axis.set_ylabel("state NLL")
    axis.set_title("Reference posterior projection NLL")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
