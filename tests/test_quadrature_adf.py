import importlib.util
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import yaml

from vbf.data import LinearGaussianParams
from vbf.models.cells import (
    init_component_mixture_mlp_params,
    run_component_mixture_mlp_filter,
)
from vbf.nonlinear import NonlinearDataConfig, make_nonlinear_batch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sweep_quadrature_adf.py"
SPEC = importlib.util.spec_from_file_location("sweep_quadrature_adf", MODULE_PATH)
assert SPEC is not None
quadrature_adf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = quadrature_adf
SPEC.loader.exec_module(quadrature_adf)
run_quadrature_adf_filter = quadrature_adf.run_quadrature_adf_filter


def test_quadrature_adf_gaussian_outputs_are_finite() -> None:
    config = NonlinearDataConfig(batch_size=3, time_steps=5)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=201)

    outputs = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        components=1,
        likelihood_power=1.0,
        init_span=0.0,
        num_points=16,
        em_steps=5,
    )

    assert outputs.filter_mean.shape == (3, 5)
    assert outputs.filter_var.shape == (3, 5)
    assert outputs.weights.shape == (3, 5, 1)
    assert np.all(np.isfinite(outputs.predictive_y_log_prob))
    assert np.all(outputs.filter_var > 0.0)


def test_quadrature_adf_mixture_weights_are_normalized() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=4)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=202)

    outputs = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        components=4,
        likelihood_power=0.5,
        init_span=2.0 * np.pi,
        num_points=16,
        em_steps=5,
    )

    assert outputs.weights.shape == (2, 4, 4)
    assert np.allclose(np.sum(outputs.weights, axis=-1), 1.0)
    assert np.all(outputs.component_var > 0.0)


def test_quadrature_alias_projection_preserves_2pi_component_spacing() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=4)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=206)

    outputs = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        components=5,
        likelihood_power=1.0,
        init_span=4.0 * np.pi,
        projection="mode_preserving",
        alias_spacing=2.0 * np.pi,
        num_points=16,
        em_steps=5,
    )

    assert outputs.weights.shape == (2, 4, 5)
    assert np.allclose(np.sum(outputs.weights, axis=-1), 1.0)
    assert np.all(outputs.component_var > 0.0)
    assert np.all(np.diff(outputs.component_mean, axis=-1) > 0.0)


def test_quadrature_alias_prior_weighting_downweights_distant_aliases() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=4)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=207)

    outputs = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        components=5,
        likelihood_power=1.0,
        init_span=4.0 * np.pi,
        projection="mode_preserving",
        alias_spacing=2.0 * np.pi,
        initial_weighting="prior_alias",
        num_points=16,
        em_steps=5,
    )

    initial_predictive_weights = outputs.predictive_weights[:, 0]
    assert np.all(initial_predictive_weights[:, 2] > initial_predictive_weights[:, 1])
    assert np.all(initial_predictive_weights[:, 1] > initial_predictive_weights[:, 0])
    assert np.allclose(np.sum(outputs.weights, axis=-1), 1.0)


def test_quadrature_alias_pruning_limits_active_components() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=4)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=208)

    outputs = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        components=5,
        likelihood_power=0.5,
        init_span=4.0 * np.pi,
        projection="mode_preserving",
        alias_spacing=2.0 * np.pi,
        initial_weighting="prior_alias",
        max_active_aliases=2,
        num_points=16,
        em_steps=5,
    )

    active = outputs.weights > 1e-5
    assert np.all(np.sum(active, axis=-1) <= 2)
    assert np.allclose(np.sum(outputs.weights, axis=-1), 1.0)


def test_quadrature_alias_mean_shrink_reduces_filter_variance() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=4)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=209)
    common_kwargs = {
        "components": 5,
        "likelihood_power": 0.5,
        "init_span": 4.0 * np.pi,
        "projection": "mode_preserving",
        "alias_spacing": 2.0 * np.pi,
        "initial_weighting": "prior_alias",
        "num_points": 16,
        "em_steps": 5,
    }

    unshrunk = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        **common_kwargs,
    )
    shrunk = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        alias_mean_shrink=0.7,
        **common_kwargs,
    )

    assert np.mean(shrunk.filter_var) < np.mean(unshrunk.filter_var)
    assert np.allclose(np.sum(shrunk.weights, axis=-1), 1.0)


def test_quadrature_alias_entropy_gate_can_disable_shrink() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=4)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=210)
    common_kwargs = {
        "components": 5,
        "likelihood_power": 0.5,
        "init_span": 4.0 * np.pi,
        "projection": "mode_preserving",
        "alias_spacing": 2.0 * np.pi,
        "initial_weighting": "prior_alias",
        "alias_mean_shrink": 0.7,
        "num_points": 16,
        "em_steps": 5,
    }

    disabled = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        alias_entropy_threshold=1.0,
        **common_kwargs,
    )
    unshrunk = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        **{**common_kwargs, "alias_mean_shrink": 1.0},
    )

    assert np.allclose(disabled.component_mean, unshrunk.component_mean)
    assert np.allclose(disabled.filter_var, unshrunk.filter_var)


def test_zero_x_predictive_y_matches_observation_noise() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=4, x_pattern="zero")
    params = LinearGaussianParams(q=0.1, r=0.3, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=203)

    outputs = run_quadrature_adf_filter(
        np.asarray(batch.x),
        np.asarray(batch.y),
        params,
        components=2,
        likelihood_power=1.0,
        init_span=0.0,
        num_points=16,
        em_steps=5,
    )
    expected = -0.5 * (
        np.log(2.0 * np.pi * float(params.r)) + np.asarray(batch.y) ** 2 / float(params.r)
    )

    assert np.allclose(outputs.predictive_y_log_prob, expected)


def test_component_mixture_cell_outputs_are_finite() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=4)
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, params, seed=204)
    mlp_params = init_component_mixture_mlp_params(
        jax.random.PRNGKey(205),
        hidden_dim=4,
        num_components=3,
        component_mean_init_span=1.0,
    )

    outputs = run_component_mixture_mlp_filter(
        mlp_params,
        batch,
        params,
        num_components=3,
        component_mean_init_span=1.0,
    )

    assert outputs.filter_weights.shape == (2, 4, 3)
    assert np.allclose(np.asarray(np.sum(outputs.filter_weights, axis=-1)), 1.0)
    assert np.all(np.isfinite(np.asarray(outputs.component_mean)))
    assert np.all(np.asarray(outputs.component_var) > 0.0)


def test_quadrature_adf_distillation_trainer_smoke(tmp_path: Path) -> None:
    config = {
        "name": "quadrature_distillation_test",
        "benchmark": "nonlinear",
        "model": "direct_mixture_quadrature_power_ep_distilled",
        "seed": 301,
        "output_dir": str(tmp_path / "run"),
        "data": {
            "batch_size": 4,
            "time_steps": 5,
            "x_pattern": "sinusoidal",
            "x_cycles": 1.0,
            "x_amplitude": 1.0,
            "x_constant": 1.0,
            "x_missing_period": 2,
            "observation": "x_sine",
        },
        "evaluation": {"seed_offset": 10000, "data": {"batch_size": 4}},
        "state_space": {"q": 0.1, "r": 0.1, "m0": 1.0, "p0": 2.0},
        "training": {
            "steps": 2,
            "learning_rate": 0.001,
            "hidden_dim": 4,
            "log_every": 1,
            "min_var": 1e-6,
            "cell_type": "component_mixture",
            "mixture_components": 2,
            "mixture_component_mean_init_span": 1.0,
            "target_likelihood_power": 0.5,
            "target_num_points": 8,
            "target_em_steps": 2,
            "target_density_num_points": 4,
            "density_loss_weight": 1.0,
            "predictive_carry_weight": 1.0,
            "hybrid_refinement_steps": 1,
        },
        "reference": {"grid_min": -8.0, "grid_max": 8.0, "num_grid": 101},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/train_quadrature_adf_distilled.py",
            "--config",
            str(config_path),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        check=True,
    )

    metrics_path = tmp_path / "run" / "metrics.json"
    assert metrics_path.exists()
    metrics_text = metrics_path.read_text(encoding="utf-8")
    assert "predictive_carry_y_nll" in metrics_text
    assert "hybrid_predictive_y_nll" in metrics_text


def test_quadrature_pareto_sweep_smoke(tmp_path: Path) -> None:
    config = {
        "name": "quadrature_pareto_test",
        "benchmark": "nonlinear",
        "seed": 401,
        "data": {
            "batch_size": 4,
            "time_steps": 5,
            "x_pattern": "sinusoidal",
            "x_cycles": 1.0,
            "x_amplitude": 1.0,
            "x_constant": 1.0,
            "x_missing_period": 2,
            "observation": "x_sine",
        },
        "state_space": {"q": 0.1, "r": 0.1, "m0": 1.0, "p0": 2.0},
        "reference": {"grid_min": -8.0, "grid_max": 8.0, "num_grid": 101},
    }
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "pareto"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/sweep_quadrature_pareto.py",
            "--configs",
            str(config_path),
            "--num-points",
            "8",
            "--em-steps",
            "2",
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        check=True,
    )

    metrics_text = (output_dir / "metrics.csv").read_text(encoding="utf-8")
    assert "pareto_state_nll" in metrics_text
    assert "pareto_predictive_y_nll" in metrics_text
