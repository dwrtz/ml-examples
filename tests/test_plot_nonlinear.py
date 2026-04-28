import importlib.util
from pathlib import Path

import numpy as np


def _load_plot_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "plot_nonlinear.py"
    spec = importlib.util.spec_from_file_location("plot_nonlinear", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gap_recovery_metrics_no_mask_uses_zero_gap_age() -> None:
    plot_nonlinear = _load_plot_module()
    data = {
        "x": np.ones((2, 3)),
        "y": np.zeros((2, 3)),
        "z": np.zeros((2, 3)),
        "learned_filter_mean": np.zeros((2, 3)),
        "learned_filter_var": np.ones((2, 3)),
        "reference_filter_var": np.ones((2, 3)),
        "learned_predictive_mean": np.zeros((2, 3)),
        "learned_predictive_var": np.ones((2, 3)),
    }

    metrics = plot_nonlinear._gap_recovery_metrics(data)

    assert metrics["gap_age"].tolist() == [0]
    assert metrics["count"].tolist() == [6.0]
    assert np.isfinite(metrics["gap_state_nll"][0])
