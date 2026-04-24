"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_linear_gaussian_diagnostics(run_dir: Path) -> dict[str, np.ndarray]:
    """Load saved linear-Gaussian training diagnostics."""

    diagnostics_path = run_dir / "diagnostics.npz"
    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing diagnostics file: {diagnostics_path}")
    with np.load(diagnostics_path) as data:
        return {key: data[key] for key in data.files}
