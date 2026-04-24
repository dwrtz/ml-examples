from pathlib import Path

import numpy as np

from vbf.plotting import load_linear_gaussian_diagnostics


def test_load_linear_gaussian_diagnostics(tmp_path: Path) -> None:
    np.savez(tmp_path / "diagnostics.npz", x=np.array([1.0]), y=np.array([2.0]))

    diagnostics = load_linear_gaussian_diagnostics(tmp_path)

    np.testing.assert_array_equal(diagnostics["x"], np.array([1.0]))
    np.testing.assert_array_equal(diagnostics["y"], np.array([2.0]))
