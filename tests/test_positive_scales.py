"""Tests that learned posterior scales remain positive."""

import numpy as np
import jax

from vbf.data import LinearGaussianDataConfig, LinearGaussianParams, make_linear_gaussian_batch
from vbf.models.cells import init_structured_mlp_params, run_structured_mlp_filter


def test_structured_mlp_outputs_positive_variances() -> None:
    params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=10.0)
    batch = make_linear_gaussian_batch(
        LinearGaussianDataConfig(batch_size=4, time_steps=7),
        params,
        seed=11,
    )
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(0), hidden_dim=8)

    outputs = run_structured_mlp_filter(mlp_params, batch, params)

    assert np.all(np.asarray(outputs.filter_var) > 0.0)
    assert np.all(np.asarray(outputs.backward_var) > 0.0)
