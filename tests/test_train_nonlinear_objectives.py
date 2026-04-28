import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp

from vbf.data import LinearGaussianParams
from vbf.models.cells import init_structured_mlp_params
from vbf.nonlinear import (
    NonlinearDataConfig,
    make_nonlinear_batch,
    run_nonlinear_structured_mlp_filter,
)


def _load_train_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_nonlinear.py"
    spec = importlib.util.spec_from_file_location("train_nonlinear", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_windowed_joint_elbo_h1_matches_edge_elbo() -> None:
    train_nonlinear = _load_train_module()
    config = NonlinearDataConfig(batch_size=2, time_steps=5, observation="x_sine")
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, state_params, seed=201)
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(202), hidden_dim=8)
    outputs = run_nonlinear_structured_mlp_filter(
        mlp_params,
        batch,
        state_params,
        observation="x_sine",
    )
    key = jax.random.PRNGKey(203)

    edge = train_nonlinear._nonlinear_edge_elbo(
        outputs,
        batch,
        state_params,
        key,
        observation="x_sine",
        num_samples=4,
    )
    joint_h1 = train_nonlinear._nonlinear_windowed_joint_elbo(
        outputs,
        batch,
        state_params,
        key,
        observation="x_sine",
        horizon=1,
        num_samples=4,
        num_windows=3,
    )

    assert jnp.allclose(joint_h1, edge)
