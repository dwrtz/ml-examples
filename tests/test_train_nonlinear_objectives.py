import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp

from vbf.data import LinearGaussianParams
from vbf.models.cells import (
    init_direct_mixture_mlp_params,
    init_structured_mixture_mlp_params,
    init_structured_mlp_params,
    run_direct_mixture_mlp_filter,
)
from vbf.nonlinear import (
    NonlinearDataConfig,
    make_nonlinear_batch,
    nonlinear_tilted_projection_loss,
    run_nonlinear_structured_mixture_mlp_filter,
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


def test_windowed_joint_iwae_k1_matches_elbo_sample() -> None:
    train_nonlinear = _load_train_module()
    config = NonlinearDataConfig(batch_size=2, time_steps=6, observation="x_sine")
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, state_params, seed=211)
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(212), hidden_dim=8)
    outputs = run_nonlinear_structured_mlp_filter(
        mlp_params,
        batch,
        state_params,
        observation="x_sine",
    )
    key = jax.random.PRNGKey(213)

    elbo = train_nonlinear._nonlinear_windowed_joint_objective(
        outputs,
        batch,
        state_params,
        key,
        observation="x_sine",
        horizon=4,
        num_samples=1,
        num_windows=3,
        objective_family="elbo",
        renyi_alpha=1.0,
    )
    iwae = train_nonlinear._nonlinear_windowed_joint_objective(
        outputs,
        batch,
        state_params,
        key,
        observation="x_sine",
        horizon=4,
        num_samples=1,
        num_windows=3,
        objective_family="iwae",
        renyi_alpha=1.0,
    )

    assert jnp.allclose(iwae, elbo)


def test_windowed_joint_divergence_objectives_are_finite() -> None:
    train_nonlinear = _load_train_module()
    config = NonlinearDataConfig(batch_size=2, time_steps=6, observation="x_sine")
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, state_params, seed=221)
    mlp_params = init_structured_mlp_params(jax.random.PRNGKey(222), hidden_dim=8)
    outputs = run_nonlinear_structured_mlp_filter(
        mlp_params,
        batch,
        state_params,
        observation="x_sine",
    )

    for objective_family, renyi_alpha in (("iwae", 1.0), ("renyi", 0.5)):
        objective = train_nonlinear._nonlinear_windowed_joint_objective(
            outputs,
            batch,
            state_params,
            jax.random.PRNGKey(223),
            observation="x_sine",
            horizon=4,
            num_samples=4,
            num_windows=3,
            objective_family=objective_family,
            renyi_alpha=renyi_alpha,
        )
        assert jnp.all(jnp.isfinite(objective))


def test_windowed_joint_mixture_objective_is_finite() -> None:
    train_nonlinear = _load_train_module()
    config = NonlinearDataConfig(batch_size=2, time_steps=6, observation="x_sine")
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, state_params, seed=231)
    mlp_params = init_direct_mixture_mlp_params(
        jax.random.PRNGKey(232),
        hidden_dim=8,
        num_components=2,
    )
    outputs = run_direct_mixture_mlp_filter(
        mlp_params,
        batch,
        state_params,
        num_components=2,
    )

    objective = train_nonlinear._nonlinear_windowed_joint_objective(
        outputs,
        batch,
        state_params,
        jax.random.PRNGKey(233),
        observation="x_sine",
        horizon=4,
        num_samples=4,
        num_windows=3,
        objective_family="iwae",
        renyi_alpha=1.0,
    )
    predictive_y = train_nonlinear._nonlinear_mixture_preassimilation_log_prob_y(
        outputs,
        batch.x,
        batch.y,
        state_params,
        observation="x_sine",
        num_points=8,
    )

    assert jnp.all(jnp.isfinite(objective))
    assert jnp.all(jnp.isfinite(predictive_y))


def test_windowed_joint_structured_mixture_objective_is_finite() -> None:
    train_nonlinear = _load_train_module()
    config = NonlinearDataConfig(batch_size=2, time_steps=6, observation="x_sine")
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, state_params, seed=241)
    mlp_params = init_structured_mixture_mlp_params(
        jax.random.PRNGKey(242),
        hidden_dim=8,
        num_components=2,
    )
    outputs = run_nonlinear_structured_mixture_mlp_filter(
        mlp_params,
        batch,
        state_params,
        num_components=2,
        observation="x_sine",
    )

    objective = train_nonlinear._nonlinear_windowed_joint_objective(
        outputs,
        batch,
        state_params,
        jax.random.PRNGKey(243),
        observation="x_sine",
        horizon=4,
        num_samples=4,
        num_windows=3,
        objective_family="iwae",
        renyi_alpha=1.0,
    )

    assert jnp.all(jnp.isfinite(objective))


def test_local_projection_loss_is_finite_for_gaussian_and_mixture() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=6, observation="x_sine")
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, state_params, seed=251)

    gaussian_params = init_structured_mlp_params(jax.random.PRNGKey(252), hidden_dim=8)
    gaussian_outputs = run_nonlinear_structured_mlp_filter(
        gaussian_params,
        batch,
        state_params,
        observation="x_sine",
    )
    gaussian_loss = nonlinear_tilted_projection_loss(
        gaussian_outputs,
        batch,
        state_params,
        observation="x_sine",
        num_points=8,
    )

    mixture_params = init_direct_mixture_mlp_params(
        jax.random.PRNGKey(253),
        hidden_dim=8,
        num_components=2,
    )
    mixture_outputs = run_direct_mixture_mlp_filter(
        mixture_params,
        batch,
        state_params,
        num_components=2,
    )
    mixture_loss = nonlinear_tilted_projection_loss(
        mixture_outputs,
        batch,
        state_params,
        observation="x_sine",
        num_points=8,
    )

    assert gaussian_loss.shape == batch.y.shape
    assert mixture_loss.shape == batch.y.shape
    assert jnp.all(jnp.isfinite(gaussian_loss))
    assert jnp.all(jnp.isfinite(mixture_loss))
