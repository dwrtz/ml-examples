import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp

from vbf.data import LinearGaussianParams
from vbf.dtypes import DEFAULT_DTYPE
from vbf.models.cells import (
    init_direct_mixture_mlp_params,
    init_structured_mixture_mlp_params,
    init_structured_mlp_params,
    run_direct_mixture_mlp_filter,
)
from vbf.nonlinear import (
    NonlinearDataConfig,
    make_nonlinear_batch,
    nonlinear_preupdate_predictive_normalizer_loss,
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


def test_non_sine_mixture_objectives_are_finite() -> None:
    train_nonlinear = _load_train_module()
    config = NonlinearDataConfig(batch_size=2, time_steps=6, observation="x_tanh")
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, state_params, seed=236)
    mlp_params = init_direct_mixture_mlp_params(
        jax.random.PRNGKey(237),
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
        jax.random.PRNGKey(238),
        observation="x_tanh",
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
        observation="x_tanh",
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


def test_fivo_mixture_objective_is_finite() -> None:
    train_nonlinear = _load_train_module()
    config = NonlinearDataConfig(batch_size=2, time_steps=6, observation="x_sine")
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, state_params, seed=251)
    mlp_params = init_direct_mixture_mlp_params(
        jax.random.PRNGKey(252),
        hidden_dim=8,
        num_components=2,
    )
    outputs = run_direct_mixture_mlp_filter(
        mlp_params,
        batch,
        state_params,
        num_components=2,
    )

    objective = train_nonlinear._nonlinear_fivo_objective(
        outputs,
        batch,
        state_params,
        jax.random.PRNGKey(253),
        observation="x_sine",
        num_particles=4,
    )

    assert objective.shape == (2,)
    assert jnp.all(jnp.isfinite(objective))

    bridge_objective = train_nonlinear._nonlinear_fivo_objective(
        outputs,
        batch,
        state_params,
        jax.random.PRNGKey(254),
        observation="x_sine",
        num_particles=4,
        proposal_family="transition_filter_bridge",
    )

    assert bridge_objective.shape == (2,)
    assert jnp.all(jnp.isfinite(bridge_objective))

    diagnostics = train_nonlinear._nonlinear_fivo_diagnostics(
        outputs,
        batch,
        state_params,
        jax.random.PRNGKey(255),
        observation="x_sine",
        num_particles=4,
        proposal_family="transition_filter_bridge",
        resampling="stopgrad_resampling",
    )

    assert diagnostics.objective.shape == (2,)
    assert jnp.all(jnp.isfinite(diagnostics.objective))
    assert jnp.isfinite(diagnostics.mean_ess)
    assert 1.0 <= diagnostics.mean_ess <= 4.0

    no_resampling_objective = train_nonlinear._nonlinear_fivo_objective(
        outputs,
        batch,
        state_params,
        jax.random.PRNGKey(256),
        observation="x_sine",
        num_particles=4,
        proposal_family="transition_filter_bridge",
        resampling="none",
    )

    assert no_resampling_objective.shape == (2,)
    assert jnp.all(jnp.isfinite(no_resampling_objective))

    proposal_params = train_nonlinear._init_auxiliary_proposal_params(
        jax.random.PRNGKey(257),
        hidden_dim=8,
    )
    auxiliary_objective = train_nonlinear._nonlinear_fivo_objective(
        outputs,
        batch,
        state_params,
        jax.random.PRNGKey(258),
        observation="x_sine",
        num_particles=4,
        proposal_family="learned_transition_filter_bridge",
        proposal_params=proposal_params,
    )

    assert auxiliary_objective.shape == (2,)
    assert jnp.all(jnp.isfinite(auxiliary_objective))

    twisted_objective = train_nonlinear._nonlinear_fivo_objective(
        outputs,
        batch,
        state_params,
        jax.random.PRNGKey(259),
        observation="x_sine",
        num_particles=4,
        proposal_family="transition_filter_bridge",
        twist_horizon=2,
        twist_num_points=5,
    )

    assert twisted_objective.shape == (2,)
    assert jnp.all(jnp.isfinite(twisted_objective))


def test_fixed_lag_twist_uses_only_future_observations() -> None:
    train_nonlinear = _load_train_module()
    x = jnp.arange(10, dtype=DEFAULT_DTYPE).reshape(1, 10)
    y = x + 100.0

    future_x, future_y, future_mask = train_nonlinear._future_observation_windows(
        x,
        y,
        horizon=3,
    )

    assert future_x.shape == (1, 10, 3)
    assert future_y.shape == (1, 10, 3)
    assert future_mask.shape == (1, 10, 3)
    assert jnp.array_equal(future_x[0, 0], jnp.asarray([1.0, 2.0, 3.0]))
    assert jnp.array_equal(future_y[0, 0], jnp.asarray([101.0, 102.0, 103.0]))
    assert jnp.array_equal(future_mask[0, -2], jnp.asarray([True, False, False]))


def test_local_projection_loss_is_finite_for_gaussian_and_mixture() -> None:
    config = NonlinearDataConfig(batch_size=2, time_steps=6, observation="x_sine")
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, state_params, seed=261)

    gaussian_params = init_structured_mlp_params(jax.random.PRNGKey(262), hidden_dim=8)
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
        jax.random.PRNGKey(263),
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
        likelihood_power=0.5,
    )

    assert gaussian_loss.shape == batch.y.shape
    assert mixture_loss.shape == batch.y.shape
    assert jnp.all(jnp.isfinite(gaussian_loss))
    assert jnp.all(jnp.isfinite(mixture_loss))

    mixture_alpha_loss = nonlinear_tilted_projection_loss(
        mixture_outputs,
        batch,
        state_params,
        observation="x_sine",
        num_points=8,
        likelihood_power=0.5,
        divergence="alpha",
        alpha=0.5,
    )

    assert mixture_alpha_loss.shape == batch.y.shape
    assert jnp.all(jnp.isfinite(mixture_alpha_loss))


def test_preupdate_predictive_normalizer_loss_matches_legacy_mixture_path() -> None:
    train_nonlinear = _load_train_module()
    config = NonlinearDataConfig(batch_size=2, time_steps=6, observation="x_sine")
    state_params = LinearGaussianParams(q=0.1, r=0.1, m0=1.0, p0=2.0)
    batch = make_nonlinear_batch(config, state_params, seed=271)
    mixture_params = init_direct_mixture_mlp_params(
        jax.random.PRNGKey(272),
        hidden_dim=8,
        num_components=2,
        component_mean_init_span=1.0,
    )
    mixture_outputs = run_direct_mixture_mlp_filter(
        mixture_params,
        batch,
        state_params,
        num_components=2,
    )

    loss = nonlinear_preupdate_predictive_normalizer_loss(
        mixture_outputs,
        batch,
        state_params,
        observation="x_sine",
        num_points=8,
    )
    legacy_loss = -train_nonlinear._nonlinear_mixture_preassimilation_log_prob_y(
        mixture_outputs,
        batch.x,
        batch.y,
        state_params,
        observation="x_sine",
        num_points=8,
    )

    assert loss.shape == batch.y.shape
    assert jnp.all(jnp.isfinite(loss))
    assert jnp.allclose(loss, legacy_loss, atol=1e-10)
