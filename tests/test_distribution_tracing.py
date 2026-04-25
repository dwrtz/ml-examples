"""Tests for JAX-facing distribution helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vbf.distributions import GaussianBelief


def test_gaussian_belief_rejects_static_nonpositive_variance() -> None:
    with pytest.raises(ValueError, match="var must be positive"):
        GaussianBelief(mean=jnp.array(0.0), var=jnp.array(0.0))


def test_gaussian_belief_can_be_constructed_under_jit() -> None:
    @jax.jit
    def evaluate(var):
        belief = GaussianBelief(mean=jnp.array(0.0), var=var)
        return belief.log_prob(jnp.array(0.0))

    value = evaluate(jnp.array(1.0))

    np.testing.assert_allclose(np.asarray(value), -0.5 * np.log(2.0 * np.pi))
