"""Distribution objects for Gaussian filtering and edge posteriors."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


LOG_2PI = jnp.log(2.0 * jnp.pi)


@dataclass(frozen=True)
class GaussianBelief:
    """Scalar Gaussian filtering belief `q^F_t(z_t)`."""

    mean: jax.Array
    var: jax.Array

    def __post_init__(self) -> None:
        _validate_static_positive("var", self.var)

    @property
    def std(self) -> jax.Array:
        return jnp.sqrt(self.var)

    def log_prob(self, z: jax.Array) -> jax.Array:
        return _normal_log_prob(z, self.mean, self.var)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        noise = jax.random.normal(key, shape=sample_shape + self.mean.shape, dtype=self.mean.dtype)
        return self.mean + self.std * noise


@dataclass(frozen=True)
class ConditionalGaussianBackward:
    """Scalar Gaussian backward conditional `q^B_t(z_tm1 | z_t)`."""

    a: jax.Array
    b: jax.Array
    var: jax.Array

    def __post_init__(self) -> None:
        _validate_static_positive("var", self.var)

    @property
    def std(self) -> jax.Array:
        return jnp.sqrt(self.var)

    def mean_given(self, z_t: jax.Array) -> jax.Array:
        return self.a * z_t + self.b

    def log_prob(self, z_tm1: jax.Array, z_t: jax.Array) -> jax.Array:
        return _normal_log_prob(z_tm1, self.mean_given(z_t), self.var)

    def sample(self, key: jax.Array, z_t: jax.Array) -> jax.Array:
        noise = jax.random.normal(key, shape=z_t.shape, dtype=z_t.dtype)
        return self.mean_given(z_t) + self.std * noise


@dataclass(frozen=True)
class GaussianEdgePosterior:
    """Factorized scalar edge posterior `q^F_t(z_t) q^B_t(z_tm1 | z_t)`."""

    q_filter: GaussianBelief
    q_backward: ConditionalGaussianBackward

    @classmethod
    def from_mean_cov(
        cls,
        edge_mean: jax.Array,
        edge_cov: jax.Array,
        *,
        min_var: float = 1e-12,
    ) -> "GaussianEdgePosterior":
        """Build the marginal-preserving factorization from a bivariate Gaussian.

        `edge_mean` has final dimension `[z_t, z_tm1]`, and `edge_cov` has final
        dimensions ordered as `[[Var(z_t), Cov], [Cov, Var(z_tm1)]]`.
        """

        if edge_mean.shape[-1] != 2:
            raise ValueError("edge_mean must have final dimension 2")
        if edge_cov.shape[-2:] != (2, 2):
            raise ValueError("edge_cov must have final dimensions [2, 2]")
        if edge_mean.shape[:-1] != edge_cov.shape[:-2]:
            raise ValueError("edge_mean and edge_cov batch shapes must match")

        mean_z_t = edge_mean[..., 0]
        mean_z_tm1 = edge_mean[..., 1]
        var_z_t = jnp.maximum(edge_cov[..., 0, 0], min_var)
        var_z_tm1 = jnp.maximum(edge_cov[..., 1, 1], min_var)
        cov = edge_cov[..., 0, 1]

        a = cov / var_z_t
        b = mean_z_tm1 - a * mean_z_t
        backward_var = jnp.maximum(var_z_tm1 - cov**2 / var_z_t, min_var)

        return cls(
            q_filter=GaussianBelief(mean=mean_z_t, var=var_z_t),
            q_backward=ConditionalGaussianBackward(a=a, b=b, var=backward_var),
        )

    @property
    def filter_marginal(self) -> GaussianBelief:
        return self.q_filter

    def log_prob(self, z_t: jax.Array, z_tm1: jax.Array) -> jax.Array:
        return self.q_filter.log_prob(z_t) + self.q_backward.log_prob(z_tm1, z_t)

    def sample(
        self, key: jax.Array, sample_shape: tuple[int, ...] = ()
    ) -> tuple[jax.Array, jax.Array]:
        key_z_t, key_z_tm1 = jax.random.split(key)
        z_t = self.q_filter.sample(key_z_t, sample_shape=sample_shape)
        z_tm1 = self.q_backward.sample(key_z_tm1, z_t)
        return z_t, z_tm1

    def joint_mean(self) -> jax.Array:
        mean_z_t = self.q_filter.mean
        mean_z_tm1 = self.q_backward.mean_given(mean_z_t)
        return jnp.stack((mean_z_t, mean_z_tm1), axis=-1)

    def joint_cov(self) -> jax.Array:
        var_z_t = self.q_filter.var
        cov = self.q_backward.a * var_z_t
        var_z_tm1 = self.q_backward.a**2 * var_z_t + self.q_backward.var
        row_0 = jnp.stack((var_z_t, cov), axis=-1)
        row_1 = jnp.stack((cov, var_z_tm1), axis=-1)
        return jnp.stack((row_0, row_1), axis=-2)


def _normal_log_prob(value: jax.Array, mean: jax.Array, var: jax.Array) -> jax.Array:
    return -0.5 * (LOG_2PI + jnp.log(var) + (value - mean) ** 2 / var)


def _validate_static_positive(name: str, value: jax.Array) -> None:
    """Validate host-known arrays without forcing a Python bool on tracers."""

    try:
        value_np = np.asarray(jax.device_get(value))
    except Exception:
        return
    if value_np.shape == ():
        if float(value_np) <= 0.0:
            raise ValueError(f"{name} must be positive")
        return
    if bool(np.any(value_np <= 0)):
        raise ValueError(f"{name} must be positive")
