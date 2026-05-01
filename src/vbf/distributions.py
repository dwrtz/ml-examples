"""Distribution objects for Gaussian filtering and edge posteriors."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
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


@dataclass(frozen=True)
class GaussianMixtureBelief:
    """Scalar Gaussian mixture filtering belief `q^F_t(z_t)`.

    The final axis indexes mixture components.
    """

    weights: jax.Array
    mean: jax.Array
    var: jax.Array

    def __post_init__(self) -> None:
        if self.weights.shape != self.mean.shape or self.mean.shape != self.var.shape:
            raise ValueError("weights, mean, and var must have the same shape")
        if len(self.weights.shape) == 0:
            raise ValueError("mixture arrays must include a final component axis")
        _validate_static_positive("weights", self.weights)
        _validate_static_positive("var", self.var)
        _validate_static_normalized("weights", self.weights)

    @property
    def log_weights(self) -> jax.Array:
        return jnp.log(self.weights)

    @property
    def num_components(self) -> int:
        return int(self.weights.shape[-1])

    def filtering_belief(self) -> "GaussianMixtureBelief":
        return self

    def mean_and_var(self) -> tuple[jax.Array, jax.Array]:
        mean = jnp.sum(self.weights * self.mean, axis=-1)
        second_moment = jnp.sum(self.weights * (self.var + self.mean**2), axis=-1)
        return mean, jnp.maximum(second_moment - mean**2, 0.0)

    def log_prob(self, z: jax.Array) -> jax.Array:
        component_log_prob = self.log_weights + _normal_log_prob(
            z[..., None],
            self.mean,
            self.var,
        )
        return jax.nn.logsumexp(component_log_prob, axis=-1)

    def sample(
        self,
        key: jax.Array,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array]:
        key_component, key_noise = jax.random.split(key)
        logits = jnp.log(self.weights)
        component = jax.random.categorical(
            key_component,
            logits=logits,
            axis=-1,
            shape=sample_shape + self.weights.shape[:-1],
        )
        selected_mean = _take_component(self.mean, component)
        selected_var = _take_component(self.var, component)
        noise = jax.random.normal(key_noise, shape=selected_mean.shape, dtype=self.mean.dtype)
        return selected_mean + jnp.sqrt(selected_var) * noise, component


@dataclass(frozen=True)
class ConditionalGaussianMixtureBackward:
    """Component-conditional `q^B_t(z_tm1 | z_t, k)`.

    The final axis indexes mixture components and must align with the filtering
    mixture component axis.
    """

    a: jax.Array
    b: jax.Array
    var: jax.Array

    def __post_init__(self) -> None:
        if self.a.shape != self.b.shape or self.b.shape != self.var.shape:
            raise ValueError("a, b, and var must have the same shape")
        if len(self.a.shape) == 0:
            raise ValueError("mixture arrays must include a final component axis")
        _validate_static_positive("var", self.var)

    def mean_given(self, z_t: jax.Array) -> jax.Array:
        return self.a * z_t[..., None] + self.b

    def log_prob(self, z_tm1: jax.Array, z_t: jax.Array) -> jax.Array:
        return _normal_log_prob(z_tm1[..., None], self.mean_given(z_t), self.var)

    def sample(
        self,
        key: jax.Array,
        z_t: jax.Array,
        component: jax.Array,
    ) -> jax.Array:
        selected_a = _take_component(self.a, component)
        selected_b = _take_component(self.b, component)
        selected_var = _take_component(self.var, component)
        noise = jax.random.normal(key, shape=z_t.shape, dtype=z_t.dtype)
        return selected_a * z_t + selected_b + jnp.sqrt(selected_var) * noise


@dataclass(frozen=True)
class GaussianMixtureEdgePosterior:
    """Mixture edge posterior `sum_k pi_k q^F_k(z_t) q^B_k(z_tm1 | z_t)`."""

    q_filter: GaussianMixtureBelief
    q_backward: ConditionalGaussianMixtureBackward

    def __post_init__(self) -> None:
        if self.q_filter.weights.shape != self.q_backward.a.shape:
            raise ValueError("filter and backward component shapes must match")

    @classmethod
    def from_gaussian_edge(cls, edge: GaussianEdgePosterior) -> "GaussianMixtureEdgePosterior":
        return cls(
            q_filter=GaussianMixtureBelief(
                weights=jnp.ones(edge.q_filter.mean.shape + (1,), dtype=edge.q_filter.mean.dtype),
                mean=edge.q_filter.mean[..., None],
                var=edge.q_filter.var[..., None],
            ),
            q_backward=ConditionalGaussianMixtureBackward(
                a=edge.q_backward.a[..., None],
                b=edge.q_backward.b[..., None],
                var=edge.q_backward.var[..., None],
            ),
        )

    @property
    def filter_marginal(self) -> GaussianMixtureBelief:
        return self.q_filter

    def filtering_belief(self) -> GaussianMixtureBelief:
        return self.q_filter

    def log_prob(self, z_t: jax.Array, z_tm1: jax.Array) -> jax.Array:
        component_log_prob = (
            self.q_filter.log_weights
            + _normal_log_prob(z_t[..., None], self.q_filter.mean, self.q_filter.var)
            + self.q_backward.log_prob(z_tm1, z_t)
        )
        return jax.nn.logsumexp(component_log_prob, axis=-1)

    def sample(
        self,
        key: jax.Array,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        key_z_t, key_z_tm1 = jax.random.split(key)
        z_t, component = self.q_filter.sample(key_z_t, sample_shape=sample_shape)
        z_tm1 = self.q_backward.sample(key_z_tm1, z_t, component)
        return z_t, z_tm1, component

    def mean_and_var(self) -> tuple[jax.Array, jax.Array]:
        return self.q_filter.mean_and_var()

    def edge_mean_cov(self) -> tuple[jax.Array, jax.Array]:
        mean_z_t_k = self.q_filter.mean
        var_z_t_k = self.q_filter.var
        mean_z_tm1_k = self.q_backward.a * mean_z_t_k + self.q_backward.b
        cov_k = self.q_backward.a * var_z_t_k
        var_z_tm1_k = self.q_backward.a**2 * var_z_t_k + self.q_backward.var

        weights = self.q_filter.weights
        mean_z_t = jnp.sum(weights * mean_z_t_k, axis=-1)
        mean_z_tm1 = jnp.sum(weights * mean_z_tm1_k, axis=-1)
        edge_mean = jnp.stack((mean_z_t, mean_z_tm1), axis=-1)

        second_z_t = jnp.sum(weights * (var_z_t_k + mean_z_t_k**2), axis=-1)
        second_z_tm1 = jnp.sum(weights * (var_z_tm1_k + mean_z_tm1_k**2), axis=-1)
        cross = jnp.sum(weights * (cov_k + mean_z_t_k * mean_z_tm1_k), axis=-1)
        var_z_t = jnp.maximum(second_z_t - mean_z_t**2, 0.0)
        var_z_tm1 = jnp.maximum(second_z_tm1 - mean_z_tm1**2, 0.0)
        cov = cross - mean_z_t * mean_z_tm1
        row_0 = jnp.stack((var_z_t, cov), axis=-1)
        row_1 = jnp.stack((cov, var_z_tm1), axis=-1)
        return edge_mean, jnp.stack((row_0, row_1), axis=-2)


def _normal_log_prob(value: jax.Array, mean: jax.Array, var: jax.Array) -> jax.Array:
    return -0.5 * (LOG_2PI + jnp.log(var) + (value - mean) ** 2 / var)


def _take_component(values: jax.Array, component: jax.Array) -> jax.Array:
    sample_ndim = component.ndim - (values.ndim - 1)
    if sample_ndim < 0:
        raise ValueError("component shape is not compatible with values")
    broadcast_values = jnp.reshape(values, (1,) * sample_ndim + values.shape)
    broadcast_values = jnp.broadcast_to(broadcast_values, component.shape + values.shape[-1:])
    return jnp.take_along_axis(broadcast_values, component[..., None], axis=-1)[..., 0]


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


def _validate_static_normalized(name: str, value: jax.Array) -> None:
    try:
        value_np = np.asarray(jax.device_get(value))
    except Exception:
        return
    total = np.sum(value_np, axis=-1)
    if not bool(np.allclose(total, 1.0, atol=1e-6)):
        raise ValueError(f"{name} must sum to one along the final axis")
