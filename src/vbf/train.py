"""Shared training utilities."""

from __future__ import annotations

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


class AdamState(NamedTuple):
    step: jax.Array
    m: dict[str, jax.Array]
    v: dict[str, jax.Array]


def init_adam(params: dict[str, jax.Array]) -> AdamState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(step=jnp.array(0, dtype=jnp.int64), m=zeros, v=zeros)


def adam_update(
    params: dict[str, jax.Array],
    grads: dict[str, jax.Array],
    state: AdamState,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[dict[str, jax.Array], AdamState]:
    step = state.step + 1
    m = jax.tree_util.tree_map(lambda m_i, g_i: beta1 * m_i + (1.0 - beta1) * g_i, state.m, grads)
    v = jax.tree_util.tree_map(
        lambda v_i, g_i: beta2 * v_i + (1.0 - beta2) * (g_i**2),
        state.v,
        grads,
    )
    m_hat = jax.tree_util.tree_map(lambda item: item / (1.0 - beta1**step), m)
    v_hat = jax.tree_util.tree_map(lambda item: item / (1.0 - beta2**step), v)
    next_params = jax.tree_util.tree_map(
        lambda p_i, m_i, v_i: p_i - learning_rate * m_i / (jnp.sqrt(v_i) + eps),
        params,
        m_hat,
        v_hat,
    )
    return next_params, AdamState(step=step, m=m, v=v)
