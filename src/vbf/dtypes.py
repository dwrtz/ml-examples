"""Shared dtype policy for VBF experiments."""

from __future__ import annotations

import os

import jax


ENABLE_X64 = os.environ.get("VBF_ENABLE_X64", "").lower() in {"1", "true", "yes", "on"}
jax.config.update("jax_enable_x64", ENABLE_X64)

import jax.numpy as jnp  # noqa: E402


DEFAULT_DTYPE = jnp.float64 if ENABLE_X64 else jnp.float32

