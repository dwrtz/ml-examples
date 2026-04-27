"""Disk cache for expensive nonlinear grid-reference computations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from vbf.data import EpisodeBatch, LinearGaussianParams
from vbf.nonlinear import (
    GridReferenceConfig,
    NonlinearDataConfig,
    NonlinearReferenceOutputs,
    make_nonlinear_batch,
    nonlinear_grid_filter,
)


CACHE_VERSION = 1
DEFAULT_NONLINEAR_REFERENCE_CACHE_DIR = Path("outputs/cache/nonlinear_reference")


class CachedNonlinearReference(NamedTuple):
    batch: EpisodeBatch
    reference: NonlinearReferenceOutputs
    cache_path: Path
    cache_hit: bool


def load_or_compute_nonlinear_reference(
    data_config: NonlinearDataConfig,
    state_params: LinearGaussianParams,
    *,
    seed: int,
    grid_config: GridReferenceConfig,
    cache_dir: Path = DEFAULT_NONLINEAR_REFERENCE_CACHE_DIR,
    use_cache: bool = True,
) -> CachedNonlinearReference:
    """Load or compute a deterministic nonlinear grid-reference run."""

    cache_path = cache_dir / f"{nonlinear_reference_cache_key(data_config, state_params, seed=seed, grid_config=grid_config)}.npz"
    if use_cache and cache_path.exists():
        return CachedNonlinearReference(
            batch=_load_batch(cache_path),
            reference=_load_reference(cache_path),
            cache_path=cache_path,
            cache_hit=True,
        )

    batch = make_nonlinear_batch(data_config, state_params, seed=seed)
    reference = nonlinear_grid_filter(
        batch,
        state_params,
        data_config=data_config,
        grid_config=grid_config,
    )
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp.npz")
        np.savez(
            tmp_path,
            x=np.asarray(batch.x),
            y=np.asarray(batch.y),
            z=np.asarray(batch.z),
            reference_filter_mean=np.asarray(reference.filter_mean),
            reference_filter_var=np.asarray(reference.filter_var),
            reference_predictive_mean=np.asarray(reference.predictive_mean),
            reference_predictive_var=np.asarray(reference.predictive_var),
        )
        tmp_path.replace(cache_path)
    return CachedNonlinearReference(
        batch=batch,
        reference=reference,
        cache_path=cache_path,
        cache_hit=False,
    )


def nonlinear_reference_cache_key(
    data_config: NonlinearDataConfig,
    state_params: LinearGaussianParams,
    *,
    seed: int,
    grid_config: GridReferenceConfig,
) -> str:
    """Return a stable hash key for a nonlinear reference computation."""

    payload = {
        "version": CACHE_VERSION,
        "data": _json_ready(asdict(data_config)),
        "state_space": _json_ready(asdict(state_params)),
        "seed": int(seed),
        "reference": _json_ready(asdict(grid_config)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _load_batch(path: Path) -> EpisodeBatch:
    with np.load(path) as data:
        return EpisodeBatch(
            x=jnp.asarray(data["x"]),
            y=jnp.asarray(data["y"]),
            z=jnp.asarray(data["z"]),
        )


def _load_reference(path: Path) -> NonlinearReferenceOutputs:
    with np.load(path) as data:
        return NonlinearReferenceOutputs(
            filter_mean=jnp.asarray(data["reference_filter_mean"]),
            filter_var=jnp.asarray(data["reference_filter_var"]),
            predictive_mean=jnp.asarray(data["reference_predictive_mean"]),
            predictive_var=jnp.asarray(data["reference_predictive_var"]),
        )


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
