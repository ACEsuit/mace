"""Taking the process back to how it was before legacy was loaded.

Importing and running the frozen tree changes things that outlive the call.
``mace/__init__`` sets an environment variable so pickled checkpoints load under
newer PyTorch defaults. e3nn's optimization settings are process-wide. The
default dtype is global, and so is every random number generator.

Run legacy and then the rewrite in one process without putting those back and
the comparison is **silently invalid**: the second stack inherits the first
one's settings, the numbers agree, and what was compared is not what either
would do on its own.

So this snapshots the lot and restores it, and a test asserts the restoration
rather than trusting it. The list is not hypothetical: the frozen tree's own
suite already defends the dtype half of it, because a calculator that changed
the global dtype broke unrelated tests downstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = ["ProcessState", "capture_state", "restore_state", "state_differences"]

#: Environment variables the frozen tree sets on import. Restoring a variable
#: to *absent* is not the same as setting it to its old value, so the snapshot
#: records the difference.
WATCHED_ENVIRONMENT = ("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",)


@dataclass
class ProcessState:
    """Everything a legacy run can change that outlives it."""

    environment: dict[str, str | None]
    default_dtype: torch.dtype
    torch_rng: torch.Tensor
    numpy_rng: tuple[Any, ...]
    e3nn_defaults: dict[str, Any] | None


def capture_state() -> ProcessState:
    """Take the snapshot. Cheap enough to take around every legacy call."""
    defaults = None
    try:
        from e3nn import get_optimization_defaults

        defaults = dict(get_optimization_defaults())
    except ImportError:
        # e3nn belongs to the frozen tree alone. Its absence is a fact about
        # the environment, not a failure: the rewrite does not depend on it.
        defaults = None

    return ProcessState(
        environment={name: os.environ.get(name) for name in WATCHED_ENVIRONMENT},
        default_dtype=torch.get_default_dtype(),
        torch_rng=torch.get_rng_state().clone(),
        numpy_rng=np.random.get_state(),
        e3nn_defaults=defaults,
    )


def restore_state(state: ProcessState) -> None:
    """Put it all back, including unsetting what was unset."""
    for name, value in state.environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    torch.set_default_dtype(state.default_dtype)
    torch.set_rng_state(state.torch_rng)
    np.random.set_state(state.numpy_rng)
    if state.e3nn_defaults is not None:
        from e3nn import set_optimization_defaults

        set_optimization_defaults(**state.e3nn_defaults)


def state_differences(before: ProcessState, after: ProcessState) -> list[str]:
    """What changed between two snapshots, named one per line.

    Returns a list rather than a bool so a failure says which global moved,
    which is the whole difficulty of debugging one.
    """
    changed = []
    for name, value in before.environment.items():
        if after.environment.get(name) != value:
            changed.append(
                f"environment {name}: {value!r} became {after.environment.get(name)!r}"
            )
    if before.default_dtype != after.default_dtype:
        changed.append(
            f"default dtype: {before.default_dtype} became {after.default_dtype}"
        )
    if not torch.equal(before.torch_rng, after.torch_rng):
        changed.append("the torch random number generator advanced")
    if before.numpy_rng[0] != after.numpy_rng[0] or not np.array_equal(
        before.numpy_rng[1], after.numpy_rng[1]
    ):
        changed.append("the numpy random number generator advanced")
    if before.e3nn_defaults != after.e3nn_defaults:
        changed.append(
            f"e3nn optimization defaults: {before.e3nn_defaults} became "
            f"{after.e3nn_defaults}"
        )
    return changed
