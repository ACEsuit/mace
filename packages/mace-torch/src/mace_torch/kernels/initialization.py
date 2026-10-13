"""Giving a freshly built model a set of weights to start from.

An op that holds its own weights allocates them at zero and waits to be loaded,
which is right for every model that comes from a checkpoint and builds a model
that cannot be trained: zero-valued tensors multiply to zero, so does the
gradient, and the run finishes reporting a model that never moved. A model
built to be trained is initialised here, once, right after it is built.

**The seed is per op, derived from the run's seed and the op's path in the
module tree.** One seed for the whole model would give two ops of the same
shape the same weights, and drawing from the global generator would make a
model depend on how many random numbers anything else had drawn first. The
derivation is a hash of the path, so it is stable across processes and across
python versions, which `hash` is not.
"""

from __future__ import annotations

import hashlib

from mace_core.kernels.protocol import InternalWeights
from torch import nn

__all__ = ["initialize_model_weights", "op_seed"]

#: Keeps the derived seeds inside the range a torch generator accepts.
_SEED_MODULUS = 2**31


def op_seed(seed: int, path: str) -> int:
    """The seed one op draws from, given the run's seed and where it sits.

    Stable across processes: the path is hashed with blake2b rather than with
    `hash`, whose salt changes every run.
    """
    digest = hashlib.blake2b(path.encode("utf-8"), digest_size=8).digest()
    return (seed + int.from_bytes(digest, "big")) % _SEED_MODULUS


def initialize_model_weights(model: nn.Module, seed: int = 0) -> tuple[str, ...]:
    """Draw a fresh set of weights for every op that holds its own.

    Args:
        model: Anything holding ops. The tree is walked; an op that does not
            hold weights contributes nothing.
        seed: The run's seed. Two runs with the same seed and the same model
            get the same weights, and that is the whole reason it is recorded.

    Returns:
        The paths that were initialised, in tree order. A caller that gets an
        empty tuple back has a model with no weights of its own, which is worth
        knowing rather than discovering as a loss that will not move.
    """
    initialized: list[str] = []
    for path, module in model.named_modules():
        if not isinstance(module, InternalWeights):
            continue
        module.initialize_weights(op_seed(seed, path))
        initialized.append(path)
    return tuple(initialized)
