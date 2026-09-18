"""Dtypes as names, never as a framework's dtype object.

``mace_core`` imports no framework, so it cannot hold a ``torch.dtype`` or a
``jax`` one. It holds the name, and the framework layer binds it. That is not a
workaround for the purity rule: a checkpoint records a name, and a name is what
survives being written to disk and read back by the other implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

__all__ = ["PRECISIONS", "Precision", "PrecisionConfig"]

Precision = Literal["float64", "float32", "bfloat16"]

#: The accepted names, for error messages and for callers that enumerate them.
PRECISIONS: tuple[str, ...] = get_args(Precision)


@dataclass(frozen=True)
class PrecisionConfig:
    """What the model computes in, and what it accumulates sums in.

    The two differ on purpose. A sum over ten thousand site energies loses
    accuracy the model's own arithmetic does not, so the reduction is allowed a
    wider type than the blocks that feed it.

    ``accumulate`` is a **floor the device has to be able to satisfy**, not a
    promise. A device with no float64 degrades to ``model``, which is what the
    frozen tree already does on MPS, and the degradation is reported when the
    model is built rather than found later as a dtype error.

    Attributes:
        model: What the blocks compute in.
        accumulate: What reductions and per-atom energies are accumulated in.
    """

    model: Precision = "float64"
    accumulate: Precision = "float64"

    def resolved_accumulate(self, supports_float64: bool) -> Precision:
        """The accumulation type this device can actually give.

        Args:
            supports_float64: Whether the device has float64 at all.
        """
        if self.accumulate == "float64" and not supports_float64:
            return self.model
        return self.accumulate
