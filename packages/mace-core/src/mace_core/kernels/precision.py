"""Dtypes as names, never as a framework's dtype object.

``mace_core`` imports no framework, so it cannot hold a ``torch.dtype`` or a
``jax`` one. It holds the name, and the framework layer binds it. That is not a
workaround for the purity rule: a checkpoint records a name, and a name is what
survives being written to disk and read back by the other implementation.

Which name each op uses is configuration, and lives in
:mod:`mace_core.config.precision`.
"""

from __future__ import annotations

from typing import Literal, get_args

__all__ = ["PRECISIONS", "Precision", "widest"]

Precision = Literal["float64", "float32", "bfloat16"]

#: The accepted names, for error messages and for callers that enumerate them.
PRECISIONS: tuple[str, ...] = get_args(Precision)

#: How wide each precision is, in bits.
WIDTH: dict[str, int] = {"bfloat16": 16, "float32": 32, "float64": 64}


def widest(*precisions: Precision) -> Precision:
    """The widest of the names given."""
    return max(precisions, key=lambda name: WIDTH[name])
