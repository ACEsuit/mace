"""Dtypes as names, never as a framework's dtype object.

``mace_core`` imports no framework, so it cannot hold a ``torch.dtype`` or a
``jax`` one. It holds the name, and the framework layer binds it. That is not a
workaround for the purity rule: a checkpoint records a name, and a name is what
survives being written to disk and read back by the other implementation.
"""

from __future__ import annotations

from typing import Literal, get_args

__all__ = ["PRECISIONS", "Precision"]

Precision = Literal["float64", "float32", "bfloat16"]

#: The accepted names, for error messages and for callers that enumerate them.
PRECISIONS: tuple[str, ...] = get_args(Precision)
