"""Kernel backends of the v1 PyTorch stack.

``reference`` holds the plain-torch implementations every other backend is
measured against, and :func:`resolve_backend` puts it behind any other one for
the ops that one does not build.
"""

from mace_torch.backends.composite import (
    BuildDecision,
    CompositeBackend,
    require_double_backward,
    resolve_backend,
)
from mace_torch.backends.reference import ReferenceBackend

__all__ = [
    "BuildDecision",
    "CompositeBackend",
    "ReferenceBackend",
    "require_double_backward",
    "resolve_backend",
]
