"""The torch binding of the flat graph dict.

``collate``, ``collate_targets`` and ``build_batch`` turn graph samples, or
parsed structures, into one batched dict of tensors with its
:class:`~mace_core.graph.GraphInfo` beside it. The core and the layout are
:mod:`mace_core.graph`'s; nothing here redefines them, and nothing here
imports torch_geometric.
"""

from mace_torch.graph.collate import build_batch, collate, collate_targets

__all__ = [
    "build_batch",
    "collate",
    "collate_targets",
]
