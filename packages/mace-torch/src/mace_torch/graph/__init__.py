"""The torch binding of the flat graph dict.

``collate``, ``collate_targets`` and ``build_batch`` turn graph samples, or
parsed structures, into one batched dict of tensors with its
:class:`~mace_core.graph.GraphInfo` beside it; ``pad_batch`` and
``pad_targets`` implement the static-shape padding contract, and
:func:`mace_core.graph.unpad_outputs` strips it from outputs. The core
and the layout are :mod:`mace_core.graph`'s; nothing here redefines them, and
nothing here imports torch_geometric.
"""

from mace_torch.graph.collate import build_batch, collate, collate_targets
from mace_torch.graph.padding import pad_batch, pad_targets

__all__ = [
    "build_batch",
    "collate",
    "collate_targets",
    "pad_batch",
    "pad_targets",
]
