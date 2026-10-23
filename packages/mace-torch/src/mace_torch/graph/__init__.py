"""The torch binding of the flat graph dict.

``collate`` and ``build_batch`` turn graph samples, or parsed structures, into
one batched dict of tensors. The schema itself is
:data:`mace_core.graph.GRAPH_SCHEMA`; nothing here redefines it, and nothing
here imports torch_geometric.
"""

from mace_torch.graph.collate import build_batch, collate, dtype_name

__all__ = ["build_batch", "collate", "dtype_name"]
