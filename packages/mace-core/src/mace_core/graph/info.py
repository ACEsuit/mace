"""What a batch knows about itself, kept beside the graph dict on the host.

The graph dict holds arrays only. A value in it that a compiled region reads is
traced under ``jax.jit`` and guarded on by Dynamo, and per-structure sizes
change almost every batch, so carrying them in the dict, even as a static
leaf, would recompile almost every batch. They live here instead, and code
around the model reads them: unpadding, splitting results per structure,
per-structure metrics, and the builders of batch requirements.

A reader that needs a :class:`GraphInfo` and does not get one raises. None
falls back to reading the device, which would both synchronise and lose the
padding flags.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GraphInfo", "PaddingInfo"]


@dataclass(frozen=True)
class PaddingInfo:
    """The real sizes of a padded batch. Real slots come first in each.

    Attributes:
        n_real_nodes: Nodes before the first padding atom.
        n_real_edges: Edges before the first padding edge.
        n_real_graphs: Structures before the first padding structure.
    """

    n_real_nodes: int
    n_real_edges: int
    n_real_graphs: int


@dataclass(frozen=True)
class GraphInfo:
    """The sizes of a batch, on the host.

    Attributes:
        num_graphs: ``len(ptr) - 1``, padding structures included.
        num_nodes: ``positions.shape[0]``, padding atoms included.
        num_edges: ``edge_index.shape[1]``, padding edges included; 0 for a
            batch whose edges a device search has not joined yet.
        structure_sizes: Atoms per structure, ``diff(ptr)``, one entry per
            structure, padding included.
        padding: The real sizes when the batch is padded, ``None`` otherwise.
    """

    num_graphs: int
    num_nodes: int
    num_edges: int
    structure_sizes: tuple[int, ...]
    padding: PaddingInfo | None = None

    def __post_init__(self) -> None:
        if len(self.structure_sizes) != self.num_graphs:
            raise ValueError(
                f"structure_sizes has {len(self.structure_sizes)} entries for "
                f"{self.num_graphs} structures."
            )
        if sum(self.structure_sizes) != self.num_nodes:
            raise ValueError(
                f"structure_sizes sums to {sum(self.structure_sizes)} atoms for "
                f"{self.num_nodes} nodes."
            )
