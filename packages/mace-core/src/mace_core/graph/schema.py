"""The flat graph dict: its structural core, and how a field is described.

The data-to-model boundary is a plain mapping from key to array. It is the
structural core below, fixed here, plus the fields a run declares in its
:class:`~mace_core.graph.layout.GraphLayout`: the model's inputs and embedding
features, and the fields of its batch requirements. Labels never travel in the
graph: they are targets, beside it. Each key has a :class:`FieldSpec` stating
its shape over symbolic dimensions, its dtype by name and whether every graph
carries it. The core is a file-format contract: a change to it is a breaking
schema change.

**Dtypes are names.** ``"float"`` is the run's floating dtype, which the
layout names; the others are literal (``"int64"``, ``"bool"``). Nothing here
knows a framework's dtype objects.

**Shapes are symbolic.** ``n_nodes``, ``n_edges`` and ``n_graphs`` include the
padding slots of a padded batch, and every key using one symbol must agree on
its size. ``n_graphs`` is defined by ``ptr`` alone (``len(ptr) - 1``), which is
why ``ptr`` has the shape symbol ``n_graphs_plus_one``. Any other name is a
dimension a batch requirement declares for itself, bound by the first key that
uses it. Every other entry is a fixed size.

The dict holds arrays only. Counts, per-structure sizes and the padding live
beside it, in :class:`~mace_core.graph.info.GraphInfo`.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import Field

from mace_core.config.base import ConfigSection

__all__ = [
    "EDGE_KEYS",
    "EDGE_LEVEL",
    "FIXED_DIMENSIONS",
    "FLOAT",
    "GRAPH_CORE",
    "GRAPH_LEVEL",
    "NODE_LEVEL",
    "PRESENCE_SUFFIX",
    "WEIGHT_SUFFIX",
    "FieldSpec",
    "presence_key",
    "weight_key",
]

#: The run's floating dtype, which the layout names.
FLOAT = "float"

#: The suffix of a target's presence flag.
PRESENCE_SUFFIX = "_present"

#: The suffix of a target's loss weight.
WEIGHT_SUFFIX = "_weight"

NODE_LEVEL = "n_nodes"
EDGE_LEVEL = "n_edges"
GRAPH_LEVEL = "n_graphs"

#: The dimensions every batch binds from its own arrays.
FIXED_DIMENSIONS: frozenset[str] = frozenset(
    {NODE_LEVEL, EDGE_LEVEL, GRAPH_LEVEL, "n_graphs_plus_one"}
)

_LEVELS = (NODE_LEVEL, EDGE_LEVEL, GRAPH_LEVEL)

#: A fixed size in a shape.
Size = Annotated[int, Field(ge=0, strict=True)]

#: A symbolic dimension in a shape.
DimensionName = Annotated[str, Field(min_length=1, strict=True)]


class FieldSpec(ConfigSection):
    """One key of the flat graph dict, or of the targets beside it.

    Attributes:
        shape: Symbolic dimensions and fixed sizes.
        dtype: ``"float"`` for the run's floating dtype, ``"int64"`` or
            ``"bool"``.
        required: Whether every graph carries the key.
        absent_value: What a structure without a value for a declared input
            carries, and what padding writes.
    """

    shape: tuple[Size | DimensionName, ...]
    dtype: Literal["float", "int64", "bool"]
    required: bool = True
    absent_value: float = 0.0

    def level(self) -> str | None:
        """``n_nodes``, ``n_edges`` or ``n_graphs``: the dimension the key is
        concatenated or stacked along when graphs are batched. ``None`` for
        ``ptr``, which collation builds."""
        for dimension in self.shape:
            if dimension in _LEVELS:
                return str(dimension)
        return None

    def sample_shape(self, n_atoms: int) -> tuple[int, ...]:
        """The shape of one structure's value of a fixed-shape node or graph
        field: ``(n_atoms, ...)`` per node, the trailing sizes per graph."""
        trailing = tuple(int(size) for size in self.shape[1:])
        return (n_atoms, *trailing) if self.shape[0] == NODE_LEVEL else trailing


def presence_key(name: str) -> str:
    """The name of target ``name``'s presence flag."""
    return name + PRESENCE_SUFFIX


def weight_key(name: str) -> str:
    """The name of target ``name``'s loss weight."""
    return name + WEIGHT_SUFFIX


#: The structural core. Read-only: a change here is a schema change.
GRAPH_CORE: Mapping[str, FieldSpec] = MappingProxyType(
    {
        # Cartesian positions, Å.
        "positions": FieldSpec(shape=(NODE_LEVEL, 3), dtype=FLOAT),
        # Atomic number Z of each atom; 0 for a padding atom. The model forms
        # its element one-hot from it.
        "node_atomic_numbers": FieldSpec(shape=(NODE_LEVEL,), dtype="int64"),
        # Row 0 the sender, row 1 the receiver, both global node indices.
        "edge_index": FieldSpec(shape=(2, EDGE_LEVEL), dtype="int64"),
        # Edge displacement, Å: unit_shifts @ cell.
        "shifts": FieldSpec(shape=(EDGE_LEVEL, 3), dtype=FLOAT),
        # Integer lattice-vector counts, stored as float. Authoritative under
        # strain, where shifts are recomputed from them.
        "unit_shifts": FieldSpec(shape=(EDGE_LEVEL, 3), dtype=FLOAT),
        # The parsed cell per structure, rows as lattice vectors, Å; zero for a
        # structure without one. Never extended.
        "cell": FieldSpec(shape=(GRAPH_LEVEL, 3, 3), dtype=FLOAT),
        # Periodicity per axis.
        "pbc": FieldSpec(shape=(GRAPH_LEVEL, 3), dtype="bool"),
        # The graph each node belongs to, non-decreasing. Written by collation.
        "batch": FieldSpec(shape=(NODE_LEVEL,), dtype="int64"),
        # Node offsets: graph g owns nodes ptr[g]:ptr[g + 1]. Written by
        # collation.
        "ptr": FieldSpec(shape=("n_graphs_plus_one",), dtype="int64"),
        # Index of the structure's head in the layout's head list.
        "head": FieldSpec(shape=(GRAPH_LEVEL,), dtype="int64", required=False),
        # A caller-supplied strain, used in place of the zero strain the
        # derivative engine otherwise creates. Transported only.
        "displacement": FieldSpec(
            shape=(GRAPH_LEVEL, 3, 3), dtype=FLOAT, required=False
        ),
    }
)

#: The keys a neighbour list writes. A batch carries all three or none: none
#: while a device search is still to join its edges.
EDGE_KEYS: frozenset[str] = frozenset({"edge_index", "shifts", "unit_shifts"})
