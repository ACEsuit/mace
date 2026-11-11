"""The flat graph dict: its structural core, and how a field is described.

The data-to-model boundary is a plain mapping from key to array. It is the
structural core below, fixed here, plus the fields a run declares in its
:class:`~mace_core.graph.layout.GraphLayout`: the model's inputs and its batch
requirements. Labels never travel in the graph: they are targets, beside it.
Each key has a :class:`FieldSpec` stating its shape over symbolic dimensions,
its dtype by name, whether it is required and who produces it. The core is a
file-format contract: a change to it is a breaking schema change.

**Dtypes are names.** ``"float"`` is the model's floating dtype (``float64``
by default) and is bound when a batch is tensorized or checked; the others are
literal (``"int64"``, ``"bool"``). Nothing here knows a framework's dtype
objects.

**Shapes are symbolic.** ``n_nodes``, ``n_edges`` and ``n_graphs`` include the
padding slots of a padded batch, and every key using one symbol must agree on
its size. ``n_graphs`` is defined by ``ptr`` alone (``len(ptr) - 1``), which is
why ``ptr`` has the shape symbol ``n_graphs_plus_one``. Any other name is a
dimension a batch requirement declares for itself, bound by the first key that
uses it. :data:`ANY_TRAILING` closes a shape whose trailing sizes are the
data's own, as for an embedding feature.

The dict holds arrays only. Counts, per-structure sizes and the padding live
beside it, in :class:`~mace_core.graph.info.GraphInfo`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

__all__ = [
    "ANY_TRAILING",
    "EDGE_KEYS",
    "EDGE_LEVEL",
    "FLOAT",
    "GRAPH_CORE",
    "GRAPH_LEVEL",
    "NODE_LEVEL",
    "NOT_KEYS",
    "PRESENCE_SUFFIX",
    "WEIGHT_SUFFIX",
    "FieldSpec",
    "Producer",
    "SymDim",
    "level_of",
    "presence_key",
    "weight_key",
]

SymDim = Literal["n_nodes", "n_edges", "n_graphs", "n_graphs_plus_one"]

#: Who writes a key: the dataset (``data``), the neighbour list (``nbr``),
#: collation (``coll``), graph construction (``derived``), the caller of the
#: model (``caller``), or a batch requirement built on the collated batch
#: (``requirement``).
Producer = Literal["data", "nbr", "coll", "derived", "caller", "requirement"]

#: The model's floating dtype, bound per framework.
FLOAT = "float"

#: The last entry of a shape whose trailing sizes are the data's own.
ANY_TRAILING = "..."

#: The suffix of a target's presence flag.
PRESENCE_SUFFIX = "_present"

#: The suffix of a target's loss weight.
WEIGHT_SUFFIX = "_weight"

NODE_LEVEL = "n_nodes"
EDGE_LEVEL = "n_edges"
GRAPH_LEVEL = "n_graphs"


@dataclass(frozen=True)
class FieldSpec:
    """One key of the flat graph dict, or of the targets beside it.

    Attributes:
        shape: Symbolic dimensions and fixed integers, possibly closed by
            :data:`ANY_TRAILING`.
        dtype: A dtype name; :data:`FLOAT` is the model's floating dtype.
        required: Whether every graph carries the key.
        producer: Who writes the key.
        doc: Units and meaning.
        absent_value: What a structure without a value for a declared input
            carries, and what padding writes. 1 for ``total_spin``, 0 for
            everything else.
    """

    shape: tuple[SymDim | int | str, ...]
    dtype: str
    required: bool
    producer: Producer
    doc: str
    absent_value: float = 0.0


def presence_key(name: str) -> str:
    """The name of target ``name``'s presence flag."""
    return name + PRESENCE_SUFFIX


def weight_key(name: str) -> str:
    """The name of target ``name``'s loss weight."""
    return name + WEIGHT_SUFFIX


def level_of(spec: FieldSpec) -> str | None:
    """``n_nodes``, ``n_edges`` or ``n_graphs``: the dimension a key is
    concatenated or stacked along when graphs are batched. ``None`` for
    ``ptr``, which collation builds."""
    for dimension in spec.shape:
        if dimension in (NODE_LEVEL, EDGE_LEVEL, GRAPH_LEVEL):
            return dimension
    return None


#: The structural core. Read-only: a change here is a schema change.
GRAPH_CORE: Mapping[str, FieldSpec] = MappingProxyType(
    {
        "positions": FieldSpec(
            ("n_nodes", 3), FLOAT, True, "data", "Cartesian positions, Å."
        ),
        "node_atomic_numbers": FieldSpec(
            ("n_nodes",),
            "int64",
            True,
            "data",
            "Atomic number Z of each atom; 0 for a padding atom. The model forms "
            "its element one-hot from it.",
        ),
        "edge_index": FieldSpec(
            (2, "n_edges"),
            "int64",
            True,
            "nbr",
            "Row 0 the sender, row 1 the receiver, both global node indices.",
        ),
        "shifts": FieldSpec(
            ("n_edges", 3),
            FLOAT,
            True,
            "nbr",
            "Edge displacement, Å: unit_shifts @ cell.",
        ),
        "unit_shifts": FieldSpec(
            ("n_edges", 3),
            FLOAT,
            True,
            "nbr",
            "Integer lattice-vector counts, stored as float. Authoritative under "
            "strain, where shifts are recomputed from them.",
        ),
        "cell": FieldSpec(
            ("n_graphs", 3, 3),
            FLOAT,
            True,
            "data",
            "The parsed cell per structure, rows as lattice vectors, Å; zero for "
            "a structure without one. Never extended.",
        ),
        "pbc": FieldSpec(
            ("n_graphs", 3), "bool", True, "data", "Periodicity per axis."
        ),
        "batch": FieldSpec(
            ("n_nodes",),
            "int64",
            True,
            "coll",
            "The graph each node belongs to, non-decreasing.",
        ),
        "ptr": FieldSpec(
            ("n_graphs_plus_one",),
            "int64",
            True,
            "coll",
            "Node offsets: graph g owns nodes ptr[g]:ptr[g + 1].",
        ),
        "head": FieldSpec(
            ("n_graphs",),
            "int64",
            False,
            "derived",
            "Index of the structure's head in the layout's head list; 0 when absent.",
        ),
        "displacement": FieldSpec(
            ("n_graphs", 3, 3),
            FLOAT,
            False,
            "caller",
            "A caller-supplied strain, used in place of the zero strain the "
            "derivative engine otherwise creates. Transported only.",
        ),
    }
)

#: The keys a neighbour list writes. A batch carries all three or none: none
#: while a device search is still to join its edges.
EDGE_KEYS: frozenset[str] = frozenset({"edge_index", "shifts", "unit_shifts"})

#: Names no graph may carry: each is derived from the arrays, computed in the
#: forward, or kept beside the dict, so a stored copy could only be stale.
NOT_KEYS: frozenset[str] = frozenset(
    {
        "num_nodes",
        "num_edges",
        "num_graphs",
        "edge_vectors",
        "edge_lengths",
        "node_attrs",
        "rcell",
        "volume",
        "structure_sizes",
        "structure_periodic",
        "structure_padding",
    }
)
