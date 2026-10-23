"""The flat graph dict: every key a model may be handed, as data.

The data-to-model boundary is a plain mapping from key to array, and this
module is its specification. Each key has a :class:`FieldSpec` stating its
shape over symbolic dimensions, its dtype by name, whether it is required, how
a padded batch fills it, and who produces it. The table is a file-format
contract: a change to it is a breaking schema change.

**Dtypes are names.** ``"float"`` is the model's floating dtype (``float64``
by default) and is bound when a batch is tensorized or validated; the others
are literal (``"int64"``, ``"bool"``). Nothing here knows a framework's dtype
objects.

**Shapes are symbolic.** ``n_nodes``, ``n_edges`` and ``n_graphs`` include the
padding slots of a padded batch, and every key using one symbol must agree on
its size. ``n_graphs`` is defined by ``ptr`` alone (``len(ptr) - 1``), which is
why ``ptr`` has the shape symbol ``n_graphs_plus_one``.

**Host keys** are Python tuples rather than arrays (``kind="host"``), one entry
per structure. They exist so that whatever branches per structure decides on
the host, and they are never read inside a compiled region.

Deliberately **not** keys: ``num_nodes``, ``num_edges`` and ``num_graphs``,
derived from the arrays by :class:`~mace_core.graph.validation.GraphInfo`;
``edge_vectors`` and ``edge_lengths``, computed inside the forward after the
strain is applied; and every model output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

__all__ = [
    "EDGE_LEVEL",
    "FLOAT",
    "GRAPH_LEVEL",
    "GRAPH_SCHEMA",
    "HOST_KEYS",
    "INPUT_KEYS",
    "LABEL_KEYS",
    "NODE_LEVEL",
    "NOT_KEYS",
    "PRESENCE_SUFFIX",
    "FieldSpec",
    "PadValue",
    "Producer",
    "SymDim",
    "level_of",
    "presence_key",
]

SymDim = Literal[
    "n_nodes",
    "n_edges",
    "n_graphs",
    "n_graphs_plus_one",
    "n_elements",
    "n_density_coefficients",
]

#: How a padded batch fills a key's padding slots.
#:
#: ``zero``, ``one``, ``false`` and ``true`` are constants.
#: ``element_zero_one_hot``: the one-hot of the first element of the table.
#: ``element_zero_atomic_number``: that element's atomic number.
#: ``self_loop``: an edge from the last padding atom to itself.
#: ``padding_shift``: ``max(2 * r_max, 1)`` on x and zero elsewhere, so a
#: padding edge is longer than the cutoff and contributes exactly nothing.
#: ``unit_x``: the unit shift ``(1, 0, 0)``.
#: ``padding_cell``: ``eye(3) * max(2 * r_max, 1)``.
#: ``padding_reciprocal_cell`` and ``padding_volume``: the ``rcell`` and the
#: ``volume`` derived from that padding cell, exactly as for a real structure,
#: so a padding structure is an ordinary aperiodic one with a finite volume.
#: ``padding_graph_id``: the index of the last padding graph.
#: ``node_count``: the padded node count, closing ``ptr``.
#: ``padding_structure_size``: the number of atoms in that padding structure.
PadValue = Literal[
    "zero",
    "one",
    "false",
    "true",
    "element_zero_one_hot",
    "element_zero_atomic_number",
    "self_loop",
    "padding_shift",
    "unit_x",
    "padding_cell",
    "padding_reciprocal_cell",
    "padding_volume",
    "padding_graph_id",
    "node_count",
    "padding_structure_size",
]

#: Who writes a key: the dataset (``data``), the neighbour list (``nbr``),
#: collation (``coll``), graph construction (``derived``) or the caller of the
#: model (``caller``).
Producer = Literal["data", "nbr", "coll", "derived", "caller"]

#: The model's floating dtype, bound per framework.
FLOAT = "float"

#: The suffix of a label's presence flag.
PRESENCE_SUFFIX = "_present"

NODE_LEVEL = "n_nodes"
EDGE_LEVEL = "n_edges"
GRAPH_LEVEL = "n_graphs"


@dataclass(frozen=True)
class FieldSpec:
    """One key of the flat graph dict.

    Attributes:
        shape: Symbolic dimensions and fixed integers.
        dtype: A dtype name; :data:`FLOAT` is the model's floating dtype.
        required: Whether every graph carries the key. A label is not
            required, and its presence flag travels with it.
        pad_value: How a padded batch fills the padding slots.
        producer: Who writes the key.
        doc: Units and meaning.
        kind: ``tensor`` for an array, ``host`` for a Python tuple with one
            entry per structure.
        label: Whether the key is a training label. Every label has a
            presence flag, :func:`presence_key`, that is present exactly when
            the label is.
        flags: For a presence flag, the label it belongs to.
    """

    shape: tuple[SymDim | int, ...]
    dtype: str
    required: bool
    pad_value: PadValue
    producer: Producer
    doc: str
    kind: Literal["tensor", "host"] = "tensor"
    label: bool = False
    flags: str | None = None


def presence_key(label: str) -> str:
    """The name of ``label``'s presence flag."""
    return label + PRESENCE_SUFFIX


def level_of(spec: FieldSpec) -> str | None:
    """``n_nodes``, ``n_edges`` or ``n_graphs``: the dimension a key is
    concatenated or stacked along when graphs are batched. ``None`` for
    ``ptr``, which collation builds."""
    for dimension in spec.shape:
        if dimension in (NODE_LEVEL, EDGE_LEVEL, GRAPH_LEVEL):
            return dimension
    return None


def _labels() -> dict[str, FieldSpec]:
    """The label keys, each followed by its presence flag."""
    labels: dict[str, tuple[tuple[SymDim | int, ...], str]] = {
        "energy": (("n_graphs",), "Total energy, eV."),
        "forces": (("n_nodes", 3), "Forces, eV/Å."),
        "magforces": (("n_nodes", 3), "Magnetic forces; read as REF_magforces."),
        "stress": (("n_graphs", 3, 3), "Stress, eV/Å³; Voigt read as the matrix."),
        "virials": (("n_graphs", 3, 3), "Virials, eV; Voigt read as the matrix."),
        "dipole": (("n_graphs", 3), "Total dipole, e·Å."),
        "charges": (("n_nodes",), "Per-atom charges, e."),
        "polarizability": (("n_graphs", 3, 3), "Polarizability tensor."),
    }
    specs: dict[str, FieldSpec] = {}
    for name, (shape, doc) in labels.items():
        specs[name] = FieldSpec(shape, FLOAT, False, "zero", "data", doc, label=True)
        level: tuple[SymDim | int, ...] = (shape[0],)
        specs[presence_key(name)] = FieldSpec(
            level,
            "bool",
            False,
            "false",
            "derived",
            f"Whether {name} is labelled, from Configuration.is_labelled. A "
            f"zero weight does not make a present label absent.",
            flags=name,
        )
    return specs


def _weights() -> dict[str, FieldSpec]:
    graph_weights = (
        "weight",
        "energy_weight",
        "forces_weight",
        "stress_weight",
        "virials_weight",
        "charges_weight",
        "magforces_weight",
    )
    specs = {
        name: FieldSpec(
            ("n_graphs",),
            FLOAT,
            False,
            "zero",
            "data",
            "Loss weight per structure; 1 when the file gives none. Training only.",
        )
        for name in graph_weights
    }
    specs["dipole_weight"] = FieldSpec(
        ("n_graphs", 3), FLOAT, False, "zero", "data", "Dipole loss weight."
    )
    specs["polarizability_weight"] = FieldSpec(
        ("n_graphs", 3, 3), FLOAT, False, "zero", "data", "Polarizability weight."
    )
    return specs


_STRUCTURE: dict[str, FieldSpec] = {
    "positions": FieldSpec(
        ("n_nodes", 3), FLOAT, True, "zero", "data", "Cartesian positions, Å."
    ),
    "node_attrs": FieldSpec(
        ("n_nodes", "n_elements"),
        FLOAT,
        True,
        "element_zero_one_hot",
        "data",
        "One-hot of each atom's element, indexed by the model's element table.",
    ),
    "node_atomic_numbers": FieldSpec(
        ("n_nodes",),
        "int64",
        True,
        "element_zero_atomic_number",
        "data",
        "Atomic number Z of each atom, beside the one-hot.",
    ),
    "edge_index": FieldSpec(
        (2, "n_edges"),
        "int64",
        True,
        "self_loop",
        "nbr",
        "Row 0 the sender, row 1 the receiver, both global node indices.",
    ),
    "shifts": FieldSpec(
        ("n_edges", 3),
        FLOAT,
        True,
        "padding_shift",
        "nbr",
        "Edge displacement, Å: unit_shifts @ cell with the effective cell.",
    ),
    "unit_shifts": FieldSpec(
        ("n_edges", 3),
        FLOAT,
        True,
        "unit_x",
        "nbr",
        "Integer lattice-vector counts, stored as float. Authoritative under "
        "strain, where shifts are recomputed from them.",
    ),
    "cell": FieldSpec(
        ("n_graphs", 3, 3),
        FLOAT,
        True,
        "padding_cell",
        "nbr",
        "The effective cell per structure, rows as lattice vectors, Å.",
    ),
    "pbc": FieldSpec(
        ("n_graphs", 3), "bool", True, "false", "data", "Periodicity per axis."
    ),
    "rcell": FieldSpec(
        ("n_graphs", 3, 3),
        FLOAT,
        True,
        "padding_reciprocal_cell",
        "derived",
        "2π · inv(cellᵀ), exactly zero where det(cell) == 0. Not recomputed "
        "under strain.",
    ),
    "volume": FieldSpec(
        ("n_graphs",),
        FLOAT,
        True,
        "padding_volume",
        "derived",
        "det(cell), Å³. Not recomputed under strain.",
    ),
    "batch": FieldSpec(
        ("n_nodes",),
        "int64",
        True,
        "padding_graph_id",
        "coll",
        "The graph each node belongs to, non-decreasing.",
    ),
    "ptr": FieldSpec(
        ("n_graphs_plus_one",),
        "int64",
        True,
        "node_count",
        "coll",
        "Node offsets: graph g owns nodes ptr[g]:ptr[g + 1].",
    ),
    "head": FieldSpec(
        ("n_graphs",),
        "int64",
        False,
        "zero",
        "derived",
        "Index of the structure's head in the model's ordered head list; 0 "
        "when absent.",
    ),
    "displacement": FieldSpec(
        ("n_graphs", 3, 3),
        FLOAT,
        False,
        "zero",
        "caller",
        "A caller-supplied strain, used in place of the zero strain the "
        "derivative engine otherwise creates. Transported only.",
    ),
}

_INPUTS: dict[str, FieldSpec] = {
    "magmom": FieldSpec(
        ("n_nodes", 3), FLOAT, False, "zero", "data", "Magnetic moments, μB."
    ),
    "total_charge": FieldSpec(
        ("n_graphs",), FLOAT, False, "zero", "data", "Total charge, e."
    ),
    "total_spin": FieldSpec(
        ("n_graphs",),
        FLOAT,
        False,
        "one",
        "data",
        "Total spin multiplicity; 1 when absent, and 1 in padding.",
    ),
    "elec_temp": FieldSpec(
        ("n_graphs",), FLOAT, False, "zero", "data", "Electronic temperature."
    ),
    "fermi_level": FieldSpec(
        ("n_graphs",), FLOAT, False, "zero", "data", "Fermi level, eV."
    ),
    "external_field": FieldSpec(
        ("n_graphs", 3), FLOAT, False, "zero", "data", "External electric field."
    ),
    "density_coefficients": FieldSpec(
        ("n_nodes", "n_density_coefficients"),
        FLOAT,
        False,
        "zero",
        "data",
        "Per-atom density-fit coefficients for the electrostatics models.",
    ),
}

_HOST: dict[str, FieldSpec] = {
    "structure_sizes": FieldSpec(
        ("n_graphs",),
        "int",
        True,
        "padding_structure_size",
        "coll",
        "Atoms per structure, equal to the differences of ptr.",
        kind="host",
    ),
    "structure_periodic": FieldSpec(
        ("n_graphs",),
        "bool",
        True,
        "false",
        "coll",
        "pbc[g].any() for each structure, padding included.",
        kind="host",
    ),
    "structure_padding": FieldSpec(
        ("n_graphs",),
        "bool",
        True,
        "true",
        "coll",
        "True for a structure pad_batch appended.",
        kind="host",
    ),
}

#: The schema. Read-only: a change here is a schema change, made in the source.
GRAPH_SCHEMA: Mapping[str, FieldSpec] = MappingProxyType(
    {**_STRUCTURE, **_INPUTS, **_weights(), **_labels(), **_HOST}
)

#: The training labels, each with a presence flag.
LABEL_KEYS: frozenset[str] = frozenset(k for k, s in GRAPH_SCHEMA.items() if s.label)

#: The per-structure and per-atom inputs a dataset supplies beside the
#: structure itself.
INPUT_KEYS: frozenset[str] = frozenset(_INPUTS)

#: The keys held as Python tuples.
HOST_KEYS: frozenset[str] = frozenset(
    k for k, s in GRAPH_SCHEMA.items() if s.kind == "host"
)

#: Names a graph must not carry: each is derived from the arrays or computed
#: inside the forward, so a stored copy could only be stale.
NOT_KEYS: frozenset[str] = frozenset(
    {"num_nodes", "num_edges", "num_graphs", "edge_vectors", "edge_lengths"}
)
