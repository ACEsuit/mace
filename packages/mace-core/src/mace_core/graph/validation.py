"""Checking a flat graph dict against the schema, and reading it back typed.

:func:`validate_graph` runs at boundaries only: on the output of collation and
padding, in a backend adapter, at a calculator's entry. Never inside a model
forward, which takes the dict as it is. It is ``O(n_nodes + n_edges)``.

It reads arrays of any framework without importing one. Shapes come from
``.shape`` and dtypes from the name of ``.dtype`` (``torch.float64`` and numpy's
``float64`` both read as ``float64``). The value checks copy the index arrays
to the host once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mace_core.graph.schema import (
    FLOAT,
    GRAPH_SCHEMA,
    LABEL_KEYS,
    NOT_KEYS,
    PRESENCE_SUFFIX,
    FieldSpec,
    presence_key,
)

__all__ = [
    "FLOAT_DTYPES",
    "GraphInfo",
    "GraphValidationError",
    "GraphView",
    "validate_graph",
]

#: The dtype names a model's floating dtype can take.
FLOAT_DTYPES: frozenset[str] = frozenset({"float16", "bfloat16", "float32", "float64"})


class GraphValidationError(ValueError):
    """A graph dict that does not satisfy the schema. The message names the key."""


@dataclass(frozen=True)
class GraphInfo:
    """The sizes of a graph, derived from its arrays.

    Attributes:
        num_graphs: ``len(ptr) - 1``, the only sanctioned derivation.
        num_nodes: ``positions.shape[0]``; padding slots included.
        num_edges: ``edge_index.shape[1]``; padding slots included.
    """

    num_graphs: int
    num_nodes: int
    num_edges: int


def validate_graph(
    data: Mapping[str, Any],
    *,
    require_labels: bool = False,
    allow_extra: bool = True,
    float_dtype: str = "float64",
) -> GraphInfo:
    """Check ``data`` against :data:`~mace_core.graph.schema.GRAPH_SCHEMA`.

    Checked: every required key is present; every schema key has its shape,
    with each symbolic dimension the same size wherever it appears; dtype names
    match, :data:`~mace_core.graph.schema.FLOAT` being ``float_dtype``;
    ``edge_index`` stays in ``[0, n_nodes)``; ``ptr`` starts at 0, never
    decreases and ends at ``n_nodes``; ``batch`` is
    ``repeat_interleave(arange(n_graphs), diff(ptr))``; each label travels with
    its presence flag; the host tuples agree with ``ptr`` and ``pbc``; and the
    padding structures, if any, come last, are aperiodic, keep their atoms at
    the origin and carry no present label.

    Args:
        data: The graph.
        require_labels: Also require at least one label to be present in at
            least one structure, as a training batch must have.
        allow_extra: Accept keys the schema does not name. They are carried as
            they are and nothing is inferred about their shape. A key the
            schema forbids (:data:`~mace_core.graph.schema.NOT_KEYS`) or a
            presence flag for an unknown label is refused either way.
        float_dtype: The model's floating dtype name, one of
            :data:`FLOAT_DTYPES`.

    Returns:
        The sizes of the graph.

    Raises:
        GraphValidationError: naming the first key that breaks a rule.
        ValueError: if ``float_dtype`` is not a floating dtype name.
    """
    if float_dtype not in FLOAT_DTYPES:
        raise ValueError(
            f"float_dtype is {float_dtype!r}; use one of {sorted(FLOAT_DTYPES)}."
        )
    missing = sorted(k for k, s in GRAPH_SCHEMA.items() if s.required and k not in data)
    if missing:
        raise GraphValidationError(f"the graph is missing the required keys {missing}.")
    for key in data:
        if key in GRAPH_SCHEMA:
            continue
        if key in NOT_KEYS:
            raise GraphValidationError(
                f"{key!r} is derived, not stored: read it from GraphInfo or "
                f"compute it inside the forward. Remove the key."
            )
        if key.endswith(PRESENCE_SUFFIX):
            raise GraphValidationError(
                f"{key!r} is a presence flag for {key[: -len(PRESENCE_SUFFIX)]!r}, "
                f"which is not a label of the schema. The labels are "
                f"{sorted(LABEL_KEYS)}."
            )
        if not allow_extra:
            raise GraphValidationError(
                f"{key!r} is not a key of the graph schema, and extra keys are "
                f"not allowed here."
            )

    # ptr fixes n_graphs, positions n_nodes and edge_index n_edges, so those
    # three are checked first and every other key is measured against them.
    sizes: dict[str, int] = {}
    _check_array("ptr", data["ptr"], GRAPH_SCHEMA["ptr"], sizes, float_dtype)
    sizes["n_graphs"] = sizes["n_graphs_plus_one"] - 1
    if sizes["n_graphs"] < 0:
        raise GraphValidationError("ptr is empty; it holds n_graphs + 1 offsets.")
    ordered = [
        "positions",
        "edge_index",
        *(k for k in data if k not in ("ptr", "positions", "edge_index")),
    ]
    for key in ordered:
        spec = GRAPH_SCHEMA.get(key)
        if spec is None:
            continue
        if spec.kind == "host":
            _check_host(key, data[key], spec, sizes["n_graphs"])
        else:
            _check_array(key, data[key], spec, sizes, float_dtype)

    for label in LABEL_KEYS:
        flag = presence_key(label)
        if (label in data) != (flag in data):
            have, lack = (label, flag) if label in data else (flag, label)
            raise GraphValidationError(
                f"{have!r} is present without {lack!r}. A label and its "
                f"presence flag travel together."
            )
    if require_labels and not any(
        _host_array(data[presence_key(label)]).any()
        for label in LABEL_KEYS
        if label in data
    ):
        raise GraphValidationError(
            f"no structure of the graph carries a present label, and labels are "
            f"required here. The labels are {sorted(LABEL_KEYS)}."
        )

    _check_indices(data, sizes)
    return GraphInfo(
        num_graphs=sizes["n_graphs"],
        num_nodes=sizes["n_nodes"],
        num_edges=sizes["n_edges"],
    )


def _dtype_name(value: Any) -> str:
    name = str(getattr(value, "dtype", type(value).__name__))
    return name.removeprefix("torch.")


def _check_array(
    key: str, value: Any, spec: FieldSpec, sizes: dict[str, int], float_dtype: str
) -> None:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise GraphValidationError(
            f"{key!r} is a {type(value).__name__}; the schema stores it as an array."
        )
    try:
        shape = tuple(int(n) for n in shape)
    except TypeError:
        raise GraphValidationError(
            f"{key!r} has shape {shape!r}, which is not a sequence of sizes."
        ) from None
    if len(shape) != len(spec.shape):
        raise GraphValidationError(
            f"{key!r} has shape {list(shape)}; the schema gives {list(spec.shape)}."
        )
    for size, dimension in zip(shape, spec.shape, strict=True):
        if isinstance(dimension, int):
            expected = dimension
        else:
            expected = sizes.setdefault(dimension, size)
        if size != expected:
            raise GraphValidationError(
                f"{key!r} has shape {list(shape)}; the schema gives "
                f"{list(spec.shape)} and {dimension} is {expected} in this graph."
            )
    expected_dtype = float_dtype if spec.dtype == FLOAT else spec.dtype
    if _dtype_name(value) != expected_dtype:
        raise GraphValidationError(
            f"{key!r} has dtype {_dtype_name(value)}; the schema gives "
            f"{expected_dtype}."
        )


def _check_host(key: str, value: Any, spec: FieldSpec, n_graphs: int) -> None:
    if not isinstance(value, tuple):
        raise GraphValidationError(
            f"{key!r} is a {type(value).__name__}. It is a host value: a tuple "
            f"with one entry per structure, never an array."
        )
    if len(value) != n_graphs:
        raise GraphValidationError(
            f"{key!r} has {len(value)} entries for {n_graphs} structures."
        )
    kind = bool if spec.dtype == "bool" else int
    for entry in value:
        if type(entry) is not kind:
            raise GraphValidationError(
                f"{key!r} holds {entry!r}; each entry is a Python {kind.__name__}."
            )


def _host_array(value: Any) -> np.ndarray:
    """A host numpy copy of an array of any framework."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _check_indices(data: Mapping[str, Any], sizes: dict[str, int]) -> None:
    n_nodes, n_graphs = sizes["n_nodes"], sizes["n_graphs"]
    edge_index = _host_array(data["edge_index"])
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= n_nodes):
        raise GraphValidationError(
            f"'edge_index' holds values in [{int(edge_index.min())}, "
            f"{int(edge_index.max())}]; they must be in [0, {n_nodes})."
        )
    ptr = _host_array(data["ptr"])
    if ptr[0] != 0 or ptr[-1] != n_nodes or np.any(np.diff(ptr) < 0):
        raise GraphValidationError(
            f"'ptr' is {ptr.tolist()}; it starts at 0, never decreases and "
            f"ends at n_nodes = {n_nodes}."
        )
    sizes_per_graph = np.diff(ptr)
    batch = _host_array(data["batch"])
    if not np.array_equal(batch, np.repeat(np.arange(n_graphs), sizes_per_graph)):
        raise GraphValidationError(
            "'batch' does not match 'ptr': node n belongs to graph g exactly "
            "when ptr[g] <= n < ptr[g + 1]."
        )
    if "structure_sizes" in data and list(data["structure_sizes"]) != (
        sizes_per_graph.tolist()
    ):
        raise GraphValidationError(
            f"'structure_sizes' is {list(data['structure_sizes'])}; 'ptr' gives "
            f"{sizes_per_graph.tolist()}."
        )
    padding = list(data["structure_padding"])
    first_padding = padding.index(True) if True in padding else len(padding)
    if not all(padding[first_padding:]):
        raise GraphValidationError(
            f"'structure_padding' is {padding}; padding structures are appended, "
            f"so every structure after the first padding one is padding too."
        )
    if _host_array(data["pbc"])[first_padding:].any():
        raise GraphValidationError(
            f"'structure_padding' is {padding} but a padding structure is "
            f"periodic in 'pbc'; padding structures are aperiodic."
        )
    if first_padding < n_graphs:
        first_padding_atom = int(ptr[first_padding])
        if _host_array(data["positions"])[first_padding_atom:].any():
            raise GraphValidationError(
                f"'structure_padding' is {padding} but an atom of a padding "
                f"structure is away from the origin; padding atoms sit at 0."
            )
        for label in sorted(LABEL_KEYS):
            if label not in data:
                continue
            flag = presence_key(label)
            flags = _host_array(data[flag])
            per_atom = GRAPH_SCHEMA[flag].shape[0] == "n_nodes"
            start = first_padding_atom if per_atom else first_padding
            if flags[start:].any():
                raise GraphValidationError(
                    f"{presence_key(label)!r} is true on a padding structure; "
                    f"padding carries no label."
                )
    if "structure_periodic" in data:
        periodic = _host_array(data["pbc"]).any(axis=1).tolist()
        if list(data["structure_periodic"]) != periodic:
            raise GraphValidationError(
                f"'structure_periodic' is {list(data['structure_periodic'])}; "
                f"'pbc' gives {periodic}."
            )


class GraphView:
    """Attribute access over a validated graph dict, without copying it.

    ``view.positions`` is ``data["positions"]`` and ``view.info.num_graphs``
    the graph count. Sugar for code outside the model: a forward takes the
    dict itself.

    Args:
        data: The graph. It is validated once, here.
        float_dtype: The model's floating dtype name.
    """

    __slots__ = ("_data", "info")

    def __init__(self, data: Mapping[str, Any], *, float_dtype: str = "float64"):
        self.info = validate_graph(data, float_dtype=float_dtype)
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._data:
            known = "a schema key" if name in GRAPH_SCHEMA else "not a schema key"
            raise AttributeError(f"this graph has no {name!r} ({known}).")
        return self._data[name]

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def __contains__(self, name: object) -> bool:
        return name in self._data
