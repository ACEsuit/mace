"""Checking a batch, split by what each check costs.

* :func:`check_layout` runs on **every batch** and reads no value: it compares
  the batch's signature, ``{key: (shape, dtype)}``, with the one the layout
  gives once the dimensions are bound from ``ptr``, ``positions``,
  ``edge_index`` and the requirements' own dimensions. Only on a mismatch does
  it work out which key is missing, undeclared or misshapen, so the message
  names the key. The expected signature is cached, so a padded batch of a
  fixed budget computes it once.
* The **per-sample value checks** run in
  :func:`~mace_core.graph.build.graph_sample`, on every structure, in the
  worker that builds it.
* :func:`check_batch_invariants` reads the index arrays: ``edge_index`` in
  range, ``ptr`` monotone from 0 to ``n_nodes``, ``batch`` consistent with
  ``ptr``, no edge between structures. Every value it reads is written by
  collation, padding or a provider, so it runs in their test suites and in a
  provider's conformance, and on a hand-built dict through ``validate=True``,
  never in the training loop.

Nothing here imports a framework. Shapes come from ``.shape`` and dtypes from
the name of ``.dtype``, so ``torch.float64`` and numpy's ``float64`` both read
as ``float64``. None of it runs inside a forward.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np

from mace_core.graph.info import GraphInfo
from mace_core.graph.layout import GraphLayout
from mace_core.graph.schema import (
    ANY_TRAILING,
    EDGE_KEYS,
    FLOAT,
    NOT_KEYS,
    FieldSpec,
)

__all__ = [
    "FLOAT_DTYPES",
    "GraphValidationError",
    "GraphView",
    "check_batch_invariants",
    "check_layout",
    "dtype_name",
]

#: The dtype names a model's floating dtype can take.
FLOAT_DTYPES: frozenset[str] = frozenset({"float16", "bfloat16", "float32", "float64"})

_FIXED_DIMS = ("n_nodes", "n_edges", "n_graphs", "n_graphs_plus_one")


class GraphValidationError(ValueError):
    """A batch that does not satisfy its layout. The message names the key."""


def dtype_name(value: Any) -> str:
    """The dtype name of an array of any framework: ``torch.float64`` is
    ``float64``."""
    name = str(getattr(value, "dtype", type(value).__name__))
    return name.removeprefix("torch.")


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(n) for n in shape)
    except TypeError:
        return None


def _expected_dtype(spec: FieldSpec, float_dtype: str) -> str:
    return float_dtype if spec.dtype == FLOAT else spec.dtype


@lru_cache(maxsize=256)
def _expected_signature(
    layout: GraphLayout,
    keys: frozenset[str],
    sizes: tuple[tuple[str, int], ...],
    float_dtype: str,
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """The signature of a batch with these keys and sizes, sorted by key.
    A shape closed by :data:`ANY_TRAILING` keeps only its leading dimension."""
    bound = dict(sizes)
    fields = layout.graph_fields()
    signature = []
    for key in sorted(keys):
        spec = fields[key]
        shape = []
        for dimension in spec.shape:
            if dimension == ANY_TRAILING:
                break
            shape.append(dimension if isinstance(dimension, int) else bound[dimension])
        signature.append((key, tuple(shape), _expected_dtype(spec, float_dtype)))
    return tuple(signature)


def _actual_signature(
    graph: Mapping[str, Any], fields: Mapping[str, FieldSpec]
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    signature = []
    for key in sorted(graph):
        shape = _shape(graph[key]) or ()
        spec = fields.get(key)
        if spec is not None and spec.shape and spec.shape[-1] == ANY_TRAILING:
            shape = shape[: len(spec.shape) - 1]
        signature.append((key, shape, dtype_name(graph[key])))
    return tuple(signature)


def check_layout(
    graph: Mapping[str, Any],
    layout: GraphLayout,
    info: GraphInfo,
    *,
    float_dtype: str = "float64",
) -> None:
    """Check that ``graph`` carries exactly the keys ``layout`` declares, each
    with its shape and dtype, and agrees with ``info``. Reads no value.

    Required: the structural core and every declared input. The edge keys
    come all together or not at all, the latter while a device search has
    still to join its edges, and so do the fields of the batch requirements,
    which are built on the padded batch after collation. ``head`` and
    ``displacement`` may be present.

    Raises:
        GraphValidationError: naming the first key that breaks a rule.
        ValueError: if ``float_dtype`` is not a floating dtype name.
    """
    if float_dtype not in FLOAT_DTYPES:
        raise ValueError(
            f"float_dtype is {float_dtype!r}; use one of {sorted(FLOAT_DTYPES)}."
        )
    fields = layout.graph_fields()
    sizes = _bound_sizes(graph, fields)
    keys = frozenset(graph)
    built = frozenset(layout.requirement_fields())
    if keys - set(fields) or not _required(fields, keys, built) <= keys:
        _explain_keys(graph, fields, keys, built)
    expected = _expected_signature(
        layout, keys, tuple(sorted(sizes.items())), float_dtype
    )
    if _actual_signature(graph, fields) != expected:
        _explain_signature(graph, fields, expected)
    if (info.num_graphs, info.num_nodes, info.num_edges) != (
        sizes["n_graphs"],
        sizes["n_nodes"],
        sizes["n_edges"],
    ):
        raise GraphValidationError(
            f"the batch has {sizes['n_graphs']} structures, {sizes['n_nodes']} "
            f"nodes and {sizes['n_edges']} edges, but its GraphInfo says "
            f"{info.num_graphs}, {info.num_nodes} and {info.num_edges}."
        )


def _required(
    fields: Mapping[str, FieldSpec], keys: frozenset[str], built: frozenset[str]
) -> set[str]:
    """The keys this batch must carry. The edge keys and the requirement fields
    each come as a group, all or none: none while a device search has still to
    join the edges, or before the requirements are built on the padded batch."""
    required = {key for key, spec in fields.items() if spec.required}
    if not keys & EDGE_KEYS:
        required -= EDGE_KEYS
    if not keys & built:
        required -= built
    return required


def _bound_sizes(
    graph: Mapping[str, Any], fields: Mapping[str, FieldSpec]
) -> dict[str, int]:
    """Every symbolic dimension of the batch, bound from the arrays."""
    for key in ("ptr", "positions"):
        if key not in graph:
            raise GraphValidationError(
                f"the batch is missing the required key {key!r}."
            )
    ptr, positions = _shape(graph["ptr"]), _shape(graph["positions"])
    if ptr is None or len(ptr) != 1 or ptr[0] < 1:
        raise GraphValidationError(
            f"'ptr' has shape {ptr}; it holds n_graphs + 1 offsets."
        )
    if positions is None or len(positions) != 2:
        raise GraphValidationError(
            f"'positions' has shape {positions}; the layout gives [n_nodes, 3]."
        )
    edges = _shape(graph["edge_index"]) if "edge_index" in graph else (2, 0)
    if edges is None or len(edges) != 2:
        raise GraphValidationError(
            f"'edge_index' has shape {edges}; the layout gives [2, n_edges]."
        )
    sizes = {
        "n_graphs_plus_one": ptr[0],
        "n_graphs": ptr[0] - 1,
        "n_nodes": positions[0],
        "n_edges": edges[1],
    }
    for key, spec in fields.items():
        if key not in graph:
            continue
        shape = _shape(graph[key])
        if shape is None:
            continue
        for size, dimension in zip(shape, spec.shape, strict=False):
            if isinstance(dimension, str) and dimension not in (
                *_FIXED_DIMS,
                ANY_TRAILING,
            ):
                sizes.setdefault(dimension, size)
    return sizes


def _explain_keys(
    graph: Mapping[str, Any],
    fields: Mapping[str, FieldSpec],
    keys: frozenset[str],
    built: frozenset[str],
) -> None:
    for key in sorted(keys - set(fields)):
        if key in NOT_KEYS:
            raise GraphValidationError(
                f"{key!r} is never a key of the graph: it is derived from the "
                f"arrays, computed in the forward, or kept in GraphInfo. Remove it."
            )
        raise GraphValidationError(
            f"{key!r} is not declared by the layout. Declare it as an input, a "
            f"feature or a requirement field, or leave it out of the batch."
        )
    present_edges = keys & EDGE_KEYS
    if present_edges and present_edges != EDGE_KEYS:
        raise GraphValidationError(
            f"the batch carries {sorted(present_edges)} but not "
            f"{sorted(EDGE_KEYS - present_edges)}. The edge keys come together."
        )
    missing = sorted(_required(fields, keys, built) - keys)
    raise GraphValidationError(f"the batch is missing the required keys {missing}.")


def _explain_signature(
    graph: Mapping[str, Any],
    fields: Mapping[str, FieldSpec],
    expected: tuple[tuple[str, tuple[int, ...], str], ...],
) -> None:
    actual = {
        key: (shape, dtype) for key, shape, dtype in _actual_signature(graph, fields)
    }
    for key, shape, dtype in expected:
        have_shape, have_dtype = actual[key]
        if have_shape != shape:
            raise GraphValidationError(
                f"{key!r} has shape {list(_shape(graph[key]) or ())}; the layout "
                f"gives {list(fields[key].shape)}, which is {list(shape)} in this "
                f"batch."
            )
        if have_dtype != dtype:
            raise GraphValidationError(
                f"{key!r} has dtype {have_dtype}; the layout gives {dtype}."
            )


def _host(value: Any) -> np.ndarray:
    """A host numpy copy of an array of any framework."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def check_batch_invariants(
    graph: Mapping[str, Any], info: GraphInfo | None = None
) -> None:
    """Check the values collation, padding and providers write. Copies the
    index arrays to the host, so it belongs in tests and at a hand-built
    boundary, never in the training loop.

    Raises:
        GraphValidationError: naming the array that breaks a rule.
    """
    ptr = _host(graph["ptr"])
    n_nodes = len(_host(graph["positions"]))
    if ptr[0] != 0 or ptr[-1] != n_nodes or np.any(np.diff(ptr) < 0):
        raise GraphValidationError(
            f"'ptr' is {ptr.tolist()}; it starts at 0, never decreases and ends "
            f"at n_nodes = {n_nodes}."
        )
    batch = _host(graph["batch"])
    sizes = np.diff(ptr)
    if not np.array_equal(batch, np.repeat(np.arange(len(sizes)), sizes)):
        raise GraphValidationError(
            "'batch' does not match 'ptr': node n belongs to graph g exactly when "
            "ptr[g] <= n < ptr[g + 1]."
        )
    if "edge_index" in graph:
        edge_index = _host(graph["edge_index"])
        if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= n_nodes):
            raise GraphValidationError(
                f"'edge_index' holds values in [{int(edge_index.min())}, "
                f"{int(edge_index.max())}]; they must be in [0, {n_nodes})."
            )
        if edge_index.size and np.any(batch[edge_index[0]] != batch[edge_index[1]]):
            raise GraphValidationError(
                "'edge_index' has an edge between two structures; every edge "
                "joins two atoms of one structure."
            )
    if info is not None and tuple(sizes.tolist()) != info.structure_sizes:
        raise GraphValidationError(
            f"GraphInfo.structure_sizes is {list(info.structure_sizes)}; 'ptr' "
            f"gives {sizes.tolist()}."
        )


class GraphView:
    """Attribute access over a checked graph dict, without copying it.

    ``view.positions`` is ``graph["positions"]`` and ``view.info.num_graphs``
    the graph count. Sugar for code outside the model: a forward takes the
    dict itself.
    """

    __slots__ = ("_graph", "info")

    def __init__(
        self,
        graph: Mapping[str, Any],
        layout: GraphLayout,
        info: GraphInfo,
        *,
        float_dtype: str = "float64",
    ):
        check_layout(graph, layout, info, float_dtype=float_dtype)
        self._graph = graph
        self.info = info

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._graph:
            raise AttributeError(f"this graph has no {name!r}.")
        return self._graph[name]

    def __getitem__(self, name: str) -> Any:
        return self._graph[name]

    def __contains__(self, name: object) -> bool:
        return name in self._graph
