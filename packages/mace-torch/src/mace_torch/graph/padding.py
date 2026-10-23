"""Padding a batched torch graph to a budget, and stripping it from outputs.

The contract is :mod:`mace_core.graph.padding`'s: the padded dict is a valid
graph whose padding structures, atoms and edges are physically inert, and the
real sizes travel beside it in a :class:`~mace_core.graph.PaddingInfo`.
Nothing is reordered, so :func:`unpad_outputs` keeps the leading real slots of
each output and that is the whole inverse.

:func:`pad_batch` does not validate its output, since a device-resident batch
would have to be copied to the host to do so. A caller at a boundary validates.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, overload

import torch
from mace_core.elements import AtomicNumberTable
from mace_core.elements.default_keys import Storage
from mace_core.graph import (
    GRAPH_SCHEMA,
    AutoPadding,
    FixedPadding,
    NoPadding,
    PaddingBudget,
    PaddingInfo,
    padding_budget,
    padding_length,
)
from mace_core.graph.collate import feature_levels
from mace_core.graph.schema import EDGE_LEVEL, GRAPH_LEVEL, NODE_LEVEL, level_of
from mace_core.observables import DEFAULT_CATALOGUE, ObservableCatalogue
from mace_core.outputs import CORE_FIELD_NAMES, MACEOutput

from mace_torch.graph.collate import derived_cell_quantities

__all__ = ["pad_batch", "unpad_outputs"]

logger = logging.getLogger(__name__)

#: Whether each core field of a model output is per atom.
_CORE_FIELD_PER_ATOM = {
    "total_energy": False,
    "node_energies": True,
    "forces": True,
    "stress": False,
    "virials": False,
    "dipole": False,
}


def pad_batch(
    data: Mapping[str, Any],
    policy: NoPadding | FixedPadding | AutoPadding,
    r_max: float,
    *,
    atomic_number_table: AtomicNumberTable,
    features: Mapping[str, Storage] | None = None,
) -> tuple[dict[str, Any], PaddingInfo]:
    """``data`` padded to the budget ``policy`` gives, and its real sizes.

    Args:
        data: A collated graph dict. Not modified.
        policy: The padding policy.
        r_max: The model's cutoff, Angstrom; sets the padding edge length and
            the padding cell.
        atomic_number_table: The model's element table. Padding atoms are its
            first element, in the one-hot and in ``node_atomic_numbers``.
        features: The declared embedding features, padded with zeros.

    Raises:
        PaddingOverflowError: for a fixed budget the batch does not fit, when
            the policy says ``error``.
        ValueError: for a key whose level is unknown and that is not a tuple,
            since it cannot be padded without guessing its shape.
    """
    n_nodes = int(data["positions"].shape[0])
    n_edges = int(data["edge_index"].shape[1])
    n_graphs = int(data["ptr"].shape[0]) - 1
    info = PaddingInfo(
        n_real_nodes=n_nodes, n_real_edges=n_edges, n_real_graphs=n_graphs
    )
    budget = padding_budget(policy, n_nodes=n_nodes, n_edges=n_edges, n_graphs=n_graphs)
    if budget is None or budget == PaddingBudget(n_nodes, n_edges, n_graphs):
        return dict(data), info
    if isinstance(policy, FixedPadding) and budget != PaddingBudget(
        policy.nodes_budget, policy.edges_budget, policy.graphs_budget
    ):
        logger.warning(
            "the batch of %d nodes, %d edges and %d structures exceeds the "
            "padding budget; padding to %d, %d and %d instead, which recompiles "
            "a compiled model once",
            n_nodes,
            n_edges,
            n_graphs,
            budget.nodes,
            budget.edges,
            budget.graphs,
        )

    padding = _Padding(
        nodes=budget.nodes - n_nodes,
        edges=budget.edges - n_edges,
        graphs=budget.graphs - n_graphs,
        budget=budget,
        length=padding_length(r_max),
        element_zero=atomic_number_table.index_to_z(0),
    )
    feature_level = feature_levels(features)
    padded: dict[str, Any] = {}
    for key, value in data.items():
        if key == "ptr":
            padded[key] = torch.cat((value, padding.ptr_tail(value)))
        elif key in GRAPH_SCHEMA and GRAPH_SCHEMA[key].kind == "host":
            padded[key] = value + padding.host_tail(key)
        elif key in GRAPH_SCHEMA:
            level = level_of(GRAPH_SCHEMA[key])
            padded[key] = padding.extend(key, value, level, GRAPH_SCHEMA[key].pad_value)
        elif key in feature_level:
            padded[key] = padding.extend(key, value, feature_level[key], "zero")
        elif isinstance(value, tuple):
            padded[key] = value + (None,) * padding.graphs
        else:
            raise ValueError(
                f"{key!r} is neither a schema key nor a declared feature, so "
                f"pad_batch cannot tell which dimension to pad. Declare it in "
                f"`features`, or leave it out of the batch."
            )
    return padded, info


class _Padding:
    """The padding rows of one batch, by key."""

    def __init__(
        self,
        *,
        nodes: int,
        edges: int,
        graphs: int,
        budget: PaddingBudget,
        length: float,
        element_zero: int,
    ) -> None:
        self.nodes, self.edges, self.graphs = nodes, edges, graphs
        self.budget = budget
        self.length = length
        self.element_zero = element_zero

    def count(self, level: str | None) -> int:
        if level == NODE_LEVEL:
            return self.nodes
        if level == EDGE_LEVEL:
            return self.edges
        if level == GRAPH_LEVEL:
            return self.graphs
        raise ValueError(f"{level!r} is not a level padding extends.")

    def extend(
        self, key: str, value: torch.Tensor, level: str | None, pad_value: str
    ) -> torch.Tensor:
        axis = 1 if key == "edge_index" else 0
        shape = list(value.shape)
        shape[axis] = self.count(level)
        rows = torch.zeros(shape, dtype=value.dtype, device=value.device)
        if pad_value in ("one", "true"):
            rows.fill_(1)
        elif pad_value == "element_zero_one_hot":
            rows[:, 0] = 1
        elif pad_value == "element_zero_atomic_number":
            rows.fill_(self.element_zero)
        elif pad_value == "self_loop":
            rows.fill_(self.budget.nodes - 1)
        elif pad_value == "padding_shift":
            rows[:, 0] = self.length
        elif pad_value == "unit_x":
            rows[:, 0] = 1
        elif pad_value == "padding_cell":
            rows[:] = self._cells(rows)
        elif pad_value == "padding_reciprocal_cell":
            rows[:] = derived_cell_quantities(self._cells(rows))[0]
        elif pad_value == "padding_volume":
            rows[:] = derived_cell_quantities(self._cells(rows[:, None, None]))[1]
        elif pad_value == "padding_graph_id":
            rows.fill_(self.budget.graphs - 1)
        elif pad_value not in ("zero", "false"):
            raise ValueError(f"{key!r} has the pad value {pad_value!r}, unknown here.")
        return torch.cat((value, rows), dim=axis)

    def _cells(self, like: torch.Tensor) -> torch.Tensor:
        """``len(like)`` padding cells in ``like``'s dtype and device."""
        eye = torch.eye(3, dtype=like.dtype, device=like.device)
        return (self.length * eye).expand(len(like), 3, 3)

    def ptr_tail(self, ptr: torch.Tensor) -> torch.Tensor:
        """Empty padding structures, then the last one holding every padding atom."""
        tail = torch.full(
            (self.graphs,), int(ptr[-1]), dtype=ptr.dtype, device=ptr.device
        )
        tail[-1] = self.budget.nodes
        return tail

    def host_tail(self, key: str) -> tuple[Any, ...]:
        if key == "structure_sizes":
            return (0,) * (self.graphs - 1) + (self.nodes,)
        if key == "structure_periodic":
            return (False,) * self.graphs
        return (True,) * self.graphs


@overload
def unpad_outputs(
    outputs: MACEOutput,
    info: PaddingInfo,
    *,
    catalogue: ObservableCatalogue = ...,
) -> MACEOutput: ...


@overload
def unpad_outputs(
    outputs: Mapping[str, Any],
    info: PaddingInfo,
    *,
    catalogue: ObservableCatalogue = ...,
) -> dict[str, Any]: ...


def unpad_outputs(
    outputs: MACEOutput | Mapping[str, Any],
    info: PaddingInfo,
    *,
    catalogue: ObservableCatalogue = DEFAULT_CATALOGUE,
) -> MACEOutput | dict[str, Any]:
    """The real slots of a padded model's outputs.

    Per-graph values keep their first ``n_real_graphs`` rows and per-atom
    values their first ``n_real_nodes``. Whether a value is per atom comes
    from the core fields of :class:`~mace_core.outputs.MACEOutput` and, for
    anything else, from ``catalogue``; it is never inferred from a shape.

    Args:
        outputs: A :class:`~mace_core.outputs.MACEOutput`, or a mapping from
            observable name to tensor. ``None`` values are kept.
        info: What :func:`pad_batch` returned.
        catalogue: The declared observables of the model.

    Returns:
        A new object of the same kind.

    Raises:
        KeyError: for a value the catalogue does not declare.
    """
    if isinstance(outputs, MACEOutput):
        unpadded: MACEOutput = MACEOutput()
        for name in CORE_FIELD_NAMES:
            value = getattr(outputs, name)
            setattr(unpadded, name, _keep_real(value, _CORE_FIELD_PER_ATOM[name], info))
        unpadded.extras = {
            name: _keep_real(value, _per_atom(name, catalogue), info)
            for name, value in outputs.extras.items()
        }
        return unpadded
    return {
        name: _keep_real(value, _per_atom(name, catalogue), info)
        for name, value in outputs.items()
    }


def _per_atom(name: str, catalogue: ObservableCatalogue) -> bool:
    if name in _CORE_FIELD_PER_ATOM:
        return _CORE_FIELD_PER_ATOM[name]
    for spec in catalogue.observables:
        if spec.name == name:
            return spec.per_atom
    for derivative in catalogue.requested_derivatives():
        if derivative.name == name:
            return derivative.per_atom
    raise KeyError(
        f"{name!r} is not an observable the catalogue declares, so whether it is "
        f"per atom is unknown. Declare it, or pass the model's catalogue."
    )


def _keep_real(value: Any, per_atom: bool, info: PaddingInfo) -> Any:
    if value is None:
        return None
    return value[: info.n_real_nodes] if per_atom else value[: info.n_real_graphs]
