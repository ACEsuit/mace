"""Padding a batch to fixed sizes: the policy, the budget and the padding structure.

**A padded batch is a valid batch, because padding is a structure.** Padding is
expressed inside the graph as physically inert structures, never as a mask the
forward has to read, so the model, its kernels and a compiler all stay unaware
of it. :func:`padding_sample` builds one: an aperiodic structure whose atoms
have Z = 0 and sit at the origin, whose cell is ``eye(3) * max(2 * cutoff, 1)``
with the layout's cutoff, whose edges are self loops on its last atom with unit
shift ``(1, 0, 0)``, so every padding edge is ``max(2 * cutoff, 1)`` long and
the polynomial cutoff is exactly zero on it, and whose declared inputs carry
their absent value (``total_spin`` 1). Padding is then ``collate([*real, *padding])``:

* real graphs, nodes and edges come first and keep their order and indices;
  padding only appends;
* ``batch`` and ``ptr`` cover the padded sizes by construction, so
  ``num_nodes`` and ``num_graphs`` are the padded counts a compiler sees;
* when more than one padding structure is needed, the extra ones are empty
  and the last holds every padding atom.

So under :class:`FixedPadding` the padding structure holds every padding atom,
all at the origin. A consumer that works per structure reads the
:class:`~mace_core.graph.info.PaddingInfo` in the batch's ``GraphInfo`` and
skips padding structures; whatever rows it must write for one are self-pairs on
one padding atom. Classifying it from ``pbc`` alone would see a large open
structure with every pair at ``r = 0``.

**The policy** is a config section, a union discriminated on ``kind``, and it
follows the regime. ``none`` is eager training's, with dynamic shapes.
``fixed`` is compiled training's, with budgets computed once before the first
step and identical on every DDP rank, and ``on_overflow: error``. ``auto``
sizes each budget from its own batch, by the frozen calculator's rule; keeping
one budget across calls and growing it, with one recompile per growth, is the
calculator's to do, since it suits one system that changes little and over a
training stream it would recompile again and again.

:func:`unpad_outputs` keeps the leading real slots of a model's outputs. It
only slices, so it serves every framework.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, overload

import numpy as np
from pydantic import Field

from mace_core.config.base import ConfigSection
from mace_core.graph.info import GraphInfo, PaddingInfo
from mace_core.graph.layout import GraphLayout
from mace_core.observables import DEFAULT_CATALOGUE, ObservableCatalogue
from mace_core.outputs import CORE_FIELD_NAMES, MACEOutput

__all__ = [
    "AutoPadding",
    "FixedPadding",
    "NoPadding",
    "PaddingBudget",
    "PaddingInfo",
    "PaddingOverflowError",
    "PaddingPolicy",
    "padding_budget",
    "padding_length",
    "padding_sample",
    "padding_samples",
    "unpad_outputs",
]


@dataclass(frozen=True)
class PaddingBudget:
    """The padded sizes of a batch: nodes, edges and structures."""

    nodes: int
    edges: int
    graphs: int


class PaddingOverflowError(ValueError):
    """A batch does not fit its fixed budget, and the policy says not to grow."""


class NoPadding(ConfigSection):
    """Leave the batch as it is: dynamic shapes, eager training."""

    kind: Literal["none"] = "none"


class FixedPadding(ConfigSection):
    """Pad every batch to the same three sizes.

    Attributes:
        nodes_budget: Padded node count.
        edges_budget: Padded edge count.
        graphs_budget: Padded structure count.
        on_overflow: ``grow`` raises a budget a batch does not fit to the
            smallest that fits, which costs one recompile; ``error`` raises
            :class:`PaddingOverflowError`.
    """

    kind: Literal["fixed"] = "fixed"
    nodes_budget: int = Field(ge=1)
    edges_budget: int = Field(ge=0)
    graphs_budget: int = Field(ge=1)
    on_overflow: Literal["grow", "error"] = "grow"


class AutoPadding(ConfigSection):
    """Size the budgets from the batch as the frozen calculator does: the edge
    budget is ``int(n_edges * headroom)`` rounded up to a multiple of
    ``edge_multiple``, and when that leaves padding edges they go on one
    padding structure with one atom. A batch without edges is not padded.

    Attributes:
        edge_multiple: The edge budget is a multiple of this.
        headroom: The real edge count is scaled by this, then truncated,
            before rounding up.
    """

    kind: Literal["auto"] = "auto"
    edge_multiple: int = Field(default=64, ge=1)
    headroom: float = Field(default=1.25, ge=1.0, allow_inf_nan=False)


PaddingPolicy = Annotated[
    NoPadding | FixedPadding | AutoPadding, Field(discriminator="kind")
]


def padding_length(r_max: float) -> float:
    """``max(2 * r_max, 1)``: the length of a padding edge and the side of the
    padding cell, in Angstrom. Longer than the cutoff, so the cutoff
    envelope is exactly zero on a padding edge."""
    return max(2.0 * float(r_max), 1.0)


def padding_budget(
    policy: NoPadding | FixedPadding | AutoPadding,
    *,
    n_nodes: int,
    n_edges: int,
    n_graphs: int,
) -> PaddingBudget | None:
    """The padded sizes for a batch of these real sizes, or ``None`` for none.

    A budget always fits: it is at least the real sizes, it has a padding
    structure whenever it has a padding atom or edge, and a padding atom
    whenever it has a padding edge, since a padding edge is a self loop on one.

    A batch without edges is held to the fixed edge budget all the same. A
    device search joins its real edges and then its padding edges after the
    batch is padded, and it can add neither an atom nor a structure, so the
    padding atom and structure those edges loop on are reserved here.

    Raises:
        PaddingOverflowError: for a fixed policy with ``on_overflow="error"``
            that the batch does not fit. The message names each size.
    """
    if isinstance(policy, NoPadding):
        return None
    if isinstance(policy, AutoPadding):
        wanted = int(n_edges * policy.headroom)
        multiple = policy.edge_multiple
        edges = (wanted + multiple - 1) // multiple * multiple
        if edges == n_edges:
            return PaddingBudget(nodes=n_nodes, edges=n_edges, graphs=n_graphs)
        return PaddingBudget(nodes=n_nodes + 1, edges=edges, graphs=n_graphs + 1)

    nodes = max(policy.nodes_budget, n_nodes)
    edges = max(policy.edges_budget, n_edges)
    graphs = max(policy.graphs_budget, n_graphs)
    if (nodes > n_nodes or edges > n_edges) and graphs == n_graphs:
        graphs += 1
    if edges > n_edges and nodes == n_nodes:
        nodes += 1
    grown = PaddingBudget(nodes=nodes, edges=edges, graphs=graphs)
    asked = PaddingBudget(
        nodes=policy.nodes_budget,
        edges=policy.edges_budget,
        graphs=policy.graphs_budget,
    )
    if grown != asked and policy.on_overflow == "error":
        over = [
            f"{name} ({needed} needed, budget {budget})"
            for name, needed, budget in (
                ("nodes", grown.nodes, asked.nodes),
                ("edges", grown.edges, asked.edges),
                ("structures", grown.graphs, asked.graphs),
            )
            if needed > budget
        ]
        raise PaddingOverflowError(
            f"a batch of {n_nodes} nodes, {n_edges} edges and {n_graphs} "
            f"structures is over its fixed budget in {', '.join(over)}, counting "
            f"one padding structure and one padding atom whenever anything is "
            f"padded. Raise the budget, or set on_overflow to 'grow'."
        )
    return grown


def padding_sample(
    n_atoms: int,
    n_edges: int | None,
    layout: GraphLayout,
) -> dict[str, Any]:
    """One padding structure, as a graph sample :func:`collate` batches.

    Args:
        n_atoms: Its atoms, at the origin with Z = 0.
        n_edges: Its edges, self loops on its last atom, or ``None`` for a
            sample without edge keys, as a batch a device search completes.
        layout: What the run batches; every declared input takes its absent
            value, every embedding feature zero.

    Raises:
        ValueError: for edges without an atom to loop on.
    """
    if n_atoms < 0 or (n_edges is not None and n_edges < 0):
        raise ValueError(
            f"a padding structure of {n_atoms} atoms and {n_edges} edges; both "
            f"are counts."
        )
    if n_edges and not n_atoms:
        raise ValueError(
            f"a padding structure with {n_edges} edges and no atom: a padding "
            f"edge is a self loop on one."
        )
    length = padding_length(layout.cutoff)
    sample: dict[str, Any] = {
        "positions": np.zeros((n_atoms, 3)),
        "node_atomic_numbers": np.zeros(n_atoms, dtype=np.int64),
        "cell": np.eye(3) * length,
        "pbc": np.zeros(3, dtype=bool),
        "head": np.int64(0),
    }
    if n_edges is not None:
        unit_shifts = np.zeros((n_edges, 3))
        unit_shifts[:, 0] = 1.0
        sample["edge_index"] = np.full((2, n_edges), n_atoms - 1, dtype=np.int64)
        sample["unit_shifts"] = unit_shifts
        sample["shifts"] = unit_shifts * length
    for name, spec in {**layout.inputs, **layout.features}.items():
        sample[name] = np.full(spec.sample_shape(n_atoms), spec.absent_value)
    return sample


def padding_samples(
    budget: PaddingBudget,
    real: PaddingInfo,
    layout: GraphLayout,
    *,
    with_edges: bool = True,
) -> list[dict[str, Any]]:
    """The padding structures that take a batch of ``real`` sizes to
    ``budget``: empty ones first, then the one holding every padding atom."""
    graphs = budget.graphs - real.n_real_graphs
    nodes = budget.nodes - real.n_real_nodes
    edges = budget.edges - real.n_real_edges if with_edges else None
    if graphs < 0 or nodes < 0 or (edges is not None and edges < 0):
        raise ValueError(
            f"the budget {budget} is below the real sizes {real}; a budget is "
            f"at least the batch it pads."
        )
    if graphs == 0:
        if nodes or edges:
            raise ValueError(
                f"the budget {budget} pads atoms or edges but no structure to "
                f"hold them; a padding atom needs a padding structure."
            )
        return []
    empty = padding_sample(0, 0 if with_edges else None, layout)
    last = padding_sample(nodes, edges, layout)
    return [*(dict(empty) for _ in range(graphs - 1)), last]


#: Whether each core field of a model output is per atom.
_CORE_FIELD_PER_ATOM = {
    "total_energy": False,
    "node_energies": True,
    "forces": True,
    "stress": False,
    "virials": False,
    "dipole": False,
}


@overload
def unpad_outputs(
    outputs: MACEOutput,
    info: GraphInfo,
    *,
    catalogue: ObservableCatalogue = ...,
) -> MACEOutput: ...


@overload
def unpad_outputs(
    outputs: Mapping[str, Any],
    info: GraphInfo,
    *,
    catalogue: ObservableCatalogue = ...,
) -> dict[str, Any]: ...


def unpad_outputs(
    outputs: MACEOutput | Mapping[str, Any],
    info: GraphInfo,
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
            observable name to array, of any framework. ``None`` values are
            kept.
        info: What ``pad_batch`` returned. An unpadded batch's info keeps
            every slot.
        catalogue: The declared observables of the model.

    Returns:
        A new object of the same kind.

    Raises:
        KeyError: for a value the catalogue does not declare.
    """
    real = info.padding or PaddingInfo(
        n_real_nodes=info.num_nodes,
        n_real_edges=info.num_edges,
        n_real_graphs=info.num_graphs,
    )
    if isinstance(outputs, MACEOutput):
        unpadded: MACEOutput = MACEOutput()
        for name in CORE_FIELD_NAMES:
            value = getattr(outputs, name)
            setattr(unpadded, name, _keep_real(value, _CORE_FIELD_PER_ATOM[name], real))
        unpadded.extras = {
            name: _keep_real(value, _per_atom(name, catalogue), real)
            for name, value in outputs.extras.items()
        }
        return unpadded
    return {
        name: _keep_real(value, _per_atom(name, catalogue), real)
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


def _keep_real(value: Any, per_atom: bool, real: PaddingInfo) -> Any:
    if value is None:
        return None
    return value[: real.n_real_nodes] if per_atom else value[: real.n_real_graphs]
