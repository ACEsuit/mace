"""Padding a batch to fixed sizes: the policy, the budget and the padding structure.

**A padded batch is a valid batch, because padding is a structure.** Padding is
expressed inside the graph as physically inert structures, never as a mask the
forward has to read, so the model, its kernels and a compiler all stay unaware
of it. :func:`padding_sample` builds one: an aperiodic structure whose atoms
have Z = 0 and sit at the origin, whose cell is ``eye(3) * max(2 * r_max, 1)``,
whose edges are self loops on its last atom with unit shift ``(1, 0, 0)``, so
every padding edge is ``max(2 * r_max, 1)`` long and the polynomial cutoff is
exactly zero on it, and whose declared inputs carry their absent value
(``total_spin`` 1). Padding is then ``collate([*real, *padding])``:

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
step and identical on every DDP rank, and ``on_overflow: error``. ``auto``,
which grows its budget with one recompile per growth, is the calculator's: it
suits one system that changes little, and over a training stream it recompiles
again and again.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field, field_validator

from mace_core.config.base import ConfigSection
from mace_core.graph.info import PaddingInfo
from mace_core.graph.layout import GraphLayout
from mace_core.graph.schema import ANY_TRAILING, NODE_LEVEL

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
    """Size the budgets from the batch: one padding structure with one atom,
    and ``int(n_edges * headroom)`` edges rounded up to a multiple of
    ``edge_multiple``, which is the frozen calculator's rule.

    Attributes:
        edge_multiple: The edge budget is a multiple of this.
        headroom: The real edge count is scaled by this, then truncated,
            before rounding up.
        on_overflow: For a caller that keeps the first budget across calls,
            what to do when a later batch exceeds it. A single
            :func:`padding_budget` call sizes from its own batch and cannot
            overflow.
    """

    kind: Literal["auto"] = "auto"
    edge_multiple: int = Field(default=64, ge=1)
    headroom: float = Field(default=1.25, ge=1.0)
    on_overflow: Literal["grow", "error"] = "grow"

    @field_validator("headroom")
    @classmethod
    def _finite(cls, headroom: float) -> float:
        if not math.isfinite(headroom):
            raise ValueError("headroom must be finite.")
        return headroom


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
    r_max: float,
    layout: GraphLayout,
    *,
    feature_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """One padding structure, as a graph sample :func:`collate` batches.

    Args:
        n_atoms: Its atoms, at the origin with Z = 0.
        n_edges: Its edges, self loops on its last atom, or ``None`` for a
            sample without edge keys, as a batch a device search completes.
        r_max: The model's cutoff, Angstrom.
        layout: What the run batches; every declared input takes its absent
            value, every embedding feature zero.
        feature_shapes: The trailing sizes of each embedding feature, whose
            shape is the data's own, as the batch being padded carries them.

    Raises:
        ValueError: for edges without an atom to loop on, or an embedding
            feature whose trailing sizes are not given.
    """
    if n_edges and not n_atoms:
        raise ValueError(
            f"a padding structure with {n_edges} edges and no atom: a padding "
            f"edge is a self loop on one."
        )
    length = padding_length(r_max)
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
    for name, spec in layout.inputs.items():
        if spec.shape[-1] == ANY_TRAILING:
            if feature_shapes is None or name not in feature_shapes:
                raise ValueError(
                    f"feature {name!r} has trailing sizes that are the data's "
                    f"own; give them in feature_shapes, as the batch carries them."
                )
            trailing = tuple(feature_shapes[name])
        else:
            trailing = tuple(int(n) for n in spec.shape[1:])
        rows = (n_atoms,) if spec.shape[0] == NODE_LEVEL else ()
        sample[name] = np.full((*rows, *trailing), spec.absent_value)
    return sample


def padding_samples(
    budget: PaddingBudget,
    real: PaddingInfo,
    r_max: float,
    layout: GraphLayout,
    *,
    with_edges: bool = True,
    feature_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> list[dict[str, Any]]:
    """The padding structures that take a batch of ``real`` sizes to
    ``budget``: empty ones first, then the one holding every padding atom."""
    graphs = budget.graphs - real.n_real_graphs
    nodes = budget.nodes - real.n_real_nodes
    edges = budget.edges - real.n_real_edges if with_edges else None
    if graphs <= 0:
        return []
    empty = padding_sample(
        0, 0 if with_edges else None, r_max, layout, feature_shapes=feature_shapes
    )
    last = padding_sample(nodes, edges, r_max, layout, feature_shapes=feature_shapes)
    return [*(dict(empty) for _ in range(graphs - 1)), last]
