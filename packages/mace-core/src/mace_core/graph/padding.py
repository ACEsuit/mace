"""Padding a batch to fixed sizes: the policy, the budget and the sidecar.

**A padded batch is a valid batch.** Padding is expressed inside the schema as
physically inert structures, atoms and edges, never as a mask the forward has
to read, so the model, its kernels and a compiler all stay unaware of it:

* real graphs, nodes and edges come first and keep their order and indices;
  padding only appends;
* every padding atom belongs to the last padding structure. Its element is the
  first of the element table, its position is the origin, it is not periodic,
  and its cell is ``eye(3) * max(2 * r_max, 1)``, with the ``rcell`` and
  ``volume`` derived from that cell as for any structure, so nothing that
  divides by a volume meets a zero;
* every padding edge is a self loop on the last padding atom with unit shift
  ``(1, 0, 0)`` and shift ``max(2 * r_max, 1)`` on x, so its length is at least
  ``2 * r_max`` and the polynomial cutoff is exactly zero on it;
* ``total_spin`` pads to 1, everything else to its schema pad value;
* ``batch`` and ``ptr`` cover the padded sizes, so ``num_nodes`` and
  ``num_graphs`` are the padded counts a compiler sees. A padding structure
  other than the last holds no atoms.

The real counts are kept beside the dict, in :class:`PaddingInfo`, never in it.

**The policy** is a config section, a union discriminated on ``kind``:
``none`` (training's default, dynamic shapes), ``fixed`` (explicit budgets) and
``auto`` (budgets sized from the batch, edges rounded up with headroom).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, field_validator

from mace_core.config.base import ConfigSection

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
]


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
        raise PaddingOverflowError(
            f"a batch of {n_nodes} nodes, {n_edges} edges and {n_graphs} "
            f"structures does not fit the budget of {asked.nodes} nodes, "
            f"{asked.edges} edges and {asked.graphs} structures; it needs "
            f"{grown.nodes}, {grown.edges} and {grown.graphs}, counting one "
            f"padding structure and one padding atom whenever anything is "
            f"padded. Raise the budget, or set on_overflow to 'grow'."
        )
    return grown
