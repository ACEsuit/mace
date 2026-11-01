"""Padding a structure to a fixed size, and cutting the result back.

A compiled model with static shapes runs one graph size; a structure whose atom
or edge count changes from one step to the next would recompile it every time.
Padding appends a second structure of fake atoms so the batch always has the
budgeted size, and the fake structure contributes nothing to the real one.

**Three regimes, and the default pads nothing.** ``none`` is what an uncompiled
calculator does, as the frozen tree does unless asked. ``fixed`` takes its
budgets from the caller or from ``MACE_ASE_PAD_NUM_ATOMS`` and
``MACE_ASE_PAD_NUM_EDGES``. ``auto`` is what the compiled path selects: the
first structure sets the budgets, its own atom count and its edge count with
headroom, rounded up to a multiple.

**The fake structure is one graph on its own.** Its edges are self-loops
shifted by twice the cutoff, so every one of them is beyond the cutoff and its
message is zero, and its atoms have no edge to a real atom. The real
structure's energy, forces and stress are therefore the same with and without
it.

The calculator caches its real-node and edge budgets. The common graph
padding implementation owns all tensor operations and reserves one final fake
atom for its self-loops.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal, cast

from mace_core.graph import FixedPadding, NoPadding, PaddingInfo, PaddingOverflowError
from mace_core.observables import RequestedOutputs
from mace_core.outputs import (
    CORE_FIELD_NAMES,
    CORE_FIELD_ROWS,
    FIELD_BY_OBSERVABLE,
    MACEOutput,
)
from torch import Tensor

from mace_torch.physics.outputs import ENGINE_EXTRA_ROWS

__all__ = [
    "PADDING_ENVIRONMENT",
    "PaddingInfo",
    "PaddingOverflowError",
    "PaddingPolicy",
    "output_rows",
    "resolve_budget",
    "unpad_outputs",
]

#: The two variables a fixed budget can come from, atoms then edges.
PADDING_ENVIRONMENT = ("MACE_ASE_PAD_NUM_ATOMS", "MACE_ASE_PAD_NUM_EDGES")

Row = Literal["atom", "graph", "edge"]


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


@dataclass(frozen=True)
class PaddingPolicy:
    """How a structure is padded, and to what.

    Attributes:
        mode: ``none``, ``fixed`` or ``auto``.
        nodes_budget: Real atoms the batch has room for. Zero is unset.
        edges_budget: Edges the batch has room for, real and padding. Zero is
            unset.
        edge_multiple: What an estimated edge budget is rounded up to.
        headroom: How much larger than the structure that sets it an
            estimated edge budget is.
        on_overflow: ``grow`` to raise the budget past a structure that
            exceeds it, which changes the shape once; ``error`` to refuse it.
    """

    mode: Literal["none", "fixed", "auto"] = "none"
    nodes_budget: int = 0
    edges_budget: int = 0
    edge_multiple: int = 64
    headroom: float = 1.25
    on_overflow: Literal["grow", "error"] = "grow"

    def graph_policy(self) -> NoPadding | FixedPadding:
        """The shared graph policy for the currently cached budgets."""
        if self.mode == "none":
            return NoPadding()
        return FixedPadding(
            nodes_budget=self.nodes_budget + 1,
            edges_budget=self.edges_budget,
            graphs_budget=2,
            on_overflow=self.on_overflow,
        )

    @classmethod
    def requested(
        cls,
        nodes: int = 0,
        edges: int = 0,
        *,
        compiled: bool = False,
        environ: Mapping[str, str] | None = None,
    ) -> PaddingPolicy:
        """The policy a calculator's arguments ask for.

        A budget left at zero is read from its environment variable, and any
        budget at all selects ``fixed``. With none, a compiled model gets
        ``auto`` and an uncompiled one ``none``.
        """
        environ = os.environ if environ is None else environ
        if nodes <= 0:
            nodes = int(environ.get(PADDING_ENVIRONMENT[0], "0"))
        if edges <= 0:
            edges = int(environ.get(PADDING_ENVIRONMENT[1], "0"))
        nodes, edges = max(nodes, 0), max(edges, 0)
        if nodes or edges:
            return cls(mode="fixed", nodes_budget=nodes, edges_budget=edges)
        return cls(mode="auto" if compiled else "none")


def resolve_budget(
    policy: PaddingPolicy, nodes: int, edges: int
) -> tuple[PaddingPolicy, bool]:
    """The policy with budgets that fit this structure.

    Args:
        policy: The current policy.
        nodes: The structure's atom count.
        edges: Its edge count.

    Returns:
        The policy to pad with, and whether its budgets changed, which under
        static shapes is a recompile.

    Raises:
        PaddingOverflowError: If the structure exceeds a budget and the policy
            does not grow.
    """
    if policy.mode == "none":
        return policy, False
    updated = policy
    if policy.mode == "auto" and not policy.edges_budget:
        updated = replace(
            updated,
            nodes_budget=nodes,
            edges_budget=_round_up(int(edges * policy.headroom), policy.edge_multiple),
        )
    over = [
        name
        for name, real, budget in (
            ("atoms", nodes, updated.nodes_budget),
            ("edges", edges, updated.edges_budget),
        )
        if budget and real > budget
    ]
    if over and policy.on_overflow == "error":
        raise PaddingOverflowError(
            f"the structure has {nodes} atoms and {edges} edges, over the "
            f"budget of {updated.nodes_budget} atoms and {updated.edges_budget} "
            f"edges in {over}. Raise the budget, or let it grow."
        )
    if "atoms" in over:
        updated = replace(updated, nodes_budget=nodes)
    if "edges" in over:
        updated = replace(
            updated,
            edges_budget=_round_up(int(edges * policy.headroom), policy.edge_multiple),
        )
    if not updated.nodes_budget:
        updated = replace(updated, nodes_budget=nodes)
    if not updated.edges_budget:
        updated = replace(updated, edges_budget=edges)
    return updated, updated != policy


def output_rows(
    requested: RequestedOutputs, extra_rows: Mapping[str, str]
) -> dict[str, Row]:
    """What every quantity a model can return has a row for.

    Read off the declarations: the core fields from the output type, each
    declared observable and derivative from its spec, the extras the model
    declares as it adds them, and the derivative engine's.

    The derivatives of a quantity that is not a scalar, a dipole's ``dmu_dr``,
    have no row: their leading axis is the quantity's components, not atoms,
    and they are taken on a batch with no padding, as the Hessian is.
    """
    rows: dict[str, Row] = dict(CORE_FIELD_ROWS)
    for spec in requested.observables:
        rows.setdefault(
            FIELD_BY_OBSERVABLE.get(spec.name, spec.name),
            "atom" if spec.per_atom else "graph",
        )
        for request in spec.derivatives if spec.is_scalar else ():
            rows.setdefault(
                spec.derivative_name(request.wrt),
                "atom" if requested.per_atom_derivative(request.wrt) else "graph",
            )
    for name, row in {**extra_rows, **ENGINE_EXTRA_ROWS}.items():
        rows.setdefault(name, cast(Row, row))
    return rows


def unpad_outputs(
    output: MACEOutput[Tensor], info: PaddingInfo, rows: Mapping[str, Row]
) -> MACEOutput[Tensor]:
    """The real structure's part of every quantity, and nothing else.

    Raises:
        KeyError: For a quantity no declaration says the row of. Passing it
            through would hand the caller the fake structure's rows with it.
    """

    def cut(name: str, value: Tensor) -> Tensor:
        if name not in rows:
            raise KeyError(
                f"the model returned {name!r} and nothing declares whether it "
                f"has a row per atom, per structure or per edge, so its padding "
                f"cannot be cut off. Declare it where it is produced."
            )
        return value[
            : {
                "atom": info.n_real_nodes,
                "graph": info.n_real_graphs,
                "edge": info.n_real_edges,
            }[rows[name]]
        ]

    fields = {
        name: None
        if getattr(output, name) is None
        else cut(name, getattr(output, name))
        for name in CORE_FIELD_NAMES
    }
    extras = {name: cut(name, value) for name, value in output.extras.items()}
    return MACEOutput(**fields, extras=extras)
