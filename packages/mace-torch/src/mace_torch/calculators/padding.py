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

**Its edges are spread over its atoms.** Stacked on one fake atom, as the frozen
tree places them, they hand the message scatter one segment thousands of times
larger than any real atom's, and on an MI300A with a quarter of the edges
padded that alone cost a factor of 2.7 to 3.2. One fake atom per
:data:`EDGES_PER_PADDING_ATOM` padding edges removes the imbalance, and more
atoms than that buy nothing.

**How many fake atoms there are is fixed with the budget.** Counted from each
call's shortfall, as the frozen tree counts it, the number of nodes moves
whenever the edge count crosses a multiple of the spread, and under static
shapes every new node count is a recompile. Here the budget fixes the shape of
the whole batch, nodes included, so the shape changes only when the budget
grows.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal, cast

import numpy as np
from mace_core.graph import GRAPH_INPUT_DEFAULTS, NODE_INPUT_DEFAULTS
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
    "EDGES_PER_PADDING_ATOM",
    "PADDING_ENVIRONMENT",
    "PaddingInfo",
    "PaddingOverflowError",
    "PaddingPolicy",
    "output_rows",
    "pad_batch",
    "resolve_budget",
    "unpad_outputs",
]

#: Padding edges per fake atom. Above roughly this the message scatter pays for
#: the imbalance; below it the cost is flat, so a smaller value only adds nodes.
EDGES_PER_PADDING_ATOM = 1024

#: The two variables a fixed budget can come from, atoms then edges.
PADDING_ENVIRONMENT = ("MACE_ASE_PAD_NUM_ATOMS", "MACE_ASE_PAD_NUM_EDGES")

Row = Literal["atom", "graph", "edge"]


class PaddingOverflowError(RuntimeError):
    """A structure larger than the budget, under a policy that does not grow."""


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
        padding_atoms: The fake atoms, fixed with the budget. Zero until a
            structure has been seen.
    """

    mode: Literal["none", "fixed", "auto"] = "none"
    nodes_budget: int = 0
    edges_budget: int = 0
    edge_multiple: int = 64
    headroom: float = 1.25
    on_overflow: Literal["grow", "error"] = "grow"
    padding_atoms: int = 0

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


@dataclass(frozen=True)
class PaddingInfo:
    """Where the real structure ends in a padded batch.

    Attributes:
        nodes: Its atoms, which come first.
        edges: Its edges, which come first.
        graphs: How many structures are real, which is the first ones.
    """

    nodes: int
    edges: int
    graphs: int = 1


def _padding_atoms(edges_budget: int, real_edges: int) -> int:
    return max(1, math.ceil(max(edges_budget - real_edges, 0) / EDGES_PER_PADDING_ATOM))


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
    if updated.padding_atoms == 0 or updated.edges_budget != policy.edges_budget:
        updated = replace(
            updated, padding_atoms=_padding_atoms(updated.edges_budget, edges)
        )
    return updated, updated != policy


def pad_batch(
    structure: Mapping[str, np.ndarray], policy: PaddingPolicy, r_max: float
) -> tuple[list[dict[str, np.ndarray]], PaddingInfo]:
    """The structure, and the fake one that pads it to the budget.

    Args:
        structure: One structure's graph, as the graph builder returns it.
        policy: A policy whose budgets were resolved for this structure.
        r_max: The cutoff, in Angstrom.

    Returns:
        The structures to collate, real first, and where the real one ends.
    """
    real_nodes = int(np.asarray(structure["atomic_numbers"]).shape[0])
    real_edges = int(np.asarray(structure["edge_index"]).shape[1])
    info = PaddingInfo(nodes=real_nodes, edges=real_edges)
    if policy.mode == "none":
        return [dict(structure)], info
    nodes = policy.nodes_budget + policy.padding_atoms - real_nodes
    edges = max(policy.edges_budget - real_edges, 0)
    length = max(2.0 * float(r_max), 1.0)
    loops = np.arange(edges, dtype=np.int64) % nodes
    unit = np.zeros((edges, 3))
    unit[:, 0] = 1.0
    number = int(np.asarray(structure["atomic_numbers"])[0])
    fake = {
        "positions": np.zeros((nodes, 3)),
        "atomic_numbers": np.full(nodes, number, dtype=np.int64),
        "edge_index": np.stack([loops, loops]),
        "shifts": unit * length,
        "unit_shifts": unit,
        "cell": np.eye(3) * length,
        "pbc": np.zeros(3, dtype=bool),
        "weight": np.asarray(0.0),
        "head": np.asarray(structure["head"]),
        # Whatever inputs the real one carries, per structure and per atom, at
        # their defaults, since a batch holds a field for every graph or for
        # none.
        **{
            name: np.asarray(default if len(default) > 1 else default[0])
            for name, default in GRAPH_INPUT_DEFAULTS.items()
            if name in structure
        },
        **{
            name: np.full(nodes, default)
            for name, default in NODE_INPUT_DEFAULTS.items()
            if name in structure
        },
    }
    return [dict(structure), fake], info


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
                "atom" if request.wrt == "pos" else "graph",
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
            : {"atom": info.nodes, "graph": info.graphs, "edge": info.edges}[rows[name]]
        ]

    fields = {
        name: None
        if getattr(output, name) is None
        else cut(name, getattr(output, name))
        for name in CORE_FIELD_NAMES
    }
    extras = {name: cut(name, value) for name, value in output.extras.items()}
    return MACEOutput(**fields, extras=extras)
