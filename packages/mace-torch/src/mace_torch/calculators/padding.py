"""ASE padding options and output declarations; sizes and slicing live in core."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal, cast

from mace_core.graph import AutoPadding, FixedPadding, NoPadding, PaddingPolicy
from mace_core.observables import RequestedOutputs
from mace_core.outputs import CORE_FIELD_ROWS, FIELD_BY_OBSERVABLE

from mace_torch.physics.outputs import ENGINE_EXTRA_ROWS

PADDING_ENVIRONMENT = ("MACE_ASE_PAD_NUM_ATOMS", "MACE_ASE_PAD_NUM_EDGES")
Row = Literal["atom", "graph", "edge"]


def requested_padding(
    nodes: int = 0,
    edges: int = 0,
    *,
    compiled: bool = False,
    environ: Mapping[str, str] | None = None,
) -> PaddingPolicy:
    """Translate ASE arguments and environment budgets to the shared policy.

    ASE's atom option counts real atoms; core budgets include the padding atom.
    Zero reads the environment. Explicit ``padding=`` uses core units directly.
    """
    environ = os.environ if environ is None else environ
    if nodes == 0:
        nodes = int(environ.get(PADDING_ENVIRONMENT[0], "0"))
    if edges == 0:
        edges = int(environ.get(PADDING_ENVIRONMENT[1], "0"))
    if nodes < 0 or edges < 0:
        raise ValueError("ASE padding atom and edge budgets must be nonnegative")
    if nodes or edges:
        return FixedPadding(nodes_budget=nodes + 1, edges_budget=edges, graphs_budget=2)
    return AutoPadding() if compiled else NoPadding()


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
                "atom" if requested.per_atom_derivative(request.wrt, spec) else "graph",
            )
    for name, row in {**extra_rows, **ENGINE_EXTRA_ROWS}.items():
        rows.setdefault(name, cast(Row, row))
    return rows
