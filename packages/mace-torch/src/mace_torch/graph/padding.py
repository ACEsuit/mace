"""Padding a batched torch graph to a budget, and stripping it from outputs.

The contract is :mod:`mace_core.graph.padding`'s: padding is a structure. The
padding structures of :func:`~mace_core.graph.padding_samples` are collated
and appended after the real ones, exactly as ``collate([*real, *padding])``
would batch them, and the real sizes travel in the returned
:class:`~mace_core.graph.GraphInfo`. Nothing is reordered, so
:func:`unpad_outputs` keeps the leading real slots of each output and that is
the whole inverse.

:func:`pad_batch` checks its output with
:func:`~mace_core.graph.check_layout`, which reads no value, so a batch already
on the device is not copied back.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, overload

import torch
from mace_core.graph import (
    ANY_TRAILING,
    AutoPadding,
    FixedPadding,
    GraphInfo,
    GraphLayout,
    NoPadding,
    PaddingBudget,
    PaddingInfo,
    check_layout,
    padding_budget,
    padding_samples,
)
from mace_core.graph.schema import EDGE_LEVEL, GRAPH_LEVEL, NODE_LEVEL, level_of
from mace_core.observables import DEFAULT_CATALOGUE, ObservableCatalogue
from mace_core.outputs import CORE_FIELD_NAMES, MACEOutput

from mace_torch.graph.collate import collate, dtype_name

__all__ = ["pad_batch", "pad_targets", "unpad_outputs"]

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
    graph: Mapping[str, torch.Tensor],
    info: GraphInfo,
    policy: NoPadding | FixedPadding | AutoPadding,
    r_max: float,
    layout: GraphLayout,
) -> tuple[dict[str, torch.Tensor], GraphInfo]:
    """``graph`` padded to the budget ``policy`` gives, and its new info.

    Args:
        graph: A collated, unpadded graph dict. Not modified.
        info: Its :class:`~mace_core.graph.GraphInfo`.
        policy: The padding policy.
        r_max: The model's cutoff, Angstrom; sets the padding edge length and
            the padding cell.
        layout: What the run batches.

    Returns:
        The padded graph, on the device and in the dtype of ``graph``, and a
        ``GraphInfo`` whose ``padding`` holds the real sizes.

    Raises:
        PaddingOverflowError: for a fixed budget the batch does not fit, when
            the policy says ``error``.
        ValueError: for a batch that is padded already, or that carries the
            field of a batch requirement, which is built after padding.
    """
    if info.padding is not None:
        raise ValueError("the batch is padded already; pad a batch once.")
    built = sorted(set(graph) & set(layout.requirement_fields()))
    if built:
        raise ValueError(
            f"the batch carries the batch-requirement fields {built}. Pad first, "
            f"then build the requirements on the padded batch."
        )
    real = PaddingInfo(
        n_real_nodes=info.num_nodes,
        n_real_edges=info.num_edges,
        n_real_graphs=info.num_graphs,
    )
    budget = padding_budget(
        policy, n_nodes=info.num_nodes, n_edges=info.num_edges, n_graphs=info.num_graphs
    )
    if (
        isinstance(policy, FixedPadding)
        and budget is not None
        and budget
        != PaddingBudget(policy.nodes_budget, policy.edges_budget, policy.graphs_budget)
    ):
        logger.warning(
            "the batch of %d nodes, %d edges and %d structures exceeds the "
            "padding budget; batching it at %d, %d and %d instead, which "
            "recompiles a compiled model once",
            info.num_nodes,
            info.num_edges,
            info.num_graphs,
            budget.nodes,
            budget.edges,
            budget.graphs,
        )
    unpadded = PaddingBudget(info.num_nodes, info.num_edges, info.num_graphs)
    if budget is None or budget == unpadded:
        return dict(graph), _with_padding(info, real)

    positions = graph["positions"]
    dtype, device = positions.dtype, positions.device
    samples = padding_samples(
        budget,
        real,
        r_max,
        layout,
        with_edges="edge_index" in graph,
        feature_shapes=_feature_shapes(graph, layout),
    )
    padding, padding_info = collate(samples, layout, dtype=dtype, device=device)
    padded = _append(graph, padding, info, layout)
    padded_info = GraphInfo(
        num_graphs=budget.graphs,
        num_nodes=budget.nodes,
        num_edges=budget.edges if "edge_index" in graph else 0,
        structure_sizes=info.structure_sizes + padding_info.structure_sizes,
        padding=real,
    )
    check_layout(padded, layout, padded_info, float_dtype=dtype_name(dtype))
    return padded, padded_info


def _with_padding(info: GraphInfo, real: PaddingInfo) -> GraphInfo:
    return GraphInfo(
        num_graphs=info.num_graphs,
        num_nodes=info.num_nodes,
        num_edges=info.num_edges,
        structure_sizes=info.structure_sizes,
        padding=real,
    )


def _feature_shapes(
    graph: Mapping[str, torch.Tensor], layout: GraphLayout
) -> dict[str, tuple[int, ...]]:
    """The trailing sizes of each embedding feature, as the batch carries them."""
    shapes: dict[str, tuple[int, ...]] = {}
    for name, spec in layout.inputs.items():
        if spec.shape[-1] == ANY_TRAILING:
            shapes[name] = tuple(int(n) for n in graph[name].shape[1:])
    return shapes


def _append(
    graph: Mapping[str, torch.Tensor],
    padding: Mapping[str, torch.Tensor],
    info: GraphInfo,
    layout: GraphLayout,
) -> dict[str, torch.Tensor]:
    """``padding`` batched after ``graph``, as one collation of both would."""
    fields = layout.graph_fields()
    graphs = len(padding["ptr"]) - 1
    joined: dict[str, torch.Tensor] = {}
    for key, value in graph.items():
        if key not in fields:
            raise ValueError(
                f"{key!r} is not declared by the layout, so pad_batch cannot tell "
                f"which dimension to pad. Declare it, or leave it out of the batch."
            )
        if key in padding:
            extra = padding[key]
        elif level_of(fields[key]) == GRAPH_LEVEL:
            # A per-structure key the caller adds, such as ``displacement``:
            # padding structures carry zero.
            extra = value.new_zeros((graphs, *value.shape[1:]))
        else:
            raise ValueError(f"{key!r} has no level padding extends.")
        if key == "edge_index":
            joined[key] = torch.cat((value, extra + info.num_nodes), dim=1)
        elif key == "batch":
            joined[key] = torch.cat((value, extra + info.num_graphs))
        elif key == "ptr":
            joined[key] = torch.cat((value, extra[1:] + info.num_nodes))
        elif key in fields and level_of(fields[key]) in (
            NODE_LEVEL,
            EDGE_LEVEL,
            GRAPH_LEVEL,
        ):
            joined[key] = torch.cat((value, extra.to(value.dtype)))
        else:
            raise ValueError(f"{key!r} has no level padding extends.")
    return joined


def pad_targets(
    targets: Mapping[str, torch.Tensor], info: GraphInfo, layout: GraphLayout
) -> dict[str, torch.Tensor]:
    """Targets extended to a padded batch: zero labels, false presence flags and
    zero weights on every padding slot, so a loss over the padded batch counts
    nothing there.

    Raises:
        ValueError: if ``info`` is not a padded batch's.
    """
    if info.padding is None:
        raise ValueError("pad_targets needs the GraphInfo pad_batch returned.")
    extra = {
        NODE_LEVEL: info.num_nodes - info.padding.n_real_nodes,
        GRAPH_LEVEL: info.num_graphs - info.padding.n_real_graphs,
    }
    fields = layout.target_fields()
    padded: dict[str, torch.Tensor] = {}
    for key, value in targets.items():
        rows = extra[level_of(fields[key]) or GRAPH_LEVEL]
        tail = torch.zeros(
            (rows, *value.shape[1:]), dtype=value.dtype, device=value.device
        )
        padded[key] = torch.cat((value, tail))
    return padded


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
            observable name to tensor. ``None`` values are kept.
        info: What :func:`pad_batch` returned. An unpadded batch's info keeps
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
