"""Padding a batched torch graph to a budget.

The contract is :mod:`mace_core.graph.padding`'s: padding is a structure. The
padding structures of :func:`~mace_core.graph.padding_samples` are collated
and appended after the real ones, exactly as ``collate([*real, *padding])``
would batch them, and the real sizes travel in the returned
:class:`~mace_core.graph.GraphInfo`. Nothing is reordered, so
:func:`~mace_core.graph.unpad_outputs` keeps the leading real slots of each
output and that is the whole inverse.

:func:`pad_batch` checks its output with
:func:`~mace_core.graph.check_layout`, which reads no value, so a batch already
on the device is not copied back. It checks whether or not anything was
padded, so it is the batch's one check: collate the real structures with
``check=False`` when they are padded next, and the padding structures are
collated unchecked here.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import cast

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
from mace_core.graph.schema import GRAPH_LEVEL, NODE_LEVEL

from mace_torch.graph.collate import collate

__all__ = ["pad_batch", "pad_targets"]

logger = logging.getLogger(__name__)


def pad_batch(
    graph: Mapping[str, torch.Tensor],
    info: GraphInfo,
    policy: NoPadding | FixedPadding | AutoPadding,
    layout: GraphLayout,
) -> tuple[dict[str, torch.Tensor], GraphInfo]:
    """``graph`` padded to the budget ``policy`` gives, and its new info.

    Args:
        graph: A collated, unpadded graph dict. Not modified.
        info: Its :class:`~mace_core.graph.GraphInfo`.
        policy: The padding policy.
        layout: What the run batches. Its cutoff sets the padding edge
            length and the padding cell.

    Returns:
        The padded graph, on the device and in the dtype of ``graph``, and a
        ``GraphInfo`` whose ``padding`` holds the real sizes.

    Raises:
        PaddingOverflowError: for a fixed budget the batch does not fit, when
            the policy says ``error``.
        ValueError: for a batch that is padded already, or that carries the
            field of a batch requirement, which is built after padding.
        GraphValidationError: for a result that breaks its layout.
    """
    if info.padding is not None:
        raise ValueError("the batch is padded already; pad a batch once.")
    built = sorted(set(graph) & set(layout.requirement_fields))
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
        result, result_info = dict(graph), dataclasses.replace(info, padding=real)
        check_layout(result, layout, result_info)
        return result, result_info

    device = graph["positions"].device
    samples = padding_samples(
        budget,
        real,
        layout,
        with_edges="edge_index" in graph,
        feature_shapes=_feature_shapes(graph, layout),
    )
    padding, padding_info = collate(samples, layout, device=device, check=False)
    padded = _append(graph, padding, info, layout)
    padded_info = GraphInfo(
        num_graphs=budget.graphs,
        num_nodes=budget.nodes,
        num_edges=budget.edges if "edge_index" in graph else 0,
        structure_sizes=info.structure_sizes + padding_info.structure_sizes,
        padding=real,
    )
    check_layout(padded, layout, padded_info)
    return padded, padded_info


def _feature_shapes(
    graph: Mapping[str, torch.Tensor], layout: GraphLayout
) -> dict[str, tuple[int, ...]]:
    """The trailing sizes of each embedding feature, as the batch carries them."""
    shapes: dict[str, tuple[int, ...]] = {}
    for name, spec in layout.inputs.items():
        if spec.shape[-1] != ANY_TRAILING:
            continue
        if name not in graph:
            raise ValueError(
                f"the batch is missing the embedding feature {name!r}, which the "
                f"layout declares."
            )
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
        # A padding structure carries every declared field except the
        # caller's ``displacement``, which is per structure and zero there.
        extra = (
            padding[key]
            if key in padding
            else value.new_zeros((graphs, *value.shape[1:]))
        )
        if key == "edge_index":
            joined[key] = torch.cat((value, extra + info.num_nodes), dim=1)
        elif key == "batch":
            joined[key] = torch.cat((value, extra + info.num_graphs))
        elif key == "ptr":
            joined[key] = torch.cat((value, extra[1:] + info.num_nodes))
        else:
            joined[key] = torch.cat((value, extra))
    return joined


def pad_targets(
    targets: Mapping[str, torch.Tensor], info: GraphInfo, layout: GraphLayout
) -> dict[str, torch.Tensor]:
    """Targets extended to a padded batch: zero labels, false presence flags and
    zero weights on every padding slot, so a loss over the padded batch counts
    nothing there.

    Raises:
        ValueError: if ``info`` is not a padded batch's, a key is not a declared
            target, or a value does not have the real batch's rows.
    """
    if info.padding is None:
        raise ValueError("pad_targets needs the GraphInfo pad_batch returned.")
    real = {
        NODE_LEVEL: info.padding.n_real_nodes,
        GRAPH_LEVEL: info.padding.n_real_graphs,
    }
    total = {NODE_LEVEL: info.num_nodes, GRAPH_LEVEL: info.num_graphs}
    fields = layout.target_fields()
    padded: dict[str, torch.Tensor] = {}
    for key, value in targets.items():
        if key not in fields:
            raise ValueError(
                f"{key!r} is not a target the layout declares, so pad_targets "
                f"cannot tell which dimension to pad."
            )
        level = cast(str, fields[key].level())
        if value.shape[0] != real[level]:
            raise ValueError(
                f"target {key!r} has {value.shape[0]} rows; the real batch has "
                f"{real[level]} along {level}. Pad the targets of the real batch, "
                f"once."
            )
        rows = total[level] - real[level]
        tail = torch.zeros(
            (rows, *value.shape[1:]), dtype=value.dtype, device=value.device
        )
        padded[key] = torch.cat((value, tail))
    return padded
