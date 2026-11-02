"""Complete edge-order requirements on device, once after collation/padding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from mace_core.edges import EdgePlan
from mace_core.graph.schema import DERIVED_KEYS, EDGE_LEVEL, GRAPH_SCHEMA, level_of


@dataclass(frozen=True)
class EdgeRuns:
    """A stable edge permutation and its node offsets, both int64 on device.

    ``order[offsets[node]:offsets[node + 1]]`` gives that node's edge indices.
    Zero-degree nodes have equal adjacent offsets. Construct outside a forward
    and pass the tensors to the consumer; no host values are read here.
    """

    order: torch.Tensor
    offsets: torch.Tensor

    @classmethod
    def from_graph(
        cls, graph: Mapping[str, Any], side: Literal["sender", "receiver"]
    ) -> EdgeRuns:
        if side not in ("sender", "receiver"):
            raise ValueError(f"edge runs side is {side!r}; use 'sender' or 'receiver'.")
        counts = graph[f"{side}_counts"]
        return cls(
            graph[f"{side}_order"], torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        )


def _runs(graph: dict[str, Any], side: str, *, ordered: bool = False) -> None:
    nodes = graph["edge_index"][0 if side == "sender" else 1]
    graph[f"{side}_order"] = (
        torch.arange(len(nodes), device=nodes.device)
        if ordered
        else torch.argsort(nodes, stable=True)
    )
    graph[f"{side}_counts"] = torch.bincount(
        nodes, minlength=graph["positions"].shape[0]
    )


def apply_edge_plan(graph: Mapping[str, Any], plan: EdgePlan) -> dict[str, Any]:
    """Return a new dict satisfying ``plan``; never modify input tensors.

    Sorting applies the same permutation to every schema edge field. Existing
    runs are rebuilt against the new edge indices, so they remain valid even
    if a later adapter sorts. Unknown extra keys are carried unchanged.
    """
    result = dict(graph)
    runs_to_build: set[str] = set()
    sender_sorted = "sorted_by_sender" in plan.guaranteed
    for adapter in plan.adapters:
        if adapter.name == "sort_by_sender":
            sender_sorted = True
            order = torch.argsort(result["edge_index"][0], stable=True)
            for key, spec in GRAPH_SCHEMA.items():
                if (
                    key in result
                    and key not in DERIVED_KEYS
                    and level_of(spec) == EDGE_LEVEL
                ):
                    result[key] = (
                        result[key][:, order]
                        if key == "edge_index"
                        else result[key][order]
                    )
            for side in ("sender", "receiver"):
                if f"{side}_order" in result or f"{side}_counts" in result:
                    runs_to_build.add(side)
        elif adapter.name in ("sender_runs", "receiver_runs"):
            runs_to_build.add(adapter.name.removesuffix("_runs"))
        else:
            raise ValueError(f"unknown torch edge adapter {adapter.name!r}.")
    for side in sorted(runs_to_build):
        _runs(result, side, ordered=side == "sender" and sender_sorted)
    return result
