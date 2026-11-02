"""Edge-order requirements and their cheapest completion outside a forward.

Costs are relative adapter costs, not timings. There are only eight feature
sets, so planning explores that finite graph; no provider is selected here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

EdgeFeature = Literal["sorted_by_sender", "receiver_runs", "sender_runs"]
EDGE_FEATURES: frozenset[EdgeFeature] = frozenset(get_args(EdgeFeature))


class EdgeRequirementError(ValueError):
    """An edge feature is unknown or cannot be supplied by the adapters."""


def edge_features(features: Iterable[EdgeFeature]) -> frozenset[EdgeFeature]:
    """Validate the closed vocabulary and return an immutable feature set."""
    result = frozenset(features)
    unknown = result - EDGE_FEATURES
    if unknown:
        raise EdgeRequirementError(
            f"unknown edge features {sorted(unknown)}; use {sorted(EDGE_FEATURES)}."
        )
    return result


@dataclass(frozen=True)
class NeighborListCapabilities:
    """A provider's name, device kinds and guarantees before adaptation.

    ``stateful`` providers belong to one system and cannot implement
    ``search_batch``. Device names are kinds (``cpu``, ``cuda``), not ordinals.
    """

    name: str
    devices: frozenset[str] = frozenset({"cpu"})
    features: frozenset[EdgeFeature] = frozenset()
    stateful: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", edge_features(self.features))
        object.__setattr__(self, "devices", frozenset(self.devices))
        if not self.name.strip() or not self.devices:
            raise ValueError(
                "a neighbour provider needs a name and at least one device."
            )


@dataclass(frozen=True)
class EdgeAdapter:
    """One named, non-destructive completion of an edge feature set."""

    name: str
    requires: frozenset[EdgeFeature]
    provides: frozenset[EdgeFeature]
    cost: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "requires", edge_features(self.requires))
        object.__setattr__(self, "provides", edge_features(self.provides))
        if not self.name or not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError(
                "an edge adapter needs a name and a finite non-negative cost."
            )


EDGE_ADAPTERS = (
    EdgeAdapter("sort_by_sender", frozenset(), frozenset({"sorted_by_sender"}), 2),
    EdgeAdapter("receiver_runs", frozenset(), frozenset({"receiver_runs"}), 2),
    EdgeAdapter(
        "sender_runs", frozenset({"sorted_by_sender"}), frozenset({"sender_runs"}), 1
    ),
)


@dataclass(frozen=True)
class EdgePlan:
    """The ordered adapters and the guarantees at the two ends of a batch boundary."""

    required: frozenset[EdgeFeature]
    guaranteed: frozenset[EdgeFeature]
    adapters: tuple[EdgeAdapter, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "required", edge_features(self.required))
        object.__setattr__(self, "guaranteed", edge_features(self.guaranteed))
        object.__setattr__(self, "adapters", tuple(self.adapters))
        supplied = self.guaranteed
        for adapter in self.adapters:
            if not adapter.requires <= supplied:
                raise EdgeRequirementError(
                    f"adapter {adapter.name!r} needs "
                    f"{sorted(adapter.requires - supplied)}."
                )
            supplied |= adapter.provides
        if not self.required <= supplied:
            raise EdgeRequirementError(
                f"edge plan is missing {sorted(self.required - supplied)}."
            )

    @property
    def cost(self) -> float:
        return sum(adapter.cost for adapter in self.adapters)


def plan_edges(
    required: Iterable[EdgeFeature],
    guaranteed: Iterable[EdgeFeature],
    *,
    adapters: Sequence[EdgeAdapter] = EDGE_ADAPTERS,
) -> EdgePlan:
    """Find the cheapest completion, breaking equal-cost ties by adapter name.

    Raises ``EdgeRequirementError`` naming missing features if the available
    adapters cannot provide them. Inputs and adapter declarations are not mutated.
    """
    required, guaranteed = edge_features(required), edge_features(guaranteed)
    if len({adapter.name for adapter in adapters}) != len(adapters):
        raise ValueError("edge adapter names must be unique.")
    frontier: list[
        tuple[float, tuple[str, ...], frozenset[EdgeFeature], tuple[EdgeAdapter, ...]]
    ] = [(0.0, (), guaranteed, ())]
    visited: set[frozenset[EdgeFeature]] = set()
    reachable = set(guaranteed)
    while frontier:
        frontier.sort(key=lambda item: (item[0], item[1]))
        cost, names, features, steps = frontier.pop(0)
        if features in visited:
            continue
        visited.add(features)
        reachable.update(features)
        if required <= features:
            return EdgePlan(required, guaranteed, steps)
        for adapter in adapters:
            completed = features | adapter.provides
            if adapter.requires <= features and completed != features:
                frontier.append(
                    (
                        cost + adapter.cost,
                        (*names, adapter.name),
                        completed,
                        (*steps, adapter),
                    )
                )
    raise EdgeRequirementError(
        f"cannot provide edge features {sorted(required - reachable)}; "
        f"the provider guarantees {sorted(guaranteed)}. "
        "Choose a compatible provider or adapters."
    )
