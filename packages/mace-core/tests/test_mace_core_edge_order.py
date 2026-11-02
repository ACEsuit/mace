"""The finite edge-feature contract: optimal plans, immutable declarations, purity."""

import itertools
import subprocess
import sys
from typing import Any, cast

import pytest
from mace_core.edges import (
    EDGE_ADAPTERS,
    EDGE_FEATURES,
    EdgeAdapter,
    EdgeRequirementError,
    NeighborListCapabilities,
    plan_edges,
)
from mace_core.kernels import KERNEL_SPEC_VERSION, BackendCapabilities

FEATURE_SETS = [
    frozenset(combination)
    for size in range(4)
    for combination in itertools.combinations(sorted(EDGE_FEATURES), size)
]
INVALID_FEATURES = cast(Any, frozenset({"invented"}))


@pytest.mark.parametrize("required", FEATURE_SETS)
@pytest.mark.parametrize("guaranteed", FEATURE_SETS)
def test_every_plan_has_the_exhaustively_minimal_cost(required, guaranteed):
    possibilities = []
    for size in range(len(EDGE_ADAPTERS) + 1):
        for steps in itertools.permutations(EDGE_ADAPTERS, size):
            features = guaranteed
            for step in steps:
                if not step.requires <= features:
                    break
                features |= step.provides
            else:
                if required <= features:
                    possibilities.append(sum(step.cost for step in steps))
    plan = plan_edges(required, guaranteed)
    assert plan.cost == min(possibilities)
    features = guaranteed
    for step in plan.adapters:
        assert step.requires <= features
        features |= step.provides
    assert required <= features
    assert plan == plan_edges(
        required, guaranteed, adapters=tuple(reversed(EDGE_ADAPTERS))
    )


def test_planning_uses_cost_rather_than_fewest_steps():
    expensive = EdgeAdapter("direct", frozenset(), frozenset({"sender_runs"}), 10)
    assert (
        plan_edges({"sender_runs"}, set(), adapters=(*EDGE_ADAPTERS, expensive)).cost
        == 3
    )


def test_a_missing_completion_names_the_feature():
    with pytest.raises(EdgeRequirementError, match="sender_runs"):
        plan_edges({"sender_runs"}, set(), adapters=EDGE_ADAPTERS[:2])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: plan_edges(INVALID_FEATURES, set()),
        lambda: plan_edges(set(), INVALID_FEATURES),
        lambda: BackendCapabilities(edge_features=INVALID_FEATURES),
        lambda: NeighborListCapabilities("bad", features=INVALID_FEATURES),
    ],
)
def test_unknown_features_are_refused_at_the_boundary(factory):
    with pytest.raises(EdgeRequirementError, match="invented"):
        factory()


def test_old_backend_declarations_require_no_edge_features():
    assert BackendCapabilities().edge_features == frozenset()
    assert tuple(map(int, KERNEL_SPEC_VERSION.split("."))) >= (1, 1)


def test_the_edge_contract_imports_no_framework():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import mace_core.edges, sys; "
            "assert not {'torch', 'jax'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_optional_host_batch_search_is_used_and_preserves_order():
    import numpy as np
    from mace_core.neighbors import AllPairsBackend, get_neighborhood, get_neighborhoods

    class Batched(AllPairsBackend):
        calls = 0

        def search_batch(self, positions, cutoff, geometries):
            self.calls += 1
            return [
                self.search(p, cutoff, g)
                for p, g in zip(positions, geometries, strict=True)
            ]

    positions = [np.array([[0.0, 0.0, 0.0], [0.4, 0.2, 0.0]]), np.zeros((1, 3))]
    backend = Batched()
    actual = get_neighborhoods(
        positions, 1.0, [None, None], [None, None], backend=backend
    )
    assert backend.calls == 1
    for points, result in zip(positions, actual, strict=True):
        expected = get_neighborhood(points, 1.0, backend=AllPairsBackend())
        assert np.array_equal(result.edge_index, expected.edge_index)
        assert np.array_equal(result.unit_shifts, expected.unit_shifts)


def test_stateful_provider_cannot_advertise_batch_search():
    from mace_core.neighbors import (
        NeighborBackendNotAvailableError,
        neighbor_capabilities,
    )

    class Invalid:
        def capabilities(self):
            return NeighborListCapabilities("invalid", stateful=True)

        def search_batch(self, *args):
            raise AssertionError("must be refused at resolution")

    with pytest.raises(
        NeighborBackendNotAvailableError, match=r"invalid.*search_batch"
    ):
        neighbor_capabilities(Invalid(), "invalid")
