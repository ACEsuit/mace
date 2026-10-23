"""The neighbour list against the frozen legacy one, on the committed fixtures.

It must reproduce the legacy ``get_neighborhood`` exactly, the returned cell
included, and every registered backend must reproduce the independent
brute-force oracle in ``tests/neighbour_oracle.py``, which the package suites
cannot reach.
"""

import json

import numpy as np
import pytest
from mace_core.neighbors import (
    available_neighbor_backends,
    get_neighbor_backend,
    get_neighborhood,
)

from mace.data.neighborhood import get_neighborhood as legacy_get_neighborhood
from tests.golden import harness
from tests.neighbour_oracle import (
    assert_neighbourhoods_match,
    brute_force_neighborhood,
    canonical_edges,
)


def _fixtures():
    with open(harness.MANIFEST_PATH, encoding="utf-8") as handle:
        r_max = json.load(handle)["r_max_hint"]
    return harness.load_fixtures(), r_max


FIXTURES, R_MAX = _fixtures()


def _geometry(atoms):
    return (
        atoms.get_positions(),
        tuple(bool(p) for p in atoms.get_pbc()),
        np.array(atoms.get_cell()),
    )


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_neighbor_list_is_the_legacy_one_exactly(name):
    positions, pbc, cell = _geometry(FIXTURES[name])
    edge_index, shifts, unit_shifts, returned = legacy_get_neighborhood(
        positions, cutoff=R_MAX, pbc=pbc, cell=cell
    )
    neighbors = get_neighborhood(positions, R_MAX, pbc, cell)
    assert np.array_equal(neighbors.edge_index, edge_index)
    assert np.array_equal(neighbors.unit_shifts, unit_shifts)
    assert np.array_equal(neighbors.shifts, shifts)
    assert np.array_equal(neighbors.cell, returned)


@pytest.mark.parametrize(
    "backend", sorted(b.name for b in available_neighbor_backends())
)
@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_neighbor_backend_reproduces_the_oracle(name, backend):
    positions, pbc, cell = _geometry(FIXTURES[name])
    neighbors = get_neighborhood(
        positions, R_MAX, pbc, cell, backend=get_neighbor_backend(backend)
    )
    reference_index, _, reference_unit_shifts = brute_force_neighborhood(
        positions, R_MAX, pbc, cell
    )
    assert_neighbourhoods_match(
        canonical_edges(neighbors.edge_index, neighbors.unit_shifts),
        canonical_edges(reference_index, reference_unit_shifts),
        context=f"{name}, {backend}",
    )
