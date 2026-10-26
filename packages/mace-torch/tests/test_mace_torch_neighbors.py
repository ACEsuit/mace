"""mace-torch registers vesin and builds graphs with it unless told otherwise."""

import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from mace_core.neighbors import (
    BruteForceBackend,
    NeighborBackendNotAvailableError,
    NeighborListConfig,
    available_neighbor_backends,
    get_neighbor_backend,
    get_neighborhood,
)
from mace_core.neighbors.matscipy_backend import MatscipyBackend
from mace_torch.neighbors import DEFAULT_NEIGHBOR_BACKEND, resolve_neighbor_backend
from mace_torch.neighbors.vesin_backend import VesinBackend

TRICLINIC = np.array([[3.1, 0.2, 0.1], [0.9, 2.8, -0.2], [0.4, -0.3, 3.3]])
SLAB = np.array([[3.0, 0.0, 0.0], [1.5, 2.598, 0.0], [0.0, 0.0, 12.0]])


def _cluster(n_atoms, spread, seed):
    return np.random.default_rng(seed).uniform(0.0, spread, size=(n_atoms, 3))


CASES = {
    "triclinic": (_cluster(5, 2.5, 1), 3.2, (True, True, True), TRICLINIC),
    "cell_smaller_than_cutoff": (
        np.zeros((1, 3)),
        3.0,
        (True, True, True),
        np.eye(3) * 1.5,
    ),
    "aperiodic_cluster": (_cluster(8, 1.8, 2), 2.0, (False, False, False), None),
    "slab": (_cluster(6, 2.0, 3), 3.5, (True, True, False), SLAB),
    "unit_cube_at_sqrt_two": (np.zeros((1, 3)), np.sqrt(2), (True,) * 3, np.eye(3)),
}


def _edges(neighbors):
    rows = np.concatenate((neighbors.edge_index.T, neighbors.unit_shifts), axis=1)
    return sorted(map(tuple, rows.tolist()))


def test_mace_torch_defaults_to_vesin():
    assert DEFAULT_NEIGHBOR_BACKEND == "vesin"
    assert isinstance(resolve_neighbor_backend(), VesinBackend)


def test_a_backend_named_in_the_config_wins():
    config = NeighborListConfig(kind="matscipy")
    assert isinstance(resolve_neighbor_backend(config), MatscipyBackend)


def test_vesin_is_registered_from_mace_torch():
    listed = {backend.name: backend for backend in available_neighbor_backends()}
    assert listed["vesin"].loaded
    assert listed["vesin"].factory is VesinBackend


@pytest.mark.parametrize("name", sorted(CASES))
def test_vesin_reproduces_the_brute_force_edge_set(name):
    positions, cutoff, pbc, cell = CASES[name]
    vesin = get_neighborhood(positions, cutoff, pbc, cell, backend=VesinBackend())
    brute = get_neighborhood(positions, cutoff, pbc, cell, backend=BruteForceBackend())
    assert _edges(vesin) == _edges(brute)
    assert np.array_equal(vesin.cell, brute.cell)


def test_vesin_takes_its_thread_count_from_the_settings():
    """The thread count changes the speed and never the edges or their order."""
    positions, cutoff, pbc, cell = CASES["triclinic"]
    config = NeighborListConfig(kind="vesin", settings={"n_threads": 4})
    threaded: Any = resolve_neighbor_backend(config)
    single: Any = get_neighbor_backend("vesin")
    assert (threaded.settings.n_threads, single.settings.n_threads) == (4, 1)
    many = get_neighborhood(positions, cutoff, pbc, cell, backend=threaded)
    one = get_neighborhood(positions, cutoff, pbc, cell, backend=single)
    assert np.array_equal(many.edge_index, one.edge_index)
    assert np.array_equal(many.unit_shifts, one.unit_shifts)


@pytest.mark.parametrize("n_threads", [-1, 1.5, True, "4"])
def test_vesin_refuses_a_thread_count_that_is_not_a_count(n_threads):
    with pytest.raises(NeighborBackendNotAvailableError, match="n_threads"):
        get_neighbor_backend("vesin", {"n_threads": n_threads})


def test_vesin_refuses_an_unknown_setting():
    with pytest.raises(NeighborBackendNotAvailableError, match="threads"):
        get_neighbor_backend("vesin", {"threads": 2})


def test_resolving_another_backend_does_not_import_vesin():
    probe = (
        "import sys; from mace_core.neighbors import get_neighbor_backend; "
        "get_neighbor_backend('matscipy'); print('vesin' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["False"]
