"""The neighbour list: the pinned edge set and the three-regime effective cell.

Every registered backend is held to the brute-force backend's edge set on the
characterization cases below (triclinic, a cell smaller than the cutoff, an
aperiodic cluster, mixed periodicity with and without a zero vacuum row, and a
structure with no edges). The brute-force backend itself is checked against
counts known from crystallography and against an independent oracle on the
committed fixtures in ``tests/parity/test_graph_parity.py``.
"""

from __future__ import annotations

import sys
import textwrap

import numpy as np
import pytest
from mace_core.config import ConfigSection
from mace_core.neighbors import (
    BruteForceBackend,
    DuplicateNeighborBackendError,
    NeighborBackendNotAvailableError,
    NeighborListConfig,
    available_neighbor_backends,
    get_neighbor_backend,
    get_neighborhood,
    search_geometry,
)
from mace_core.neighbors import geometry as geometry_module


def _edges(neighbors):
    """The edge set as sorted ``(sender, receiver, n0, n1, n2)`` integers."""
    rows = np.concatenate((neighbors.edge_index.T, neighbors.unit_shifts), axis=1)
    return sorted(map(tuple, rows.tolist()))


def _physical(cell):
    """The case's cell, for the cases that have one."""
    assert cell is not None
    return cell


def _cluster(n_atoms=8, spread=1.8, seed=20260810):
    return np.random.default_rng(seed).uniform(0.0, spread, size=(n_atoms, 3))


TRICLINIC = np.array([[3.1, 0.2, 0.1], [0.9, 2.8, -0.2], [0.4, -0.3, 3.3]])
HEXAGONAL_SLAB = np.array([[3.0, 0.0, 0.0], [1.5, 2.598, 0.0], [0.0, 0.0, 0.0]])

CASES = {
    "triclinic": (_cluster(5, 2.5, 1), 3.2, (True, True, True), TRICLINIC),
    "cell_smaller_than_cutoff": (
        np.zeros((1, 3)),
        3.0,
        (True, True, True),
        np.eye(3) * 1.5,
    ),
    "aperiodic_cluster": (_cluster(), 2.0, (False, False, False), None),
    "mixed_pbc_with_vacuum": (
        _cluster(6, 2.0, 2),
        3.5,
        (True, True, False),
        np.array([[3.0, 0.0, 0.0], [1.5, 2.598, 0.0], [0.0, 0.0, 12.0]]),
    ),
    "mixed_pbc_zero_vacuum_row": (
        _cluster(6, 2.0, 3),
        3.5,
        (True, True, False),
        HEXAGONAL_SLAB,
    ),
    "mixed_pbc_one_axis": (
        _cluster(4, 2.0, 4),
        3.0,
        (True, False, False),
        np.eye(3) * 2.7,
    ),
    "zero_edges": (np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]), 2.0, None, None),
}


@pytest.fixture(params=sorted(CASES), name="case")
def fixture_case(request):
    return CASES[request.param]


# --- the brute-force backend is falsifiable ---------------------------------


@pytest.mark.parametrize(
    ("cell", "neighbours"),
    [
        # Simple cubic, a = 1, cutoff between the first and second shell: 6.
        (np.eye(3), 6),
        # Face-centred cubic primitive cell, nearest-neighbour distance 1: 12.
        (
            np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * np.sqrt(2),
            12,
        ),
    ],
)
def test_brute_force_reproduces_known_coordination_numbers(cell, neighbours):
    neighbors = get_neighborhood(
        np.zeros((1, 3)), 1.2, (True, True, True), cell, backend=BruteForceBackend()
    )
    assert neighbors.edge_index.shape[1] == neighbours


def test_brute_force_reaches_images_of_a_strongly_skewed_cell():
    """The image bound comes from the reciprocal vectors, so a cell whose
    second vector nearly lies along the first still finds every image."""
    cell = np.array([[1.0, 0.0, 0.0], [3.9, 0.3, 0.0], [0.0, 0.0, 1.0]])
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.1, 0.5]])
    brute = get_neighborhood(
        positions, 2.5, (True, True, True), cell, backend=BruteForceBackend()
    )
    matscipy = get_neighborhood(positions, 2.5, (True, True, True), cell)
    assert _edges(brute) == _edges(matscipy)
    assert brute.edge_index.shape[1] > 0


# --- the pinned edge set ------------------------------------------------------


def test_every_registered_backend_reproduces_the_brute_force_edge_set(case):
    positions, cutoff, pbc, cell = case
    expected = _edges(
        get_neighborhood(positions, cutoff, pbc, cell, backend=BruteForceBackend())
    )
    for discovered in available_neighbor_backends():
        backend = get_neighbor_backend(discovered.name)
        neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=backend)
        assert _edges(neighbors) == expected, discovered.name


def test_the_returned_cell_does_not_depend_on_the_backend(case):
    positions, cutoff, pbc, cell = case
    cells = [
        get_neighborhood(
            positions, cutoff, pbc, cell, backend=get_neighbor_backend(d.name)
        ).cell
        for d in available_neighbor_backends()
    ]
    for returned in cells[1:]:
        assert np.array_equal(returned, cells[0])


def test_the_list_is_full_and_directed(case):
    positions, cutoff, pbc, cell = case
    edges = _edges(get_neighborhood(positions, cutoff, pbc, cell))
    reversed_edges = sorted((j, i, -a, -b, -c) for i, j, a, b, c in edges)
    assert edges == reversed_edges


def test_every_edge_is_strictly_shorter_than_the_cutoff(case):
    positions, cutoff, pbc, cell = case
    neighbors = get_neighborhood(positions, cutoff, pbc, cell)
    sender, receiver = neighbors.edge_index
    lengths = np.linalg.norm(
        positions[receiver] - positions[sender] + neighbors.shifts, axis=1
    )
    assert np.all(lengths < cutoff)


def test_shifts_are_unit_shifts_times_the_returned_cell(case):
    positions, cutoff, pbc, cell = case
    neighbors = get_neighborhood(positions, cutoff, pbc, cell)
    assert np.array_equal(
        neighbors.shifts, np.dot(neighbors.unit_shifts, neighbors.cell)
    )
    assert neighbors.edge_index.dtype == np.int64
    assert neighbors.unit_shifts.dtype == np.int64


def test_periodic_self_images_survive_and_the_zero_shift_self_pair_does_not():
    positions, cutoff, pbc, cell = CASES["cell_smaller_than_cutoff"]
    neighbors = get_neighborhood(positions, cutoff, pbc, cell)
    assert neighbors.edge_index.shape[1] > 0
    assert np.all(neighbors.edge_index == 0)
    assert not np.any(np.all(neighbors.unit_shifts == 0, axis=1))


def test_asking_for_the_zero_shift_self_edge_is_refused():
    """The frozen implementation accepts the flag and never emits the edge, so
    honouring it would change every edge count that passes it."""
    with pytest.raises(ValueError, match="true_self_interaction"):
        get_neighborhood(_cluster(4), 2.0, true_self_interaction=True)


def test_a_structure_with_no_edges_is_legal():
    positions, cutoff, pbc, cell = CASES["zero_edges"]
    neighbors = get_neighborhood(positions, cutoff, pbc, cell)
    assert neighbors.edge_index.shape == (2, 0)
    assert neighbors.unit_shifts.shape == (0, 3)
    assert neighbors.shifts.shape == (0, 3)


def test_a_single_aperiodic_atom_has_no_edges():
    neighbors = get_neighborhood(np.zeros((1, 3)), 5.0)
    assert neighbors.edge_index.shape == (2, 0)


def test_nothing_passed_in_is_modified(case):
    positions, cutoff, pbc, cell = case
    positions_before = positions.copy()
    cell_before = None if cell is None else cell.copy()
    get_neighborhood(positions, cutoff, pbc, cell)
    assert np.array_equal(positions, positions_before)
    if cell is not None:
        assert np.array_equal(cell, _physical(cell_before))


def test_the_returned_arrays_share_no_memory_with_the_inputs_and_are_read_only():
    positions, cutoff, pbc, cell = CASES["mixed_pbc_with_vacuum"]
    cell = _physical(cell)
    neighbors = get_neighborhood(positions, cutoff, pbc, cell)
    assert not np.shares_memory(neighbors.cell, cell)
    for array in (
        neighbors.edge_index,
        neighbors.unit_shifts,
        neighbors.shifts,
        neighbors.cell,
    ):
        assert not array.flags.writeable


# --- the three effective-cell regimes -----------------------------------------


def test_a_fully_aperiodic_cluster_returns_the_extended_search_cell():
    positions = _cluster()
    neighbors = get_neighborhood(positions, 2.0)
    extent = positions.max(axis=0) - positions.min(axis=0)
    assert np.array_equal(neighbors.cell, np.diag(extent + 2 * 2.0 + 1))
    assert np.linalg.det(neighbors.cell) > 0


def test_the_extended_cell_depends_on_the_extent_not_the_origin():
    positions = _cluster()
    near = get_neighborhood(positions, 2.0).cell
    far = get_neighborhood(positions + 100.0, 2.0).cell
    np.testing.assert_allclose(near, far, rtol=0, atol=1e-12)


def test_a_periodic_structure_returns_its_physical_cell():
    positions, cutoff, pbc, cell = CASES["triclinic"]
    cell = _physical(cell)
    assert np.array_equal(get_neighborhood(positions, cutoff, pbc, cell).cell, cell)


def test_a_slab_with_vacuum_returns_its_physical_cell():
    positions, cutoff, pbc, cell = CASES["mixed_pbc_with_vacuum"]
    cell = _physical(cell)
    assert np.array_equal(get_neighborhood(positions, cutoff, pbc, cell).cell, cell)


def test_a_zero_vacuum_row_is_patched_from_the_search_cell():
    positions, cutoff, pbc, cell = CASES["mixed_pbc_zero_vacuum_row"]
    cell = _physical(cell)
    returned = get_neighborhood(positions, cutoff, pbc, cell).cell
    extent = positions[:, 2].max() - positions[:, 2].min()
    assert np.array_equal(returned[:2], cell[:2])
    assert np.array_equal(returned[2], [0.0, 0.0, extent + 2 * cutoff + 1])
    assert np.linalg.det(returned) != 0


def test_a_slab_stress_from_the_returned_cell_uses_the_physical_volume():
    """stress = -virials / volume: the vacuum box the search used must not
    enter the volume."""
    positions, cutoff, pbc, cell = CASES["mixed_pbc_with_vacuum"]
    cell = _physical(cell)
    virials = np.diag([0.3, -0.1, 0.2])
    geometry = search_geometry(positions, cutoff, pbc, cell)
    returned = get_neighborhood(positions, cutoff, pbc, cell).cell
    stress = -virials / np.linalg.det(returned)
    assert np.array_equal(stress, -virials / np.linalg.det(cell))
    assert np.linalg.det(geometry.search_cell) != np.linalg.det(cell)


def test_the_three_regimes_are_three_different_matrices():
    positions = _cluster(6, 2.0, 3)
    physical = HEXAGONAL_SLAB.copy()
    physical[2] = [0.0, 0.0, 12.0]
    aperiodic = get_neighborhood(positions, 3.5, None, physical).cell
    periodic = get_neighborhood(positions, 3.5, (True, True, False), physical).cell
    patched = get_neighborhood(positions, 3.5, (True, True, False), HEXAGONAL_SLAB).cell
    assert not np.array_equal(aperiodic, periodic)
    assert not np.array_equal(periodic, patched)
    assert not np.array_equal(aperiodic, patched)


@pytest.mark.parametrize("cell", [None, np.zeros((3, 3))])
def test_a_missing_or_all_zero_cell_is_the_identity_before_extension(cell):
    geometry = search_geometry(np.zeros((1, 3)), 2.0, (True, False, False), cell)
    assert np.array_equal(geometry.cell[0], [1.0, 0.0, 0.0])
    assert np.array_equal(geometry.search_cell[1], [0.0, 2 * 2.0 + 1, 0.0])


def test_pbc_none_is_fully_aperiodic():
    assert search_geometry(np.zeros((2, 3)), 1.0).pbc == (False, False, False)


def test_a_periodic_axis_without_a_lattice_vector_is_refused():
    with pytest.raises(ValueError, match="axis 2 is periodic"):
        search_geometry(np.zeros((1, 3)), 1.0, (True, True, True), HEXAGONAL_SLAB)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("inf"), float("nan"), True])
def test_a_cutoff_that_is_not_a_positive_distance_is_refused(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        search_geometry(np.zeros((1, 3)), cutoff)


def test_only_search_geometry_decides_a_regime():
    """The graph builder and the brute-force backend read the box from it; no
    other function in the package computes one."""
    import inspect

    from mace_core.neighbors import brute_force, matscipy_backend, neighborhood

    assert "search_geometry(" in inspect.getsource(neighborhood.get_neighborhood)
    for module in (brute_force, matscipy_backend, neighborhood):
        source = inspect.getsource(module)
        assert "2 * cutoff" not in source, module.__name__
        assert "search_cell" not in source or module is not neighborhood
    assert "2 * cutoff + 1" in inspect.getsource(geometry_module)


# --- the registry and the config section ------------------------------------------


def test_matscipy_and_brute_force_are_registered():
    names = {backend.name for backend in available_neighbor_backends()}
    assert {"matscipy", "brute_force"} <= names


def test_the_config_section_defaults_to_matscipy():
    assert NeighborListConfig().kind == "matscipy"
    assert issubclass(NeighborListConfig, ConfigSection)
    with pytest.raises(ValueError, match="kind"):
        NeighborListConfig(kind=" ")
    with pytest.raises(ValueError):
        NeighborListConfig.model_validate({"kind": "matscipy", "cutoff": 5.0})


def test_an_unregistered_name_is_refused_with_the_registered_ones():
    with pytest.raises(NeighborBackendNotAvailableError, match="brute_force"):
        get_neighbor_backend("cell_list_on_the_moon")


def _install(path, distribution, entry_point):
    """A distribution whose metadata declares one neighbour backend."""
    info = path / f"{distribution}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 1.0\n"
    )
    (info / "entry_points.txt").write_text(f"[mace.neighbor_backends]\n{entry_point}\n")


def test_two_distributions_registering_different_backends_under_one_name_raise(
    tmp_path, monkeypatch
):
    _install(tmp_path, "first_neighbors", "duplicate = first:Backend")
    _install(tmp_path, "second_neighbors", "duplicate = second:Backend")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(DuplicateNeighborBackendError, match="duplicate"):
        available_neighbor_backends()
    with pytest.raises(DuplicateNeighborBackendError, match="duplicate"):
        get_neighbor_backend("matscipy")


def test_two_identical_registrations_are_one(tmp_path, monkeypatch):
    target = "brute = mace_core.neighbors.brute_force:BruteForceBackend"
    _install(tmp_path, "first_neighbors", target)
    _install(tmp_path, "second_neighbors", target)
    monkeypatch.syspath_prepend(str(tmp_path))
    names = [backend.name for backend in available_neighbor_backends()]
    assert names.count("brute") == 1
    assert isinstance(get_neighbor_backend("brute"), BruteForceBackend)


def test_a_backend_registered_through_an_entry_point_is_discovered(
    tmp_path, monkeypatch
):
    """A distribution installed beside this one, found through its metadata
    alone: nothing in mace_core names it."""
    (tmp_path / "dummy_neighbors.py").write_text(
        textwrap.dedent(
            """
            import numpy as np

            class DummyBackend:
                def search(self, positions, cutoff, geometry):
                    return np.zeros((2, 0), np.int64), np.zeros((0, 3), np.int64)
            """
        )
    )
    distribution = tmp_path / "dummy_neighbors-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: dummy-neighbors\nVersion: 1.0\n"
    )
    (distribution / "entry_points.txt").write_text(
        "[mace.neighbor_backends]\ndummy = dummy_neighbors:DummyBackend\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "dummy_neighbors", raising=False)

    assert "dummy" in {backend.name for backend in available_neighbor_backends()}
    backend = get_neighbor_backend("dummy")
    neighbors = get_neighborhood(_cluster(), 2.0, backend=backend)
    assert neighbors.edge_index.shape == (2, 0)
    # The effective cell is the graph builder's, whatever the backend did.
    assert np.array_equal(neighbors.cell, get_neighborhood(_cluster(), 2.0).cell)


def test_a_backend_returning_malformed_edges_is_refused():
    class Broken:
        def search(self, positions, cutoff, geometry):
            return np.zeros((3, 1), np.int64), np.zeros((1, 3), np.int64)

    with pytest.raises(ValueError, match="edge_index of shape"):
        get_neighborhood(_cluster(), 2.0, backend=Broken())
