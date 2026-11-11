"""The neighbour list: the pinned edge set, and the parsed cell its shifts use.

Every registered backend is held to the all-pairs backend's edge set on the
characterization cases below (triclinic, a cell smaller than the cutoff, an
aperiodic cluster, mixed periodicity with and without a zero vacuum row, and a
structure with no edges). The all-pairs backend itself is checked against
counts known from crystallography and against an independent oracle on the
committed fixtures in ``tests/parity/test_graph_parity.py``.
"""

from __future__ import annotations

import sys
import textwrap
from typing import Any

import numpy as np
import pytest
from mace_core.config import ConfigSection
from mace_core.neighbors import (
    DEFAULT_NEIGHBOR_BACKEND,
    AllPairsBackend,
    DuplicateNeighborBackendError,
    NeighborBackendNotAvailableError,
    NeighborListConfig,
    available_neighbor_backends,
    get_neighbor_backend,
    get_neighborhood,
    graph_cell,
    resolve_neighbor_backend,
    search_geometry,
)
from mace_core.neighbors import geometry as geometry_module
from mace_core.neighbors.matscipy_backend import MatscipyBackend

MATSCIPY = MatscipyBackend()


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


# --- the all-pairs backend is falsifiable ---------------------------------


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
def test_all_pairs_reproduces_known_coordination_numbers(cell, neighbours):
    neighbors = get_neighborhood(
        np.zeros((1, 3)), 1.2, (True, True, True), cell, backend=AllPairsBackend()
    )
    assert neighbors.edge_index.shape[1] == neighbours


def test_all_pairs_reaches_images_of_a_strongly_skewed_cell():
    """The image bound comes from the reciprocal vectors, so a cell whose
    second vector nearly lies along the first still finds every image."""
    cell = np.array([[1.0, 0.0, 0.0], [3.9, 0.3, 0.0], [0.0, 0.0, 1.0]])
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.1, 0.5]])
    pairs = get_neighborhood(
        positions, 2.5, (True, True, True), cell, backend=AllPairsBackend()
    )
    matscipy = get_neighborhood(
        positions, 2.5, (True, True, True), cell, backend=MATSCIPY
    )
    assert _edges(pairs) == _edges(matscipy)
    assert pairs.edge_index.shape[1] > 0


# --- the pinned edge set ------------------------------------------------------


def test_every_registered_backend_reproduces_the_all_pairs_edge_set(case):
    positions, cutoff, pbc, cell = case
    expected = _edges(
        get_neighborhood(positions, cutoff, pbc, cell, backend=AllPairsBackend())
    )
    for discovered in available_neighbor_backends():
        backend = get_neighbor_backend(discovered.name)
        neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=backend)
        assert _edges(neighbors) == expected, discovered.name


def test_every_backend_finds_images_along_periodic_axes_only(case):
    positions, cutoff, pbc, cell = case
    aperiodic = [axis for axis in range(3) if not (pbc or (False,) * 3)[axis]]
    for discovered in available_neighbor_backends():
        neighbors = get_neighborhood(
            positions, cutoff, pbc, cell, backend=get_neighbor_backend(discovered.name)
        )
        assert not neighbors.unit_shifts[:, aperiodic].any(), discovered.name


def test_the_list_is_full_and_directed(case):
    positions, cutoff, pbc, cell = case
    edges = _edges(get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY))
    reversed_edges = sorted((j, i, -a, -b, -c) for i, j, a, b, c in edges)
    assert edges == reversed_edges


def test_every_edge_is_strictly_shorter_than_the_cutoff(case):
    positions, cutoff, pbc, cell = case
    neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY)
    sender, receiver = neighbors.edge_index
    vectors = positions[receiver] - positions[sender] + neighbors.shifts
    assert np.all(np.sum(vectors * vectors, axis=1) < cutoff * cutoff)


def test_a_length_that_rounds_onto_the_cutoff_is_inside_for_every_backend():
    """The face diagonals of a unit cube are shorter than ``np.sqrt(2)``, whose
    float64 value is above the exact root, but their norm rounds onto it."""
    positions, cell, cutoff = np.zeros((1, 3)), np.eye(3), np.sqrt(2)
    diagonal = np.array([1.0, 1.0, 0.0])
    assert not np.linalg.norm(diagonal) < cutoff
    for discovered in available_neighbor_backends():
        neighbors = get_neighborhood(
            positions,
            cutoff,
            (True, True, True),
            cell,
            backend=get_neighbor_backend(discovered.name),
        )
        assert neighbors.edge_index.shape[1] == 6 + 12, discovered.name


def test_shifts_are_unit_shifts_times_the_graph_cell(case):
    positions, cutoff, pbc, cell = case
    neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY)
    assert np.array_equal(
        neighbors.shifts, np.dot(neighbors.unit_shifts, graph_cell(pbc, cell))
    )
    assert neighbors.edge_index.dtype == np.int64
    assert neighbors.unit_shifts.dtype == np.int64


def test_periodic_self_images_survive_and_the_zero_shift_self_pair_does_not():
    positions, cutoff, pbc, cell = CASES["cell_smaller_than_cutoff"]
    neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY)
    assert neighbors.edge_index.shape[1] > 0
    assert np.all(neighbors.edge_index == 0)
    assert not np.any(np.all(neighbors.unit_shifts == 0, axis=1))


def test_a_structure_with_no_edges_is_legal():
    positions, cutoff, pbc, cell = CASES["zero_edges"]
    neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY)
    assert neighbors.edge_index.shape == (2, 0)
    assert neighbors.unit_shifts.shape == (0, 3)
    assert neighbors.shifts.shape == (0, 3)


def test_a_single_aperiodic_atom_has_no_edges():
    neighbors = get_neighborhood(np.zeros((1, 3)), 5.0, backend=MATSCIPY)
    assert neighbors.edge_index.shape == (2, 0)


def test_nothing_passed_in_is_modified(case):
    positions, cutoff, pbc, cell = case
    positions_before = positions.copy()
    cell_before = None if cell is None else cell.copy()
    get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY)
    assert np.array_equal(positions, positions_before)
    if cell is not None:
        assert np.array_equal(cell, _physical(cell_before))


def test_the_returned_arrays_are_read_only_and_the_cutoff_is_recorded():
    positions, cutoff, pbc, cell = CASES["mixed_pbc_with_vacuum"]
    neighbors = get_neighborhood(positions, cutoff, pbc, cell, backend=MATSCIPY)
    assert neighbors.cutoff == cutoff
    for array in (neighbors.edge_index, neighbors.unit_shifts, neighbors.shifts):
        assert not array.flags.writeable
    carried = graph_cell(pbc, cell)
    assert not np.shares_memory(carried, cell)
    assert not carried.flags.writeable


# --- the cell the graph carries -------------------------------------------------


def test_a_fully_aperiodic_cluster_carries_its_parsed_cell():
    assert np.array_equal(graph_cell(None, None), np.zeros((3, 3)))
    given = np.diag([7.0, 8.0, 9.0])
    assert np.array_equal(graph_cell((False, False, False), given), given)


def test_the_graph_cell_never_depends_on_the_positions():
    positions = _cluster()
    near = get_neighborhood(positions, 2.0, backend=MATSCIPY)
    far = get_neighborhood(positions + 100.0, 2.0, backend=MATSCIPY)
    assert _edges(near) == _edges(far)
    assert np.array_equal(graph_cell(None, None), np.zeros((3, 3)))


def test_a_periodic_structure_carries_its_physical_cell():
    _, _, pbc, cell = CASES["triclinic"]
    cell = _physical(cell)
    assert np.array_equal(graph_cell(pbc, cell), cell)


def test_a_slab_with_vacuum_carries_its_physical_cell():
    _, _, pbc, cell = CASES["mixed_pbc_with_vacuum"]
    cell = _physical(cell)
    assert np.array_equal(graph_cell(pbc, cell), cell)


def test_a_zero_vacuum_row_is_kept_and_never_patched():
    _, _, pbc, cell = CASES["mixed_pbc_zero_vacuum_row"]
    cell = _physical(cell)
    carried = graph_cell(pbc, cell)
    assert np.array_equal(carried, cell)
    assert np.linalg.det(carried) == 0


def test_the_search_box_is_wider_than_the_graph_cell_along_an_aperiodic_axis():
    positions, cutoff, pbc, cell = CASES["mixed_pbc_with_vacuum"]
    cell = _physical(cell)
    geometry = search_geometry(positions, cutoff, pbc, cell)
    extent = positions[:, 2].max() - positions[:, 2].min()
    assert np.array_equal(geometry.search_cell[:2], cell[:2])
    assert np.array_equal(geometry.search_cell[2], [0.0, 0.0, extent + 2 * cutoff + 1])
    assert not hasattr(geometry, "cell")


@pytest.mark.parametrize("cell", [None, np.zeros((3, 3))])
def test_a_missing_or_all_zero_cell_is_the_identity_along_a_periodic_axis(cell):
    assert np.array_equal(graph_cell((True, False, False), cell), np.eye(3))
    geometry = search_geometry(np.zeros((1, 3)), 2.0, (True, False, False), cell)
    assert np.array_equal(geometry.search_cell[0], [1.0, 0.0, 0.0])
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


def test_a_backend_returning_an_image_along_an_aperiodic_axis_is_refused():
    class Imaging:
        def search(self, positions, cutoff, geometry):
            return np.array([[0], [1]]), np.array([[0, 0, 1]])

    with pytest.raises(ValueError, match="aperiodic axes"):
        get_neighborhood(
            _cluster(2), 2.0, (True, True, False), TRICLINIC, backend=Imaging()
        )


def test_only_the_search_box_is_sized_from_the_extent():
    """No function but search_geometry sizes a box, and no box reaches the list."""
    import inspect

    from mace_core.neighbors import all_pairs, matscipy_backend, neighborhood

    assert "search_geometry(" in inspect.getsource(neighborhood.get_neighborhood)
    for module in (all_pairs, matscipy_backend, neighborhood):
        source = inspect.getsource(module)
        assert "2 * cutoff" not in source, module.__name__
        assert "search_cell" not in source or module is not neighborhood
    assert "2 * cutoff + 1" in inspect.getsource(geometry_module)
    assert "cell" not in neighborhood.NeighborList.__dataclass_fields__


# --- the registry and the config section ------------------------------------------


def test_matscipy_and_all_pairs_are_registered():
    names = {backend.name for backend in available_neighbor_backends()}
    assert {"matscipy", "all_pairs"} <= names


def test_the_config_section_leaves_the_choice_to_the_framework_by_default():
    assert NeighborListConfig().kind is None
    assert NeighborListConfig().settings == {}
    with pytest.raises(ValueError, match=r"without neighbor_list\.kind"):
        NeighborListConfig(settings={"n_threads": 2})
    assert issubclass(NeighborListConfig, ConfigSection)
    with pytest.raises(ValueError, match="kind"):
        NeighborListConfig(kind=" ")
    with pytest.raises(ValueError):
        NeighborListConfig.model_validate({"kind": "matscipy", "cutoff": 5.0})


def test_an_unregistered_name_is_refused_with_the_registered_ones():
    with pytest.raises(NeighborBackendNotAvailableError, match="all_pairs"):
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
    target = "pairs = mace_core.neighbors.all_pairs:AllPairsBackend"
    _install(tmp_path, "first_neighbors", target)
    _install(tmp_path, "second_neighbors", target)
    monkeypatch.syspath_prepend(str(tmp_path))
    names = [backend.name for backend in available_neighbor_backends()]
    assert names.count("pairs") == 1
    assert isinstance(get_neighbor_backend("pairs"), AllPairsBackend)


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


def test_a_backend_returning_malformed_edges_is_refused():
    class Broken:
        def search(self, positions, cutoff, geometry):
            return np.zeros((3, 1), np.int64), np.zeros((1, 3), np.int64)

    with pytest.raises(ValueError, match="edge_index of shape"):
        get_neighborhood(_cluster(), 2.0, backend=Broken())


def _register_module(path, monkeypatch, name, source):
    """A module and a distribution registering ``name`` as its ``Backend``."""
    (path / f"{name}_neighbors.py").write_text(textwrap.dedent(source))
    _install(path, f"{name}_neighbors", f"{name} = {name}_neighbors:Backend")
    monkeypatch.syspath_prepend(str(path))
    monkeypatch.delitem(sys.modules, f"{name}_neighbors", raising=False)


def test_resolving_a_name_imports_only_that_backend(tmp_path, monkeypatch):
    """A backend that sets up a device when imported costs nothing to a run
    that resolves another one."""
    _register_module(
        tmp_path,
        monkeypatch,
        "heavy",
        """
        raise RuntimeError("importing this backend sets up a device")
        """,
    )
    assert isinstance(get_neighbor_backend("all_pairs"), AllPairsBackend)
    assert "heavy_neighbors" not in sys.modules
    listed = {backend.name: backend for backend in available_neighbor_backends()}
    assert not listed["heavy"].loaded
    assert "sets up a device" in listed["heavy"].reason


def test_the_settings_reach_the_backend_factory(tmp_path, monkeypatch):
    _register_module(
        tmp_path,
        monkeypatch,
        "skinned",
        """
        import numpy as np
        from mace_core.config import ConfigSection
        from pydantic import Field

        class Settings(ConfigSection):
            skin: float = Field(default=0.0, ge=0)

        class Backend:
            settings_section = Settings

            def __init__(self, settings=None):
                self.settings = settings or Settings()

            def search(self, positions, cutoff, geometry):
                return np.zeros((2, 0), np.int64), np.zeros((0, 3), np.int64)
        """,
    )
    config = NeighborListConfig(kind="skinned", settings={"skin": 0.5})
    configured: Any = resolve_neighbor_backend(config)
    default: Any = get_neighbor_backend("skinned")
    assert configured.settings.skin == 0.5
    assert default.settings.skin == 0.0
    with pytest.raises(
        NeighborBackendNotAvailableError, match=r"skinned(.|\n)*rebuild"
    ):
        get_neighbor_backend("skinned", {"rebuild": 10})
    with pytest.raises(NeighborBackendNotAvailableError, match="greater than or equal"):
        get_neighbor_backend("skinned", {"skin": -1.0})


def test_a_settings_section_that_is_not_a_config_section_is_refused(
    tmp_path, monkeypatch
):
    _register_module(
        tmp_path,
        monkeypatch,
        "loose",
        """
        class Backend:
            settings_section = dict

            def __init__(self, settings=None):
                pass

            def search(self, positions, cutoff, geometry):
                raise AssertionError("never reached")
        """,
    )
    with pytest.raises(NeighborBackendNotAvailableError, match="not a ConfigSection"):
        get_neighbor_backend("loose")


def test_a_backend_without_settings_refuses_any():
    with pytest.raises(NeighborBackendNotAvailableError, match="matscipy"):
        get_neighbor_backend("matscipy", {"skin": 0.5})


# --- choosing a backend when the config names none ---------------------------


def test_with_no_backend_named_and_no_framework_all_pairs_is_used():
    assert DEFAULT_NEIGHBOR_BACKEND == "all_pairs"
    assert isinstance(resolve_neighbor_backend(), AllPairsBackend)


def test_with_no_backend_named_the_framework_default_is_used():
    assert isinstance(resolve_neighbor_backend(None, "matscipy"), MatscipyBackend)
    assert isinstance(
        resolve_neighbor_backend(NeighborListConfig(), "all_pairs"),
        AllPairsBackend,
    )


def test_a_named_backend_wins_over_the_default():
    config = NeighborListConfig(kind="matscipy")
    assert isinstance(resolve_neighbor_backend(config, "all_pairs"), MatscipyBackend)


def test_a_named_backend_that_cannot_be_built_is_not_replaced():
    config = NeighborListConfig(kind="cell_list_on_the_moon")
    with pytest.raises(NeighborBackendNotAvailableError, match="cell_list_on_the_moon"):
        resolve_neighbor_backend(config, "matscipy")


def test_a_default_that_did_not_import_is_an_error_not_a_fallback(
    tmp_path, monkeypatch
):
    _register_module(
        tmp_path,
        monkeypatch,
        "heavy",
        """
        raise RuntimeError("importing this backend sets up a device")
        """,
    )
    with pytest.raises(NeighborBackendNotAvailableError, match="sets up a device"):
        resolve_neighbor_backend(None, "heavy")


def test_a_missing_matscipy_names_the_extra_to_install(monkeypatch):
    import importlib

    monkeypatch.setitem(sys.modules, "matscipy.neighbours", None)
    monkeypatch.delitem(sys.modules, "mace_core.neighbors.matscipy_backend")
    with pytest.raises(ImportError, match=r"mace-core\[matscipy\]"):
        importlib.import_module("mace_core.neighbors.matscipy_backend")
