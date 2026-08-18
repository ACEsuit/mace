import json
from copy import deepcopy
from pathlib import Path

import ase.build
import h5py
import numpy as np
import pytest

from tests.helpers import REPO_ROOT
from tests.neighbour_oracle import (
    assert_neighbourhoods_match,
    brute_force_neighborhood,
    canonical_edges,
    canonical_edges_from_shifts,
)
import torch

from mace.data import (
    AtomicData,
    Configuration,
    HDF5Dataset,
    config_from_atoms,
    get_neighborhood,
    save_configurations_as_HDF5,
)
from mace.tools import AtomicNumberTable, torch_geometric

mace_path = REPO_ROOT


class TestAtomicData:
    config = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0, -2.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        properties={
            "forces": np.array(
                [
                    [0.0, -1.3, 0.0],
                    [1.0, 0.2, 0.0],
                    [0.0, 1.1, 0.3],
                ]
            ),
            "energy": -1.5,
        },
        property_weights={
            "forces": 1.0,
            "energy": 1.0,
        },
    )
    config_2 = deepcopy(config)
    config_2.positions = config.positions + 0.01

    table = AtomicNumberTable([1, 8])

    def test_atomic_data(self):
        data = AtomicData.from_config(self.config, z_table=self.table, cutoff=3.0)

        assert data.edge_index.shape == (2, 4)
        assert data.forces.shape == (3, 3)
        assert data.node_attrs.shape == (3, 2)

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_data_loader(self, num_workers):
        data1 = AtomicData.from_config(self.config, z_table=self.table, cutoff=3.0)
        data2 = AtomicData.from_config(self.config_2, z_table=self.table, cutoff=3.0)

        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[data1, data2],
            batch_size=2,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )

        try:
            for batch in data_loader:
                assert batch.batch.shape == (6,)
                assert batch.edge_index.shape == (2, 8)
                assert batch.shifts.shape == (8, 3)
                assert batch.positions.shape == (6, 3)
                assert batch.node_attrs.shape == (6, 2)
                assert batch.energy.shape == (2,)
                assert batch.forces.shape == (6, 3)
        except RuntimeError as exc:
            if num_workers > 0 and "torch_shm_manager" in str(exc):
                pytest.skip(
                    "Shared-memory dataloader is not permitted in this environment"
                )
            raise

    def test_to_atomic_data_dict(self):
        data1 = AtomicData.from_config(self.config, z_table=self.table, cutoff=3.0)
        data2 = AtomicData.from_config(self.config, z_table=self.table, cutoff=3.0)

        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[data1, data2],
            batch_size=2,
            shuffle=True,
            drop_last=False,
        )
        for batch in data_loader:
            batch_dict = batch.to_dict()
            assert batch_dict["batch"].shape == (6,)
            assert batch_dict["edge_index"].shape == (2, 8)
            assert batch_dict["shifts"].shape == (8, 3)
            assert batch_dict["positions"].shape == (6, 3)
            assert batch_dict["node_attrs"].shape == (6, 2)
            assert batch_dict["energy"].shape == (2,)
            assert batch_dict["forces"].shape == (6, 3)

    def test_hdf5_dataloader(self, tmp_path):
        datasets = [self.config, self.config_2] * 5
        dataset_path = tmp_path / "test.h5"
        with h5py.File(str(dataset_path), "w") as f:
            save_configurations_as_HDF5(datasets, 0, f)
        train_dataset = HDF5Dataset(str(dataset_path), z_table=self.table, r_max=3.0)
        train_loader = torch_geometric.dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=2,
            shuffle=False,
            drop_last=False,
        )
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            assert batch.batch.shape == (6,)
            assert batch.edge_index.shape == (2, 8)
            assert batch.shifts.shape == (8, 3)
            assert batch.positions.shape == (6, 3)
            assert batch.node_attrs.shape == (6, 2)
            assert batch.energy.shape == (2,)
            assert batch.forces.shape == (6, 3)
        print(batch_count, len(train_loader), len(train_dataset))
        assert batch_count == len(train_loader) == len(train_dataset) / 2
        train_loader_direct = torch_geometric.dataloader.DataLoader(
            dataset=[
                AtomicData.from_config(config, z_table=self.table, cutoff=3.0)
                for config in datasets
            ],
            batch_size=2,
            shuffle=False,
            drop_last=False,
        )
        for batch_direct, batch in zip(train_loader_direct, train_loader):
            assert torch.all(batch_direct.edge_index == batch.edge_index)
            assert torch.all(batch_direct.shifts == batch.shifts)
            assert torch.all(batch_direct.positions == batch.positions)
            assert torch.all(batch_direct.node_attrs == batch.node_attrs)
            assert torch.all(batch_direct.energy == batch.energy)
            assert torch.all(batch_direct.forces == batch.forces)


class TestNeighborhood:
    def test_basic(self):
        positions = np.array(
            [
                [-1.0, 0.0, 0.0],
                [+0.0, 0.0, 0.0],
                [+1.0, 0.0, 0.0],
            ]
        )

        indices, shifts, unit_shifts, _ = get_neighborhood(positions, cutoff=1.5)
        assert indices.shape == (2, 4)
        assert shifts.shape == (4, 3)
        assert unit_shifts.shape == (4, 3)

    def test_signs(self):
        positions = np.array(
            [
                [+0.5, 0.5, 0.0],
                [+1.0, 1.0, 0.0],
            ]
        )

        cell = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        edge_index, shifts, unit_shifts, _ = get_neighborhood(
            positions, cutoff=3.5, pbc=(True, False, False), cell=cell
        )
        num_edges = 10
        assert edge_index.shape == (2, num_edges)
        assert shifts.shape == (num_edges, 3)
        assert unit_shifts.shape == (num_edges, 3)


# Based on mir-group/nequip
def test_periodic_edge():
    atoms = ase.build.bulk("Cu", "fcc")
    dist = np.linalg.norm(atoms.cell[0]).item()
    config = config_from_atoms(atoms)
    edge_index, shifts, _, _ = get_neighborhood(
        config.positions, cutoff=1.05 * dist, pbc=(True, True, True), cell=config.cell
    )
    sender, receiver = edge_index
    vectors = (
        config.positions[receiver] - config.positions[sender] + shifts
    )  # [n_edges, 3]
    assert vectors.shape == (12, 3)  # 12 neighbors in close-packed bulk
    assert np.allclose(
        np.linalg.norm(vectors, axis=-1),
        dist,
    )


def test_half_periodic():
    atoms = ase.build.fcc111("Al", size=(3, 3, 1), vacuum=0.0)
    assert all(atoms.pbc == (True, True, False))
    config = config_from_atoms(atoms)  # first shell dist is 2.864A
    # snapshot the cell: get_neighborhood must not mutate the caller's array,
    # and comparing against a post-call cell would be a no-op guard.
    cell_before = config.cell.copy()
    edge_index, shifts, _, cell = get_neighborhood(
        config.positions, cutoff=2.9, pbc=(True, True, False), cell=config.cell
    )
    assert np.allclose(config.cell, cell_before)  # input left untouched
    # periodic rows must be the physical lattice, not the matscipy blow-up
    assert np.allclose(cell[:2], cell_before[:2])
    # the non-periodic row had zero extent here; it must stay non-degenerate so
    # det(cell) / rcell downstream don't blow up (stress would NaN otherwise)
    assert not np.isclose(np.linalg.det(cell), 0.0)
    sender, receiver = edge_index
    vectors = (
        config.positions[receiver] - config.positions[sender] + shifts
    )  # [n_edges, 3]
    # Check number of neighbors:
    _, neighbor_count = np.unique(edge_index[0], return_counts=True)
    assert (neighbor_count == 6).all()  # 6 neighbors
    # Check not periodic in z
    assert np.allclose(
        vectors[:, 2],
        np.zeros(vectors.shape[0]),
    )


def test_nonperiodic_cell_is_extent_based():
    # The fictitious cell for non-periodic directions is sized from the atom
    # extent (+ cutoff padding), not from max(|positions|). Two copies of the
    # same molecule at different absolute positions must therefore get the SAME
    # cell -- the old max(|positions|)*5*cutoff formula grew with the coordinate
    # origin and blew up (OOM) k-space electrostatics for molecules far from 0.
    rng = np.random.default_rng(0)
    positions = rng.uniform(-2.0, 2.0, size=(6, 3))
    cutoff = 5.0
    _, _, _, cell_near = get_neighborhood(
        positions, cutoff=cutoff, pbc=(False, False, False)
    )
    _, _, _, cell_far = get_neighborhood(
        positions + 100.0, cutoff=cutoff, pbc=(False, False, False)
    )
    # origin-independent: same shape -> same cell regardless of absolute position
    assert np.allclose(cell_near, cell_far)
    # and the cell tracks the extent + cutoff padding (not |positions|)
    extent = positions.max(axis=0) - positions.min(axis=0)
    assert np.allclose(np.diag(cell_near), extent + 2 * cutoff + 1)


_NONORTHO_CELLS = {
    # in-plane periodic vectors are non-orthogonal; vacuum along the 3rd axis
    "hexagonal": np.array([[3.0, 0.0, 0.0], [1.5, 2.598, 0.0], [0.0, 0.0, 0.0]]),
    "monoclinic": np.array([[3.2, 0.0, 0.0], [0.8, 2.9, 0.0], [0.0, 0.0, 0.0]]),
    "triclinic": np.array([[3.1, 0.2, 0.1], [0.9, 2.8, -0.2], [0.0, 0.0, 0.0]]),
}


@pytest.mark.parametrize("cell_name", list(_NONORTHO_CELLS))
@pytest.mark.parametrize("cutoff", [3.5, 5.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_partial_pbc_neighbours_match_bruteforce_nonorthogonal(cell_name, cutoff, seed):
    """
    get_neighborhood replaces the non-periodic (vacuum) row of the cell with an
    axis-aligned vector for the matscipy binning. Verify this still yields the
    EXACT neighbour list on non-orthogonal slabs: the periodic vectors keep their
    true skewed directions and only the vacuum row is replaced. Checked across
    hexagonal / monoclinic / triclinic in-plane lattices, two cutoffs and several
    random configurations, edge-by-edge (indices + shift vectors) against a
    brute-force reference.
    """
    cell = _NONORTHO_CELLS[cell_name]
    pbc = (True, True, False)
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, 1.0, size=(6, 2)) @ cell[:2]  # in the periodic plane
    positions[:, 2] += rng.uniform(-0.3, 0.3, size=6)  # small slab thickness

    edge_index, _, unit_shifts, _ = get_neighborhood(
        positions, cutoff=cutoff, pbc=pbc, cell=cell
    )
    ref_index, _, ref_unit_shifts = brute_force_neighborhood(
        positions, cutoff, pbc, cell
    )
    assert_neighbourhoods_match(
        canonical_edges(edge_index, unit_shifts),
        canonical_edges(ref_index, ref_unit_shifts),
        context=f"{cell_name}, cutoff {cutoff}, seed {seed}",
    )


# ===========================================================================
# get_neighborhood: the contract is the edge set AND the returned cell
#
# The four returned values are (edge_index, shifts, unit_shifts, cell), and
# the last one is not a detail: AtomicData stores it, the stress divides by
# its determinant and the long-range models use it as their k-space box. It
# is *not* always the cell that was passed in, and it is *not* always the
# blown-up cell the search ran with either -- there are three regimes, pinned
# one test each below, because "return the physical cell" and "return the
# search cell" are each a plausible-looking simplification that breaks one of
# them.
# ===========================================================================


def _cluster_positions(n_atoms=8, spread=1.8, seed=20260810):
    return np.random.default_rng(seed).uniform(0.0, spread, size=(n_atoms, 3))


# --- the oracle itself has to be falsifiable -------------------------------
#
# Every case below is "matscipy agrees with the oracle". That is worth
# nothing if the oracle agrees with everything, so it is first checked
# against two counts known from crystallography and then shown to reject a
# list that is wrong by exactly one edge.


@pytest.mark.parametrize(
    "cell,expected_neighbours",
    [
        (np.eye(3) * 2.5, 6),  # simple cubic: 6 nearest neighbours
        (np.array([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]]), 12),  # fcc
    ],
)
def test_the_oracle_reproduces_known_coordination_numbers(cell, expected_neighbours):
    edge_index, _, _ = brute_force_neighborhood(
        np.zeros((1, 3)), cutoff=1.01 * np.linalg.norm(cell[0]), pbc=(True,) * 3, cell=cell
    )
    assert edge_index.shape[1] == expected_neighbours


def test_the_oracle_rejects_a_list_that_is_wrong_by_one_edge():
    positions = _cluster_positions(n_atoms=4)
    ref_index, _, ref_unit_shifts = brute_force_neighborhood(positions, 3.0)
    reference = canonical_edges(ref_index, ref_unit_shifts)
    assert_neighbourhoods_match(reference, reference)  # identical: silent
    with pytest.raises(AssertionError, match="missing"):
        assert_neighbourhoods_match(reference[:-1], reference)
    with pytest.raises(AssertionError, match="unexpected"):
        assert_neighbourhoods_match(reference + [(0, 0, 3, 3, 3)], reference)


def test_the_oracle_refuses_a_periodic_request_without_a_cell():
    with pytest.raises(ValueError, match="non-zero cell"):
        brute_force_neighborhood(np.zeros((1, 3)), 3.0, pbc=(True, False, False))


def test_canonical_edges_refuses_displacement_vectors():
    """The integer form and the vector form are different comparisons and
    mixing them silently would compare rounded floats against integers."""
    edge_index = np.array([[0], [0]])
    with pytest.raises(ValueError, match="must be integers"):
        canonical_edges(edge_index, np.array([[0.5, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="unit shifts"):
        canonical_edges(np.array([[0, 1], [1, 0]]), np.zeros((1, 3)))


def test_full_directed_edge_set_against_the_oracle():
    """Every pair inside the cutoff appears twice, once in each direction."""
    positions = _cluster_positions()
    cutoff = 3.0
    # every pair of these 8 atoms is inside the cutoff, so the directed list
    # is the complete digraph: 8 * 7 = 56 edges
    assert (
        np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1).max()
        < cutoff
    )
    edge_index, _, unit_shifts, _ = get_neighborhood(positions, cutoff=cutoff)
    assert edge_index.shape == (2, 56)

    pairs = {(int(i), int(j)) for i, j in edge_index.T}
    assert all((j, i) in pairs for i, j in pairs), "an edge is missing its mirror"
    assert not any(i == j for i, j in pairs)

    ref_index, _, ref_unit_shifts = brute_force_neighborhood(positions, cutoff)
    assert_neighbourhoods_match(
        canonical_edges(edge_index, unit_shifts),
        canonical_edges(ref_index, ref_unit_shifts),
    )


def test_periodic_self_images_survive_the_self_edge_filter():
    """Self-edges are dropped only at zero shift, so one atom in a cell
    smaller than the cutoff is its own neighbour in every image."""
    positions = np.zeros((1, 3))
    cell = np.eye(3) * 2.0
    pbc = (True, True, True)
    edge_index, _, unit_shifts, _ = get_neighborhood(
        positions, cutoff=3.0, pbc=pbc, cell=cell
    )
    # |n| * 2 Ang < 3 Ang for the 6 face and 12 edge images, not the 8 corners
    assert edge_index.shape == (2, 18)
    assert (edge_index[0] == edge_index[1]).all()
    assert not (np.abs(unit_shifts).sum(axis=1) == 0).any()

    ref_index, _, ref_unit_shifts = brute_force_neighborhood(
        positions, 3.0, pbc, cell
    )
    assert_neighbourhoods_match(
        canonical_edges(edge_index, unit_shifts),
        canonical_edges(ref_index, ref_unit_shifts),
    )


@pytest.mark.parametrize(
    "pbc,cell", [((False, False, False), None), ((True, True, True), np.eye(3) * 2.0)]
)
def test_true_self_interaction_is_dead(pbc, cell):
    """Characterization of a parameter that does nothing.

    `true_self_interaction=True` reads like "keep the zero-shift self edge",
    and it does skip the filter that removes it -- but the matscipy call is
    made without `self_interaction=True` (the argument is commented out at
    `neighborhood.py:66-68`), so that edge is never produced and there is
    nothing to filter. Both settings return the identical list. A port that
    implements the flag as its name suggests would change every edge count
    that goes through it, which is why the dead behaviour is pinned rather
    than assumed.
    """
    positions = np.zeros((1, 3))
    off = get_neighborhood(positions, cutoff=3.0, pbc=pbc, cell=cell)
    on = get_neighborhood(
        positions, cutoff=3.0, pbc=pbc, cell=cell, true_self_interaction=True
    )
    for a, b in zip(off, on):
        assert np.array_equal(a, b)
    # and in neither case is there a zero-shift self edge
    assert not (np.abs(on[2]).sum(axis=1) == 0).any()


def test_zero_edge_systems_are_legal():
    """An isolated pair beyond the cutoff returns empty arrays of the right
    shape rather than raising -- the padded-batch machinery depends on it."""
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    edge_index, shifts, unit_shifts, _ = get_neighborhood(positions, cutoff=2.0)
    assert edge_index.shape == (2, 0)
    assert shifts.shape == (0, 3)
    assert unit_shifts.shape == (0, 3)


def test_missing_or_all_zero_cell_becomes_the_identity_before_extension():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    _, _, _, from_none = get_neighborhood(positions, cutoff=2.0, cell=None)
    _, _, _, from_zeros = get_neighborhood(
        positions, cutoff=2.0, cell=np.zeros((3, 3))
    )
    assert np.array_equal(from_none, from_zeros)
    # identity rows, each then blown up along its own axis: extent + 2*cutoff + 1
    assert np.allclose(np.diag(from_none), [1.0 + 5.0, 0.0 + 5.0, 0.0 + 5.0])
    assert np.allclose(from_none - np.diag(np.diag(from_none)), 0.0)


# --- the three returned-cell regimes ---------------------------------------


def test_returned_cell_regime_aperiodic_is_the_extended_search_cell():
    """Fully aperiodic: the extended cell comes back, deliberately. Stress is
    meaningless without periodicity and the electrostatic models need a
    non-degenerate k-space box, so suppressing this in a port is a bug."""
    positions = _cluster_positions()
    cutoff = 3.0
    _, _, _, cell = get_neighborhood(positions, cutoff=cutoff)
    extent = positions.max(axis=0) - positions.min(axis=0)
    assert np.allclose(np.diag(cell), extent + 2 * cutoff + 1)
    assert np.allclose(cell - np.diag(np.diag(cell)), 0.0)
    assert not np.isclose(np.linalg.det(cell), 0.0)


def test_returned_cell_regime_periodic_is_the_physical_cell():
    """Any periodic axis: the physical cell comes back untouched, including
    the vacuum row. Returning the search cell here would rescale every slab
    stress by a fabricated volume."""
    physical = np.diag([4.0, 4.0, 12.0])
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.5]])
    _, _, _, cell = get_neighborhood(
        positions, cutoff=3.0, pbc=(True, True, False), cell=physical
    )
    assert np.array_equal(cell, physical)


def test_returned_cell_regime_periodic_with_a_zero_row_is_patched():
    """The one case where a row of the physical cell is replaced: an all-zero
    non-periodic row would make det(cell) zero, which NaNs the stress and
    blows up the reciprocal cell."""
    degenerate = np.diag([4.0, 4.0, 0.0])
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.5]])
    cutoff = 3.0
    _, _, _, cell = get_neighborhood(
        positions, cutoff=cutoff, pbc=(True, True, False), cell=degenerate
    )
    # the periodic rows are the physical ones ...
    assert np.array_equal(cell[:2], degenerate[:2])
    # ... and only the zero row is taken from the search cell
    extent_z = positions[:, 2].max() - positions[:, 2].min()
    assert np.allclose(cell[2], [0.0, 0.0, extent_z + 2 * cutoff + 1])
    assert not np.isclose(np.linalg.det(cell), 0.0)


def test_the_three_returned_cell_regimes_are_actually_three():
    """A port that collapsed any two of them would still pass every test
    above that it happened to keep. Assert the regimes differ from each
    other on the same geometry."""
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.5]])
    physical = np.diag([4.0, 4.0, 12.0])
    _, _, _, aperiodic = get_neighborhood(positions, cutoff=3.0, cell=physical)
    _, _, _, periodic = get_neighborhood(
        positions, cutoff=3.0, pbc=(True, True, False), cell=physical
    )
    _, _, _, patched = get_neighborhood(
        positions, cutoff=3.0, pbc=(True, True, False), cell=np.diag([4.0, 4.0, 0.0])
    )
    assert not np.allclose(aperiodic, periodic)
    assert not np.allclose(periodic, patched)
    assert not np.allclose(aperiodic, patched)


def test_shifts_are_unit_shifts_times_the_returned_cell():
    """The identity every consumer relies on, in the regime where the search
    cell and the returned cell differ. It holds because matscipy emits no
    shift along a non-periodic axis."""
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.5]])
    _, shifts, unit_shifts, cell = get_neighborhood(
        positions, cutoff=3.0, pbc=(True, True, False), cell=np.diag([4.0, 4.0, 0.0])
    )
    assert np.array_equal(shifts, unit_shifts @ cell)
    # nothing shifted along the aperiodic axis, which is what makes the
    # patched row harmless
    assert np.array_equal(unit_shifts[:, 2], np.zeros(unit_shifts.shape[0]))


@pytest.mark.parametrize(
    "pbc", [(True, False, False), (False, True, False), (False, False, True)]
)
def test_mixed_pbc_is_passed_through_per_axis(pbc):
    positions = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    cell = np.diag([2.0, 2.5, 3.0])
    edge_index, _, unit_shifts, _ = get_neighborhood(
        positions, cutoff=3.5, pbc=pbc, cell=cell
    )
    for dim, periodic in enumerate(pbc):
        if not periodic:
            assert np.array_equal(unit_shifts[:, dim], np.zeros(len(unit_shifts[:, dim])))
        else:
            assert np.abs(unit_shifts[:, dim]).sum() > 0
    ref_index, _, ref_unit_shifts = brute_force_neighborhood(positions, 3.5, pbc, cell)
    assert_neighbourhoods_match(
        canonical_edges(edge_index, unit_shifts),
        canonical_edges(ref_index, ref_unit_shifts),
        context=str(pbc),
    )


@pytest.mark.parametrize(
    "pbc,cell",
    [
        ((False, False, False), np.diag([4.0, 4.0, 4.0])),
        ((True, True, True), np.diag([4.0, 4.0, 4.0])),
        ((True, True, False), np.diag([4.0, 4.0, 12.0])),
        ((True, True, False), np.diag([4.0, 4.0, 0.0])),
    ],
)
def test_get_neighborhood_does_not_mutate_its_arguments(pbc, cell):
    """Purity is a property of the port, not an accident of the caller:
    AtomicData.from_config deep-copies before calling, so a mutation here
    would be invisible there and would surface only in a caller that does
    not. Every regime is checked, because the periodic branch and the
    aperiodic branch build their cell differently."""
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.5]])
    cell_before, positions_before = cell.copy(), positions.copy()
    _, _, _, returned = get_neighborhood(positions, cutoff=3.0, pbc=pbc, cell=cell)
    assert np.array_equal(cell, cell_before)
    assert np.array_equal(positions, positions_before)
    # and the returned cell is a distinct object, so mutating it downstream
    # cannot reach back into the caller's array
    returned += 1.0
    assert np.array_equal(cell, cell_before)


# --- the same three regimes, on the committed golden fixtures --------------
#
# tests/golden/fixtures exists to reach exactly these regimes (its manifest
# names the regime each structure was built for), and the anchors' stresses
# depend on the cell that comes back. Reading the regime out of the manifest
# rather than restating it here is what keeps the two suites pinning the same
# thing: a fixture retagged there fails here.


def _golden_fixtures():
    from tests.golden import harness  # noqa: PLC0415  (optional, tests-only)

    manifest = harness.load_manifest()
    with open(harness.MANIFEST_PATH, encoding="utf-8") as handle:
        r_max = json.load(handle)["r_max_hint"]
    return harness.load_fixtures(), manifest, r_max


#: Which returned-cell regime each committed fixture reaches. Written out
#: rather than derived, because deriving it from the same rule the function
#: uses would make the assertion circular.
_FIXTURE_REGIME = {
    "dimer_short": "extended",
    "isolated_atom": "extended",
    "water_cluster": "extended",
    "slab_vacuum": "physical",
    "triclinic_bulk": "physical",
    "slab_zero_vacuum": "patched",
    # The magnetic group, committed by the magnetic goldens. All five are
    # aperiodic clusters, so all five reach the extended-search-cell
    # regime; they are listed here rather than left out because the point
    # of the table is that no committed fixture reaches a regime nobody
    # wrote down.
    "mag_fe_atom": "extended",
    "mag_fe_dimer_fm": "extended",
    "mag_fe_dimer_afm": "extended",
    "mag_fe3_canted": "extended",
    "mag_feo_cluster": "extended",
}


def _tabled_fixtures():
    """The table entries this checkout actually has files for.

    The table names every fixture any golden family commits, but a given
    branch holds only its own: naming a fixture that is not here yet is a
    row waiting for it, not a failure. The assertion that matters is the
    other direction -- a fixture in the manifest with no row -- and that
    is checked below.
    """
    from tests.golden import harness  # noqa: PLC0415  (optional, tests-only)

    return sorted(set(_FIXTURE_REGIME) & set(harness.load_manifest()))


def test_the_fixture_regime_table_agrees_with_the_golden_manifest():
    """The manifest's `regime` field is prose for a human; this table is what
    the assertions below use. Where the prose names a cell regime, the two
    must say the same thing -- otherwise one suite is pinning a structure the
    other has since repurposed."""
    _, manifest, _ = _golden_fixtures()
    # Manifest-subset, not equality: see _tabled_fixtures. A fixture with
    # no row still fails here, which is the direction that matters.
    missing = set(manifest) - set(_FIXTURE_REGIME)
    assert not missing, (
        f"{sorted(missing)} are committed fixtures with no regime row. Add "
        "one: the table is what the neighbourhood oracle below is driven "
        "from, so a fixture missing from it is never checked at all."
    )
    for name in _tabled_fixtures():
        regime = _FIXTURE_REGIME[name]
        prose = manifest[name]["regime"]
        if "cell" not in prose:
            continue  # e.g. "short-range repulsion envelope": not a cell claim
        expected = {
            "extended": "extended search cell",
            "physical": "physical cell",
            "patched": "patched",
        }[regime]
        assert expected in prose, (name, prose)


@pytest.mark.parametrize("name", _tabled_fixtures())
def test_golden_fixture_neighbourhood_matches_the_oracle(name):
    fixtures, _, r_max = _golden_fixtures()
    atoms = fixtures[name]
    positions = atoms.get_positions()
    pbc = tuple(bool(p) for p in atoms.get_pbc())
    cell = np.array(atoms.get_cell())

    edge_index, shifts, unit_shifts, returned = get_neighborhood(
        positions, cutoff=r_max, pbc=pbc, cell=cell
    )
    ref_index, _, ref_unit_shifts = brute_force_neighborhood(
        positions, r_max, pbc, cell
    )
    assert_neighbourhoods_match(
        canonical_edges(edge_index, unit_shifts),
        canonical_edges(ref_index, ref_unit_shifts),
        context=name,
    )
    assert np.array_equal(shifts, unit_shifts @ returned)

    regime = _FIXTURE_REGIME[name]
    if regime == "extended":
        assert not any(pbc)
        extent = positions.max(axis=0) - positions.min(axis=0)
        assert np.allclose(np.diag(returned), extent + 2 * r_max + 1)
    elif regime == "physical":
        assert np.array_equal(returned, cell)
    else:
        assert np.array_equal(returned[:2], cell[:2])  # the periodic rows
        assert not cell[2].any() and returned[2].any()  # the patched one
    assert not np.isclose(np.linalg.det(returned), 0.0)


def test_oracle_and_matscipy_agree_on_displacement_vectors_too():
    """The integer comparison everywhere else would not catch a cell whose
    rows were permuted, since the unit shifts would permute with it. Compare
    the vectors once, on the triclinic fixture."""
    fixtures, _, r_max = _golden_fixtures()
    atoms = fixtures["triclinic_bulk"]
    positions, cell = atoms.get_positions(), np.array(atoms.get_cell())
    pbc = tuple(bool(p) for p in atoms.get_pbc())
    edge_index, shifts, _, _ = get_neighborhood(
        positions, cutoff=r_max, pbc=pbc, cell=cell
    )
    ref_index, ref_shifts, _ = brute_force_neighborhood(positions, r_max, pbc, cell)
    assert canonical_edges_from_shifts(
        edge_index, shifts
    ) == canonical_edges_from_shifts(ref_index, ref_shifts)


# ===========================================================================
# AtomicData: one-hot, shifts, Voigt expansion, and collation
# ===========================================================================


class TestAtomicDataContract:
    z_table = AtomicNumberTable([1, 6, 8])

    @staticmethod
    def water(cell_length=6.0, **overrides):
        properties = {
            "energy": -1.5,
            "forces": np.arange(9.0).reshape(3, 3),
        }
        properties.update(overrides)
        return Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [0.0, 0.95, 0.0]]),
            properties=properties,
            property_weights={"energy": 1.0, "forces": 1.0},
            cell=np.diag([cell_length] * 3),
            pbc=(True, True, True),
        )

    def test_one_hot_follows_the_z_table_not_the_atomic_number(self):
        data = AtomicData.from_config(self.water(), z_table=self.z_table, cutoff=3.0)
        # z_table is [1, 6, 8]; O is index 2, H is index 0
        assert data.node_attrs.shape == (3, 3)
        assert torch.equal(
            data.node_attrs,
            torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        assert torch.equal(data.atomic_numbers, torch.tensor([8, 1, 1]))
        # one-hot rows are exactly one hot
        assert torch.equal(data.node_attrs.sum(dim=1), torch.ones(3))

    def test_shifts_are_unit_shifts_times_the_stored_cell(self):
        # a cell smaller than the cutoff, so the identity is not vacuous
        data = AtomicData.from_config(
            self.water(cell_length=3.0), z_table=self.z_table, cutoff=4.0
        )
        assert data.cell.shape == (3, 3)
        assert data.unit_shifts.abs().sum() > 0
        assert torch.allclose(data.shifts, data.unit_shifts @ data.cell)

    def test_voigt_stress_and_virials_expand_at_parse_time(self):
        config = self.water(
            stress=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            virials=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        )
        data = AtomicData.from_config(config, z_table=self.z_table, cutoff=3.0)
        # Voigt order is (xx, yy, zz, yz, xz, xy) and the result is symmetric
        assert torch.equal(
            data.stress,
            torch.tensor([[[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]]]),
        )
        assert torch.allclose(
            data.virials,
            torch.tensor([[[0.1, 0.6, 0.5], [0.6, 0.2, 0.4], [0.5, 0.4, 0.3]]]),
        )
        assert data.stress.shape == (1, 3, 3)

    def test_graph_level_inputs_reach_the_graph(self):
        config = self.water(total_charge=1.0, total_spin=2.0, elec_temp=300.0)
        data = AtomicData.from_config(config, z_table=self.z_table, cutoff=3.0)
        assert data.total_charge.item() == 1.0
        assert data.total_spin.item() == 2.0
        assert data.elec_temp.item() == 300.0
        # the defaults are not all zero: an unspecified spin is 1.0
        plain = AtomicData.from_config(self.water(), z_table=self.z_table, cutoff=3.0)
        assert plain.total_charge.item() == 0.0
        assert plain.total_spin.item() == 1.0
        assert plain.elec_temp.item() == 0.0

    def test_collating_two_graphs_offsets_the_edge_index(self):
        first = AtomicData.from_config(self.water(), z_table=self.z_table, cutoff=4.0)
        second_config = Configuration(
            atomic_numbers=np.array([1, 1]),
            positions=np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
            properties={"energy": -0.5},
            property_weights={"energy": 1.0},
            cell=np.diag([5.0, 5.0, 5.0]),
            pbc=(True, True, True),
        )
        second = AtomicData.from_config(
            second_config, z_table=self.z_table, cutoff=4.0
        )
        batch = torch_geometric.batch.Batch.from_data_list([first, second])

        n_first = first.num_nodes
        assert batch.num_graphs == 2
        assert torch.equal(
            batch.batch, torch.tensor([0] * n_first + [1] * second.num_nodes)
        )
        assert torch.equal(
            batch.ptr, torch.tensor([0, n_first, n_first + second.num_nodes])
        )
        # the second graph's edges are the same edges, shifted by n_first
        assert torch.equal(
            batch.edge_index,
            torch.cat([first.edge_index, second.edge_index + n_first], dim=1),
        )
        # per-edge data concatenates in the same order
        assert torch.equal(
            batch.shifts, torch.cat([first.shifts, second.shifts], dim=0)
        )
        assert torch.equal(batch.energy, torch.stack([first.energy, second.energy]))

    def test_the_collated_cell_is_stacked_rows_and_views_back(self):
        """Characterization, not endorsement: a per-config cell is stored
        [3, 3] and collates by concatenation into [3 * n_graphs, 3], which
        every consumer un-flattens with .view(-1, 3, 3). The rewrite stores
        [n_graphs, 3, 3] instead -- value-identical modulo this reshape, which
        is why the values are pinned here and the storage accident is not."""
        first = AtomicData.from_config(self.water(), z_table=self.z_table, cutoff=3.0)
        second = AtomicData.from_config(self.water(), z_table=self.z_table, cutoff=3.0)
        batch = torch_geometric.batch.Batch.from_data_list([first, second])
        assert batch.cell.shape == (6, 3)
        assert torch.equal(
            batch.cell.view(-1, 3, 3), torch.stack([first.cell, second.cell])
        )
