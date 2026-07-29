from copy import deepcopy
from pathlib import Path

import ase.build
import h5py
import numpy as np
import pytest

from tests.helpers import REPO_ROOT
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


def _edge_multiset(idx_pairs, shift_vecs):
    return sorted(
        (
            int(i),
            int(j),
            round(float(s[0]), 4),
            round(float(s[1]), 4),
            round(float(s[2]), 4),
        )
        for (i, j), s in zip(idx_pairs, shift_vecs)
    )


def _bruteforce_neighbours(positions, cell, pbc, cutoff):
    # Reference neighbour list: enumerate periodic images along the periodic axes
    # (none along non-periodic ones) and keep pairs within cutoff. The image range
    # is grown until the edge set stops changing, so the reference is complete no
    # matter how skewed the cell is (a fixed range could miss images for a very
    # non-orthogonal cell).
    prev, radius = None, 2
    while True:
        ranges = [range(-radius, radius + 1) if pbc[d] else range(1) for d in range(3)]
        pairs, shift_vecs = [], []
        for i in range(len(positions)):
            for j in range(len(positions)):
                for n0 in ranges[0]:
                    for n1 in ranges[1]:
                        for n2 in ranges[2]:
                            if i == j and n0 == 0 and n1 == 0 and n2 == 0:
                                continue
                            shift = n0 * cell[0] + n1 * cell[1] + n2 * cell[2]
                            if (
                                np.linalg.norm(positions[j] + shift - positions[i])
                                < cutoff
                            ):
                                pairs.append((i, j))
                                shift_vecs.append(shift)
        edges = _edge_multiset(pairs, shift_vecs)
        if edges == prev:
            return edges
        prev, radius = edges, radius + 1


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

    edge_index, shifts, _, _ = get_neighborhood(
        positions, cutoff=cutoff, pbc=pbc, cell=cell
    )
    got = _edge_multiset(edge_index.T, shifts)
    ref = _bruteforce_neighbours(positions, cell, pbc, cutoff)
    assert got == ref
