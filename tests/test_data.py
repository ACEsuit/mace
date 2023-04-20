import ase.build
import numpy as np

from mace.data import AtomicData, Configuration, config_from_atoms, get_neighborhood
from mace.tools import AtomicNumberTable, torch_geometric


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
        forces=np.array(
            [
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
                [0.0, 1.1, 0.3],
            ]
        ),
        energy=-1.5,
    )

    table = AtomicNumberTable([1, 8])

    def test_atomic_data(self):
        data = AtomicData.from_config(self.config, z_table=self.table, cutoff=3.0)

        assert data.edge_index.shape == (2, 4)
        assert data.forces.shape == (3, 3)
        assert data.node_attrs.shape == (3, 2)

    def test_data_loader(self):
        data1 = AtomicData.from_config(self.config, z_table=self.table, cutoff=3.0)
        data2 = AtomicData.from_config(self.config, z_table=self.table, cutoff=3.0)

        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[data1, data2],
            batch_size=2,
            shuffle=True,
            drop_last=False,
        )

        for batch in data_loader:
            assert batch.batch.shape == (6,)
            assert batch.edge_index.shape == (2, 8)
            assert batch.shifts.shape == (8, 3)
            assert batch.positions.shape == (6, 3)
            assert batch.node_attrs.shape == (6, 2)
            assert batch.energy.shape == (2,)
            assert batch.forces.shape == (6, 3)

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


class TestNeighborhood:
    def test_basic(self):
        positions = np.array(
            [
                [-1.0, 0.0, 0.0],
                [+0.0, 0.0, 0.0],
                [+1.0, 0.0, 0.0],
            ]
        )

        indices, shifts, unit_shifts = get_neighborhood(positions, cutoff=1.5)
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
        edge_index, shifts, unit_shifts = get_neighborhood(
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
    edge_index, shifts, _ = get_neighborhood(
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
    edge_index, shifts, _ = get_neighborhood(
        config.positions, cutoff=2.9, pbc=(True, True, False), cell=config.cell
    )
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
