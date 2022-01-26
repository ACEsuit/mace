# pylint: disable=no-self-use
import numpy as np
from LieACE.data.atomic_data import AtomicData, get_data_loader

from LieACE.data.utils import Configuration
from LieACE.tools.utils import AtomicNumberTable

config = Configuration(
    atomic_numbers=np.array([1,1, 1]),
    positions=np.array([
        [0.0, -2.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]),
    forces=np.array([
        [0.0, -1.3, 0.0],
        [1.0, 0.2, 0.0],
        [0.0, 1.1, 0.3],
    ]),
    energy=-1.5,
    degree = [{'type' : 'NaiveMaxDeg',1:[4,2],2:[4,2]}]
)

table = AtomicNumberTable([1])


class TestAtomicData:
    def test_atomic_data(self):
        data = AtomicData.from_config(config, z_table=table, cutoff=3.0)

        assert data.edge_index.shape == (2, 4)
        assert data.forces.shape == (3, 3)
        assert data.node_attrs.shape == (3, 2)

    def test_collate(self):
        data1 = AtomicData.from_config(config, z_table=table, cutoff=3.0)
        data2 = AtomicData.from_config(config, z_table=table, cutoff=3.0)

        data_loader = get_data_loader([data1, data2], batch_size=32)

        for i in data_loader:
            print(i)