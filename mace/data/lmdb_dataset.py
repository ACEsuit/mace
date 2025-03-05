import os

import numpy as np
from torch.utils.data import Dataset

from mace.data.atomic_data import AtomicData
from mace.data.utils import Configuration
from mace.tools.fairchem_dataset import AseDBDataset


class LMDBDataset(Dataset):
    def __init__(self, file_path, r_max, z_table, **kwargs):
        dataset_paths = file_path.split(":")  # using : split multiple paths
        # make sure each of the path exist
        for path in dataset_paths:
            assert os.path.exists(path)
        config_kwargs = {}
        super(LMDBDataset, self).__init__()  # pylint: disable=super-with-arguments
        self.AseDB = AseDBDataset(config=dict(src=dataset_paths, **config_kwargs))
        self.r_max = r_max
        self.z_table = z_table

        self.kwargs = kwargs
        self.transform = kwargs["transform"] if "transform" in kwargs else None

    def __len__(self):
        return len(self.AseDB)

    def __getitem__(self, index):
        try:
            atoms = self.AseDB.get_atoms(self.AseDB.ids[index])
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error in index {index}")
            print(e)
            return None
        assert np.sum(atoms.get_cell() == atoms.cell) == 9

        config = Configuration(
            atomic_numbers=atoms.numbers,
            positions=atoms.positions,
            energy=atoms.calc.results["energy"],
            forces=atoms.calc.results["forces"],
            stress=atoms.calc.results["stress"],
            virials=np.zeros(atoms.get_stress().shape),
            dipole=np.zeros(atoms.get_forces()[0].shape),
            charges=np.zeros(atoms.numbers.shape),
            weight=1.0,
            head=None,  # do not asign head according to h5
            energy_weight=1.0,
            forces_weight=1.0,
            stress_weight=1.0,
            virials_weight=1.0,
            config_type=None,
            pbc=np.array(atoms.pbc),
            cell=np.array(atoms.cell),
        )
        if config.head is None:
            config.head = self.kwargs.get("head")
        try:
            atomic_data = AtomicData.from_config(
                config,
                z_table=self.z_table,
                cutoff=self.r_max,
                heads=self.kwargs.get("heads", ["Default"]),
            )
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error in index {index}")
            print(e)

        if self.transform:
            atomic_data = self.transform(atomic_data)
        return atomic_data
