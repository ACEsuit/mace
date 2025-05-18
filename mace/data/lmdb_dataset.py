import os

import numpy as np
from torch.utils.data import Dataset

from mace.data.atomic_data import AtomicData
from mace.data.utils import KeySpecification, config_from_atoms
from mace.tools.default_keys import DefaultKeys
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

        if hasattr(atoms, "calc") and hasattr(atoms.calc, "results"):
            if "energy" in atoms.calc.results:
                atoms.info[DefaultKeys.ENERGY.value] = atoms.calc.results["energy"]
            if "forces" in atoms.calc.results:
                atoms.arrays[DefaultKeys.FORCES.value] = atoms.calc.results["forces"]
            if "stress" in atoms.calc.results:
                atoms.info[DefaultKeys.STRESS.value] = atoms.calc.results["stress"]

        config = config_from_atoms(
            atoms,
            key_specification=KeySpecification.from_defaults(),
        )

        # Set head if not already set
        if config.head == "Default":
            config.head = self.kwargs.get("head", "Default")

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
