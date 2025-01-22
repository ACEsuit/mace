from mace.data.atomic_data import AtomicData
from mace.data.utils import Configuration, config_from_atoms
from fairchem.core.datasets import AseDBDataset
from torch.utils.data import Dataset
from mace.tools.utils import AtomicNumberTable
from ase.io.extxyz import save_calc_results
import numpy as np
import os

class LMDBDataset(Dataset):
    def __init__(self, file_path, r_max, z_table, **kwargs):
        dataset_paths = file_path.split(":") # using : split multiple paths
        # make sure each of the path exist
        for path in dataset_paths:
            assert os.path.exists(path)
        config_kwargs = {}
        super(LMDBDataset, self).__init__() # pylint: disable=super-with-arguments
        self.AseDB = AseDBDataset(config=dict(src=dataset_paths, **config_kwargs))
        self.r_max = r_max
        self.z_table = z_table

        self.kwargs = kwargs
        self.transform = kwargs['transform'] if 'transform' in kwargs else None

    def __len__(self):
        return len(self.AseDB)

    #def __getitem__(self, indices):
    #    single_datum = False
    #    if isinstance(indices, int):  # Handle single index case for compatibility
    #        indices = [indices]
    #        single_datum = True
    #    
    #    atomic_data_list = []
    #    for index in indices:
    #        try:
    #            atoms = self.AseDB.get_atoms(self.AseDB.ids[index])
    #        except Exception as e:
    #            import ipdb; ipdb.set_trace()
    #            print("Error at index:", index)
    #            print("Total IDs:", len(self.AseDB.ids))
    #            raise e

    #        assert np.sum(atoms.get_cell() == atoms.cell) == 9

    #        config = Configuration(
    #            atomic_numbers=atoms.numbers,
    #            positions=atoms.positions,
    #            energy=atoms.calc.results['energy'],
    #            forces=atoms.calc.results['forces'],
    #            stress=atoms.calc.results['stress'],
    #            virials=np.zeros(atoms.get_stress().shape),
    #            dipole=np.zeros(atoms.get_forces()[0].shape),
    #            charges=np.zeros(atoms.numbers.shape),
    #            weight=1.0,
    #            head=None,
    #            energy_weight=1.0,
    #            forces_weight=1.0,
    #            stress_weight=1.0,
    #            virials_weight=1.0,
    #            config_type=None,
    #            pbc=np.array(atoms.pbc),
    #            cell=np.array(atoms.cell),
    #            alex_config_id=None,
    #        )

    #        if config.head is None:
    #            config.head = self.kwargs.get("head")
    #        
    #        try:
    #            atomic_data = AtomicData.from_config(
    #                config,
    #                z_table=self.z_table,
    #                cutoff=self.r_max,
    #                heads=self.kwargs.get("heads", ["Default"]),
    #            )
    #        except Exception as e:
    #            import ipdb; ipdb.set_trace()
    #            raise e

    #        if self.transform:
    #            atomic_data = self.transform(atomic_data)
    #        
    #        atomic_data_list.append(atomic_data)

    #    if single_datum:
    #        return atomic_data_list[0]

    #    return atomic_data_list 
    def __getitem__(self, index):
        try:
            atoms = self.AseDB.get_atoms(self.AseDB.ids[index])
        except:
            import ipdb; ipdb.set_trace()
            print(index)
            print(len(self.AseDB.ids))
            raise NotImplementedError

        assert np.sum(atoms.get_cell() == atoms.cell) == 9

        #import ipdb; ipdb.set_trace()
        config = Configuration(
            atomic_numbers=atoms.numbers,
            positions=atoms.positions,
            energy=atoms.calc.results['energy'],
            forces=atoms.calc.results['forces'],
            stress=atoms.calc.results['stress'],
            virials=np.zeros(atoms.get_stress().shape),
            dipole=np.zeros(atoms.get_forces()[0].shape),
            charges=np.zeros(atoms.numbers.shape),
            weight=1.0,
            head=None, # do not asign head according to h5
            energy_weight=1.0,
            forces_weight=1.0,
            stress_weight=1.0,
            virials_weight=1.0,
            config_type=None,
            pbc=np.array(atoms.pbc),
            cell=np.array(atoms.cell),
            alex_config_id=None,
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
        except Exception as e:
            import ipdb; ipdb.set_trace()
            raise e

        if self.transform:
            atomic_data = self.transform(atomic_data)
        return atomic_data

if __name__ == "__main__":
    db = LMDBDataset(None, 5.0, AtomicNumberTable(range(1, 120)))
    print(db[0])

    from mace.tools import torch_geometric 
    loader = torch_geometric.dataloader.DataLoader(
        db, batch_size=128, num_workers=12, shuffle=False
    )
    for b in loader:
        print(b)
