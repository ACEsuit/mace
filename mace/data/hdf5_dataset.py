import h5py
import torch
from torch.utils.data import Dataset
from mace.data import AtomicData

class HDF5Dataset(Dataset):
    def __init__(self, file, **kwargs):
        super(HDF5Dataset, self).__init__()
        # it might be dangerous to open the file here
        # move opening of file to __getitem__?
        self.file = h5py.File(file, 'r')  
        self.length = len(self.file.keys())
        # self.file = file
        # self.length = len(h5py.File(file, 'r').keys())

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # file = h5py.File(self.file, 'r')  
        # grp = file["config_" + str(index)] 
        grp = self.file["config_" + str(index)] 
        edge_index = grp['edge_index'][()]
        positions = grp['positions'][()]
        shifts = grp['shifts'][()]
        unit_shifts = grp['unit_shifts'][()]
        cell = grp['cell'][()]
        node_attrs = grp['node_attrs'][()]
        weight = grp['weight'][()]
        energy_weight = grp['energy_weight'][()]
        forces_weight = grp['forces_weight'][()]
        stress_weight = grp['stress_weight'][()]
        virials_weight = grp['virials_weight'][()]
        forces = grp['forces'][()]
        energy = grp['energy'][()]
        stress = grp['stress'][()]
        virials = grp['virials'][()]
        dipole = grp['dipole'][()]
        charges = grp['charges'][()]
        return AtomicData( 
            edge_index =  torch.tensor(edge_index, dtype=torch.long),
            positions = torch.tensor(positions, dtype=torch.get_default_dtype()),
            shifts = torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
            cell=torch.tensor(cell, dtype=torch.get_default_dtype()),
            node_attrs=torch.tensor(node_attrs, dtype=torch.get_default_dtype()),
            weight=torch.tensor(weight, dtype=torch.get_default_dtype()),
            energy_weight=torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
            forces_weight=torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
            stress_weight=torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
            virials_weight=torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
            forces=torch.tensor(forces, dtype=torch.get_default_dtype()),
            energy=torch.tensor(energy, dtype=torch.get_default_dtype()),
            stress=torch.tensor(stress, dtype=torch.get_default_dtype()),
            virials=torch.tensor(virials, dtype=torch.get_default_dtype()),
            dipole=torch.tensor(dipole, dtype=torch.get_default_dtype()),
            charges=torch.tensor(charges, dtype=torch.get_default_dtype()),
        )
      