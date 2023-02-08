import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from mace.data import AtomicData

from mace.tools import torch_geometric

# Define the custom HDF5 dataset
class HDF5Dataset(Dataset):
    def __init__(self, file):
        self.file = h5py.File(file, 'r')

    def __len__(self):
        return len(self.file.keys())

    def __getitem__(self, index):
        grp = self.file["config_" + str(index.item())]
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
    
# AtomicData_dict = { 
#         "edge_index" : torch.tensor(edge_index, dtype=torch.long),
#     "positions" : torch.tensor(positions, dtype=torch.get_default_dtype()),
#     "shifts" : torch.tensor(shifts, dtype=torch.get_default_dtype()),
#     "unit_shifts" :torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
#     "cell" : cell,
#     "node_attrs" : node_attrs,
#     "weight" : weight,
#     "energy_weight" : energy_weight,
#     "forces_weight" : forces_weight,
#     "stress_weight" : stress_weight,
#     "virials_weight" : virials_weight,
#     "forces" : forces,
#     "energy" : energy,
#     "stress" : stress,
#     "virials" : virials,
#     "dipole" : dipole,
#     "charges" : charges,
# }
    

class HDF5DataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.kwargs = kwargs
        
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            **kwargs,
        )

    def __iter__(self):
        indices = torch.randperm(len(self.dataset)) if self.shuffle else torch.arange(len(self.dataset))
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = torch_geometric.Batch.from_data_list([self.dataset[j] for j in batch_indices])
            yield batch