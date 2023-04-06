import h5py
import torch
from torch.utils.data import Dataset
from mace import data
from mace.data import AtomicData
from mace.data.utils import Configuration

import h5py
import torch
from torch.utils.data import Dataset, IterableDataset, ChainDataset
from mace import data
from mace.data import AtomicData
from mace.data.utils import Configuration


class HDF5ChainDataset(ChainDataset):
    def __init__(self, file, r_max, z_table, **kwargs):
        super(HDF5ChainDataset, self).__init__()
        self.file = file
        self.length = len(h5py.File(file, "r").keys())
        self.r_max = r_max
        self.z_table = z_table

    def __call__(self):
        self.file = h5py.File(self.file, "r")
        datasets = []
        for i in range(self.length):
            grp = self.file["config_" + str(i)]
            datasets.append(
                HDF5IterDataset(iter_group=grp, r_max=self.r_max, z_table=self.z_table,)
            )
        return ChainDataset(datasets)


class HDF5IterDataset(IterableDataset):
    def __init__(self, iter_group, r_max, z_table, **kwargs):
        super(HDF5IterDataset, self).__init__()
        # it might be dangerous to open the file here
        # move opening of file to __getitem__?
        self.iter_group = iter_group
        self.length = len(self.iter_group.keys())
        self.r_max = r_max
        self.z_table = z_table
        # self.file = file
        # self.length = len(h5py.File(file, 'r').keys())

    def __len__(self):
        return self.length

    def __iter__(self):
        # file = h5py.File(self.file, 'r')
        # grp = file["config_" + str(index)]
        grp = self.iter_group
        len_subgrp = len(grp.keys())
        grp_list = []
        for i in range(len_subgrp):
            subgrp = grp["config_" + str(i)]
            config = Configuration(
                atomic_numbers=subgrp["atomic_numbers"][()],
                positions=subgrp["positions"][()],
                energy=subgrp["energy"][()],
                forces=subgrp["forces"][()],
                stress=subgrp["stress"][()],
                virials=subgrp["virials"][()],
                dipole=subgrp["dipole"][()],
                charges=subgrp["charges"][()],
                weight=subgrp["weight"][()],
                energy_weight=subgrp["energy_weight"][()],
                forces_weight=subgrp["forces_weight"][()],
                stress_weight=subgrp["stress_weight"][()],
                virials_weight=subgrp["virials_weight"][()],
                config_type=subgrp["config_type"][()],
                pbc=subgrp["pbc"][()],
                cell=subgrp["cell"][()],
            )
            atomic_data = data.AtomicData.from_config(
                config, z_table=self.z_table, cutoff=self.r_max
            )
            grp_list.append(atomic_data)

        return iter(grp_list)


class HDF5Dataset(Dataset):
    def __init__(self, file, r_max, z_table, **kwargs):
        super(HDF5Dataset, self).__init__()
        self.file = h5py.File(file, "r")  # this is dangerous to open the file here
        self.batch_size = len(self.file["config_0"].keys())
        self.length = len(self.file.keys()) * len(self.file["config_0"].keys())
        self.r_max = r_max
        self.z_table = z_table

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # compute the index of the batch
        batch_index = index // self.batch_size
        config_index = index % self.batch_size
        grp = self.file["config_batch" + str(batch_index)]
        subgrp = grp["config_" + str(config_index)]
        config = Configuration(
            atomic_numbers=subgrp["atomic_numbers"][()],
            positions=subgrp["positions"][()],
            energy=subgrp["energy"][()],
            forces=subgrp["forces"][()],
            stress=subgrp["stress"][()],
            virials=subgrp["virials"][()],
            dipole=subgrp["dipole"][()],
            charges=subgrp["charges"][()],
            weight=subgrp["weight"][()],
            energy_weight=subgrp["energy_weight"][()],
            forces_weight=subgrp["forces_weight"][()],
            stress_weight=subgrp["stress_weight"][()],
            virials_weight=subgrp["virials_weight"][()],
            config_type=subgrp["config_type"][()],
            pbc=subgrp["pbc"][()],
            cell=subgrp["cell"][()],
        )
        atomic_data = data.AtomicData.from_config(
            config, z_table=self.z_table, cutoff=self.r_max
        )
        return atomic_data

