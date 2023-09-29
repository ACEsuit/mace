import h5py
import torch
from torch.utils.data import Dataset, IterableDataset, ChainDataset
from mace import data
from mace.data.utils import Configuration
from torch.utils.data import ConcatDataset
from glob import glob
from typing import List
from mace.tools.utils import AtomicNumberTable


class HDF5ChainDataset(ChainDataset):
    def __init__(self, file_path, r_max, z_table, **kwargs):
        super(HDF5ChainDataset, self).__init__()
        self.file_path = file_path
        self._file = None

        self.length = len(self.file.keys())
        self.r_max = r_max
        self.z_table = z_table

    @property
    def file(self):
        if self._file is None:
            # If a file has not already been opened, open one here
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __getstate__(self):
        _d = dict(self.__dict__)
        
        # An opened h5py.File cannot be pickled, so we must exclude it from the state
        _d["_file"] = None
        return _d

    def __call__(self):
        datasets = []
        for i in range(self.length):
            grp = self.file["config_" + str(i)]
            datasets.append(
                HDF5IterDataset(
                    iter_group=grp,
                    r_max=self.r_max,
                    z_table=self.z_table,
                )
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
    def __init__(self, file_path, r_max, z_table, **kwargs):
        super(HDF5Dataset, self).__init__()
        self.file_path = file_path
        self._file = None
        batch_key = list(self.file.keys())[0]
        self.batch_size = len(self.file[batch_key].keys())
        self.length = len(self.file.keys()) * self.batch_size
        self.r_max = r_max
        self.z_table = z_table
        try:
            self.drop_last = bool(self.file.attrs["drop_last"])
        except KeyError:
            self.drop_last = False

    @property
    def file(self):
        if self._file is None:
            # If a file has not already been opened, open one here
            self._file = h5py.File(self.file_path, "r")
        return self._file

    def __getstate__(self):
        _d = dict(self.__dict__)

        # An opened h5py.File cannot be pickled, so we must exclude it from the state
        _d["_file"] = None
        return _d

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # compute the index of the batch
        batch_index = index // self.batch_size
        config_index = index % self.batch_size
        grp = self.file["config_batch_" + str(batch_index)]
        subgrp = grp["config_" + str(config_index)]
        config = Configuration(
            atomic_numbers=subgrp["atomic_numbers"][()],
            positions=subgrp["positions"][()],
            energy=unpack_value(subgrp["energy"][()]),
            forces=unpack_value(subgrp["forces"][()]),
            stress=unpack_value(subgrp["stress"][()]),
            virials=unpack_value(subgrp["virials"][()]),
            dipole=unpack_value(subgrp["dipole"][()]),
            charges=unpack_value(subgrp["charges"][()]),
            weight=unpack_value(subgrp["weight"][()]),
            energy_weight=unpack_value(subgrp["energy_weight"][()]),
            forces_weight=unpack_value(subgrp["forces_weight"][()]),
            stress_weight=unpack_value(subgrp["stress_weight"][()]),
            virials_weight=unpack_value(subgrp["virials_weight"][()]),
            config_type=unpack_value(subgrp["config_type"][()]),
            pbc=unpack_value(subgrp["pbc"][()]),
            cell=unpack_value(subgrp["cell"][()]),
        )
        atomic_data = data.AtomicData.from_config(
            config, z_table=self.z_table, cutoff=self.r_max
        )
        return atomic_data

def dataset_from_sharded_hdf5(files: List, z_table: AtomicNumberTable, r_max: float):
    files = glob(files+'/*')
    datasets = []
    for file in files:
        datasets.append(data.HDF5Dataset(file, z_table=z_table, r_max=r_max))
    full_dataset = ConcatDataset(datasets)
    return full_dataset

def unpack_value(value):
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value
