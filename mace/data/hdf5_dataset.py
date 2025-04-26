from glob import glob
from typing import List

import h5py
from torch.utils.data import ConcatDataset, Dataset

from mace.data.atomic_data import AtomicData
from mace.data.utils import Configuration
from mace.tools.utils import AtomicNumberTable


class HDF5Dataset(Dataset):
    def __init__(
        self, file_path, r_max, z_table, atomic_dataclass=AtomicData, **kwargs
    ):
        super(HDF5Dataset, self).__init__()  # pylint: disable=super-with-arguments
        self.file_path = file_path
        self._file = None
        batch_key = list(self.file.keys())[0]
        self.batch_size = len(self.file[batch_key].keys())
        self.length = len(self.file.keys()) * self.batch_size
        self.r_max = r_max
        self.z_table = z_table
        self.atomic_dataclass = atomic_dataclass
        try:
            self.drop_last = bool(self.file.attrs["drop_last"])
        except KeyError:
            self.drop_last = False
        self.kwargs = kwargs

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

        properties = {}
        property_weights = {}
        for key in subgrp["properties"]:
            properties[key] = unpack_value(subgrp["properties"][key][()])
        for key in subgrp["property_weights"]:
            property_weights[key] = unpack_value(subgrp["property_weights"][key][()])

        config = Configuration(
            atomic_numbers=subgrp["atomic_numbers"][()],
            positions=subgrp["positions"][()],
            properties=properties,
            weight=unpack_value(subgrp["weight"][()]),
            property_weights=property_weights,
            config_type=unpack_value(subgrp["config_type"][()]),
            pbc=unpack_value(subgrp["pbc"][()]),
            cell=unpack_value(subgrp["cell"][()]),
        )
        if config.head is None:
            config.head = self.kwargs.get("head")
        atomic_data = self.atomic_dataclass.from_config(
            config,
            z_table=self.z_table,
            cutoff=self.r_max,
            heads=self.kwargs.get("heads", ["Default"]),
            **{k: v for k, v in self.kwargs.items() if k != "heads"},
        )
        return atomic_data


def dataset_from_sharded_hdf5(
    files: List, z_table: AtomicNumberTable, r_max: float, **kwargs
):
    files = glob(files + "/*")
    datasets = []
    for file in files:
        datasets.append(HDF5Dataset(file, z_table=z_table, r_max=r_max, **kwargs))
    full_dataset = ConcatDataset(datasets)
    return full_dataset


def unpack_value(value):
    value = value.decode("utf-8") if isinstance(value, bytes) else value
    return None if str(value) == "None" else value
