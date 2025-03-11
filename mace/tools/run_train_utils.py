import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
from torch.utils.data import ConcatDataset

from mace import data
from mace.tools.scripts_utils import check_path_ase_read
from mace.tools.torch_geometric.dataset import Dataset
from mace.tools.utils import AtomicNumberTable


def normalize_file_paths(file_paths: Union[str, List[str]]) -> List[str]:
    """
    Normalize file paths to a list format.

    Args:
        file_paths: Either a string or a list of strings representing file paths

    Returns:
        A list of file paths
    """
    if isinstance(file_paths, str):
        return [file_paths]
    if isinstance(file_paths, list):
        return file_paths
    raise ValueError(f"Unexpected file paths format: {type(file_paths)}")


def load_dataset_for_path(
    file_path: Union[str, Path, List[str]],
    r_max: float,
    z_table: AtomicNumberTable,
    heads: List[str],
    head_config: Any,
    collection: Optional[Any] = None,
) -> Union[Dataset, List]:
    """
    Load a dataset from a file path based on its format.

    Args:
        file_path: Path to the dataset file
        r_max: Cutoff radius
        z_table: Atomic number table
        heads: List of head names
        head_name: Current head name
        **kwargs: Additional arguments

    Returns:
        Loaded dataset
    """
    if isinstance(file_path, list):
        if len(file_path) == 1:
            file_path = file_path[0]
    if isinstance(file_path, list):
        is_ase_readable = all(check_path_ase_read(p) for p in file_path)
        if not is_ase_readable:
            raise ValueError(
                "Not all paths in the list are ASE readable, not supported"
            )
    if isinstance(file_path, str):
        is_ase_readable = check_path_ase_read(file_path)

    if is_ase_readable:
        assert (
            collection is not None
        ), "Collection must be provided for ASE readable files"
        return [
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=r_max, heads=heads
            )
            for config in collection
        ]

    filepath = Path(file_path)
    if filepath.is_dir():

        if filepath.name.endswith("_lmdb") or any(
            f.endswith(".lmdb") or f.endswith(".aselmdb") for f in os.listdir(filepath)
        ):
            logging.info(f"Loading LMDB dataset from {file_path}")
            return data.LMDBDataset(
                file_path,
                r_max=r_max,
                z_table=z_table,
                heads=heads,
                head=head_config.head_name,
            )

        h5_files = list(filepath.glob("*.h5")) + list(filepath.glob("*.hdf5"))
        if h5_files:
            logging.info(f"Loading HDF5 dataset from directory {file_path}")
            try:
                return data.dataset_from_sharded_hdf5(
                    file_path,
                    r_max=r_max,
                    z_table=z_table,
                    heads=heads,
                    head=head_config.head_name,
                )
            except Exception as e:
                logging.error(f"Error loading sharded HDF5 dataset: {e}")
                raise

        if "lmdb" in str(filepath).lower() or "aselmdb" in str(filepath).lower():
            logging.info(f"Loading LMDB dataset based on path name: {file_path}")
            return data.LMDBDataset(
                file_path,
                r_max=r_max,
                z_table=z_table,
                heads=heads,
                head=head_config.head_name,
            )

        logging.info(f"Attempting to load directory as HDF5 dataset: {file_path}")
        try:
            return data.dataset_from_sharded_hdf5(
                file_path,
                r_max=r_max,
                z_table=z_table,
                heads=heads,
                head=head_config.head_name,
            )
        except Exception as e:
            logging.error(f"Error loading as sharded HDF5: {e}")
            raise

    suffix = filepath.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        logging.info(f"Loading single HDF5 file: {file_path}")
        return data.HDF5Dataset(
            file_path,
            r_max=r_max,
            z_table=z_table,
            heads=heads,
            head=head_config.head_name,
        )

    if suffix in (".lmdb", ".aselmdb", ".db"):
        logging.info(f"Loading single LMDB file: {file_path}")
        return data.LMDBDataset(
            file_path,
            r_max=r_max,
            z_table=z_table,
            heads=heads,
            head=head_config.head_name,
        )

    logging.info(f"Attempting to load as LMDB: {file_path}")
    return data.LMDBDataset(
        file_path,
        r_max=r_max,
        z_table=z_table,
        heads=heads,
        head=head_config.head_name,
    )


def combine_datasets(datasets, head_name):
    """
    Combine multiple datasets which might be of different types.

    Args:
        datasets: List of datasets (can be mixed types)
        head_name: Name of the current head

    Returns:
        Combined dataset
    """
    if not datasets:
        return []

    if all(isinstance(ds, list) for ds in datasets):
        logging.info(f"Combining {len(datasets)} list datasets for head '{head_name}'")
        return [item for sublist in datasets for item in sublist]

    if all(not isinstance(ds, list) for ds in datasets):
        logging.info(
            f"Combining {len(datasets)} Dataset objects for head '{head_name}'"
        )
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    logging.info(f"Converting mixed dataset types for head '{head_name}'")

    try:
        all_items = []
        for ds in datasets:
            if isinstance(ds, list):
                all_items.extend(ds)
            else:
                all_items.extend([ds[i] for i in range(len(ds))])
        return all_items
    except Exception as e:  # pylint: disable=W0703
        logging.warning(f"Failed to convert mixed datasets to list: {e}")

    try:
        dataset_objects = []
        for ds in datasets:
            if isinstance(ds, list):
                from torch.utils.data import TensorDataset

                # Convert list to a Dataset
                dataset_objects.append(
                    TensorDataset(*[torch.tensor([i]) for i in range(len(ds))])
                )
            else:
                dataset_objects.append(ds)
        return ConcatDataset(dataset_objects)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(f"Failed to convert mixed datasets to ConcatDataset: {e}")

    logging.warning(
        "Could not combine datasets of different types. Using only the first dataset."
    )
    return datasets[0]
