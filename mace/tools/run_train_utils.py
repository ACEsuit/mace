import ast
import logging
import os
from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import ConcatDataset

from mace import data
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
    file_path: str,
    r_max: float,
    z_table: AtomicNumberTable,
    heads: List[str],
    head_name: str,
    **kwargs,
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
    filepath = Path(file_path)

    # Handle XYZ files
    if filepath.suffix.lower() in [".xyz", ".extxyz"]:
        logging.info(f"Loading XYZ file dataset: {file_path}")

        config_type_weights = kwargs.get("config_type_weights", {"Default": 1.0})
        if isinstance(config_type_weights, str):
            config_type_weights = ast.literal_eval(config_type_weights)

        try:
            _, configs = data.load_from_xyz(
                file_path=file_path,
                config_type_weights=config_type_weights,
                energy_key=kwargs.get("energy_key", "REF_energy"),
                forces_key=kwargs.get("forces_key", "REF_forces"),
                stress_key=kwargs.get("stress_key", "REF_stress"),
                virials_key=kwargs.get("virials_key", "REF_virials"),
                dipole_key=kwargs.get("dipole_key", "REF_dipole"),
                charges_key=kwargs.get("charges_key", "REF_charges"),
                head_key="head",
                head_name=head_name,
                extract_atomic_energies=False,
                keep_isolated_atoms=kwargs.get("keep_isolated_atoms", False),
            )

            return [
                data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=r_max, heads=heads
                )
                for config in configs
            ]
        except Exception as e:
            logging.error(f"Error processing XYZ file {file_path}: {e}")
            raise

    # Handle directories
    if filepath.is_dir():

        if filepath.name.endswith("_lmdb") or any(
            f.endswith(".lmdb") for f in os.listdir(filepath)
        ):
            logging.info(f"Loading LMDB dataset from {file_path}")
            return data.LMDBDataset(
                file_path, r_max=r_max, z_table=z_table, heads=heads, head=head_name
            )

        h5_files = list(filepath.glob("*.h5")) + list(filepath.glob("*.hdf5"))
        if h5_files:
            logging.info(f"Loading HDF5 dataset from directory {file_path}")
            try:
                return data.dataset_from_sharded_hdf5(
                    file_path, r_max=r_max, z_table=z_table, heads=heads, head=head_name
                )
            except Exception as e:
                logging.error(f"Error loading sharded HDF5 dataset: {e}")
                raise

        if "lmdb" in str(filepath).lower():
            logging.info(f"Loading LMDB dataset based on path name: {file_path}")
            return data.LMDBDataset(
                file_path, r_max=r_max, z_table=z_table, heads=heads, head=head_name
            )

        logging.info(f"Attempting to load directory as HDF5 dataset: {file_path}")
        try:
            return data.dataset_from_sharded_hdf5(
                file_path, r_max=r_max, z_table=z_table, heads=heads, head=head_name
            )
        except Exception as e:
            logging.error(f"Error loading as sharded HDF5: {e}")
            raise

    suffix = filepath.suffix.lower()
    if suffix in (".h5", ".hdf5"):
        logging.info(f"Loading single HDF5 file: {file_path}")
        return data.HDF5Dataset(
            file_path, r_max=r_max, z_table=z_table, heads=heads, head=head_name
        )

    logging.info(f"Attempting to load as sharded HDF5: {file_path}")
    return data.dataset_from_sharded_hdf5(
        file_path, r_max=r_max, z_table=z_table, heads=heads, head=head_name
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
