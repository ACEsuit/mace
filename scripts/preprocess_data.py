# This file loads an xyz dataset and prepares
# new hdf5 file that is ready for training with on-the-fly dataloading

import logging
import ast
import numpy as np
import json
import random
import h5py

from ase.io import read
import torch

from mace import tools, data
from mace.data.utils import save_AtomicData_to_HDF5 #, save_dataset_as_HDF5
from mace.tools.scripts_utils import (get_dataset_from_xyz, 
                                    get_atomic_energies)
from mace.tools import torch_geometric
from mace.modules import compute_avg_num_neighbors, scaling_classes

def compute_statistics(train_loader: torch.utils.data.DataLoader, 
                       scaling: str, 
                       atomic_energies: np.ndarray):
    """
    Compute the average number of neighbors and the mean energy and standard
    deviation of the force components"""
    avg_num_neighbors = compute_avg_num_neighbors(train_loader)
    mean, std = scaling_classes[scaling](train_loader, atomic_energies)
    return avg_num_neighbors, mean, std

def main():
    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """

    args = tools.build_preprocess_arg_parser().parse_args()

    # Setup
    tools.set_seeds(args.seed)
    random.seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler()]
    )

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    # Data preparation
    collections, atomic_energies_dict = get_dataset_from_xyz(
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file, 
        seed=args.seed,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        virials_key=args.virials_key,
        dipole_key=args.dipole_key,
        charges_key=args.charges_key,
    )

    # Atomic number table
    # yapf: disable
    if args.atomic_numbers is None:
        z_table = tools.get_atomic_number_table_from_zs(
            z
            for configs in (collections.train, collections.valid)
            for config in configs
            for z in config.atomic_numbers
        )
    else:
        logging.info("Using atomic numbers from command line argument")
        zs_list = ast.literal_eval(args.atomic_numbers)
        assert isinstance(zs_list, list)
        z_table = tools.get_atomic_number_table_from_zs(zs_list)

    logging.info("Preparing training set")
    if args.shuffle:
        random.shuffle(collections.train)

    with h5py.File(args.h5_prefix + "train.h5", "w") as f:
        for i, config in enumerate(collections.train):
            atomic_data = data.AtomicData.from_config(
                config, z_table=z_table, cutoff=args.r_max)
            save_AtomicData_to_HDF5(atomic_data, i, f)
        

    if args.compute_statistics:
        # Compute statistics
        logging.info("Computing statistics")
        if len(atomic_energies_dict) == 0:
            atomic_energies_dict = get_atomic_energies(args.E0s, collections.train, z_table)
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        logging.info(f"Atomic energies: {atomic_energies.tolist()}")
        train_loader = torch_geometric.dataloader.DataLoader(
            data.HDF5Dataset(args.h5_prefix + "train.h5"), 
            batch_size=args.batch_size, 
            shuffle=False,
            drop_last=False,
        )
        avg_num_neighbors, mean, std = compute_statistics(
            train_loader, args.scaling, atomic_energies
        )
        logging.info(f"Average number of neighbors: {avg_num_neighbors}")
        logging.info(f"Mean: {mean}")
        logging.info(f"Standard deviation: {std}")
            
        # save the statistics as a json
        statistics = {
            "atomic_energies": str(atomic_energies_dict),
            "avg_num_neighbors": avg_num_neighbors,
            "mean": mean,
            "std": std,
            "atomic_numbers": str(z_table.zs),
            "r_max": args.r_max,
        }
        with open(args.h5_prefix + "statistics.json", "w") as f:
            json.dump(statistics, f)
    
    logging.info("Preparing validation set")
    if args.shuffle:
        random.shuffle(collections.valid)

    with h5py.File(args.h5_prefix + "valid.h5", "w") as f:    
        for i, config in enumerate(collections.valid):
            atomic_data = data.AtomicData.from_config(
                config, z_table=z_table, cutoff=args.r_max)
            save_AtomicData_to_HDF5(atomic_data, i, f)

    if args.test_file is not None:
        logging.info("Preparing test sets")
        for name, subset in collections.tests:
            with h5py.File(args.h5_prefix + name + "_test.h5", "w") as f:
                for i, config in enumerate(subset):
                    atomic_data = data.AtomicData.from_config(
                        config, z_table=z_table, cutoff=args.r_max)
                    save_AtomicData_to_HDF5(atomic_data, i, f)


if __name__ == "__main__":
    main()