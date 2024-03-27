# This file loads an xyz dataset and prepares
# new hdf5 file that is ready for training with on-the-fly dataloading

import ast
import json
import logging
import multiprocessing as mp
import os
import random
from glob import glob
from typing import List, Tuple

import h5py
import numpy as np
import tqdm

from mace import data, tools
from mace.data.utils import save_configurations_as_HDF5
from mace.modules import compute_statistics
from mace.tools import torch_geometric
from mace.tools.scripts_utils import get_atomic_energies, get_dataset_from_xyz
from mace.tools.utils import AtomicNumberTable


def compute_stats_target(
    file: str,
    z_table: AtomicNumberTable,
    r_max: float,
    atomic_energies: Tuple,
    batch_size: int,
):
    train_dataset = data.HDF5Dataset(file, z_table=z_table, r_max=r_max)
    train_loader = torch_geometric.dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    avg_num_neighbors, mean, std = compute_statistics(train_loader, atomic_energies)
    output = [avg_num_neighbors, mean, std]
    return output


def pool_compute_stats(inputs: List):
    path_to_files, z_table, r_max, atomic_energies, batch_size, num_process = inputs

    with mp.Pool(processes=num_process) as pool:
        re = [
            pool.apply_async(
                compute_stats_target,
                args=(
                    file,
                    z_table,
                    r_max,
                    atomic_energies,
                    batch_size,
                ),
            )
            for file in glob(path_to_files + "/*")
        ]

        pool.close()
        pool.join()

    results = [r.get() for r in tqdm.tqdm(re)]
    return np.average(results, axis=0)


def split_array(a: np.ndarray, max_size: int):
    drop_last = False
    if len(a) % 2 == 1:
        a = np.append(a, a[-1])
        drop_last = True
    factors = get_prime_factors(len(a))
    max_factor = 1
    for i in range(1, len(factors) + 1):
        for j in range(0, len(factors) - i + 1):
            if np.prod(factors[j : j + i]) <= max_size:
                test = np.prod(factors[j : j + i])
                if test > max_factor:
                    max_factor = test
    return np.array_split(a, max_factor), drop_last


def get_prime_factors(n: int):
    factors = []
    for i in range(2, n + 1):
        while n % i == 0:
            factors.append(i)
            n = n / i
    return factors


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
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    try:
        config_type_weights = ast.literal_eval(args.config_type_weights)
        assert isinstance(config_type_weights, dict)
    except Exception as e:  # pylint: disable=W0703
        logging.warning(
            f"Config type weights not specified correctly ({e}), using Default"
        )
        config_type_weights = {"Default": 1.0}

    folders = ["train", "val", "test"]
    for sub_dir in folders:
        if not os.path.exists(args.h5_prefix + sub_dir):
            os.makedirs(args.h5_prefix + sub_dir)

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

    # split collections.train into batches and save them to hdf5
    split_train = np.array_split(collections.train,args.num_process)
    drop_last = False
    if len(collections.train) % 2 == 1:
        drop_last = True

    # Define Task for Multiprocessiing
    def multi_train_hdf5(process):
        with h5py.File(args.h5_prefix + "train/train_" + str(process)+".h5", "w") as f:
            f.attrs["drop_last"] = drop_last
            save_configurations_as_HDF5(split_train[process], process, f)

    processes = []
    for i in range(args.num_process):
        p = mp.Process(target=multi_train_hdf5, args=[i])
        p.start()
        processes.append(p)

    for i in processes:
        i.join()


    logging.info("Computing statistics")
    if len(atomic_energies_dict) == 0:
        atomic_energies_dict = get_atomic_energies(args.E0s, collections.train, z_table)
    atomic_energies: np.ndarray = np.array(
        [atomic_energies_dict[z] for z in z_table.zs]
    )
    logging.info(f"Atomic energies: {atomic_energies.tolist()}")
    _inputs = [args.h5_prefix+'train', z_table, args.r_max, atomic_energies, args.batch_size, args.num_process]
    avg_num_neighbors, mean, std=pool_compute_stats(_inputs)
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

    with open(args.h5_prefix + "statistics.json", "w") as f: # pylint: disable=W1514
        json.dump(statistics, f)

    logging.info("Preparing validation set")
    if args.shuffle:
        random.shuffle(collections.valid)
    split_valid = np.array_split(collections.valid, args.num_process)
    drop_last = False
    if len(collections.valid) % 2 == 1:
        drop_last = True

    def multi_valid_hdf5(process):
        with h5py.File(args.h5_prefix + "val/val_" + str(process)+".h5", "w") as f:
            f.attrs["drop_last"] = drop_last
            save_configurations_as_HDF5(split_valid[process], process, f)

    processes = []
    for i in range(args.num_process):
        p = mp.Process(target=multi_valid_hdf5, args=[i])
        p.start()
        processes.append(p)

    for i in processes:
        i.join()

    if args.test_file is not None:
        def multi_test_hdf5(process, name):
            with h5py.File(args.h5_prefix + "test/" + name + "_" + str(process) + ".h5", "w") as f:
                f.attrs["drop_last"] = drop_last
                save_configurations_as_HDF5(split_test[process], process, f)

        logging.info("Preparing test sets")
        for name, subset in collections.tests:
            drop_last = False
            if len(subset) % 2 == 1:
                drop_last = True
            split_test = np.array_split(subset, args.num_process)

            processes = []
            for i in range(args.num_process):
                p = mp.Process(target=multi_test_hdf5, args=[i, name])
                p.start()
                processes.append(p)

            for i in processes:
                i.join()


if __name__ == "__main__":
    main()
