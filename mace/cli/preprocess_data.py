# This file loads an xyz dataset and prepares
# new hdf5 file that is ready for training with on-the-fly dataloading

import argparse
import ast
import json
import logging
import multiprocessing as mp
import os
import random
from functools import partial
from glob import glob
from typing import List, Tuple

import h5py
import numpy as np
import tqdm

from mace import data, tools
from mace.data import KeySpecification, update_keyspec_from_kwargs
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

    if not results:
        raise ValueError(
            "No results were computed. Check if the input files exist and are readable."
        )

    # Separate avg_num_neighbors, mean, and std
    avg_num_neighbors = np.mean([r[0] for r in results])
    means = np.array([r[1] for r in results])
    stds = np.array([r[2] for r in results])

    # Compute averages
    mean = np.mean(means, axis=0).item()
    std = np.mean(stds, axis=0).item()

    return avg_num_neighbors, mean, std


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
                max_factor = max(test, max_factor)
    return np.array_split(a, max_factor), drop_last


def get_prime_factors(n: int):
    factors = []
    for i in range(2, n + 1):
        while n % i == 0:
            factors.append(i)
            n = n / i
    return factors


# Define Task for Multiprocessiing
def multi_train_hdf5(process, args, split_train, drop_last):
    with h5py.File(args.h5_prefix + "train/train_" + str(process) + ".h5", "w") as f:
        f.attrs["drop_last"] = drop_last
        save_configurations_as_HDF5(split_train[process], process, f)


def multi_valid_hdf5(process, args, split_valid, drop_last):
    with h5py.File(args.h5_prefix + "val/val_" + str(process) + ".h5", "w") as f:
        f.attrs["drop_last"] = drop_last
        save_configurations_as_HDF5(split_valid[process], process, f)


def multi_test_hdf5(process, name, args, split_test, drop_last):
    with h5py.File(
        args.h5_prefix + "test/" + name + "_" + str(process) + ".h5", "w"
    ) as f:
        f.attrs["drop_last"] = drop_last
        save_configurations_as_HDF5(split_test[process], process, f)


def main() -> None:
    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """
    args = tools.build_preprocess_arg_parser().parse_args()
    run(args)


def run(args: argparse.Namespace):
    """
    This script loads an xyz dataset and prepares
    new hdf5 file that is ready for training with on-the-fly dataloading
    """

    # currently support only command line property_key syntax
    args.key_specification = KeySpecification()
    update_keyspec_from_kwargs(args.key_specification, vars(args))

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
        work_dir=args.work_dir,
        train_path=args.train_file,
        valid_path=args.valid_file,
        valid_fraction=args.valid_fraction,
        config_type_weights=config_type_weights,
        test_path=args.test_file,
        seed=args.seed,
        key_specification=args.key_specification,
        head_name="",
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

    multi_train_hdf5_ = partial(multi_train_hdf5, args=args, split_train=split_train, drop_last=drop_last)
    processes = []
    for i in range(args.num_process):
        p = mp.Process(target=multi_train_hdf5_, args=[i])
        p.start()
        processes.append(p)

    for i in processes:
        i.join()

    if args.compute_statistics:
        logging.info("Computing statistics")
        if atomic_energies_dict is None or len(atomic_energies_dict) == 0:
            atomic_energies_dict = get_atomic_energies(args.E0s, collections.train, z_table)

        # Remove atomic energies if element not in z_table
        removed_atomic_energies = {}
        for z in list(atomic_energies_dict):
            if z not in z_table.zs:
                removed_atomic_energies[z] = atomic_energies_dict.pop(z)
        if len(removed_atomic_energies) > 0:
            logging.warning("Atomic energies for elements not present in the atomic number table have been removed.")
            logging.warning(f"Removed atomic energies (eV): {str(removed_atomic_energies)}")
            logging.warning("To include these elements in the model, specify all atomic numbers explicitly using the --atomic_numbers argument.")

        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        logging.info(f"Atomic Energies: {atomic_energies.tolist()}")
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
            "atomic_numbers": str([int(z) for z in z_table.zs]),
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

    multi_valid_hdf5_ = partial(multi_valid_hdf5, args=args, split_valid=split_valid, drop_last=drop_last)
    processes = []
    for i in range(args.num_process):
        p = mp.Process(target=multi_valid_hdf5_, args=[i])
        p.start()
        processes.append(p)

    for i in processes:
        i.join()

    if args.test_file is not None:
        logging.info("Preparing test sets")
        for name, subset in collections.tests:
            drop_last = False
            if len(subset) % 2 == 1:
                drop_last = True
            split_test = np.array_split(subset, args.num_process)
            multi_test_hdf5_ = partial(multi_test_hdf5, args=args, split_test=split_test, drop_last=drop_last)

            processes = []
            for i in range(args.num_process):
                p = mp.Process(target=multi_test_hdf5_, args=[i, name])
                p.start()
                processes.append(p)

            for i in processes:
                i.join()


if __name__ == "__main__":
    main()
