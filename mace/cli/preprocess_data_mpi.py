# This file loads an xyz dataset and prepares
# new hdf5 file that is ready for training with on-the-fly dataloading

import argparse
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
from ase import Atoms
from ase.io import iread
from mpi4py import MPI

from mace import data, tools
from mace.data.utils import (
    Configurations,
    config_from_atoms,
    save_configurations_as_HDF5,
)
from mace.modules import compute_statistics
from mace.tools import torch_geometric
from mace.tools.scripts_utils import SubsetCollection, get_atomic_energies
from mace.tools.utils import AtomicNumberTable

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
                max_factor = max(test, max_factor)
    return np.array_split(a, max_factor), drop_last


def get_prime_factors(n: int):
    factors = []
    for i in range(2, n + 1):
        while n % i == 0:
            factors.append(i)
            n = n / i
    return factors


def chunkify(data: list, size: int) -> list:
    """
    Splits an iterable into a specified number of chunks without knowing its length.

    Args:
        data (Iterable): The iterable to split into chunks.
        size (int): The number of chunks to split the iterable into.

    Yields:
        Generator: A generator yielding chunks of the input iterable.
    """
    # def create_chunks(data, size):
    total_items = len(data)
    chunk_size = total_items // size
    remainder = total_items % size

    chunks = []
    for i in range(size):
        start_index = i * chunk_size + min(i, remainder)
        end_index = start_index + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start_index:end_index])

    return chunks


def process_chunk(
    chunk: List[Atoms], config_type_weights: dict, args: argparse.Namespace
) -> Configurations:
    """
    Processes a single chunk of data.
    Modify this function to include the actual processing logic.

    Args:
        chunk (List): A chunk of data to process.

    Returns:
        Any: The result of processing the chunk.
    """

    atoms_list = []
    for atoms in chunk:
        atoms_list.append(
            config_from_atoms(
                atoms,
                energy_key=args.energy_key,
                forces_key=args.forces_key,
                stress_key=args.stress_key,
                virials_key=args.virials_key,
                dipole_key=args.dipole_key,
                charges_key=args.charges_key,
                config_type_weights=config_type_weights,
            )
        )
    return atoms_list


def read_configs(
    file: str, args: argparse.Namespace, config_type_weights: dict
) -> Configurations | None:

    # MPI: Read the xyz file

    data = list(iread(file, ":"))

    if rank == 0:
        logging.info(f"Read the file from {file}: {len(data)} configurations")
        chunks = chunkify(data, size)
    else:
        chunks = []

    # Scatter chunks to all processes
    chunk = comm.scatter(chunks, root=0)

    # Each process processes its chunk
    result = process_chunk(chunk, config_type_weights, args)
    print(f"Process {rank} processed chunk of size {len(chunk)}")

    # Gather results from all processes
    results = comm.gather(result, root=0)

    if rank == 0:
        logging.info("Gathered configurations from all processes")

        configs = []
        for result in results:
            configs.extend(result)
        return configs
    else:
        return


def get_z_table(
    configs: Configurations, args: argparse.Namespace
) -> AtomicNumberTable | None:
    """
    Extracts the atomic numbers from the configurations and creates an atomic number table.

    Args:
        configs (List): The list of configurations.
        args (argparse.Namespace): The command line arguments.
    Returns:
        AtomicNumberTable: The atomic number table.
    """

    if args.atomic_numbers is None:
        # MPI: Extract all the atomic numbers
        if rank == 0:
            logging.info("Extracting atomic numbers...")
            chunks = chunkify(configs, size)
        else:
            chunks = []

        chunk = comm.scatter(chunks, root=0)

        z_set = set()
        zs = [config.atomic_numbers for config in chunk]
        for z in zs:
            z_set.add(z)

        print(f"Rank {rank} processed chunk of size {len(chunk)}")

        z_sets = comm.gather(z_set, root=0)

        if rank == 0:
            z_table = tools.get_atomic_number_table_from_zs(
                z for z_set in z_sets for z in z_set
            )
            logging.info("Extracted atomic numbers")
            return z_table

            # comm.bcast(z_table, root=0)

        # comm.barrier()
        return
    else:
        logging.info("Using atomic numbers from command line argument")
        zs_list = ast.literal_eval(args.atomic_numbers)
        assert isinstance(zs_list, list)
        z_table = tools.get_atomic_number_table_from_zs(zs_list)

        return z_table


# def write_hdf5(
#         configs: Configurations,
#         filename: str
#         ) -> None:

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     if rank == 0:
#         logging.info("Writing the hdf5 file...")
#         chunks = list(chunkify(configs, size))
#     else:
#         chunks = None

#     chunk = comm.scatter(chunks, root=0)

#     with h5py.File(filename + str(rank) , "w") as f:
#         f.attrs["drop_last"] = len(configs) % 2 == 1
#         save_configurations_as_HDF5(chunk, rank, f)


def main() -> None:
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

    # - iread the xyz file
    # - save hdf5 files for each process
    # - extract all the atomic numbers
    # - compute statistics

    comm.barrier()

    train_configs = read_configs(args.train_file, args, config_type_weights)

    comm.barrier()

    if args.valid_file is not None:
        valid_configs = read_configs(args.valid_file, args, config_type_weights)
        if rank == 0:
            logging.info(f"Number of training configurations: {len(train_configs)}")
            logging.info(f"Number of validation configurations: {len(valid_configs)}")
    else:
        if rank == 0:
            logging.info(
                f"Using {args.valid_fraction*100}% of the training data for validation"
            )

            train_configs, valid_configs = data.random_train_valid_split(
                train_configs, args.valid_fraction, seed=args.seed
            )
            logging.info(f"Number of training configurations: {len(train_configs)}")
            logging.info(f"Number of validation configurations: {len(valid_configs)}")

    comm.barrier()

    if args.test_file is not None:
        test_configs = read_configs(args.test_file, args, config_type_weights)

        if rank == 0:
            test_configs = data.test_config_types(test_configs)
            logging.info(f"Number of test configurations: {len(test_configs)}")
    else:
        test_configs = []
        logging.info("No test set provided")

    # comm.bcast(train_configs, root=0)
    # comm.bcast(valid_configs, root=0)
    # comm.bcast(test_configs, root=0)

    # Atomic number table
    if rank == 0:
        # assert isinstance(train_configs, list)
        # assert isinstance(valid_configs, list)
        configs = [c for c in train_configs] + [c for c in valid_configs]
        z_table = get_z_table(configs, args)

    # comm.bcast(z_table, root=0)
    comm.barrier()

    if rank == 0:

        collections = SubsetCollection(
            train=train_configs, valid=valid_configs, tests=test_configs
        )

        if args.shuffle:
            random.shuffle(collections.train)

        # split collections.train into batches and save them to hdf5
        split_train = np.array_split(collections.train, args.num_process)
        drop_last = False
        if len(collections.train) % 2 == 1:
            drop_last = True

        # Define Task for Multiprocessiing
        def multi_train_hdf5(process):
            with h5py.File(
                args.h5_prefix + "train/train_" + str(process) + ".h5", "w"
            ) as f:
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
        atomic_energies_dict = get_atomic_energies(args.E0s, collections.train, z_table)
        atomic_energies: np.ndarray = np.array(
            [atomic_energies_dict[z] for z in z_table.zs]
        )
        logging.info(f"Atomic energies: {atomic_energies.tolist()}")
        _inputs = [
            args.h5_prefix + "train",
            z_table,
            args.r_max,
            atomic_energies,
            args.batch_size,
            args.num_process,
        ]
        avg_num_neighbors, mean, std = pool_compute_stats(_inputs)
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

        with open(
            args.h5_prefix + "statistics.json", "w"
        ) as f:  # pylint: disable=W1514
            json.dump(statistics, f)

        logging.info("Preparing validation set")
        if args.shuffle:
            random.shuffle(collections.valid)
        split_valid = np.array_split(collections.valid, args.num_process)
        drop_last = False
        if len(collections.valid) % 2 == 1:
            drop_last = True

        def multi_valid_hdf5(process):
            with h5py.File(
                args.h5_prefix + "val/val_" + str(process) + ".h5", "w"
            ) as f:
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
                with h5py.File(
                    args.h5_prefix + "test/" + name + "_" + str(process) + ".h5", "w"
                ) as f:
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
