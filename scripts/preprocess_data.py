# This file loads an xyz or a hdf5 dataset and prepares a 
# new hdf5 file that is ready for training with on-the-fly dataloading

import h5py
import logging
import ast

from ase.io import read

import mace
from mace import tools, data
from mace.data.utils import save_dataset_as_HDF5
from mace.tools.scripts_utils import get_dataset_from_xyz

def main():
    args = tools.build_default_arg_parser().parse_args()

    # Setup
    tools.set_seeds(args.seed)
    tools.set_default_dtype(args.default_dtype)

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

    z_table = tools.get_atomic_number_table_from_zs(
        z
        for configs in (collections.train, collections.valid)
        for config in configs
        for z in config.atomic_numbers
    )

    # prepare the datasets containing the AtomicData objects
    training_set = [data.AtomicData.from_config(
        config, z_table=z_table, cutoff=args.r_max)
        for config in collections.train]  
    
    save_dataset_as_HDF5(training_set, args.h5_prefix + "train.h5")
    
    valid_set = [data.AtomicData.from_config(
        config, z_table=z_table, cutoff=args.r_max)
        for config in collections.valid]

    save_dataset_as_HDF5(valid_set, args.h5_prefix + "valid.h5")


if __name__ == "__main__":
    main()