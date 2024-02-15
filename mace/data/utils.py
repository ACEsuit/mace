###########################################################################################
# Data parsing utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
import h5py
from multiprocessing import Pool
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import ase.data
import ase.io
import numpy as np

from mace.tools import AtomicNumberTable

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]
Stress = np.ndarray  # [6, ], [3,3], [9, ]
Virials = np.ndarray  # [6, ], [3,3], [9, ]
Charges = np.ndarray  # [..., 1]
Cell = np.ndarray  # [3,3]
Pbc = tuple  # (3,)

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom
    stress: Optional[Stress] = None  # eV/Angstrom^3
    virials: Optional[Virials] = None  # eV
    dipole: Optional[Vector] = None  # Debye
    charges: Optional[Charges] = None  # atomic unit
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

    weight: float = 1.0  # weight of config in loss
    energy_weight: float = 1.0  # weight of config energy in loss
    forces_weight: float = 1.0  # weight of config forces in loss
    stress_weight: float = 1.0  # weight of config stress in loss
    virials_weight: float = 1.0  # weight of config virial in loss
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config


Configurations = List[Configuration]


def random_train_valid_split(
    items: Sequence, valid_fraction: float, seed: int
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    train_size = size - int(valid_fraction * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )


def config_from_atoms_list(
    atoms_list: List[ase.Atoms],
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    config_type_weights: Dict[str, float] = None,
) -> Configurations:
    """Convert list of ase.Atoms into Configurations"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    all_configs = []
    for atoms in atoms_list:
        all_configs.append(
            config_from_atoms(
                atoms,
                energy_key=energy_key,
                forces_key=forces_key,
                stress_key=stress_key,
                virials_key=virials_key,
                dipole_key=dipole_key,
                charges_key=charges_key,
                config_type_weights=config_type_weights,
            )
        )
    return all_configs


def config_from_atoms(
    atoms: ase.Atoms,
    energy_key="energy",
    forces_key="forces",
    stress_key="stress",
    virials_key="virials",
    dipole_key="dipole",
    charges_key="charges",
    config_type_weights: Dict[str, float] = None,
) -> Configuration:
    """Convert ase.Atoms to Configuration"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    energy = atoms.info.get(energy_key, None)  # eV
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    stress = atoms.info.get(stress_key, None)  # eV / Ang
    virials = atoms.info.get(virials_key, None)
    dipole = atoms.info.get(dipole_key, None)  # Debye
    # Charges default to 0 instead of None if not found
    charges = atoms.arrays.get(charges_key, np.zeros(len(atoms)))  # atomic unit
    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc())
    cell = np.array(atoms.get_cell())
    config_type = atoms.info.get("config_type", "Default")
    weight = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
        config_type, 1.0
    )
    energy_weight = atoms.info.get("config_energy_weight", 1.0)
    forces_weight = atoms.info.get("config_forces_weight", 1.0)
    stress_weight = atoms.info.get("config_stress_weight", 1.0)
    virials_weight = atoms.info.get("config_virials_weight", 1.0)

    # fill in missing quantities but set their weight to 0.0
    if energy is None:
        energy = 0.0
        energy_weight = 0.0
    if forces is None:
        forces = np.zeros(np.shape(atoms.positions))
        forces_weight = 0.0
    if stress is None:
        stress = np.zeros(6)
        stress_weight = 0.0
    if virials is None:
        virials = np.zeros((3, 3))
        virials_weight = 0.0
    if dipole is None:
        dipole = np.zeros(3)
        # dipoles_weight = 0.0

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        energy=energy,
        forces=forces,
        stress=stress,
        virials=virials,
        dipole=dipole,
        charges=charges,
        weight=weight,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
    )


def test_config_types(
    test_configs: Configurations,
) -> List[Tuple[Optional[str], List[Configuration]]]:
    """Split test set based on config_type-s"""
    test_by_ct = []
    all_cts = []
    for conf in test_configs:
        if conf.config_type not in all_cts:
            all_cts.append(conf.config_type)
            test_by_ct.append((conf.config_type, [conf]))
        else:
            ind = all_cts.index(conf.config_type)
            test_by_ct[ind][1].append(conf)
    return test_by_ct


def load_from_xyz(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    extract_atomic_energies: bool = False,
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = ase.io.read(file_path, index=":")

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            if len(atoms) == 1:
                isolated_atom_config = atoms.info.get("config_type") == "IsolatedAtom"
                if isolated_atom_config:
                    if energy_key in atoms.info.keys():
                        atomic_energies_dict[
                            atoms.get_atomic_numbers()[0]
                        ] = atoms.info[energy_key]
                    else:
                        logging.warning(
                            f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                            "but does not contain an energy."
                        )
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")

        atoms_list = atoms_without_iso_atoms

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
    )
    return atomic_energies_dict, configs


def compute_average_E0s(
    collections_train: Configurations, z_table: AtomicNumberTable
) -> Dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(collections_train)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i in range(len_train):
        B[i] = collections_train[i].energy
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(collections_train[i].atomic_numbers == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.warning(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict


def save_dataset_as_HDF5(dataset: List, out_name: str) -> None:
    with h5py.File(out_name, "w") as f:
        for i, data in enumerate(dataset):
            grp = f.create_group(f"config_{i}")
            grp["num_nodes"] = data.num_nodes
            grp["edge_index"] = data.edge_index
            grp["positions"] = data.positions
            grp["shifts"] = data.shifts
            grp["unit_shifts"] = data.unit_shifts
            grp["cell"] = data.cell
            grp["node_attrs"] = data.node_attrs
            grp["weight"] = data.weight
            grp["energy_weight"] = data.energy_weight
            grp["forces_weight"] = data.forces_weight
            grp["stress_weight"] = data.stress_weight
            grp["virials_weight"] = data.virials_weight
            grp["forces"] = data.forces
            grp["energy"] = data.energy
            grp["stress"] = data.stress
            grp["virials"] = data.virials
            grp["dipole"] = data.dipole
            grp["charges"] = data.charges


def save_AtomicData_to_HDF5(data, i, h5_file) -> None:
    grp = h5_file.create_group(f"config_{i}")
    grp["num_nodes"] = data.num_nodes
    grp["edge_index"] = data.edge_index
    grp["positions"] = data.positions
    grp["shifts"] = data.shifts
    grp["unit_shifts"] = data.unit_shifts
    grp["cell"] = data.cell
    grp["node_attrs"] = data.node_attrs
    grp["weight"] = data.weight
    grp["energy_weight"] = data.energy_weight
    grp["forces_weight"] = data.forces_weight
    grp["stress_weight"] = data.stress_weight
    grp["virials_weight"] = data.virials_weight
    grp["forces"] = data.forces
    grp["energy"] = data.energy
    grp["stress"] = data.stress
    grp["virials"] = data.virials
    grp["dipole"] = data.dipole
    grp["charges"] = data.charges


def save_configurations_as_HDF5(configurations: Configurations, i, h5_file) -> None:
    grp = h5_file.create_group(f"config_batch_{i}")
    for i, config in enumerate(configurations):
        subgroup_name = f"config_{i}"
        subgroup = grp.create_group(subgroup_name)
        subgroup["atomic_numbers"] = write_value(config.atomic_numbers)
        subgroup["positions"] = write_value(config.positions)
        subgroup["energy"] = write_value(config.energy)
        subgroup["forces"] = write_value(config.forces)
        subgroup["stress"] = write_value(config.stress)
        subgroup["virials"] = write_value(config.virials)
        subgroup["dipole"] = write_value(config.dipole)
        subgroup["charges"] = write_value(config.charges)
        subgroup["cell"] = write_value(config.cell)
        subgroup["pbc"] = write_value(config.pbc)
        subgroup["weight"] = write_value(config.weight)
        subgroup["energy_weight"] = write_value(config.energy_weight)
        subgroup["forces_weight"] = write_value(config.forces_weight)
        subgroup["stress_weight"] = write_value(config.stress_weight)
        subgroup["virials_weight"] = write_value(config.virials_weight)
        subgroup["config_type"] = write_value(config.config_type)


def write_value(value):
    return value if value is not None else "None"

