###########################################################################################
# Data parsing utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ase.data
import ase.io
import h5py
import numpy as np

from mace.tools import AtomicNumberTable, DefaultKeys

Positions = np.ndarray  # [..., 3]
Cell = np.ndarray  # [3,3]
Pbc = tuple  # (3,)

DEFAULT_CONFIG_TYPE = "Default"
DEFAULT_CONFIG_TYPE_WEIGHTS = {DEFAULT_CONFIG_TYPE: 1.0}


@dataclass
class KeySpecification:
    info_keys: Dict[str, str] = field(default_factory=dict)
    arrays_keys: Dict[str, str] = field(default_factory=dict)

    def update(
        self,
        info_keys: Optional[Dict[str, str]] = None,
        arrays_keys: Optional[Dict[str, str]] = None,
    ):
        if info_keys is not None:
            self.info_keys.update(info_keys)
        if arrays_keys is not None:
            self.arrays_keys.update(arrays_keys)
        return self

    @classmethod
    def from_defaults(cls):
        instance = cls()
        return update_keyspec_from_kwargs(instance, DefaultKeys.keydict())


def update_keyspec_from_kwargs(
    keyspec: KeySpecification, keydict: Dict[str, str]
) -> KeySpecification:
    # convert command line style property_key arguments into a keyspec
    infos = [
        "energy_key",
        "stress_key",
        "virials_key",
        "dipole_key",
        "head_key",
        "elec_temp_key",
        "total_charge_key",
        "polarizability_key",
        "total_spin_key",
    ]
    arrays = ["forces_key", "charges_key"]
    info_keys = {}
    arrays_keys = {}
    for key in infos:
        if key in keydict:
            info_keys[key[:-4]] = keydict[key]
    for key in arrays:
        if key in keydict:
            arrays_keys[key[:-4]] = keydict[key]

    # automagically add properties for embeddings
    if keydict.get("embedding_specs") is not None:
        for embed_name, embed_spec in keydict["embedding_specs"].items():
            key = embed_spec.get("key", embed_name)
            if embed_spec["per"] == "atom":
                arrays_keys[embed_name] = key
            elif embed_spec["per"] == "graph":
                info_keys[embed_name] = key
            else:
                raise ValueError(f"Unsupported embedding_specs per {embed_spec['per']} for {embed_name}")

    keyspec.update(info_keys=info_keys, arrays_keys=arrays_keys)
    return keyspec


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    properties: Dict[str, Any]
    property_weights: Dict[str, float]
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None

    weight: float = 1.0  # weight of config in loss
    config_type: str = DEFAULT_CONFIG_TYPE  # config_type of config
    head: str = "Default"  # head used to compute the config


Configurations = List[Configuration]


def random_train_valid_split(
    items: Sequence,
    valid_fraction: float,
    seed: int,
    work_dir: str,
    prefix: Optional[str] = None,
) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    # guarantee at least one validation, mostly for tests with tiny fitting databases
    assert size > 1
    train_size = min(size - int(valid_fraction * size), size - 1)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if len(indices[train_size:]) < 10:
        logging.info(
            f"Using random {100 * valid_fraction:.0f}% of training set for validation with following indices: {indices[train_size:]}"
        )
    else:
        # Save indices to file (optionally prefixed with experiment name)
        filename = f"valid_indices_{seed}.txt"
        if prefix is not None and len(prefix) > 0:
            filename = f"{prefix}_" + filename
        path = os.path.join(work_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for index in indices[train_size:]:
                f.write(f"{index}\n")

        logging.info(
            f"Using random {100 * valid_fraction:.0f}% of training set for validation with indices saved in: {path}"
        )

    return (
        [items[i] for i in indices[:train_size]],
        [items[i] for i in indices[train_size:]],
    )


def config_from_atoms_list(
    atoms_list: List[ase.Atoms],
    key_specification: KeySpecification,
    config_type_weights: Optional[Dict[str, float]] = None,
    head_name: str = "Default",
) -> Configurations:
    """Convert list of ase.Atoms into Configurations"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    all_configs = []
    for atoms in atoms_list:
        all_configs.append(
            config_from_atoms(
                atoms,
                key_specification=key_specification,
                config_type_weights=config_type_weights,
                head_name=head_name,
            )
        )
    return all_configs


def config_from_atoms(
    atoms: ase.Atoms,
    key_specification: KeySpecification = KeySpecification(),
    config_type_weights: Optional[Dict[str, float]] = None,
    head_name: str = "Default",
) -> Configuration:
    """Convert ase.Atoms to Configuration"""
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    atomic_numbers = np.array(
        [ase.data.atomic_numbers[symbol] for symbol in atoms.symbols]
    )
    pbc = tuple(atoms.get_pbc().tolist())
    cell = np.array(atoms.get_cell())
    config_type = atoms.info.get("config_type", "Default")
    weight = atoms.info.get("config_weight", 1.0) * config_type_weights.get(
        config_type, 1.0
    )

    properties = {}
    property_weights = {}
    for name in list(key_specification.arrays_keys) + list(key_specification.info_keys):
        property_weights[name] = atoms.info.get(f"config_{name}_weight", 1.0)

    for name, atoms_key in key_specification.info_keys.items():
        properties[name] = atoms.info.get(atoms_key, None)
        if not atoms_key in atoms.info:
            property_weights[name] = 0.0

    for name, atoms_key in key_specification.arrays_keys.items():
        properties[name] = atoms.arrays.get(atoms_key, None)
        if not atoms_key in atoms.arrays:
            property_weights[name] = 0.0

    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=atoms.get_positions(),
        properties=properties,
        weight=weight,
        property_weights=property_weights,
        head=head_name,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
    )


def test_config_types(
    test_configs: Configurations,
) -> List[Tuple[str, List[Configuration]]]:
    """Split test set based on config_type-s"""
    test_by_ct = []
    all_cts = []
    for conf in test_configs:
        if conf.head is None:
            conf.head = ""
        config_type_name = conf.config_type + "_" + conf.head
        if config_type_name not in all_cts:
            all_cts.append(config_type_name)
            test_by_ct.append((config_type_name, [conf]))
        else:
            ind = all_cts.index(config_type_name)
            test_by_ct[ind][1].append(conf)
    return test_by_ct


def load_from_xyz(
    file_path: str,
    key_specification: KeySpecification,
    head_name: str = "Default",
    config_type_weights: Optional[Dict] = None,
    extract_atomic_energies: bool = False,
    keep_isolated_atoms: bool = False,
    no_data_ok: bool = False,
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = ase.io.read(file_path, index=":")
    energy_key = key_specification.info_keys["energy"]
    forces_key = key_specification.arrays_keys["forces"]
    stress_key = key_specification.info_keys["stress"]
    head_key = key_specification.info_keys["head"]
    original_energy_key = energy_key
    original_forces_key = forces_key
    original_stress_key = stress_key
    if energy_key == "energy":
        logging.warning(
            "Since ASE version 3.23.0b1, using energy_key 'energy' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'energy' to 'REF_energy'. You need to use --energy_key='REF_energy' to specify the chosen key name."
        )
        key_specification.info_keys["energy"] = "REF_energy"
        for atoms in atoms_list:
            try:
                # print("OK")
                atoms.info["REF_energy"] = atoms.get_potential_energy()
                # print("atoms.info['REF_energy']:", atoms.info["REF_energy"])
            except Exception as e:  # pylint: disable=W0703
                logging.error(f"Failed to extract energy: {e}")
                atoms.info["REF_energy"] = None
    if forces_key == "forces":
        logging.warning(
            "Since ASE version 3.23.0b1, using forces_key 'forces' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'forces' to 'REF_forces'. You need to use --forces_key='REF_forces' to specify the chosen key name."
        )
        key_specification.arrays_keys["forces"] = "REF_forces"
        for atoms in atoms_list:
            try:
                atoms.arrays["REF_forces"] = atoms.get_forces()
            except Exception as e:  # pylint: disable=W0703
                logging.error(f"Failed to extract forces: {e}")
                atoms.arrays["REF_forces"] = None
    if stress_key == "stress":
        logging.warning(
            "Since ASE version 3.23.0b1, using stress_key 'stress' is no longer safe when communicating between MACE and ASE. We recommend using a different key, rewriting 'stress' to 'REF_stress'. You need to use --stress_key='REF_stress' to specify the chosen key name."
        )
        key_specification.info_keys["stress"] = "REF_stress"
        for atoms in atoms_list:
            try:
                atoms.info["REF_stress"] = atoms.get_stress()
            except Exception as e:  # pylint: disable=W0703
                atoms.info["REF_stress"] = None

    final_energy_key = key_specification.info_keys["energy"]
    final_forces_key = key_specification.arrays_keys["forces"]
    final_dipole_key = key_specification.info_keys.get("dipole", "REF_dipole")
    has_energy = any(final_energy_key in atoms.info for atoms in atoms_list)
    has_forces = any(final_forces_key in atoms.arrays for atoms in atoms_list)
    has_dipole = any(final_dipole_key in atoms.info for atoms in atoms_list)

    if not has_energy and not has_forces and not has_dipole:
        msg = f"None of '{final_energy_key}', '{final_forces_key}', and '{final_dipole_key}' found in '{file_path}'."
        if no_data_ok:
            logging.warning(msg + " Continuing because no_data_ok=True was passed in.")
        else:
            raise ValueError(
                msg
                + " Please change the key names in the command line arguments or ensure that the file contains the required data."
            )
    if not has_energy:
        logging.warning(
            f"No energies found with key '{final_energy_key}' in '{file_path}'. If this is unexpected, please change the key name in the command line arguments or ensure that the file contains the required data."
        )
    if not has_forces:
        logging.warning(
            f"No forces found with key '{final_forces_key}' in '{file_path}'. If this is unexpected, Please change the key name in the command line arguments or ensure that the file contains the required data."
        )

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            atoms.info[head_key] = head_name
            isolated_atom_config = (
                len(atoms) == 1 and atoms.info.get("config_type") == "IsolatedAtom"
            )
            if isolated_atom_config:
                atomic_number = int(atoms.get_atomic_numbers()[0])
                if energy_key in atoms.info.keys():
                    atomic_energies_dict[atomic_number] = float(atoms.info[energy_key])
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy. Zero energy will be used."
                    )
                    atomic_energies_dict[atomic_number] = 0.0
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")
        if not keep_isolated_atoms:
            atoms_list = atoms_without_iso_atoms

    for atoms in atoms_list:
        atoms.info[head_key] = head_name

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        key_specification=key_specification,
        head_name=head_name,
    )
    key_specification.info_keys["energy"] = original_energy_key
    key_specification.arrays_keys["forces"] = original_forces_key
    key_specification.info_keys["stress"] = original_stress_key
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
        B[i] = collections_train[i].properties["energy"]
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(collections_train[i].atomic_numbers == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.error(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict


def save_dataset_as_HDF5(dataset: List, out_name: str) -> None:
    with h5py.File(out_name, "w") as f:
        for i, data in enumerate(dataset):
            save_AtomicData_to_HDF5(data, i, f)


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
    grp["polarizability"] = data.polarizability
    grp["head"] = data.head


def save_configurations_as_HDF5(configurations: Configurations, _, h5_file) -> None:
    grp = h5_file.create_group("config_batch_0")
    for j, config in enumerate(configurations):
        subgroup_name = f"config_{j}"
        subgroup = grp.create_group(subgroup_name)
        subgroup["atomic_numbers"] = write_value(config.atomic_numbers)
        subgroup["positions"] = write_value(config.positions)
        properties_subgrp = subgroup.create_group("properties")
        for key, value in config.properties.items():
            properties_subgrp[key] = write_value(value)
        subgroup["cell"] = write_value(config.cell)
        subgroup["pbc"] = write_value(config.pbc)
        subgroup["weight"] = write_value(config.weight)
        weights_subgrp = subgroup.create_group("property_weights")
        for key, value in config.property_weights.items():
            weights_subgrp[key] = write_value(value)
        subgroup["config_type"] = write_value(config.config_type)


def write_value(value):
    return value if value is not None else "None"
