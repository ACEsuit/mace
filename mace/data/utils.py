###########################################################################################
# Data parsing utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import ase.data
import ase.io
import h5py
import numpy as np

from mace.tools import AtomicNumberTable
from tqdm import tqdm

import bz2

import json
from tqdm import tqdm
from ase import Atoms

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
N_PROC=1


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
    head: Optional[str] = "Default"  # head used to compute the config
    alex_config_id: Optional[str] = None


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
    head_key="head",
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
                head_key=head_key,
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
    head_key="head",
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

    head = atoms.info.get(head_key, "Default")
    alex_config_id = atoms.info.get("alex_config_id", None)

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
        head=head,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        config_type=config_type,
        pbc=pbc,
        cell=cell,
        alex_config_id=alex_config_id
    )


def test_config_types(
    test_configs: Configurations,
) -> List[Tuple[Optional[str], List[Configuration]]]:
    """Split test set based on config_type-s"""
    test_by_ct = []
    all_cts = []
    for conf in test_configs:
        config_type_name = conf.config_type + "_" + conf.head
        if config_type_name not in all_cts:
            all_cts.append(config_type_name)
            test_by_ct.append((config_type_name, [conf]))
        else:
            ind = all_cts.index(config_type_name)
            test_by_ct[ind][1].append(conf)
    return test_by_ct

def load_from_jsonbz2s_go(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    head_key: str = "head",
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = atoms_from_alex_go(file_path)

    atomic_energies_dict = {}

    heads = set()
    for atoms in atoms_list:
        heads.add(atoms.info.get("head", "Default"))
    heads = list(heads)

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key="energy",
        forces_key="forces",
        stress_key="stress",
        virials_key="virials",
        dipole_key="dipole",
        charges_key="charges",
        head_key="head",
    )
    return atomic_energies_dict, configs, heads

def load_from_jsonbz2s(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    head_key: str = "head",
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = atoms_from_alex(file_path)

    atomic_energies_dict = {}

    heads = set()
    for atoms in atoms_list:
        heads.add(atoms.info.get("head", "Default"))
    heads = list(heads)

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key="energy",
        forces_key="forces",
        stress_key="stress",
        virials_key="virials",
        dipole_key="dipole",
        charges_key="charges",
        head_key="head",
    )
    return atomic_energies_dict, configs, heads

def load_from_extxyzs(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    head_key: str = "head",
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = atoms_from_oc20(file_path)

    atomic_energies_dict = {}

    heads = set()
    for atoms in atoms_list:
        heads.add(atoms.info.get("head", "Default"))
    heads = list(heads)

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key="energy",
        forces_key="forces",
        stress_key="stress",
        virials_key="virials",
        dipole_key="dipole",
        charges_key="charges",
        head_key="head",
    )
    return atomic_energies_dict, configs, heads

def load_from_xyz(
    file_path: str,
    config_type_weights: Dict,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    head_key: str = "head",
    extract_atomic_energies: bool = False,
    keep_isolated_atoms: bool = False,
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = ase.io.read(file_path, index=":")

    energy_from_calc = False
    forces_from_calc = False
    stress_from_calc = False

    ## Perform initial checks and log warnings
    if energy_key == "energy":
        logging.info(
            "Using energy_key 'energy' is unsafe, consider using a different key, rewriting energies to '_REF_energy'"
        )
        energy_from_calc = True
        energy_key = "_REF_energy"

    if forces_key == "forces":
        logging.info(
            "Using forces_key 'forces' is unsafe, consider using a different key, rewriting forces to '_REF_forces'"
        )
        forces_from_calc = True
        forces_key = "_REF_forces"

    if stress_key == "stress":
        logging.info(
            "Using stress_key 'stress' is unsafe, consider using a different key, rewriting stress to '_REF_stress'"
        )
        stress_from_calc = True
        stress_key = "_REF_stress"

    for atoms in atoms_list:
        if energy_from_calc:
            try:
                atoms.info["_REF_energy"] = atoms.get_potential_energy()
            except Exception as e:  # pylint: disable=W0703
                logging.warning(f"Failed to extract energy: {e}")
                atoms.info["_REF_energy"] = None

        if forces_from_calc:
            try:
                atoms.arrays["_REF_forces"] = atoms.get_forces()
            except Exception as e:  # pylint: disable=W0703
                logging.warning(f"Failed to extract forces: {e}")
                atoms.arrays["_REF_forces"] = None

        if stress_from_calc:
            try:
                atoms.info["_REF_stress"] = atoms.get_stress()
            except Exception as e:  # pylint: disable=W0703
                atoms.info["_REF_stress"] = None

    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    atomic_energies_dict = {}
    if extract_atomic_energies:
        atoms_without_iso_atoms = []

        for idx, atoms in enumerate(atoms_list):
            if atoms.info.get("config_type") == "IsolatedAtom":
                assert (
                    len(atoms) == 1
                ), f"Got config_type=IsolatedAtom for a config with len {len(atoms)}"
                if energy_key in atoms.info.keys():
                    head = atoms.info.get(head_key, "Default")
                    if head not in atomic_energies_dict:
                        atomic_energies_dict[head] = {}
                    atomic_energies_dict[head][atoms.get_atomic_numbers()[0]] = (
                        atoms.info[energy_key]
                    )
                else:
                    logging.warning(
                        f"Configuration '{idx}' is marked as 'IsolatedAtom' "
                        "but does not contain an energy. Zero energy will be used."
                    )
                    head = atoms.info.get(head_key, "Default")
                    if head not in atomic_energies_dict:
                        atomic_energies_dict[head] = {}
                    atomic_energies_dict[head][atoms.get_atomic_numbers()[0]] = (
                        np.zeros(1)
                    )
            else:
                atoms_without_iso_atoms.append(atoms)

        if len(atomic_energies_dict) > 0:
            logging.info("Using isolated atom energies from training file")
        if not keep_isolated_atoms:
            atoms_list = atoms_without_iso_atoms
    heads = set()
    for atoms in atoms_list:
        heads.add(atoms.info.get(head_key, "Default"))
    heads = list(heads)
    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        head_key=head_key,
    )
    return atomic_energies_dict, configs, heads

def load_from_h5(
    file_path: str,
    config_type_weights: Dict,
    h5_positions_key: str = 'atXYZ',
    h5_numbers_key: str = "atNUM",
    h5_energy_key: str = 'ePBE0+MBD',
    h5_forces_key: str = 'totFOR',
) -> Tuple[Dict[int, float], Configurations]:
    atoms_list = atoms_from_hdf5_ani(
        file_path, 
        positions_key=h5_positions_key,
        numbers_key=h5_numbers_key,
        energy_key=h5_energy_key,
        forces_key=h5_forces_key
    )


    atomic_energies_dict = {}

    heads = set()
    for atoms in atoms_list:
        heads.add(atoms.info.get("head", "Default"))
    heads = list(heads)

    configs = config_from_atoms_list(
        atoms_list,
        config_type_weights=config_type_weights,
        energy_key="energy",
        forces_key="forces",
        stress_key="stress",
        virials_key="virials",
        dipole_key="dipole",
        charges_key="charges",
        head_key="head",
    )
    return atomic_energies_dict, configs, heads

def atoms_from_hdf5(file_path, positions_key='atXYZ', numbers_key="atNUM", energy_key='ePBE0+MBD', forces_key='totFOR'):
    aqm = h5py.File(file_path, "r")
    atoms_list = []
    AQMmol_ids = list(aqm.keys())

    for molid in AQMmol_ids:
        AQMconf_ids = list(aqm[molid].keys())
        for confid in AQMconf_ids:
            curr_cfg = aqm[molid][confid]
            atoms = ase.Atoms(positions=np.array(curr_cfg[positions_key]), numbers=np.array(curr_cfg[numbers_key]))
            atoms.info['energy'] = np.array(curr_cfg[energy_key]).item()
            atoms.arrays['forces'] = np.array(curr_cfg[forces_key])
            # atoms.info['head'] = "Default"
            atoms_list.append(atoms)

    return atoms_list

def atoms_from_hdf5_ani(file_path, positions_key='coordinates', numbers_key="species", energy_key='energies', forces_key='forces'):
    h5 = h5py.File(file_path, "r")
    atoms_list = []
    for num_atoms, properties in tqdm(h5.items()):        #Iterate thorugh like a dictionary
        coordinates = np.array(properties['coordinates'])     #Output of properties is of type h5py Dataset
        species = np.array(properties['species'])
        energies = np.array(properties['energies'])
        forces = np.array(properties['forces'])
        for c, s, e, f in zip(coordinates, species, energies, forces):
            atoms = ase.Atoms(positions=c, numbers=s)
            atoms.info['energy'] = e * ase.units.Hartree # convert to eV
            atoms.arrays['forces'] = f  * ase.units.Hartree # convert to eV
            atoms_list.append(atoms) 
    return atoms_list

def read_atoms_file(identifier):
    return ase.io.read(identifier, index=":")

def read_atoms_jsonbz2_go(identifier):
    trajs = []
    trajs_ids = []
    with bz2.open(identifier, 'rt') as f:
        data = json.load(f)
        
        for entry in data.keys():
            for traj_idx, system in enumerate(data[entry]):
                traj = []
                config_ids = []
                for image_idx, image in enumerate(system['steps']):
                    positions = np.array(
                        [site['xyz'] for site in image['structure']['sites']]
                    )
                    atomic_numbers = np.array(
                        [
                            ase.data.atomic_numbers[site['species'][0]['element']]
                            for site in image['structure']['sites']
                        ]
                    )
                    cell = np.array(image['structure']['lattice']['matrix'])
                    pbc = np.array(image['structure']['lattice']['pbc'])
                    energy = image['energy']
                    forces = np.array(image['forces'])
                    #charges = np.array(
                    #            [site['properties']['charge'] for site in image['structure']['sites']]
                    #        )
                    #magmom = np.array(
                    #            [site['properties']['magmom'] for site in image['structure']['sites']]
                    #        )
                    stress = np.array(image['stress'])
                    
                    # Create the ase.Atoms object
                    atoms = Atoms(
                               numbers=atomic_numbers,  # Atomic numbers (list of integers)
                               positions=positions,     # Cartesian coordinates (Nx3 array)
                               cell=cell,               # Unit cell (3x3 matrix)
                               pbc=pbc                  # Periodic boundary conditions (3 boolean values)
                           )

                    # Add additional properties like energy, forces, and stress if needed
                    atoms.info['energy'] = energy
                    atoms.arrays['forces'] = forces
                    atoms.info['stress'] = stress
                    config_id = os.path.basename(identifier).split('.')[0] + f"-{entry}-{traj_idx}-{image_idx}"
                    atoms.info['alex_config_id'] = config_id
                    traj.append(atoms)
                    config_ids.append(config_id)
                # put into trajs
                trajs.append(traj)
                trajs_ids.append(config_ids)

    total_trajs = len(trajs)
    total_images = sum(len(traj) for traj in trajs)

    trajs, removed_trajs_ids = alex_traj_removing(trajs, trajs_ids)
    trajs, subsampled_trajs_ids = alex_traj_subsampling(trajs, removed_trajs_ids)

    # logging:
    selected_trajs = len([traj for traj in trajs if traj])
    selected_images = sum(len(traj) for traj in trajs)
    

    # Prepare the log content with percentages
    traj_percentage = (selected_trajs / total_trajs) * 100 if total_trajs > 0 else 0
    image_percentage = (selected_images / total_images) * 100 if total_images > 0 else 0
    
    log_content = [
        f"Trajectories: \t{selected_trajs} \t/ \t{total_trajs} \t[{traj_percentage:.2f}%]",
        f"Images: \t{selected_images} \t/ \t{total_images} \t[{image_percentage:.2f}%]\n"
    ]
    
    # Process each trajectory for detailed logging
    for orig_traj_ids, final_traj_ids in zip(trajs_ids, subsampled_trajs_ids):
        if not final_traj_ids:  # This trajectory was removed
            entry = orig_traj_ids[0].split('-')[2]  # Extract entry from the first config_id
            log_content.append(f"{entry}-{traj_idx}: removed")
        else:
            entry = final_traj_ids[0].split('-')[1]
            traj_idx = final_traj_ids[0].split('-')[2]
            kept_images = len(final_traj_ids)
            total_images = len(orig_traj_ids)
            kept_indices = [int(config_id.split('-')[-1]) for config_id in final_traj_ids]
            log_content.append(f"{entry}-{traj_idx}: {kept_images}/{total_images} [{', '.join(map(str, kept_indices))}]")
    
    # Write the log file
    log_filename = f"{os.path.basename(identifier).split('.')[0]}.processing.log"
    with open(log_filename, 'w') as log_file:
        log_file.write('\n'.join(log_content))

    return [atom for sublist in trajs for atom in sublist]

def sample_energy_time_series_reverse(energies, relative_threshold):
    """
    Sample a time series of energy values from last to first, pruning away consecutive repetitive steps.

    :param energies: List or array of energy values
    :param relative_threshold: Relative threshold for considering a change significant
    :return: Array of indices of sampled points, in ascending order
    """
    sampled_indices = [len(energies) - 1]  # Always include the last point
    last_sampled_energy = energies[-1]

    for i in range(len(energies) - 2, -1, -1):
        current_energy = energies[i]
        threshold = relative_threshold * abs(last_sampled_energy)
        under = 0.3 * abs(last_sampled_energy) # TODO add comments

        if abs(current_energy - last_sampled_energy) > threshold and abs(current_energy - last_sampled_energy) < under:
            sampled_indices.append(i)
            last_sampled_energy = current_energy

    # Always include the first point if it's not already included
    if sampled_indices[-1] != 0:
        sampled_indices.append(0)

    return np.array(sorted(sampled_indices))  # Return indices in ascending order

def max_stress(atom):
    stress = atom.info['stress']
    max_stress = np.max(np.abs(stress))
    return max_stress

def max_forces(atom):
    return np.linalg.norm(atom.arrays["forces"], axis=-1).max()

def max_stress_forces_energy_per_atom(traj):
    max_stress_value = -np.inf
    max_forces_value = -np.inf
    max_energy_per_atom_value = -np.inf
    for atoms in traj:
        stress_value = max_stress(atoms)
        forces_value = max_forces(atoms)
        energy_per_atom_value = atoms.info['energy'] / len(atoms)

        if stress_value > max_stress_value:
            max_stress_value = stress_value

        if forces_value > max_forces_value:
            max_forces_value = forces_value

        if energy_per_atom_value > max_energy_per_atom_value:
            max_energy_per_atom_value = energy_per_atom_value
    return max_stress_value, max_forces_value, max_energy_per_atom_value

def alex_traj_removing(trajs, trajs_ids):
    filtered_trajs = []
    filtered_trajs_ids = []
    for traj, config_ids in zip(trajs, trajs_ids):
        stress_value, forces_value, energy_per_atom_value = max_stress_forces_energy_per_atom(traj)
        final_forces_norm = np.linalg.norm(traj[-1].arrays['forces'], axis=-1).max()
        
        # Check if the trajectory meets the criteria
        if (stress_value <= 500 and 
            forces_value <= 300 and 
            forces_value > 0.0 and
            energy_per_atom_value <= 2.0 and 
            len(traj) >= 4 and
            final_forces_norm <= 0.005):
            filtered_trajs.append(traj)
            filtered_trajs_ids.append(config_ids)
        else:
            filtered_trajs.append([])
            filtered_trajs_ids.append([])
    
    return filtered_trajs, filtered_trajs_ids

def alex_traj_subsampling(trajs, trajs_ids):
    subsampled_trajs = []
    subsampled_trajs_ids = []
    
    alex_e0s = {1: -1.11734008, 2: 0.00096759, 3: -0.29754725, 4: -0.01781697, 5: -0.26885011, 6: -1.26173507, 7: -3.12438806, 8: -1.54838784, 9: -0.51882044, 10: -0.01241601, 11: -0.22883163, 12: -0.00951015, 13: -0.21630193, 14: -0.8263903, 15: -1.88816619, 16: -0.89160769, 17: -0.25828273, 18: -0.04925973, 19: -0.22697913, 20: -0.0927795, 21: -2.11396364, 22: -2.50054871, 23: -3.70477179, 24: -5.60261985, 25: -5.32541181, 26: -3.52004933, 27: -1.93555024, 28: -0.9351969, 29: -0.60025846, 30: -0.1651332, 31: -0.32990651, 32: -0.77971828, 33: -1.68367812, 34: -0.76941032, 35: -0.22213843, 36: -0.0335879, 37: -0.1881724, 38: -0.06826294, 39: -2.17084228, 40: -2.28579303, 41: -3.13429753, 42: -4.60211419, 43: -3.45201492, 44: -2.38073513, 45: -1.46855515, 46: -1.4773126, 47: -0.33954585, 48: -0.16843877, 49: -0.35470981, 50: -0.83642657, 51: -1.41101987, 52: -0.65740879, 53: -0.18964571, 54: -0.00857582, 55: -0.13771876, 56: -0.03457659, 57: -0.45580806, 58: -1.3309175, 59: -0.29671824, 60: -0.30391193, 61: -0.30898427, 62: -0.25470891, 63: -8.38001538, 64: -10.38896525, 65: -0.3059505, 66: -0.30676216, 67: -0.30874667, 68: -0.31610927, 69: -0.25190039, 70: -0.06431414, 71: -0.31997586, 72: -3.52770927, 73: -3.54492209, 74: -4.65658356, 75: -4.70108713, 76: -2.88257209, 77: -1.46779304, 78: -0.50269936, 79: -0.28801193, 80: -0.12454674, 81: -0.31737194, 82: -0.77644932, 83: -1.32627283, 89: -0.26827152, 90: -0.90817426, 91: -2.47653193, 92: -4.90438537, 93: -7.63378961, 94: -10.77237713}
    
    for traj, config_ids in zip(trajs, trajs_ids):
        # Skip empty trajectories
        if not traj:
            subsampled_trajs.append([])
            subsampled_trajs_ids.append([])
            continue
        
        zs = traj[0].numbers
        E0 = sum(alex_e0s[z] for z in zs if z in alex_e0s)
        
        # Remove first image if trajectory has more than one atom
        atom_list = traj[1:] if len(traj) > 1 else traj
        config_list = config_ids[1:] if len(traj) > 1 else config_ids
        
        # Extract energies
        try:
            energies = [(atom.info['energy'] - E0) for atom in atom_list]
        except KeyError:
            print(f"Warning: 'energy' not found in atom.info for a trajectory. Skipping this trajectory.")
            # subsampled_trajs.append(traj)  # Keep the original trajectory
            continue
        
        # Sample indices
        indices = sample_energy_time_series_reverse(energies, relative_threshold=0.001)
        
        # Create subsampled trajectory
        subsampled_traj = [atom_list[i] for i in indices]
        subsampled_configs = [config_list[i] for i in indices]
        subsampled_trajs.append(subsampled_traj)
        subsampled_trajs_ids.append(subsampled_configs)
    
    return subsampled_trajs, subsampled_trajs_ids

def read_atoms_jsonbz2(identifier):
    atom_list = []
    with bz2.open(identifier, 'rt') as f:
        data = json.load(f)
        
        for entry in data.keys():
            for system in data[entry]:
                image = system
                #for image in system['steps']:
                positions = np.array(
                    [site['xyz'] for site in image['structure']['sites']]
                )
                atomic_numbers = np.array(
                    [
                        ase.data.atomic_numbers[site['species'][0]['element']]
                        for site in image['structure']['sites']
                    ]
                )
                cell = np.array(image['structure']['lattice']['matrix'])
                pbc = np.array(image['structure']['lattice']['pbc'])
                energy = image['energy']
                forces = np.array(
                            [site['properties']['forces'] for site in image['structure']['sites']]
                        )
                #charges = np.array(
                #            [site['properties']['charge'] for site in image['structure']['sites']]
                #        )
                #magmom = np.array(
                #            [site['properties']['magmom'] for site in image['structure']['sites']]
                #        )
                stress = np.array(image['data']['stress'])
                
                # Create the ase.Atoms object
                atoms = Atoms(
                           numbers=atomic_numbers,  # Atomic numbers (list of integers)
                           positions=positions,     # Cartesian coordinates (Nx3 array)
                           cell=cell,               # Unit cell (3x3 matrix)
                           pbc=pbc                  # Periodic boundary conditions (3 boolean values)
                       )

                # Add additional properties like energy, forces, and stress if needed
                atoms.info['energy'] = energy
                atoms.arrays['forces'] = forces
                atoms.info['stress'] = stress
                atom_list.append(atoms)
    return atom_list


def atoms_from_oc20(file_path, positions_key='coordinates', numbers_key="species", energy_key='energies', forces_key='forces'):
    filenames = [f for f in os.listdir(file_path) if f.endswith(".extxyz")]
    identifiers = [os.path.join(file_path, f) for f in filenames if f.endswith(".extxyz")]
    
    with mp.Pool(N_PROC) as pool:
        results = list(tqdm(pool.imap(read_atoms_file, identifiers), total=len(identifiers)))
    
    # Flatten the list of lists
    atoms_list = [atom for sublist in results for atom in sublist]
    
    return atoms_list


def atoms_from_alex_go(file_path, positions_key='coordinates', numbers_key="species", energy_key='energies', forces_key='forces'):
    filenames = [f for f in os.listdir(file_path) if f.endswith(".json.bz2") and f.startswith("alex_go")]
    identifiers = [os.path.join(file_path, f) for f in filenames]
    
    with mp.Pool(N_PROC) as pool:
        results = list(tqdm(pool.imap(read_atoms_jsonbz2_go, identifiers), total=len(identifiers)))
    
    # Flatten the list of lists
    atoms_list = [atom for sublist in results for atom in sublist]
    
    return atoms_list

def atoms_from_alex(file_path, positions_key='coordinates', numbers_key="species", energy_key='energies', forces_key='forces'):
    filenames = [f for f in os.listdir(file_path) if f.endswith(".json.bz2") and f.startswith("alexandria")]
    identifiers = [os.path.join(file_path, f) for f in filenames]
    
    with mp.Pool(N_PROC) as pool:
        results = list(tqdm(pool.imap(read_atoms_jsonbz2, identifiers), total=len(identifiers)))
    
    # Flatten the list of lists
    atoms_list = [atom for sublist in results for atom in sublist]
    
    return atoms_list


def compute_average_E0s( # TODO: compute species dependant
    collections_train: Configurations, z_table: AtomicNumberTable, heads: List[str]
) -> Dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(collections_train)
    len_zs = len(z_table)
    atomic_energies_dict = {}
    for head in heads:
        A = np.zeros((len_train, len_zs))
        B = np.zeros(len_train)
        if head not in atomic_energies_dict:
            atomic_energies_dict[head] = {}
        for i in range(len_train):
            if collections_train[i].head != head:
                continue
            B[i] = collections_train[i].energy
            for j, z in enumerate(z_table.zs):
                A[i, j] = np.count_nonzero(collections_train[i].atomic_numbers == z)
        try:
            E0s = np.linalg.lstsq(A, B, rcond=None)[0]
            for i, z in enumerate(z_table.zs):
                atomic_energies_dict[head][z] = E0s[i]
        except np.linalg.LinAlgError:
            logging.warning(
                "Failed to compute E0s using least squares regression, using the same for all atoms"
            )
            for i, z in enumerate(z_table.zs):
                atomic_energies_dict[head][z] = 0.0
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
            grp["head"] = data.head


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
    grp["head"] = data.head


def save_configurations_as_HDF5(configurations: Configurations, _, h5_file) -> None:
    grp = h5_file.create_group("config_batch_0")
    for j, config in enumerate(configurations):
        subgroup_name = f"config_{j}"
        subgroup = grp.create_group(subgroup_name)
        subgroup["atomic_numbers"] = write_value(config.atomic_numbers)
        subgroup["positions"] = write_value(config.positions)
        subgroup["energy"] = write_value(config.energy)
        subgroup["forces"] = write_value(config.forces)
        subgroup["stress"] = write_value(config.stress)
        subgroup["virials"] = write_value(config.virials)
        subgroup["head"] = write_value(config.head)
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
        if config.alex_config_id is not None:
            subgroup["alex_config_id"] = write_value(config.alex_config_id)


def write_value(value):
    return value if value is not None else "None"
