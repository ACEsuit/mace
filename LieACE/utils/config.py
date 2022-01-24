import logging

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import ase.data
import ase.io


Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # kJ/mol
    forces: Optional[Forces] = None  # kJ/mol/Angstrom


Configurations = List[Configuration]

def config_from_atoms(atoms: ase.Atoms) -> Configuration:
    energy = atoms.info.get('E', None)
    if energy is not None:
        energy = float(energy)
        
    if energy is None :
        energy = float(atoms.get_potential_energy())
        
    
    forces = None
    if atoms.has('forces'):
        forces = atoms.get_forces()
        
    

    atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])

    return Configuration(atomic_numbers=atomic_numbers, positions=atoms.positions, energy=energy, forces=forces)


def load_xyz(path: str, formatting: str = 'extxyz') -> Configurations:
    logging.info(f"Loading dataset from '{path}' (format={formatting})")
    atoms_list = ase.io.read(path, ':', format=formatting)
    configs = [config_from_atoms(atoms) for atoms in atoms_list]
    logging.info(f'Number of configurations: {len(configs)}')
    return configs