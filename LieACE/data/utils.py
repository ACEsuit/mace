import io
import logging
import os
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Sequence

import ase.data
import ase.io
import numpy as np

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom


Configurations = List[Configuration]


def random_train_valid_split(items: Sequence, valid_fraction: float, seed: int) -> Tuple[List, List]:
    assert 0.0 < valid_fraction < 1.0

    size = len(items)
    train_size = size - int(valid_fraction * size)

    indices = list(range(size))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return [items[i] for i in indices[:train_size]], [items[i] for i in indices[train_size:]]


def download_url(url: str, save_path: str) -> None:
    with urllib.request.urlopen(url) as download_file:
        with open(save_path, 'wb') as out_file:
            out_file.write(download_file.read())


def fetch_archive(path: str, url: str, force_download=False) -> None:
    if not os.path.exists(path) and not force_download:
        logging.info(f'Downloading {url} to {path}')
        download_url(url=url, save_path=path)
    else:
        logging.info(f'File {path} exists')


def config_from_atoms(atoms: ase.Atoms, energy_key='energy', forces_key='forces') -> Configuration:
    energy = atoms.info.get(energy_key, None)  # eV
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])
    return Configuration(atomic_numbers=atomic_numbers, positions=atoms.positions, energy=energy, forces=forces)


def unpack_configs_from_archive(path: str, mode='r|gz') -> Dict[str, Configurations]:
    extracted_data: Dict[str, Configurations] = {}
    with tarfile.open(name=path, mode=mode) as tar_file:
        for file in tar_file:
            basename = os.path.basename(file.name)
            root, ext = os.path.splitext(basename)
            if file.isfile() and ext == '.xyz':
                extracted_file = tar_file.extractfile(file)
                if extracted_file:
                    content = io.StringIO(extracted_file.read().decode('ascii'))
                    configs = [config_from_atoms(config) for config in ase.io.read(content, format='extxyz', index=':')]
                    extracted_data[root] = configs
                else:
                    raise RuntimeError(f'Cannot read file: {file.name}')

    return extracted_data