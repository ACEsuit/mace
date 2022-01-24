from typing import Iterable, Sequence, Dict, List

import numpy as np
import torch.utils.data
import torch_geometric

from ace_torch.data.neighborhood import get_neighborhood
from ace_torch.utils.config import Configuration
import ase

import importlib
mod = importlib.import_module('ace_torch.functions.degree') #to be able to parse any degree

class AtomicNumberTable:
    """From Gregor"""
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f'AtomicNumberTable: {tuple(s for s in self.zs)}'

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


def atomic_numbers_to_indices(atomic_numbers: np.ndarray, z_table: AtomicNumberTable) -> np.ndarray:
    to_index_fn = np.vectorize(lambda z: z_table.z_to_index(z))
    return to_index_fn(atomic_numbers)


def to_one_hot(indices: torch.Tensor, num_classes: int, device=None) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes, )
    oh = torch.zeros(shape, device=device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)

def config_from_atoms(atoms: ase.Atoms) -> Configuration:
    energy = atoms.info.get('DFT_energy', None)
    if energy is not None:
        energy = float(energy)
    else :
        energy = atoms.get_potential_energy()

    forces = None
    if atoms.has('forces'):
        forces = atoms.get_forces()

    atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])

    return Configuration(atomic_numbers=atomic_numbers, positions=atoms.positions, energy=energy, forces=forces)


class AtomicData(torch_geometric.data.Data):
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    
    def __init__(
            self,
            edge_index: torch.Tensor,  # [2, n_edges]
            node_attrs: torch.Tensor,  # [n_nodes, n_node_feats]
            positions: torch.Tensor,  # [n_nodes, 3]
            shifts : torch.Tensor, # [n_edges, 3],
            forces: torch.Tensor,  # [n_nodes, 3]
            energy: torch.Tensor,  # [, ]
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]

        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert len(node_attrs.shape) == 2
        assert forces.shape == (num_nodes, 3)
        assert len(energy.shape) == 0

        # Aggregate data
        data = {
            'num_nodes': num_nodes,
            'edge_index': edge_index,
            'positions': positions,
            'shifts' : shifts,
            'node_attrs': node_attrs,
            'forces': forces,
            'energy': energy,
        }
        super().__init__(**data)

    @classmethod
    def from_config(cls, config: Configuration, z_table: AtomicNumberTable, cutoff: float) -> 'AtomicData':
        edge_index, shifts = get_neighborhood(positions=config.positions, cutoff=cutoff)
        indices = atomic_numbers_to_indices(config.atomic_numbers, z_table=z_table)
        one_hot = to_one_hot(torch.tensor(indices, dtype=torch.long).unsqueeze(-1), num_classes=len(z_table))
        
        forces = torch.tensor(config.forces, dtype=torch.get_default_dtype()) if config.forces is not None else None
        energy = torch.tensor(config.energy, dtype=torch.get_default_dtype()) if config.energy is not None else None

        return cls(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            node_attrs=one_hot,
            forces=forces,
            energy=energy,
        )

def species_degrees(degrees):
    """Return a list of degrees for each species
    ::degrees a list of dict [n_species] 
    [{'type' : 'NaiveMaxDeg',1:[5,2],2:[5,2]},{'type' : 'NaiveMaxDeg',1:[5,2],2:[5,2]}]
    return a List of degrees class"""
    node_degree = []
    for elem in range(len(degrees)): #Go over the elements 
            func = getattr(mod,degrees[elem]['type']) #select the degree type for each element
            exclude_keys = ['type',]
            arg_deg = {k: degrees[elem][k] for k in set(list(degrees[elem].keys())) - set(exclude_keys)}
            node_degree += [func(arg_deg)]
    return node_degree

def get_data_loader(
    dataset: Sequence[AtomicData],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )