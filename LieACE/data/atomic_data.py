from typing import  Sequence, Optional

import torch.utils.data
import torch_geometric

import importlib
from LieACE.data.neighborhood import get_neighborhood
from LieACE.tools.config import Configuration
from LieACE.tools.torch_tools import to_one_hot
from LieACE.tools.utils import AtomicNumberTable, atomic_numbers_to_indices
mod = importlib.import_module('LieACE.tools.degree') #to be able to parse any degree


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
            forces: Optional[torch.Tensor],  # [n_nodes, 3]
            energy: Optional[torch.Tensor],  # [, ]
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]

        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert len(node_attrs.shape) == 2
        assert forces is None or forces.shape == (num_nodes, 3)
        assert energy is None or len(energy.shape) == 0
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