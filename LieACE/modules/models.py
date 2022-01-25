from typing import Dict, Any, Type, List

import torch
import numpy as np

from .blocks import (LinearNodeEmbeddingBlock, NonLinearBlock, AtomicEnergiesBlock, ProdBasisBlock, RadialEmbeddingBlock,
                    EdgeEmbeddingBlock, AtomicBaseBlock,  VectorizeBlock)
from .utils import create_U_element, get_edge_vectors_and_lengths
from LieACE.data import AtomicData
from spherical_harmonics import SphericalHarmonics

from torch_scatter import scatter_sum


class InvariantMultiACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        degrees: List,
        num_bessel: float,
        lmax: float,
        num_polynomial_cutoff: int,
        num_elements: int,
        hidden_features: int,
        num_layers: int,
        atomic_energies: np.ndarray,
        A: Dict[int,Any],#Rot3DCoeff
        non_linear: bool,
        device = 'cpu',
    ):
        super().__init__()  
        #Embedding
        self.num_elements = num_elements
        self.node_embedding = LinearNodeEmbeddingBlock(num_in = num_elements, num_out = hidden_features) #change to higher embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        self.spherical_harmonics = SphericalHarmonics(lmax=lmax)
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)


        #Atomic Basis
        self.atomic_basis = torch.nn.ModuleList()
        self.prod_basis = torch.nn.ModuleList()

        self.atomic_basis.append(AtomicBaseBlock())
        self.prod_basis.append(ProdBasisBlock(degrees = self.degrees, num_elements = num_elements, A = A, device = device))

        for _ in range(num_layers - 1):
            self.atomic_basis.append(AtomicBaseBlock())
            self.prod_basis.append(ProdBasisBlock(degrees = self.degrees, num_elements = num_elements, A = A, device = device))
        self.readouts = torch.nn.ModuleList([(self.prod_basis[-1].out,1 )])

    def forward(self, data: AtomicData) -> Dict[str, Any]:
        #setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        #Embedding 
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lenghts = get_edge_vectors_and_lengths(positions=data.positions,
                                                        edge_index=data.edge_index,
                                                        shifts=data.shifts)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lenghts)


            
        return data