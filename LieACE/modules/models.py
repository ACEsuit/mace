from typing import Any, Dict, List, Type

import numpy as np
import torch
from LieACE.data import AtomicData
from LieACE.tools.degree import NaiveMaxDeg
from LieCG.CG_coefficients.CG_rot import Rot3DCoeffs, create_U
from e3nn import o3
from torch_scatter import scatter_sum

from .blocks import (AtomicEnergiesBlock, LinearNodeEmbeddingBlock, LinearReadoutBlock, NonLinearBlock, RadialEmbeddingBlock, 
                    InteractionBlock, ProductBasisBlock)
from .spherical_harmonics import SphericalHarmonics
from .utils import (compute_forces, create_U_element,
                    get_edge_vectors_and_lengths)



class InvariantMultiACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        num_avg_neighbors: float,
        correlation: int,
    ):
        super().__init__()

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(irreps_in=node_attr_irreps, irreps_out=node_feats_irreps)
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f'{self.radial_embedding.out_dim}x0e')

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = SphericalHarmonics(lmax=max_ell)
        A = Rot3DCoeffs(max_ell + 1)
        degree_func = NaiveMaxDeg({i : [0,max_ell]} for i in range(1,correlation+1))
        U_tensors = {nu : create_U(A=A,nu=nu,degree_func=degree_func) for nu in range(1,correlation+1)}
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
            num_avg_neighbors=num_avg_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])

        prod = ProductBasisBlock(
            U_tensors = U_tensors,
            num_features = node_feats_irreps,
            correlation = correlation,
        )
        self.product = torch.nn.ModuleList([prod]) 
        for _ in range(num_interactions - 1):
            node_feats_irreps_out = inter.irreps_out
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=hidden_irreps,
                num_avg_neighbors=num_avg_neighbors,
            )
            self.interactions.append(inter)
            prod = ProductBasisBlock(
            U_tensors = U_tensors,
                    node_feats_irreps = node_feats_irreps_out,
                    correlation = correlation,
            )
            self.product.append(prod)

        self.readouts = torch.nn.ModuleList([LinearReadoutBlock(self.interactions[-1].irreps_out)])

    def forward(self, data: AtomicData, training=False) -> Dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(src=node_e0, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(positions=data.positions,
                                                        edge_index=data.edge_index,
                                                        shifts=data.shifts)
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        for interaction in self.interactions:
            node_feats = interaction(node_attrs=data.node_attrs,
                                     node_feats=node_feats,
                                     edge_attrs=edge_attrs,
                                     edge_feats=edge_feats,
                                     edge_index=data.edge_index)
            node_feats = self.product(node_feats=node_feats)

        node_inter_es = self.readouts[0](node_feats).squeeze(-1)  # {[n_nodes, ], }

        # Sum over nodes in graph
        inter_e = scatter_sum(src=node_inter_es, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        total_e = e0 + inter_e

        return {
            'energy': total_e,
            'forces': compute_forces(energy=total_e, positions=data.positions, training=training),
        }
