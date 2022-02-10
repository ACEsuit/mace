from typing import Any, Callable, Dict, Type

import numpy as np
import torch
from LieACE.data import AtomicData
from LieACE.tools.degree import NaiveMaxDeg
from LieCG.CG_coefficients.CG_rot import Rot3DCoeffs, create_U
from e3nn import o3
from torch_scatter import scatter_sum

from LieACE.modules.blocks import (AtomicEnergiesBlock, LinearNodeEmbeddingBlock, LinearReadoutBlock, NonLinearReadoutBlock, RadialEmbeddingBlock, 
                    InteractionBlock, ProductBasisBlock, ScaleShiftBlock)
from LieACE.modules.spherical_harmonics import SphericalHarmonics
from LieACE.modules.utils import (compute_forces, get_edge_vectors_and_lengths)



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
        device: str='cpu',
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
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps*num_features).sort()[0].simplify()
        
        self.spherical_harmonics = SphericalHarmonics(lmax=max_ell,device=device)
        A = Rot3DCoeffs(max_ell + 1)
        degree_func = NaiveMaxDeg({i : [0,max_ell]for i in range(1,correlation+1)})
        U_tensors = {nu : create_U(A=A,nu=nu,degree_func=degree_func).type(torch.complex128).to(device)
                     for nu in range(1,correlation+1)}
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            num_avg_neighbors=num_avg_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])
        
        node_feats_irreps_out = inter.irreps_out
        prod = ProductBasisBlock(
                 U_tensors=U_tensors,
                 node_feats_irreps=node_feats_irreps_out,
                 target_irreps=hidden_irreps,
                 correlation=correlation,
         )
        self.product = torch.nn.ModuleList([prod]) 
        
        for _ in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                num_avg_neighbors=num_avg_neighbors,
            )
            self.interactions.append(inter)
            prod = ProductBasisBlock(
                    U_tensors=U_tensors,
                    node_feats_irreps=interaction_irreps,
                    target_irreps=hidden_irreps,
                    correlation=correlation,
            )
            self.product.append(prod)

        self.readouts = torch.nn.ModuleList([LinearReadoutBlock(hidden_irreps)])

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
        edge_attrs = self.spherical_harmonics(vectors)*torch.sqrt(4 * torch.tensor(np.pi, dtype=torch.cdouble))
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        for interaction,product in zip(self.interactions,self.product):
            node_feats = interaction(node_attrs=data.node_attrs,
                                     node_feats=node_feats,
                                     edge_attrs=edge_attrs,
                                     edge_feats=edge_feats,
                                     edge_index=data.edge_index)
            node_feats = product(node_feats=node_feats)

        node_inter_es = self.readouts[0](node_feats).squeeze(-1)  # {[n_nodes, ], }

        # Sum over nodes in graph
        inter_e = scatter_sum(src=node_inter_es, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        total_e = e0 + inter_e

        return {
            'energy': total_e,
            'forces': compute_forces(energy=total_e, positions=data.positions, training=training),
        }

class NonLinearBodyOrderedModel(torch.nn.Module):
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
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        num_avg_neighbors: float,
        correlation: int,
        gate: Callable,
        device: str='cpu',
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
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps*num_features).sort()[0].simplify()
        
        self.spherical_harmonics = SphericalHarmonics(lmax=max_ell,device=device)
        A = Rot3DCoeffs(max_ell + 1)
        degree_func = NaiveMaxDeg({i : [0,max_ell]for i in range(1,correlation+1)})
        U_tensors = {nu : create_U(A=A,nu=nu,degree_func=degree_func).type(torch.complex128).to(device)
                     for nu in range(1,correlation+1)}
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            num_avg_neighbors=num_avg_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])
        
        node_feats_irreps_out = inter.irreps_out
        prod = ProductBasisBlock(
                 U_tensors=U_tensors,
                 node_feats_irreps=node_feats_irreps_out,
                 target_irreps=hidden_irreps,
                 correlation=correlation,
         )
        self.product = torch.nn.ModuleList([prod]) 

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))
        
        for i in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                num_avg_neighbors=num_avg_neighbors,
            )
            self.interactions.append(inter)
            prod = ProductBasisBlock(
                    U_tensors=U_tensors,
                    node_feats_irreps=interaction_irreps,
                    target_irreps=hidden_irreps,
                    correlation=correlation,
            )
            self.product.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(NonLinearReadoutBlock(inter.irreps_out, MLP_irreps, gate))
            else:
                self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        self.readouts = torch.nn.ModuleList([LinearReadoutBlock(hidden_irreps)])

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
        edge_attrs = self.spherical_harmonics(vectors)*torch.sqrt(4 * torch.tensor(np.pi, dtype=torch.cdouble))
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        for interaction,product,readout in zip(self.interactions,self.product,self.readouts):
            node_feats = interaction(node_attrs=data.node_attrs,
                                     node_feats=node_feats,
                                     edge_attrs=edge_attrs,
                                     edge_feats=edge_feats,
                                     edge_index=data.edge_index)
            node_feats = product(node_feats=node_feats)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(src=node_energies, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]


        return {
           'energy': total_energy,
           'contributions': contributions,
           'forces': compute_forces(energy=total_energy, positions=data.positions, training=training),
        }



class ScaleShiftNonLinearBodyOrderedModel(NonLinearBodyOrderedModel):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(scale=atomic_inter_scale, shift=atomic_inter_shift)

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
        node_es_list = []
        for interaction,product,readout in zip(self.interactions,self.product,self.readouts):
            node_feats = interaction(node_attrs=data.node_attrs,
                                     node_feats=node_feats,
                                     edge_attrs=edge_attrs,
                                     edge_feats=edge_feats,
                                     edge_index=data.edge_index)
            node_feats = product(node_feats=node_feats)
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(src=node_inter_es, index=data.batch, dim=-1, dim_size=data.num_graphs)  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e

        output = {
            'energy': total_e,
            'forces': compute_forces(energy=total_e, positions=data.positions, training=training),
        }

        return output
