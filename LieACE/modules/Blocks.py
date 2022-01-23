import torch

from typing import Dict, List
from os import sys



sys.path.append(r'C:\Users\Lilyes\Documents\GitHub\LieCG\LieACE')
sys.path.append(r'C:\Users\Lilyes\Documents\GitHub\LieCG\LieCG')
import torch
import numpy as np
from CG_coefficients.CG_lorentz import CGDict
from functions.Zonal_functions import ZonalFunctions
from Cutoff import PolynomialCutoff
from Radial_basis import BesselBasis
from torch_scatter import scatter



class EdgeEmbeddingBlock(torch.nn.Module):
    def __init__(self,
                 lmax: int,
                 r_cut: float,
                 nmax: int = 8,
                 num_polynomial_cutoff: int = 6,
                 ):
        super().__init__()
        cg_dict = CGDict(maxdim = lmax)
        self.zf = ZonalFunctions(maxdim=lmax,cg_dict = cg_dict)
        self.radial_fn = BesselBasis(r_max=r_cut, num_basis=nmax)  #one can specify any radial_basis
        self.cutoff_fn = PolynomialCutoff(r_max=r_cut, p=num_polynomial_cutoff)  #anycutoff
        self.factor4pi = torch.sqrt(4 * torch.tensor(np.pi, dtype=torch.float64))

    def forward(
            self,
            edge_index : torch.Tensor, 
            edge_lenghts : torch.Tensor, # [n_edges, 1]
            edge_vectors : torch.Tensor,  # [n_edges, 3]
            node_attrs : torch.Tensor, 
    ) -> List[torch.tensor,torch.tensor]:
        #data.
        #data.edge_lengths: torch.Tensor,  # [n_edges, 1]
        sender, receiver = edge_index
        radial = self.radial_fn(edge_lenghts)  # [n_edges, 1, num_basis]
        cutoff = self.cutoff_fn(edge_lenghts).unsqueeze(-1)  # [n_edges, 1]
        radial = radial * cutoff  # [n_edges, 1, num_basis]
        ylm = self.zf(edge_vectors)  # [n_edges, lmax*2 + 2*lmax +1]
        ylm_r = ylm[0]  # [n_edges, lmax*2 + 2*lmax +1]
        ylm_i = ylm[1]   # [n_edges, lmax*2 + 2*lmax +1]
        combined_r = torch.einsum('bi,bk,bj -> bkij', radial.view(radial.size()[0],
                                                                      radial.size()[-1]), node_attrs[sender],
                                      ylm_r)  # [n_edges, n_basis , lmax*2 + 2*lmax +1] real part of radial embedding
        combined_i = torch.einsum('bi,bk,bj -> bkij', radial.view(radial.size()[0],
                                                                      radial.size()[-1]), node_attrs[sender],
                                      ylm_i)  # [n_edges, n_basis , lmax*2 + 2*lmax +1] imag part of radial embedding
        return (combined_r, combined_i) # [n_edges, n_basis , lmax*2 + 2*lmax +1]



class atomic_base(torch.nn.Module):
    """ Create the Atomic base from pooling 1-particle basis"""
    def __init__(self, ):
        super().__init__()

    def forward(self, 
                edge_index: torch.tensor,
                radial_feature: torch.tensor,
                node_feats: torch.tensor,) -> List[torch.tensor,torch.tensor]:
            
        sender, receiver = edge_index  # The graph connectivity
        num_nodes = node_feats.shape[0]
        combined_r = torch.einsum(
                'bkij,bl -> bkij',radial_feature[0],node_feats[sender])  # [n_edges,n_species, n_basis , lmax*2 + 2*lmax +1] real part of radial embedding
        combined_i = torch.einsum(
                'bkij,bl -> bkij',radial_feature[1],node_feats[sender])  # [n_edges,n_species, n_basis , lmax*2 + 2*lmax +1] imag part of radial embedding
        edges_features = (combined_r, combined_i)
        A_nlm_real = scatter(edges_features[0], index=receiver, dim=0, dim_size=num_nodes,
                             reduce='sum')  #size [num_nodes,n,lmax**2 + 2*lmax + 1]
        A_nlm_imag = scatter(edges_features[1], index=receiver, dim=0, dim_size=num_nodes,
                             reduce='sum')  #size [num_nodes,n,lmax**2 + 2*lmax + 1]
        node_feats = (A_nlm_real, A_nlm_imag)
        return node_feats, edges_features