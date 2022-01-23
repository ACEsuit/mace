import torch

from typing import Dict
from os import sys



sys.path.append(r'C:\Users\Lilyes\Documents\GitHub\LieCG\LieACE')
sys.path.append(r'C:\Users\Lilyes\Documents\GitHub\LieCG\LieCG')
import torch
import numpy as np
from CG_coefficients.CG_lorentz import CGDict
from functions.Zonal_functions import ZonalFunctions
from Cutoff import PolynomialCutoff
from Radial_basis import BesselBasis



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
        #TODO Species basis

    def forward(
            self,
            data: Dict[str, torch.Tensor],  # [n_nodes,1] if invariant [n_nodes,3] if equivariant
    ) -> Dict[str, torch.Tensor]:
        #data.edge_vectors: torch.Tensor,  # [n_edges, 3]
        #data.edge_lengths: torch.Tensor,  # [n_edges, 1]
        sender, receiver = data.edge_index

        radial = self.radial_fn(data.edge_lengths)  # [n_edges, 1, num_basis]
        cutoff = self.cutoff_fn(data.edge_lengths).unsqueeze(-1)  # [n_edges, 1]
        radial = radial * cutoff  # [n_edges, 1, num_basis]
    
        
        ylm = self.zf(data.edge_vectors)  # [n_edges, lmax*2 + 2*lmax +1]
        ylm_r = ylm[0]  # [n_edges, lmax*2 + 2*lmax +1]
        ylm_i = ylm[1]   # [n_edges, lmax*2 + 2*lmax +1]
    

        combined_r = torch.einsum('bi,bk,bj -> bkij', radial.view(radial.size()[0],
                                                                      radial.size()[-1]), data.node_attrs[sender],
                                      ylm_r)  # [n_edges, n_basis , lmax*2 + 2*lmax +1] real part of radial embedding
        combined_i = torch.einsum('bi,bk,bj -> bkij', radial.view(radial.size()[0],
                                                                      radial.size()[-1]), data.node_attrs[sender],
                                      ylm_i)  # [n_edges, n_basis , lmax*2 + 2*lmax +1] imag part of radial embedding


        data['radial_features'] = (combined_r, combined_i)  # [n_edges, n_basis , lmax*2 + 2*lmax +1]

        return data