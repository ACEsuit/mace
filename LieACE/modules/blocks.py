from abc import ABC, abstractmethod
from collections import OrderedDict
from os import sys
from typing import Dict, List, Union

import numpy as np
import torch
from e3nn import nn, o3
from torch_scatter import scatter_sum

from sparse_tools import tensor_contract_nd_update_sparse, vector_contract

from .irreps_tools import tp_out_irreps_with_instructions
from .radial import BesselBasis, PolynomialCutoff
from .utils import c_tildes_weight, create_U_element


class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(
            self,
            node_attrs: torch.Tensor,  # [n_nodes, irreps]
    ):
        return self.linear(node_attrs)

class NonLinearBlock(torch.nn.Module):
    def __init__(self, gate : torch.nn.Module):
        super().__init__()
        self.non_linearity = gate

    def forward(
            self,
            x: torch.Tensor  # [n_nodes, 1]
    ) -> torch.Tensor:  # [..., ]
        return self.gate(x)  # [n_nodes, 1]

class LinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=o3.Irreps('0e'))

    def forward(
            self,
            x: torch.Tensor  # [n_nodes, irreps]
    ) -> torch.Tensor:  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        assert len(atomic_energies.shape) == 1

        self.register_buffer('atomic_energies', torch.tensor(atomic_energies,
                                                             dtype=torch.get_default_dtype()))  # [n_elements, ]

    def forward(
            self,
            x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, self.atomic_energies)

    def __repr__(self):
        formatted_energies = ', '.join([f'{x:.4f}' for x in self.atomic_energies])
        return f'{self.__class__.__name__}(energies=[{formatted_energies}])'

class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(self, r_max: float, num_bessel: int, num_polynomial_cutoff: int):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
            self,
            edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        return bessel * cutoff  # [n_edges, n_basis]


class ProductBasisBlock(torch.nn.Module):
    def __init__(self,
                 U_tensors: Dict[int,torch.tensor],
                 node_feats_irreps: o3.Irreps,
                 correlation: int,
        ) -> None:
        super().__init__()  
        self.U_tensors = U_tensors   #Dict[str,[(lmax+1)**2]**correlation + [num_weights]]  
        self.num_features = node_feats_irreps.count((0,1))
        self.correlation = correlation
        
        #Tensor contraction equations
        self.equation_main = '...i' + 'k,k,bi->b' + '...'
        self.equation_weighting = '...k,k->...'
        self.equation_contract = 'b...i,bi->b' + '...'

        #Create weight for product basis
        self.weights = torch.nn.ParameterDict({})
        for i in range(1,correlation+1):
            num_params = self.U_tensors[i].size[-1]
            params_list = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(num_params)) for i in range(self.num_features)])
            self.weights[i] = torch.nn.Parameter(params_list)

        #Update linear 
        self.linear = o3.Linear(node_feats_irreps, node_feats_irreps, internal_weights=True, shared_weights=True)

    def forward(self,
                node_feats: torch.tensor,
        ) -> torch.Tensor:

        output = []
        for channel in range(self.num_features): #TODO Find a better way to implement this contraction
            out = torch.einsum(self.equation_main,
                               self.U_tensors[self.correlation],self.weights[self.correlation][channel],
                               node_feats[:,channel,:])
            for corr in range(self.correlation,0,-1):
                c_tensor = torch.einsum(self.equation_weighting,self.U_tensors[corr],self.weights[corr][channel])
                c_tensor  = c_tensor + out
                out = torch.einsum(self.equation_contract,c_tensor,node_feats[:,channel,:])
            output.append(out)

        node_feats = torch.stack(output,dim=-1)
        node_feats = self.linear(node_feats)
        return node_feats




class InteractionBlock(ABC, torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        num_avg_neighbors: float,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.num_avg_neighbors = num_avg_neighbors

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
        
nonlinearities = { 1 : torch.nn.functional.silu,
                   -1 : torch.tanh}

class AgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self,) -> None:
    

        #First linear
        self.linear_up = o3.Linear(self.node_feats_irreps,self.node_feats_irreps, internal_weights=True, shared_weights=True)

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(self.node_feats_irreps, self.edge_attrs_irreps,
                                                                   self.target_irreps)
        self.conv_tp = o3.TensorProduct(self.node_feats_irreps,
                                        self.edge_attrs_irreps,
                                        irreps_mid,
                                        instructions=instructions,
                                        shared_weights=False,
                                        internal_weights=False)

        irreps_mid = irreps_mid.simplify()

        #Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet([input_dim]
            + 3 * [64]
            + [self.conv_tp.weight_numel],
            torch.nn.functional.silu)

        #equivariant non linearity
        irreps_scalars = o3.Irreps([
                (mul, ir)
                for mul, ir in self.target_irreps
                if ir.l == 0
                and ir in irreps_mid
            ]
        )
        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.target_irreps
                if ir.l > 0
                and ir in irreps_mid
            ]
        )
        irreps_gates = o3.Irreps([mul,"0e"] for mul,_ in irreps_gated)
        self.equivariant_nonlin = nn.Gate(irreps_scalars=irreps_scalars,act_scalars=[nonlinearities[ir.p] for _,ir in irreps_scalars],
                irreps_gates=irreps_gates, act_gates=[torch.nn.functional.silu] * len(irreps_gates),irreps_gated=irreps_gated,)
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.irreps_out = self.equivariant_nonlin.irreps_out.simplify()

        # Linear
        self.linear = o3.Linear(irreps_mid, self.irreps_nonlin, internal_weights=True, shared_weights=True)

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(self.node_feats_irreps, self.node_attrs_irreps, self.irreps_nonlin)



    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]
        sc_real = self.skip_tp(node_feats.real, node_attrs)
        sc_imag = self.skip_tp(node_feats.imag, node_attrs)
        node_feats_real = self.linear_up(node_feats.real)
        node_feats_imag = self.linear_up(node_feats.imag)
        node_feats = torch.view_as_complex(torch.stack((node_feats_real,node_feats_imag)),dim=-1)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji_real = self.conv_tp(node_feats[sender].real, edge_attrs.real, tp_weights) - self.conv_tp(node_feats[sender].imag, edge_attrs.imag, tp_weights) # [n_edges, irreps]
        mji_imag = self.conv_tp(node_feats[sender].imag, edge_attrs.real, tp_weights) + self.conv_tp(node_feats[sender].real, edge_attrs.imag, tp_weights)
        message_real = scatter_sum(src=mji_real, index=receiver, dim=0, dim_size=num_nodes)  # [n_nodes, irreps]
        message_imag = scatter_sum(src=mji_imag, index=receiver, dim=0, dim_size=num_nodes) 
        message_real = self.linear(message_real)/self.num_avg_neighbors
        message_real = message_real + sc_real
        message_real = self.equivariant_nonlin(message_real)
        message_imag = self.linear(message_real)/self.num_avg_neighbors
        message_imag = message_real + sc_imag
        message_imag = self.equivariant_nonlin(message_imag)
        message = torch.stack((message_real,message_imag),dim=-1)
        return  torch.view_as_complex(message) # [n_nodes, irreps]

class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.get_default_dtype()))
        self.register_buffer('shift', torch.tensor(shift, dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.shift

    def __repr__(self):
        return f'{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})'