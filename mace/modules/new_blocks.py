from abc import abstractmethod
from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.jit import compile_mode
from e3nn.o3 import Norm
from torch.nn.modules import Embedding

from mace.tools.compile import simplify_if_compile
from mace.tools.scatter import scatter_sum

from .irreps_tools import (
    linear_out_irreps,
    mask_head,
    reshape_irreps,
    reshape_attn,
    tp_out_irreps_with_instructions,
)
from .radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    SoftTransform,
    continuous_sinous_embedding,
)
from .symmetric_contraction import SymmetricContraction, SymmetricMultiContraction
from .symmetric_cpring import SymmetricCPRing

import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial

from .plot_tools import plot_edge_data, plot_edge_data_message

@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        gate: Optional[Callable] = torch.nn.functional.silu, 
        radial_MLP: Optional[List[int]] = None,
        agnostic: Optional[bool] = False,
        num_heads: Optional[int] = 1,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP
        self.gate = gate
        self.agnostic = agnostic
        self.num_heads = num_heads

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


@compile_mode("script")
class RealAgnosticLongRangeResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,  # gate
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
        )
        self.reshape = reshape_irreps(self.irreps_out)

        # Density normalization
        #self.density_fn = nn.FullyConnectedNet(
        #    [input_dim]
        #    + [
        #        1,
        #    ],
        #    torch.nn.functional.silu,
        #)

        
        node_feats_norm_dim = [mul for mul, _ in self.node_feats_irreps][0]
        self.node_feats_norm_dim = node_feats_norm_dim
        node_attr_dim = [mul for mul, _ in self.node_attrs_irreps][0]
        self.edge_scalar_idx = irreps_mid.slices()[0]
        message_norm_dim = irreps_mid[0].mul

        self.attn = torch.nn.Sequential(
            torch.nn.Linear(node_feats_norm_dim + message_norm_dim, 16),
            torch.nn.LayerNorm(16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 1)
        )

        # Reshape
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_length: torch.Tensor,
        edge_cutoff: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        
        #edge_feats = edge_feats * torch.arange(1, 11, 1, device=edge_feats.device).unsqueeze(0)

        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)

        tp_weights = self.conv_tp_weights(edge_feats)

        #edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        node_features_cat = torch.cat([
                        #self.norm_node_feats(node_feats[sender]), 
                        mji[:, self.edge_scalar_idx],
                        node_feats[receiver][:, :self.node_feats_norm_dim],
                        ], dim=-1)

        attn = self.attn(node_features_cat).exp() * edge_cutoff
        density = scatter_sum(
            src=attn, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]

        attned_mji = attn * mji
        message = scatter_sum(
            src=attned_mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / (density + 1)
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]



@compile_mode("script")
class RealAgnosticLongRangeInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.irreps_out, self.node_attrs_irreps, self.irreps_out
        )
        self.reshape = reshape_irreps(self.irreps_out)

        ## Density normalization
        #self.density_fn = nn.FullyConnectedNet(
        #    [input_dim]
        #    + [
        #        1,
        #    ],
        #    torch.nn.functional.silu,
        #)
        node_feats_norm_dim = [mul for mul, _ in self.node_feats_irreps][0]
        self.node_feats_norm_dim = node_feats_norm_dim
        node_attr_dim = [mul for mul, _ in self.node_attrs_irreps][0]
        self.edge_scalar_idx = irreps_mid.slices()[0]
        message_norm_dim = irreps_mid[0].mul

        self.attn = torch.nn.Sequential(
            torch.nn.Linear(node_feats_norm_dim + message_norm_dim, 16),
            torch.nn.LayerNorm(16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 1)
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_length: torch.Tensor,
        edge_cutoff: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        
        #edge_feats = edge_feats * torch.arange(1, 11, 1, device=edge_feats.device).unsqueeze(0) * edge_cutoff
       
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        #edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]

        node_features_cat = torch.cat([
                        #self.norm_node_feats(node_feats[sender]), 
                        mji[:, self.edge_scalar_idx],
                        node_feats[receiver][:, :self.node_feats_norm_dim],
                        ], dim=-1)

        attn = self.attn(node_features_cat).exp() * edge_cutoff

        density = scatter_sum(
            src=attn, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        attned_mji = attn * mji
        message = scatter_sum(
            src=attned_mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]
