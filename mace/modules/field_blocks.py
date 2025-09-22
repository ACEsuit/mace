###########################################################################################
# Long Range Blocks
# Authors: Will Baldwin, Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional, Tuple, Type

from mace.modules.blocks import (
    GeneralNonLinearBiasReadoutBlock,
)
from mace.modules.radial import RadialMLP
import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.modules.irreps_tools import tp_out_irreps_with_instructions


@compile_mode("script")
class MultiLayerFeatureMixer(torch.nn.Module):
    def __init__(self, node_feats_irreps: o3.Irreps, num_interactions: int):
        super().__init__()
        self.linears = torch.nn.ModuleList(
            [
                o3.Linear(node_feats_irreps, node_feats_irreps)
                for _ in range(num_interactions)
            ]
        )

    def forward(self, all_node_feats: torch.Tensor) -> torch.Tensor:
        # all_node_feats: [num_interactions, n_nodes, irreps]
        out = torch.zeros_like(all_node_feats[0])
        for i, lin in enumerate(self.linears):
            out = out + lin(all_node_feats[i])
        return out


@compile_mode("script")
class EnvironmentDependentSpinSourceBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, max_l: int, zero_charges: bool = False):
        super().__init__()
        self.zero_charges = zero_charges
        self.irreps_out = 2 * o3.Irreps.spherical_harmonics(max_l)
        self.linear = o3.Linear(irreps_in, self.irreps_out)

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        mp = self.linear(node_feats)
        if self.zero_charges:
            mp_z = torch.zeros_like(mp)
            mp_z[:, 1:] = mp[:, 1:]
            mp = mp_z
        return mp.unsqueeze(-2)  # [n_nodes, 1, (max_l+1)^2 * 2]


def _sparse_dot_instructions(
    feat_in1: o3.Irreps, feat_in2: o3.Irreps, feat_out: o3.Irreps
):
    _, instructions = tp_out_irreps_with_instructions(feat_in1, feat_in2, feat_out)
    new = []
    for i, j, k, mode, trainable in instructions:
        new.append((i, j, 0, mode, trainable))
    return new


class PotentialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        potential_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        node_attrs_irreps: o3.Irreps,
        **kwargs,
    ):
        super().__init__()
        self.potential_irreps = potential_irreps
        self.node_feats_irreps = node_feats_irreps
        self.node_attrs_irreps = node_attrs_irreps
        self._setup(**kwargs)

    @abstractmethod
    def _setup(self, **kwargs) -> None:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def forward(
        self,
        potential_feats: torch.Tensor,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        *args,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        ...


@compile_mode("script")
class AgnosticChargeBiasedLinearPotentialEmbedding(PotentialEmbeddingBlock):
    def _setup(self, charges_irreps: o3.Irreps) -> None:
        self.potential_linear = o3.Linear(
            irreps_in=self.potential_irreps,
            irreps_out=self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.node_feats_linear = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.charges_irreps = charges_irreps
        self.charge_embedding = o3.Linear(
            self.charges_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        potential_feats: torch.Tensor,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        local_charges: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.potential_linear(potential_feats)
            + self.node_feats_linear(node_feats)
            + self.charge_embedding(local_charges)
        )


@compile_mode("script")
class NoNonLinearity(torch.nn.Module):
    def __init__(self, invar_irreps: o3.Irreps):
        super().__init__()
        self.irreps = invar_irreps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@compile_mode("script")
class MLPNonLinearity(torch.nn.Module):
    def __init__(self, invar_irreps: o3.Irreps):
        super().__init__()
        channels = invar_irreps.count(o3.Irrep(0, 1))
        self.mlp = nn.FullyConnectedNet(
            [channels, 64, 64, channels], torch.nn.functional.silu
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FieldUpdateBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        potential_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
        field_norm_factor: float,
        radial_MLP: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.potential_irreps = potential_irreps
        self.charges_irreps = charges_irreps
        self.register_buffer("field_norm_factor", torch.tensor(field_norm_factor))
        self._setup(**kwargs)

    @abstractmethod
    def _setup(self, **kwargs) -> None:  # pragma: no cover - abstract
        ...

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        potential_features: torch.Tensor,
        local_charges: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        ...


def instructions_for_sparse_tp(feat_in1, feat_in2, feat_out):
    channels1 = feat_in1.count(o3.Irrep(0, 1))
    channels2 = feat_in2.count(o3.Irrep(0, 1))
    channels3 = feat_out.count(o3.Irrep(0, 1))
    assert channels1 == channels2 and channels1 == channels3
    _, instructions = tp_out_irreps_with_instructions(feat_in1, feat_in2, feat_out)
    new_instructions = []
    for instr in instructions:
        i, j, k, mode, trainable = instr
        new_instructions.append((i, j, 0, mode, trainable))
    return new_instructions


@compile_mode("script")
class AgnosticEmbeddedOneBodyVariableUpdate(FieldUpdateBlock):
    def _setup(
        self,
        potential_embedding_cls: Type[
            PotentialEmbeddingBlock
        ] = AgnosticChargeBiasedLinearPotentialEmbedding,
        nonlinearity_cls: Type[torch.nn.Module] = NoNonLinearity,
        num_elements: Optional[int] = None,
        **kwargs,
    ) -> None:
        # product irreps is node_feats_irreps but only l=0
        invar_irreps = o3.Irreps(f"{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.potential_embedding = potential_embedding_cls(
            potential_irreps=self.potential_irreps,
            node_feats_irreps=self.node_feats_irreps,
            node_attrs_irreps=self.node_attrs_irreps,
            charges_irreps=self.charges_irreps,
        )

        self.source_embedding = o3.Linear(
            self.node_attrs_irreps,
            invar_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps, self.node_feats_irreps, invar_irreps
        )
        self.dot_products = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
        )
        self.nonlinearity = RadialMLP(
            [2 * invar_irreps.dim] + [64, 64, 64] + [invar_irreps.dim]
        )
        _, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            invar_irreps,
            self.node_feats_irreps,
        )
        self.tp_out = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=invar_irreps,
            irreps_out=self.node_feats_irreps,
            instructions=instructions,
        )

        MLP_irreps = (
            (32 * o3.Irreps.spherical_harmonics(self.charges_irreps.lmax))
            .sort()[0]
            .simplify()
        )
        self.readout = GeneralNonLinearBiasReadoutBlock(
            irreps_in=self.node_feats_irreps,
            MLP_irreps=MLP_irreps,
            gate=torch.nn.functional.silu,
            irreps_out=(self.charges_irreps + o3.Irreps(f"2x0e")),
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        potential_features: torch.Tensor,
        local_charges: torch.Tensor,
    ) -> torch.Tensor:
        # create pot feats
        mixed_feats = self.potential_embedding(
            potential_features,
            node_feats,
            node_attrs,
            local_charges,
        )
        invariant_descriptors = self.dot_products(node_feats, mixed_feats)
        source_embedding = self.source_embedding(node_attrs)
        invariant_descriptors_embedded = torch.cat(
            [invariant_descriptors, source_embedding], dim=-1
        )
        nonlin_feats = self.nonlinearity(invariant_descriptors_embedded)
        new_feats = self.tp_out(node_feats, nonlin_feats)
        multipoles = self.readout(new_feats)
        return multipoles


class PostScfReadout(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        potential_irreps: o3.Irreps,
        charges_irreps: o3.Irreps,
        radial_MLP: Optional[List[int]] = None,
        **kwargs,
    ):
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

        self.potential_irreps = potential_irreps
        self.charges_irreps = charges_irreps
        self._setup(**kwargs)

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
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


@compile_mode("script")
class OneBodyMLPFieldReadout(PostScfReadout):
    def _setup(self, **kwargs):
        invar_irreps = o3.Irreps(f"{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.linear_up_q = o3.Linear(
            self.charges_irreps, self.node_feats_irreps, biases=True
        )
        self.linear_up_v = o3.Linear(
            self.potential_irreps, self.node_feats_irreps, biases=True
        )
        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps, self.node_feats_irreps, invar_irreps
        )
        self.dot_products_q = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
        )
        new_instructions = instructions_for_sparse_tp(
            self.node_feats_irreps, self.node_feats_irreps, invar_irreps
        )
        self.dot_products_v = o3.TensorProduct(
            irreps_in1=self.node_feats_irreps,
            irreps_in2=self.node_feats_irreps,
            irreps_out=invar_irreps,
            instructions=new_instructions,
        )

        self.mlp = RadialMLP(
            [2 * invar_irreps.dim] + [128, 128, 128] + [1],
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        field_feats: torch.Tensor,
        charges_0: torch.Tensor,
        charges_induced: torch.Tensor,
    ):
        q_up = self.linear_up_q(charges_induced + charges_0)
        v_up = self.linear_up_v(field_feats)
        invar_feats = torch.cat(
            [
                self.dot_products_q(node_feats, q_up),
                self.dot_products_v(node_feats, v_up),
            ],
            dim=-1,
        )
        return self.mlp(invar_feats).squeeze(-1)


# Registries used by config files that pass strings
field_update_blocks = {
    "AgnosticEmbeddedOneBodyVariableUpdate": AgnosticEmbeddedOneBodyVariableUpdate,
}

field_readout_blocks = {
    "OneBodyMLPFieldReadout": OneBodyMLPFieldReadout,
}
