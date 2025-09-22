from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional, Tuple, Type

import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.modules.irreps_tools import tp_out_irreps_with_instructions


@compile_mode("script")
class MultiLayerFeatureMixer(torch.nn.Module):
    def __init__(self, node_feats_irreps: o3.Irreps, num_interactions: int):
        super().__init__()
        self.linears = torch.nn.ModuleList(
            [o3.Linear(node_feats_irreps, node_feats_irreps) for _ in range(num_interactions)]
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


def _sparse_dot_instructions(feat_in1: o3.Irreps, feat_in2: o3.Irreps, feat_out: o3.Irreps):
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
        self, potential_feats: torch.Tensor, node_feats: torch.Tensor, node_attrs: torch.Tensor, *args
    ) -> torch.Tensor:  # pragma: no cover - abstract
        ...


@compile_mode("script")
class AgnosticChargeBiasedLinearPotentialEmbedding(PotentialEmbeddingBlock):
    def _setup(self, charges_irreps: o3.Irreps, **kwargs) -> None:
        self.potential_linear = o3.Linear(self.potential_irreps, self.node_feats_irreps, internal_weights=True, shared_weights=True)
        self.node_feats_linear = o3.Linear(self.node_feats_irreps, self.node_feats_irreps, internal_weights=True, shared_weights=True)
        self.charge_embedding = o3.Linear(charges_irreps, self.node_feats_irreps, internal_weights=True, shared_weights=True)

    def forward(
        self,
        potential_feats: torch.Tensor,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        local_charges: torch.Tensor,
    ) -> torch.Tensor:
        return self.potential_linear(potential_feats) + self.node_feats_linear(node_feats) + self.charge_embedding(local_charges)


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
        self.mlp = nn.FullyConnectedNet([channels, 64, 64, channels], torch.nn.functional.silu)

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
        total_charges: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        ...


@compile_mode("script")
class AgnosticEmbeddedOneBodyVariableUpdate(FieldUpdateBlock):
    def _setup(
        self,
        potential_embedding_cls: Type[PotentialEmbeddingBlock],
        nonlinearity_cls: Type[torch.nn.Module],
        num_elements: Optional[int] = None,
        **kwargs,
    ) -> None:
        invar_irreps = o3.Irreps(f"{self.node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.potential_embedding = potential_embedding_cls(
            potential_irreps=self.potential_irreps,
            node_feats_irreps=self.node_feats_irreps,
            node_attrs_irreps=self.node_attrs_irreps,
            charges_irreps=self.charges_irreps,
        )
        # Sparse dot products between node feats and mixed feats
        instr = _sparse_dot_instructions(self.node_feats_irreps, self.node_feats_irreps, invar_irreps)
        self.dot_products = o3.TensorProduct(
            self.node_feats_irreps, self.node_feats_irreps, invar_irreps, instructions=instr
        )
        # Small nonlinearity over scalar invariants
        self.nonlinearity = nonlinearity_cls(invar_irreps=invar_irreps)
        # Map back to node_feats space
        _, tp_out_instr = tp_out_irreps_with_instructions(self.node_feats_irreps, invar_irreps, self.node_feats_irreps)
        self.tp_out = o3.TensorProduct(
            self.node_feats_irreps, invar_irreps, self.node_feats_irreps, instructions=tp_out_instr
        )
        # Final readout to charges with two extra scalar channels (Fukui sources)
        # Use a gated MLP-like readout to keep irreps structure consistent
        mlp_irreps = (32 * o3.Irreps.spherical_harmonics(self.charges_irreps.lmax)).sort()[0].simplify()
        self.readout = _GeneralNonLinearBiasReadout(
            irreps_in=self.node_feats_irreps,
            MLP_irreps=mlp_irreps,
            gate=torch.nn.functional.silu,
            irreps_out=(self.charges_irreps + o3.Irreps("2x0e")),
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
        total_charges: torch.Tensor,
    ) -> torch.Tensor:
        mixed = self.potential_embedding(potential_features / self.field_norm_factor, node_feats, node_attrs, local_charges)
        invariants = self.dot_products(node_feats, mixed)
        nonlin = self.nonlinearity(invariants)
        new_feats = self.tp_out(node_feats, nonlin)
        return self.readout(new_feats)


@compile_mode("script")
class OneBodyMLPFieldReadout(torch.nn.Module):
    """
    Minimal post-SCF readout: mixes induced charges and local field with node features
    via scalar invariants and an MLP to predict per-node energy contributions.
    """

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
        invar_irreps = o3.Irreps(f"{node_feats_irreps.count(o3.Irrep(0, 1))}x0e")
        self.linear_up_q = o3.Linear(charges_irreps, node_feats_irreps, biases=True)
        self.linear_up_v = o3.Linear(potential_irreps, node_feats_irreps, biases=True)
        instr = _sparse_dot_instructions(node_feats_irreps, node_feats_irreps, invar_irreps)
        self.dot_q = o3.TensorProduct(node_feats_irreps, node_feats_irreps, invar_irreps, instructions=instr)
        self.dot_v = o3.TensorProduct(node_feats_irreps, node_feats_irreps, invar_irreps, instructions=instr)
        self.mlp = nn.FullyConnectedNet([2 * invar_irreps.dim, 128, 128, 128, 1], torch.nn.functional.silu)

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
        q_up = self.linear_up_q(charges_induced + charges_0)
        v_up = self.linear_up_v(field_feats)
        inv_q = self.dot_q(node_feats, q_up)
        inv_v = self.dot_v(node_feats, v_up)
        inv = torch.cat([inv_q, inv_v], dim=-1)
        return self.mlp(inv).squeeze(-1)


class _GeneralNonLinearBiasReadout(torch.nn.Module):
    """Internal gated non-linear readout with bias, compatible with vector/tensor irreps."""

    def __init__(self, irreps_in: o3.Irreps, MLP_irreps: o3.Irreps, gate, irreps_out: o3.Irreps):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        # Split hidden irreps into scalars/gated and construct gates for non-scalars
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.hidden_irreps if ir.l == 0])
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.hidden_irreps if ir.l > 0])
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.gate = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _ in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.gate.irreps_in.simplify()
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.irreps_nonlin)
        self.linear_mid = o3.Linear(irreps_in=self.hidden_irreps, irreps_out=self.irreps_nonlin, biases=True)
        self.linear_2 = o3.Linear(irreps_in=self.hidden_irreps, irreps_out=irreps_out, biases=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate(self.linear_1(x))
        x = self.gate(self.linear_mid(x))
        return self.linear_2(x)


# Registries used by config files that pass strings
field_update_blocks = {
    "AgnosticEmbeddedOneBodyVariableUpdate": AgnosticEmbeddedOneBodyVariableUpdate,
}

field_readout_blocks = {
    "OneBodyMLPFieldReadout": OneBodyMLPFieldReadout,
}
