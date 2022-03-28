from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from e3nn import nn, o3
from torch_scatter import scatter_sum
from opt_einsum import contract


from LieACE.tools.torch_tools import get_complex_default_dtype

from .irreps_tools import (
    linear_out_irreps,
    tp_out_irreps_with_instructions,
    reshape_irreps,
)
from .radial import BesselBasis, PolynomialCutoff


class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(
        self, node_attrs: torch.Tensor,  # [n_nodes, irreps]
    ):
        return self.linear(node_attrs)


class NonLinearBlock(torch.nn.Module):
    def __init__(self, gate: torch.nn.Module):
        super().__init__()
        self.non_linearity = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, 1]  # [..., ]
        return self.gate(x)  # [n_nodes, 1]


class LinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=o3.Irreps("0e"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


class LinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(
            irreps_in=irreps_in, irreps_out=o3.Irreps("1x0e + 1x1o")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, MLP_irreps: o3.Irreps, gate: Callable):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=o3.Irreps("0e")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


class NonLinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, MLP_irreps: o3.Irreps, gate: Callable):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=o3.Irreps("1x0e + 1x0o")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, ]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, self.atomic_energies)

    def __repr__(self):
        formatted_energies = ", ".join([f"{x:.4f}" for x in self.atomic_energies])
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(self, r_max: float, num_bessel: int, num_polynomial_cutoff: int):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self, edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        return bessel * cutoff  # [n_edges, n_basis]


class ProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        U_tensors: Dict[int, torch.tensor],
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        avg_num_neighbors: int,
        use_complex: bool = True,
    ) -> None:
        super().__init__()

        if use_complex:
            self.dtype = get_complex_default_dtype()
        else:
            self.dtype = torch.get_default_dtype()
        self.U_tensors = (
            U_tensors  # Dict[str,[(lmax+1)**2]**correlation + [num_weights]]
        )
        self.num_features = node_feats_irreps.count((0, 1))
        self.correlation = correlation
        self.target_irreps = target_irreps
        self.avg_num_neighbors = avg_num_neighbors
        # Tensor contraction equations
        self.equation_main = "...ik,kc,bci -> bc..."
        self.equation_weighting = "...k,kc->c..."
        self.equation_contract = "bc...i,bci->bc..."

        # Create weight for product basis
        self.weights = torch.nn.ParameterDict({})
        for i in range(1, correlation + 1):
            num_params = self.U_tensors[i].size()[-1]
            w = torch.nn.Parameter(torch.randn(num_params, self.num_features))
            self.weights[str(i)] = w
        # Update linear
        self.linear = o3.Linear(
            self.target_irreps,
            self.target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, node_feats: torch.tensor, sc: torch.tensor,) -> torch.Tensor:
        out = contract(
            self.equation_main,
            self.U_tensors[self.correlation],
            self.weights[str(self.correlation)].type(self.dtype),
            node_feats,
        )  # TODO : use optimize library and cuTENSOR
        for corr in range(self.correlation - 1, 0, -1):
            c_tensor = contract(
                self.equation_weighting,
                self.U_tensors[corr],
                self.weights[str(corr)].type(self.dtype),
            )
            c_tensor = c_tensor + out
            out = contract(self.equation_contract, c_tensor, node_feats)
        return self.linear(out.real) + sc


class InteractionBlock(ABC, torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        avg_num_neighbors: float,
        num_radial_coupling: Optional[int],
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.num_radial_coupling = num_radial_coupling

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


nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}


class TensorProductWeightsBlock(torch.nn.Module):
    def __init__(self, num_elements: int, num_edge_feats: int, num_feats_out: int):
        super().__init__()

        weights = torch.empty(
            (num_elements, num_edge_feats, num_feats_out),
            dtype=torch.get_default_dtype(),
        )
        torch.nn.init.xavier_uniform_(weights)
        self.weights = torch.nn.Parameter(weights)

    def forward(
        self,
        sender_or_receiver_node_attrs: torch.Tensor,  # assumes that the node attributes are one-hot encoded
        edge_feats: torch.Tensor,
    ):
        return torch.einsum(
            "be, ba, aek -> bk", edge_feats, sender_or_receiver_node_attrs, self.weights
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(shape=({", ".join(str(s) for s in self.weights.shape)}), '
            f"weights={np.prod(self.weights.shape)})"
        )


class ResidualElementDependentInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.conv_tp_weights = TensorProductWeightsBlock(
            num_elements=self.node_attrs_irreps.num_irreps,
            num_edge_feats=self.edge_feats_irreps.num_irreps,
            num_feats_out=self.conv_tp.weight_numel,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.irreps_out
        )

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
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(node_attrs[sender], edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return message + sc  # [n_nodes, irreps]


class AgnosticNonlinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
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
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.irreps_out, self.node_attrs_irreps, self.irreps_out
        )

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
        tp_weights = self.conv_tp_weights(edge_feats)
        node_feats = self.linear_up(node_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return message  # [n_nodes, irreps]


class AgnosticResidualNonlinearInteractionBlock(InteractionBlock):
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
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
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
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.irreps_out
        )

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
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = message + sc
        return message  # [n_nodes, irreps]


class ComplexElementDependentResidualInteractionBlock(InteractionBlock):
    def _setup(self,) -> None:

        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps * self.num_radial_coupling,
            self.edge_attrs_irreps,
            self.target_irreps * self.num_radial_coupling,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps * self.num_radial_coupling,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        self.conv_tp_weights = TensorProductWeightsBlock(
            num_elements=self.node_attrs_irreps.num_irreps,
            num_edge_feats=self.edge_feats_irreps.num_irreps,
            num_feats_out=self.conv_tp.weight_numel,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps * self.num_radial_coupling

        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out * self.num_radial_coupling,
            internal_weights=True,
            shared_weights=True,
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.node_feats_irreps
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]

        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(node_attrs[sender], edge_feats)

        node_feats = node_feats.repeat(1, self.num_radial_coupling)

        mji_real = self.conv_tp(
            node_feats[sender], edge_attrs.real, tp_weights
        )  # [n_edges, irreps]
        mji_imag = self.conv_tp(node_feats[sender], edge_attrs.imag, tp_weights)

        message_real = scatter_sum(
            src=mji_real, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message_imag = scatter_sum(
            src=mji_imag, index=receiver, dim=0, dim_size=num_nodes
        )

        message_real = self.linear(message_real) / self.avg_num_neighbors
        message_imag = self.linear(message_imag) / self.avg_num_neighbors

        message = torch.view_as_complex(
            torch.stack((message_real, message_imag), dim=-1)
        )
        return (
            reshape_irreps(message, self.irreps_out * self.num_radial_coupling),
            sc,
        )  # [n_nodes, channels, n*(lmax + 1)**2]


class ComplexAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self,) -> None:

        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps * self.num_radial_coupling,
            self.edge_attrs_irreps,
            self.target_irreps * self.num_radial_coupling,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps * self.num_radial_coupling,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps * self.num_radial_coupling
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.node_feats_irreps
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]

        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        node_feats = node_feats.repeat(1, self.num_radial_coupling)

        mji_real = self.conv_tp(
            node_feats[sender], edge_attrs.real, tp_weights
        )  # [n_edges, irreps]
        mji_imag = self.conv_tp(node_feats[sender], edge_attrs.imag, tp_weights)

        message_real = scatter_sum(
            src=mji_real, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message_imag = scatter_sum(
            src=mji_imag, index=receiver, dim=0, dim_size=num_nodes
        )

        message_real = self.linear(message_real)
        message_imag = self.linear(message_imag)

        message = torch.view_as_complex(
            torch.stack((message_real, message_imag), dim=-1)
        )
        return (
            reshape_irreps(message, self.irreps_out),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self,) -> None:

        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps * self.num_radial_coupling,
            self.edge_attrs_irreps,
            self.target_irreps * self.num_radial_coupling,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps * self.num_radial_coupling,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [64] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps * self.num_radial_coupling
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.node_feats_irreps
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender, receiver = edge_index
        num_nodes = node_feats.shape[0]

        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        node_feats = node_feats.repeat(1, self.num_radial_coupling)

        mji_real = self.conv_tp(
            node_feats[sender], edge_attrs.real, tp_weights
        )  # [n_edges, irreps]
        mji_imag = self.conv_tp(node_feats[sender], edge_attrs.imag, tp_weights)

        message_real = scatter_sum(
            src=mji_real, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message_imag = scatter_sum(
            src=mji_imag, index=receiver, dim=0, dim_size=num_nodes
        )

        message_real = self.linear(message_real)
        message_imag = self.linear(message_imag)

        message = torch.view_as_complex(
            torch.stack((message_real, message_imag), dim=-1)
        )
        return (
            reshape_irreps(message, self.irreps_out),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "shift", torch.tensor(shift, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.shift

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"
        )

