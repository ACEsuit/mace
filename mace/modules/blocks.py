###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import math
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.tools.compile import simplify_if_compile
from mace.tools.scatter import scatter_sum

from .irreps_tools import (
    linear_out_irreps,
    mask_head,
    reshape_irreps,
    tp_out_irreps_with_instructions,
    make_tp_irreps,
    make_tucker_irreps,
    make_tucker_irreps_flexible
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
from .symmetric_contraction import SymmetricContraction

from functools import partial

AGNOSTIC = False

@compile_mode("script")
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)


@compile_mode("script")
class LinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irrep_out: o3.Irreps = o3.Irreps("0e")):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irrep_out)

    def forward(
        self,
        x: torch.Tensor,
        heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@simplify_if_compile
@compile_mode("script")
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = o3.Linear(irreps_in=self.hidden_irreps, irreps_out=irrep_out)

    def forward(
        self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        if hasattr(self, "num_heads"):
            if self.num_heads > 1 and heads is not None:
                x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)  # [n_nodes, len(heads)]


@compile_mode("script")
class LinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, dipole_only: bool = False):
        super().__init__()
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=self.irreps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@compile_mode("script")
class NonLinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable,
        dipole_only: bool = False,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.irreps_nonlin)
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=self.irreps_out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode("script")
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        # assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, n_heads]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, torch.atleast_2d(self.atomic_energies).T)

    def __repr__(self):
        formatted_energies = ", ".join(
            [
                "[" + ", ".join([f"{x:.4f}" for x in group]) + "]"
                for group in torch.atleast_2d(self.atomic_energies)
            ]
        )
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


@compile_mode("script")
class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
    ):
        super().__init__()
        if radial_type == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "gaussian":
            self.bessel_fn = GaussianBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "chebyshev":
            self.bessel_fn = ChebychevBasis(r_max=r_max, num_basis=num_bessel)
        if distance_transform == "Agnesi":
            self.distance_transform = AgnesiTransform()
        elif distance_transform == "Soft":
            self.distance_transform = SoftTransform()
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        if hasattr(self, "distance_transform"):
            edge_lengths = self.distance_transform(
                edge_lengths, node_attrs, edge_index, atomic_numbers
            )
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        return radial * cutoff  # [n_edges, n_basis]

def tensor_power_einsum(tensor, N):
    batch_size, dim, features = tensor.shape

    # Create the equation string
    indices = [chr(ord('a') + i) for i in range(N)]  # Generate indices like 'a', 'b', 'c', ...
    eq = ','.join(['bi' + 'f' for _ in range(N)]) + '->b' + ''.join(indices) + 'f'

    # Prepare the list of tensors
    tensors = [tensor] * N

    # Perform einsum
    result = torch.einsum(eq, *tensors)

    # Reshape to [batch_size, dim ** N, features]
    result = result.reshape(batch_size, dim ** N, features)
    return result

@compile_mode("script")
class TensorFormatBlock(torch.nn.Module):
    """
    Maybe useful for reshaping tensor for efficient operation later.
    """
    def __init__(self, tensor_format, correlation):
        super().__init__()

        self.tensor_format = tensor_format
        self.correlation = correlation
        #self.irreps_in = irreps_in
        #self.indices = [chr(ord('a') + i) for i in range(N)]  
        #self.eq = ','.join(['bi' + 'f' for _ in range(N)]) + '->b' + ''.join(indices) + 'f'

    def forward(self, message) -> torch.Tensor:
        batch_size, dim, features = message.shape
        if self.tensor_format in ["symmetric_cp", "symmetric_tucker", "flexible_symmetric_tucker"]:
            return message
        elif self.tensor_format in ["non_symmetric_cp", "non_symmetric_tucker"]:
            return message


@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        learned_radials_dim: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
        agnostic: Optional[bool] = False,
        tensor_format = "symmetric_cp",
        flexible_feats_L = False,
        gaussian_prior = False,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
            agnostic=agnostic,
            tensor_format=tensor_format,
            flexible_feats_L=flexible_feats_L,
            gaussian_prior=gaussian_prior,
        )
        # Update linear
        if tensor_format in ["symmetric_cp", "non_symmetric_cp"]:
            mid_irreps = target_irreps
        elif tensor_format in ["flexible_non_symmetric_tucker", "flexible_symmetric_tucker",]:
            print(target_irreps, correlation)
            mid_irreps = make_tucker_irreps_flexible(target_irreps, correlation)
        elif tensor_format in ["symmetric_tucker", "non_symmetric_tucker"]:
            mid_irreps = make_tucker_irreps(target_irreps, correlation)
        else:
            print("Tensor formatting not supported. Check your input")
        self.linear = o3.Linear(
                    mid_irreps,
                    target_irreps,
                    internal_weights=True,
                    shared_weights=True,
                )

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)        

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
        correlation: int,
        gate: Optional[Callable] = torch.nn.functional.silu, 
        radial_MLP: Optional[List[int]] = None,
        agnostic: Optional[bool] = False,
        tensor_format: str = "symmetric_cp",
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
        # === tensor format stuffs === 
        self.tensor_format = tensor_format
        self.correlation = correlation
        
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


@compile_mode("script")
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


@compile_mode("script")
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
        sender = edge_index[0]
        receiver = edge_index[1]
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


@compile_mode("script")
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
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
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
        sender = edge_index[0]
        receiver = edge_index[1]
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


@compile_mode("script")
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
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
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
        sender = edge_index[0]
        receiver = edge_index[1]
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


@compile_mode("script")
class RealAgnosticInteractionBlock(InteractionBlock):
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
        
        if self.tensor_format in ["symmetric_cp", "symmetric_tucker", "flexible_symmetric_tucker"]:
            self.linear = o3.Linear(
                irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
            )
            # Selector TensorProduct
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
            self.reshape = reshape_irreps(self.irreps_out)

        elif self.tensor_format in ["non_symmetric_cp", "non_symmetric_tucker", "flexible_non_symmetric_tucker"]:
            self.linear = torch.nn.ModuleList([])
            # Selector TensorProduct
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
            self.reshape = torch.nn.ModuleList([])
            for _ in range(self.correlation):
                self.linear.append(o3.Linear(
                    irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
                ))
                self.reshape.append(reshape_irreps(self.irreps_out))

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.irreps_out, self.node_attrs_irreps, self.irreps_out
        )
        #self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        original_message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        
        if self.tensor_format in ["symmetric_cp", "symmetric_tucker",]:
            message = self.linear(original_message) / self.avg_num_neighbors
            message = self.skip_tp(message, node_attrs)
            return (
                    self.reshape(message),
                    None,
                    )  # symmetric_cp: [n_nodes, channels, (lmax + 1)**2]
        elif self.tensor_format in ["flexible_symmetric_tucker", ]:
            raise NotImplementedError
            # message = self.linear(original_message) / self.avg_num_neighbors
            # requires further contraction in SymmetricContraction - no reshape 
            # to [n_nodes, channels, (lmax + 1) ** 2 ] yet
            # return (message, sc)
        elif self.tensor_format in ["non_symmetric_cp", "non_symmetric_tucker"]:
            message = self.reshape[0](self.linear[0](original_message))
            message = message.unsqueeze(-1)
            for idx in range(1, self.correlation):
                _message = self.linear[idx](original_message)
                _message = self.skip_tp(_message, node_attrs)
                _message = self.reshape[idx](_message).unsqueeze(-1)
                message = torch.cat((message, _message), dim = -1)
            return (
                message / self.avg_num_neighbors, 
                None,
            )
        elif self.tensor_format in ["flexible_non_symmetric_tucker"]:
            raise NotImplementedError
            # message = self.linear[0](original_message) # [n_nodes, klm]
            # message = message.unsqueeze(-1) # [n_nnodes, klm, 1]
            # for idx in range(1, self.correlation):
            #     _message = self.linear[idx](original_message).unsqueeze(-1)
            #     message = torch.cat((message, _message), dim = -1)
            # return (
            #     message / self.avg_num_neighbors,
            #     sc,
            # )

@compile_mode("script")
class RealAgnosticResidualInteractionBlock(InteractionBlock):
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
        # TODO: clena up unused reshape layer for flexible tucker formats later 
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        
        if self.tensor_format in ["symmetric_cp", "symmetric_tucker", "flexible_symmetric_tucker"]:
            self.linear = o3.Linear(
                irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
            )
            # Selector TensorProduct
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
            self.reshape = reshape_irreps(self.irreps_out)

        elif self.tensor_format in ["non_symmetric_cp", "non_symmetric_tucker", "flexible_non_symmetric_tucker"]:
            self.linear = torch.nn.ModuleList([])
            # Selector TensorProduct
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
            self.reshape = torch.nn.ModuleList([])
            for _ in range(self.correlation):
                self.linear.append(o3.Linear(
                    irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
                ))
                self.reshape.append(reshape_irreps(self.irreps_out))

        # self.reshape = reshape_irreps(self.irreps_out)
        #self.tensor_format_layer = TensorFormatBlock(self.tensor_format, self.correlation)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        original_message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        

        if self.tensor_format in ["symmetric_cp", "symmetric_tucker",]:
            message = self.linear(original_message) / self.avg_num_neighbors
            return (
                    self.reshape(message),
                    sc,
                    )  # symmetric_cp: [n_nodes, channels, (lmax + 1)**2]
        elif self.tensor_format in ["flexible_symmetric_tucker"]:
            message = self.linear(original_message) / self.avg_num_neighbors
            # requires format contraction in SymmetricContraction - no reshape 
            # to [n_nodes, channels, (lmax + 1) ** 2 ] yet
            return (message, sc)
        elif self.tensor_format in ["non_symmetric_cp", "non_symmetric_tucker"]:
            message = self.reshape[0](self.linear[0](original_message))
            message = message.unsqueeze(-1)
            for idx in range(1, self.correlation):
                _message = self.reshape[idx](self.linear[idx](original_message)).unsqueeze(-1)
                message = torch.cat((message, _message), dim = -1)
            return (
                message / self.avg_num_neighbors, 
                sc,
            )
        elif self.tensor_format in ["flexible_non_symmetric_tucker"]:
            message = self.linear[0](original_message) # [n_nodes, klm]
            message = message.unsqueeze(-1) # [n_nnodes, klm, 1]
            for idx in range(1, self.correlation):
                _message = self.linear[idx](original_message).unsqueeze(-1)
                message = torch.cat((message, _message), dim = -1)
            return (
                message / self.avg_num_neighbors,
                sc,
            )
                
            

@compile_mode("script")
class RealAgnosticDensityInteractionBlock(InteractionBlock):
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

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
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
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        original_message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticDensityResidualInteractionBlock(InteractionBlock):
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
        
        
        
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        
        if self.tensor_format in ["symmetric_cp", "symmetric_tucker", "flexible_symmetric_tucker"]:
            self.linear = o3.Linear(
                irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
            )
            # Selector TensorProduct
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
            self.reshape = reshape_irreps(self.irreps_out)

        elif self.tensor_format in ["non_symmetric_cp", "non_symmetric_tucker", "flexible_non_symmetric_tucker"]:
            self.linear = torch.nn.ModuleList([])
            # Selector TensorProduct
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
            self.reshape = torch.nn.ModuleList([])
            for _ in range(self.correlation):
                self.linear.append(o3.Linear(
                    irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
                ))
                self.reshape.append(reshape_irreps(self.irreps_out))
        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        original_message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]

        
        if self.tensor_format in ["symmetric_cp", "symmetric_tucker",]:
            message = self.linear(original_message) / (density + 1)
            return (
                    self.reshape(message),
                    sc,
                    )  # symmetric_cp: [n_nodes, channels, (lmax + 1)**2]
        elif self.tensor_format in ["flexible_symmetric_tucker"]:
            message = self.linear(original_message) / (density + 1)
            # requires format contraction in SymmetricContraction - no reshape 
            # to [n_nodes, channels, (lmax + 1) ** 2 ] yet
            return (message, sc)
        elif self.tensor_format in ["non_symmetric_cp", "non_symmetric_tucker"]:
            message = self.reshape[0](self.linear[0](original_message) / (density + 1))
            message = message.unsqueeze(-1)
            for idx in range(1, self.correlation):
                _message = self.linear[idx](original_message) / (density + 1)
                _message = self.reshape[idx](_message).unsqueeze(-1)
                message = torch.cat((message, _message), dim = -1)
            return (
                message, 
                sc,
            )
        elif self.tensor_format in ["flexible_non_symmetric_tucker"]:
            message = self.linear[0](original_message / (density + 1)) # [n_nodes, klm]
            message = message.unsqueeze(-1) # [n_nnodes, klm, 1]
            for idx in range(1, self.correlation):
                _message = self.linear[idx](original_message) / (density + 1)
                _message = _message.unsqueeze(-1)
                message = torch.cat((message, _message), dim = -1)
            return (
                message,
                sc,
            )


@compile_mode("script")
class RealAgnosticDensityInjuctedNoScaleNoBiasInteractionGateBlock(InteractionBlock):

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
        print(f"RealAgnosticInteractionGateBlock --> {self.gate}")
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            self.gate,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        # self.linear = o3.Linear(
        #     irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        # )

        if self.tensor_format in ["symmetric_cp", "symmetric_tucker", "flexible_symmetric_tucker"]:
            self.linear = o3.Linear(
                irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
            )
            # Selector TensorProduct
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
            self.reshape = reshape_irreps(self.irreps_out)

        elif self.tensor_format in ["non_symmetric_cp", "non_symmetric_tucker", "flexible_non_symmetric_tucker"]:
            self.linear = torch.nn.ModuleList([])
            # Selector TensorProduct
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
            self.reshape = torch.nn.ModuleList([])
            for _ in range(self.correlation):
                self.linear.append(o3.Linear(
                    irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
                ))
                self.reshape.append(reshape_irreps(self.irreps_out))

        if not getattr(self, "agnostic", False):
            # Selector TensorProduct
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.irreps_out, self.node_attrs_irreps, self.irreps_out
            )
        else:
            ## Selector TensorProduct
            #self.skip_tp = o3.FullyConnectedTensorProduct(
            #    self.irreps_out, self.node_feats_irreps, self.irreps_out
            #)
            pass

        #self.reshape = reshape_irreps(self.irreps_out)
        self.density_fn = nn.FullyConnectedNet(
            [input_dim] + [1,], 
            self.gate
        )
        
        self.sinous_embedding = partial(continuous_sinous_embedding, dim=32, max_density=100)
        self.density_linear = torch.nn.Linear(32, self.irreps_out[0].mul, bias=False) # TODO: density embedding model

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]        
        # learnable density funciton with 
        density = torch.tanh(self.density_fn(edge_feats)**2)
        
        # NO RESCALE
        #mji = mji * density

        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        
        node_local_density = scatter_sum(
            src=density, index=receiver, dim=0, dim_size=num_nodes
        )

        message = message / (node_local_density + 1)
 
        # density_embedding
        sin_embedding = self.sinous_embedding(node_local_density.flatten())
        density_embedding = self.density_linear(sin_embedding)
        # density inject
        message[:, self.irreps_out.slices()[0]] += density_embedding

        original_message = message
        
        if self.tensor_format in ["symmetric_cp", "symmetric_tucker",]:
            message = self.linear(original_message)
            if not getattr(self, "agnostic", False):
                message = self.skip_tp(message, node_attrs)
            else:
                pass
            return (
                self.reshape(message),
                None,
            )  # [n_nodes, channels, (lmax + 1)**2]
        elif self.tensor_format in ["flexible_symmetric_tucker", ]:
            raise NotImplementedError
            # message = self.linear(original_message) / self.avg_num_neighbors
            # requires further contraction in SymmetricContraction - no reshape 
            # to [n_nodes, channels, (lmax + 1) ** 2 ] yet
            # return (message, sc)
        elif self.tensor_format in ["non_symmetric_cp", "non_symmetric_tucker"]:
            message = self.reshape[0](self.linear[0](original_message))
            message = message.unsqueeze(-1)
            for idx in range(1, self.correlation):
                _message = self.linear[idx](original_message)
                if not getattr(self, "agnostic", False):
                    _message = self.skip_tp(_message, node_attrs)
                else:
                    pass
                _message = self.reshape[idx](_message).unsqueeze(-1)
                message = torch.cat((message, _message), dim = -1)
            return (
                message, 
                None,
            )
        elif self.tensor_format in ["flexible_non_symmetric_tucker"]:
            raise NotImplementedError
            # message = self.linear[0](original_message) # [n_nodes, klm]
            # message = message.unsqueeze(-1) # [n_nnodes, klm, 1]
            # for idx in range(1, self.correlation):
            #     _message = self.linear[idx](original_message).unsqueeze(-1)
            #     message = torch.cat((message, _message), dim = -1)
            # return (
            #     message / self.avg_num_neighbors,
            #     sc,
            # )

@compile_mode("script")
class RealAgnosticDensityInjuctedNoScaleNoBiasResidualInteractionGateBlock(InteractionBlock):
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
        print(f"RealAgnosticInteractionGateBlock --> {self.gate}")
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            self.gate,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps

        if self.tensor_format in ["symmetric_cp", "symmetric_tucker", "flexible_symmetric_tucker"]:
            self.linear = o3.Linear(
                irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
            )
            # Selector TensorProduct
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
            self.reshape = reshape_irreps(self.irreps_out)

        elif self.tensor_format in ["non_symmetric_cp", "non_symmetric_tucker", "flexible_non_symmetric_tucker"]:
            self.linear = torch.nn.ModuleList([])
            # Selector TensorProduct
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
            self.reshape = torch.nn.ModuleList([])
            for _ in range(self.correlation):
                self.linear.append(o3.Linear(
                    irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
                ))
                self.reshape.append(reshape_irreps(self.irreps_out))

        if not getattr(self, "agnostic", False):
            ValueError("agnostic not supported yet inRealAgnosticDensityInjuctedNoScaleNoBiasResidualInteractionGateBlock")
            # Selector TensorProduct
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
        else:
            ## Selector TensorProduct
            #self.skip_tp = o3.FullyConnectedTensorProduct(
            #    self.irreps_out, self.node_feats_irreps, self.irreps_out
            #)
            pass

        self.density_fn = nn.FullyConnectedNet(
            [input_dim] + [1,], 
            self.gate
        )
        
        self.sinous_embedding = partial(continuous_sinous_embedding, dim=32, max_density=100)
        self.density_linear = torch.nn.Linear(32, self.irreps_out[0].mul, bias=False) # TODO: density embedding model

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]        
        # learnable density funciton with 
        density = torch.tanh(self.density_fn(edge_feats)**2)
        
        # NO RESCALE
        #mji = mji * density

        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        
        node_local_density = scatter_sum(
            src=density, index=receiver, dim=0, dim_size=num_nodes
        )

        message = message / (node_local_density + 1)
 
        # density_embedding
        sin_embedding = self.sinous_embedding(node_local_density.flatten())
        density_embedding = self.density_linear(sin_embedding)
        # density inject
        message[:, self.irreps_out.slices()[0]] += density_embedding

        # == tensor formats ===
        original_message = message
        if self.tensor_format in ["symmetric_cp", "symmetric_tucker",]:
            message = self.linear(original_message) / (node_local_density + 1)
            return (
                    self.reshape(message),
                    sc,
                    )  # symmetric_cp: [n_nodes, channels, (lmax + 1)**2]
        elif self.tensor_format in ["flexible_symmetric_tucker"]:
            message = self.linear(original_message) / (node_local_density + 1)
            # requires format contraction in SymmetricContraction - no reshape 
            # to [n_nodes, channels, (lmax + 1) ** 2 ] yet
            return (message, sc)
        elif self.tensor_format in ["non_symmetric_cp", "non_symmetric_tucker"]:
            message = self.reshape[0](self.linear[0](original_message) / (node_local_density + 1))
            message = message.unsqueeze(-1)
            for idx in range(1, self.correlation):
                _message = self.linear[idx](original_message) / (node_local_density + 1)
                _message = self.reshape[idx](_message).unsqueeze(-1)
                message = torch.cat((message, _message), dim = -1)
            return (
                message, 
                sc,
            )
        elif self.tensor_format in ["flexible_non_symmetric_tucker"]:
            message = self.linear[0](original_message / (node_local_density + 1)) # [n_nodes, klm]
            message = message.unsqueeze(-1) # [n_nnodes, klm, 1]
            for idx in range(1, self.correlation):
                _message = self.linear[idx](original_message) / (node_local_density + 1)
                _message = _message.unsqueeze(-1)
                message = torch.cat((message, _message), dim = -1)
            return (
                message,
                sc,
            )

@compile_mode("script")
class RealAgnosticAttResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        self.node_feats_down_irreps = o3.Irreps("64x0e")
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
        self.linear_down = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_down_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        input_dim = (
            self.edge_feats_irreps.num_irreps
            + 2 * self.node_feats_down_irreps.num_irreps
        )
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + 3 * [256] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        self.reshape = reshape_irreps(self.irreps_out)

        # Skip connection.
        self.skip_linear = o3.Linear(self.node_feats_irreps, self.hidden_irreps)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_linear(node_feats)
        node_feats_up = self.linear_up(node_feats)
        node_feats_down = self.linear_down(node_feats)
        augmented_edge_feats = torch.cat(
            [
                edge_feats,
                node_feats_down[sender],
                node_feats_down[receiver],
            ],
            dim=-1,
        )
        tp_weights = self.conv_tp_weights(augmented_edge_feats)
        mji = self.conv_tp(
            node_feats_up[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "shift",
            torch.tensor(shift, dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        return (
            torch.atleast_1d(self.scale)[head] * x + torch.atleast_1d(self.shift)[head]
        )

    def __repr__(self):
        formatted_scale = (
            ", ".join([f"{x:.4f}" for x in self.scale])
            if self.scale.numel() > 1
            else f"{self.scale.item():.4f}"
        )
        formatted_shift = (
            ", ".join([f"{x:.4f}" for x in self.shift])
            if self.shift.numel() > 1
            else f"{self.shift.item():.4f}"
        )
        return f"{self.__class__.__name__}(scale={formatted_scale}, shift={formatted_shift})"
