"""The OpenEquivariance backend: the fused convolution, and nothing else.

OpenEquivariance fuses the message-passing tensor product with its gather and
scatter, and that is the one op this backend builds. It does not touch the
symmetric contraction or the linear maps, so every other op is the
reference's, through :class:`~mace_torch.backends.composite.CompositeBackend`.

The convolution is posed to it the way the pinned paths state it: the node
features as ``C`` copies of each node term, grouped by irrep; one ``uvu``
instruction per path, writing that path's own output term, in path order. So
its output is already one block of ``C`` channels per path, and the radial
weights ``[n_edges, n_paths, C]`` are its weight vector as they are. It works
in the grouped layout natively, so nothing is permuted.

Its kernels are compiled on first use, for CUDA only; without
``openequivariance`` importing this module fails, which is what the registry
records.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import openequivariance as oeq  # ty: ignore[unresolved-import]
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.descriptors import ChannelwiseTPConvDescriptor, Descriptor
from mace_core.kernels.paths import channelwise_paths
from torch import Tensor, nn

from mace_torch.nn.layout import expanded_irreps

__all__ = ["OeqBackend"]

_DTYPES = {"float64": np.float64, "float32": np.float32}


class OeqCapabilities(BackendCapabilities):
    """The convolution, when every node term has one copy per channel."""

    def supports(self, descriptor: Descriptor) -> bool:
        if not super().supports(descriptor):
            return False
        if isinstance(descriptor, ChannelwiseTPConvDescriptor):
            return all(
                multiplicity == 1
                for multiplicity, _ in Irreps.parse(descriptor.irreps_node).terms
            )
        return False


class OeqChannelwiseTPConv(nn.Module):
    """The convolution, gather, product and scatter in one kernel call."""

    def __init__(self, descriptor: ChannelwiseTPConvDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        features = descriptor.num_features
        paths = channelwise_paths(
            descriptor.irreps_node, descriptor.irreps_edge, descriptor.irreps_out
        )
        self.num_paths = len(paths)
        self.weight_width = self.num_paths * descriptor.num_features
        dtype = _DTYPES[descriptor.precision]
        problem = oeq.TPProblem(
            expanded_irreps(descriptor.irreps_node, features),
            descriptor.irreps_edge,
            "+".join(f"{features}x{path.irrep}" for path in paths),
            [
                (path.node_term, path.edge_term, index, "uvu", True)
                for index, path in enumerate(paths)
            ],
            shared_weights=False,
            internal_weights=False,
            irrep_dtype=dtype,
            weight_dtype=dtype,
        )
        self.operation = oeq.TensorProductConv(problem, deterministic=False)

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        radial_weights: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> Tensor:
        # OpenEquivariance takes the receiving rows first and the sending
        # columns second, and sizes its output from the node features.
        return self.operation(
            node_features,
            edge_attributes,
            # The width is spelled out: a structure with no edges has zero
            # rows, and `-1` cannot be inferred from an empty tensor.
            radial_weights.reshape(radial_weights.shape[0], self.weight_width),
            receiver,
            sender,
        )


class OeqBackend:
    """OpenEquivariance for the fused convolution."""

    name = "oeq"

    def capabilities(self) -> BackendCapabilities:
        return OeqCapabilities(
            ops=frozenset({"channelwise_tp_conv"}),
            devices=frozenset({"cuda"}),
            dtypes=frozenset({"float64", "float32"}),
            layouts=frozenset({"mul_ir"}),
            bases=frozenset({"reduced"}),
            supports_double_backward=True,
        )

    def make_channelwise_tp_conv(
        self, descriptor: ChannelwiseTPConvDescriptor
    ) -> OeqChannelwiseTPConv:
        self.capabilities().require(descriptor, self.name)
        return OeqChannelwiseTPConv(descriptor)

    def make_interaction_layer(self, descriptors: tuple[Any, ...]) -> None:
        return None
