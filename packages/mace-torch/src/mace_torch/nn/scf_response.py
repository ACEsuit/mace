"""Native tensor expansions used to differentiate the dense SCF response.

The numerical solver needs a Jacobian of its response, not just an ordinary
backward. Registered custom ops currently do not expose that transform (forward
AD can even return zero). These private expansions reuse the reference ops'
parameters, normalization and canonical format. They are selected at build time
inside the reference SCF solve; the backbone and energy readouts retain their
requested kernel backend. No runtime monkeypatching or parameter copies.
"""

from typing import cast

from torch import Tensor

from mace_torch.backends.reference import (
    ReferenceBackend,
    ReferenceChannelwiseTPConv,
    ReferenceFullyConnectedTP,
    ReferenceLinear,
    ReferenceSegmentReduce,
    ReferenceSymmetricContraction,
    _ConstantTensors,
)
from mace_torch.kernels.contraction import _joined_bases, _monomials, _weighted
from mace_torch.kernels.ops import (
    channelwise_tp_conv_aten,
    equivariant_linear_aten,
    segment_sum_aten,
)


class _Linear(ReferenceLinear):
    def forward(self, features):
        return equivariant_linear_aten(
            features,
            self.weight * self.path_normalization,
            self.row,
            self.column,
            self.source,
            self.bias,
            self.bias_row,
            self.dim_out,
        )


class _ElementLinear(ReferenceFullyConnectedTP):
    def forward(self, features, attributes):
        weights = self.weight * self.path_normalization
        total = features.new_zeros((features.shape[0], self.dim_out))
        for index in range(weights.shape[0]):
            mapped = equivariant_linear_aten(
                features,
                weights[index],
                self.row,
                self.column,
                self.source,
                features.new_empty(0),
                self.row.new_empty(0),
                self.dim_out,
            )
            total = total + mapped * attributes[:, index : index + 1]
        return total


class _Convolution(ReferenceChannelwiseTPConv):
    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        radial_weights: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> Tensor:
        channels = self.layout.channel_major(
            node_features, self.node_terms, self.num_features
        )
        message = channelwise_tp_conv_aten(
            channels,
            edge_attributes,
            radial_weights,
            self.coefficients,
            sender,
            receiver,
            num_nodes,
        )
        return self.layout.grouped(message, self.path_terms)


class _Contraction(ReferenceSymmetricContraction):
    def forward(self, features, element):
        channels = self.layout.channel_major(features, self.in_terms, self.num_features)
        nodes, count, dimension = channels.shape
        powers = _monomials(
            channels.reshape(-1, dimension).T.contiguous(),
            self.orders,
            differentiable=True,
        )
        bases = [
            table
            for group in cast(list[_ConstantTensors], list(self.bases))
            for table in group
        ]
        projections = [
            _joined_bases(bases, self.orders, order).T @ powers[order]
            for order in range(1, self.orders + 1)
        ]
        result = _weighted(
            projections, list(self.weights), bases, element, self.orders, nodes, count
        )
        return self.layout.grouped(result, self.out_terms)


class _Reduce(ReferenceSegmentReduce):
    def forward(self, values, index, num_segments):
        return segment_sum_aten(values, index, num_segments)


class ResponseOps(ReferenceBackend):
    """The dense reference solver's differentiable expression of its response.

    Private to the nonlinear solver, not a new registered kernel backend.
    Canonical weights are identical to the ordinary reference implementations.
    """

    def make_linear(self, kernel_spec):
        return _Linear(kernel_spec)

    def make_fully_connected_tp(self, kernel_spec):
        return _ElementLinear(kernel_spec)

    def make_channelwise_tp_conv(self, kernel_spec):
        return _Convolution(kernel_spec)

    def make_symmetric_contraction(self, kernel_spec):
        return _Contraction(kernel_spec)

    def make_segment_reduce(self, kernel_spec):
        return _Reduce(kernel_spec)
