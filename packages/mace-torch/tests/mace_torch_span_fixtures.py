"""A backend in its own wheel can implement the span using public imports only.

The test span composes reference ops. Its templates have all parameters and
buffers removed: live canonical weights are supplied on each call, and derived
constants are Python data reconstructed on device. No hidden tensor state.
"""

from dataclasses import replace

import torch
from mace_core.config import (
    AgnesiTransformConfig,
    NoDistanceTransformConfig,
    SoftTransformConfig,
)
from mace_core.kernels.canonical import KERNEL_SPEC_VERSION
from mace_core.kernels.spans import EdgeConvolutionSpan
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.backends.spans import SpanCapabilities
from mace_torch.nn.embedding import RadialEmbeddingBlock, build_distance_transform
from torch import nn


def _no_initialization(seed):
    """Templates own no weights to initialize."""


class Composition(nn.Module):
    def __init__(self, shape):
        super().__init__()
        backend = ReferenceBackend()
        radial = backend.make_radial_basis(shape.radial_basis)
        transform = (
            NoDistanceTransformConfig()
            if shape.distance_transform is None
            else AgnesiTransformConfig()
            if shape.distance_transform.kind == "agnesi"
            else SoftTransformConfig()
        )
        self.radial = RadialEmbeddingBlock(
            radial.basis, build_distance_transform(transform), radial.cutoff
        )
        self.network = backend.make_radial_mlp(shape.radial_network)
        self.harmonics = backend.make_spherical_harmonics(shape.spherical_harmonics)
        self.convolution = backend.make_channelwise_tp_conv(shape.convolution)
        self.num_paths = len(shape.convolution.paths)
        self.num_features = shape.convolution.num_features

    def forward(self, nodes, edges, sender, receiver, num_nodes):
        radial, _ = self.radial(
            edges.vectors.norm(dim=-1, keepdim=True),
            edges.node_atomic_numbers,
            edges.edge_index,
        )
        weights = self.network(radial).reshape(-1, self.num_paths, self.num_features)
        return self.convolution(
            nodes, self.harmonics(edges.vectors), weights, sender, receiver, num_nodes
        )


class ReferenceSpan(nn.Module):
    capabilities = SpanCapabilities(True, True)

    def __init__(self, shape):
        super().__init__()
        self.template = Composition(shape)
        tensors = (
            self.template.radial.span_weights() + self.template.network.span_weights()
        )
        live = {id(value): index for index, value in enumerate(tensors)}
        self.weight_names = []
        self.constants = []
        for name, value in (
            *self.template.named_parameters(),
            *self.template.named_buffers(),
        ):
            if id(value) in live:
                self.weight_names.append((name, live[id(value)]))
            else:
                self.constants.append((name, value.tolist(), value.dtype))
            parent, _, attribute = name.rpartition(".")
            owner = self.template.get_submodule(parent)
            if attribute in owner._parameters:
                owner.register_parameter(attribute, None)
            else:
                owner.register_buffer(attribute, None)
        for module in self.template.modules():
            module.__dict__.update(
                to_canonical=None,
                canonical_metadata=None,
                initialize_weights=_no_initialization,
            )

    def forward(
        self, nodes, edges, sender, receiver, num_nodes, member_weights, edge_runs
    ):
        state = {name: member_weights[index] for name, index in self.weight_names}
        state.update(
            {
                name: torch.tensor(data, dtype=dtype, device=nodes.device)
                for name, data, dtype in self.constants
            }
        )
        return torch.func.functional_call(
            self.template, state, (nodes, edges, sender, receiver, num_nodes)
        )


class SpanBackend(ReferenceBackend):
    name = "test_span"
    span_capabilities = SpanCapabilities(True, True)
    edge_features = frozenset()
    spec_version = KERNEL_SPEC_VERSION

    def __init__(self):
        self.spans = []

    def capabilities(self):
        capabilities = super().capabilities()
        return replace(
            capabilities,
            ops=capabilities.ops | {"distance_transform"},
            edge_features=self.edge_features,
            spec_version=self.spec_version,
        )

    def make_interaction_layer(self, descriptors):
        shape = EdgeConvolutionSpan.from_descriptors(descriptors)
        if shape is None:
            return None
        span = ReferenceSpan(shape)
        span.capabilities = self.span_capabilities
        self.spans.append(span)
        return span
