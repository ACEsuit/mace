"""Message passing: the interaction blocks, on the kernel backend's ops.

Every equivariant operation is an op the backend built from a descriptor when
the block was constructed: the linears, the convolution, the skip connection,
the radial networks and the reduction. Nothing resolves in ``forward``, nothing
branches on how the backend fused its kernels, and no tensor product is
written out here.

Three standard blocks and one nonlinear block, each a different function that
a trained model depends on:

* :class:`InteractionBlock` applies its skip connection **to the message**,
  which replaces it. It is the plain first layer.
* :class:`ResidualInteractionBlock` takes its skip from the **input** and
  hands it onward, for the product basis to add after its contraction.
* Either divides its messages by the average neighbour count, or, with a
  learned density, by ``1 + sum over the edges of tanh(f(r)^2)``, ``f`` one
  linear map of the radial embedding: each atom's own count, softened.
* :class:`NonLinearInteractionBlock` is a convolution conditioned on both
  elements, normalized by a learned density and two trained scalars, and
  gated. Its docstring says how.

Three things are the format rather than a choice. **The linears mix
channels**: they map between declarations that carry multiplicity. **The
convolution keeps its paths apart**, and the linear after it mixes them with
independent weights. **The normalization comes after that linear**, and it is
the average itself, not its square root.

Features are grouped by irrep between ops, the canonical ``mul_ir`` over the
declaration expanded to the channel count.
"""

from __future__ import annotations

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    Descriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    RadialMLPDescriptor,
    RadialMLPKind,
    SegmentReduceDescriptor,
)
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

from mace_torch.backends.layout import expanded_irreps
from mace_torch.backends.reference import TORCH_DTYPES, ReferenceBackend
from mace_torch.nn.edges import EdgeInputs
from mace_torch.nn.gate import Gate, gate_input_irreps

__all__ = [
    "DEFAULT_RADIAL_HIDDEN",
    "InteractionBlock",
    "NonLinearInteractionBlock",
    "ResidualInteractionBlock",
]

#: The hidden widths of the radial network, the legacy command-line default.
DEFAULT_RADIAL_HIDDEN = (64, 64, 64)

#: The hidden width of the nonlinear block's density network.
_NONLINEAR_DENSITY_HIDDEN = (64,)

#: The nonlinear block's normalization ``alpha + beta * density`` starts at
#: ``alpha``, a softened constant neighbour count.
_INITIAL_ALPHA = 20.0

#: The spread of the nonlinear block's element embeddings at the start, so that
#: the radial networks begin nearly blind to the elements.
_ELEMENT_EMBEDDING_BOUND = 1e-3


def _radial_network(backend, descriptor: RadialMLPDescriptor) -> nn.Module:
    """The backend's radial MLP, or the reference's when it declines.

    A reference-only op: declining is a normal answer, and the weights are
    canonical whoever builds them.
    """
    built = backend.make_radial_mlp(descriptor)
    if built is None:
        built = ReferenceBackend().make_radial_mlp(descriptor)
    return built


def _ask_for_span(backend, members: tuple[Descriptor, ...] | None) -> object | None:
    """The backend's answer for a whole edge convolution, asked once at build.

    ``None``, the normal answer and the reference's, builds the block op by
    op. An accepted span is held but not run here: when and how it runs is the
    span contract's. It must hold no state, so that a block's checkpoint is the
    same whatever the backend answered.
    """
    if members is None:
        return None
    span = backend.make_interaction_layer(members)
    if isinstance(span, nn.Module) and (held := list(span.state_dict())):
        raise ValueError(
            f"the span {type(span).__name__} holds {held}. A span is handed "
            f"what it computes with on every call and holds nothing, so that a "
            f"block's checkpoint does not depend on whether it was fused."
        )
    return span


class _Convolution(nn.Module):
    """What the standard blocks share: up, convolve, mix, normalize.

    Held apart from the blocks so that the difference between them, where the
    skip goes, is visible as the one thing it is.
    """

    neighbours: Tensor

    def __init__(
        self,
        backend,
        irreps_node: str,
        irreps_edge: str,
        irreps_target: str,
        num_radial: int,
        num_features: int,
        avg_num_neighbors: float,
        radial_hidden: tuple[int, ...],
        precision: Precision,
        learned_density: bool,
        envelope_outside: bool,
        edge_members: tuple[Descriptor, ...] | None,
    ) -> None:
        super().__init__()
        node_flat = expanded_irreps(irreps_node, num_features)
        convolution = ChannelwiseTPConvDescriptor(
            irreps_node=irreps_node,
            irreps_edge=irreps_edge,
            irreps_out=irreps_target,
            num_features=num_features,
            precision=precision,
        )
        self.num_features = num_features
        self.num_paths = len(convolution.paths)
        self.envelope_outside = envelope_outside
        self.irreps_out = expanded_irreps(irreps_target, num_features)

        self.linear_up = backend.make_linear(
            LinearDescriptor(
                irreps_in=node_flat, irreps_out=node_flat, precision=precision
            )
        )
        self.convolution = backend.make_channelwise_tp_conv(convolution)
        weights = RadialMLPDescriptor(
            channels=(num_radial, *radial_hidden, self.num_paths * num_features),
            kind="normalized_silu",
            precision=precision,
        )
        self.radial_mlp = _radial_network(backend, weights)
        self.linear = backend.make_linear(
            LinearDescriptor(
                irreps_in=expanded_irreps(convolution.irreps_paths, num_features),
                irreps_out=self.irreps_out,
                precision=precision,
            )
        )
        self.register_buffer(
            "neighbours",
            torch.tensor(float(avg_num_neighbors), dtype=TORCH_DTYPES[precision]),
        )
        # One linear map of the radial embedding to one number per edge.
        density = (
            RadialMLPDescriptor(
                channels=(num_radial, 1), kind="normalized_silu", precision=precision
            )
            if learned_density
            else None
        )
        self.density_mlp = (
            None if density is None else _radial_network(backend, density)
        )
        self.reduce = backend.make_segment_reduce(
            SegmentReduceDescriptor(
                reduction="sum", num_features=1, precision=precision
            )
        )
        members = (
            None
            if edge_members is None
            else (*edge_members, weights, convolution)
            + (() if density is None else (density,))
        )
        self.span_members = members
        self.span = _ask_for_span(backend, members)

    def forward(
        self,
        node_features: Tensor,
        edges: EdgeInputs,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Grouped node features in, the grouped normalized message out."""
        mapped = self.linear_up(node_features)
        weights = self.radial_mlp(edges.radial)
        if self.envelope_outside:
            weights = weights * edges.envelope
        message = self.convolution(
            mapped,
            edges.attributes,
            weights.reshape(-1, self.num_paths, self.num_features),
            sender,
            receiver,
            num_nodes,
        )
        if self.density_mlp is None:
            return self.linear(message) / self.neighbours
        edge_density = torch.tanh(self.density_mlp(edges.radial) ** 2)
        if self.envelope_outside:
            edge_density = edge_density * edges.envelope
        density = self.reduce(edge_density, receiver, num_nodes)
        return self.linear(message) / (density + 1.0)


class InteractionBlock(nn.Module):
    """The plain block: the skip is applied to the message and replaces it.

    Args:
        backend: The kernel backend. Consulted at construction only.
        irreps_node: One channel's node features.
        irreps_edge: The edge attributes, the spherical harmonics.
        irreps_target: One channel's message, the irreps the convolution keeps.
        num_radial: The radial embedding's width.
        num_features: The channel count.
        num_elements: How many elements, the skip's second input.
        avg_num_neighbors: The normalization without a learned density.
        radial_hidden: The radial network's hidden widths.
        precision: The dtype name every op is built at.
        learned_density: Normalize by a density learned per atom.
        envelope_outside: The radial embedding does not carry the cutoff
            envelope, so the radial networks' outputs are multiplied by it.
        edge_members: The descriptors of the radial basis, the harmonics and
            the distance transform this block's edge convolution starts from,
            or ``None`` when one of them has none.
    """

    def __init__(
        self,
        backend,
        irreps_node: str,
        irreps_edge: str,
        irreps_target: str,
        num_radial: int,
        num_features: int,
        num_elements: int,
        avg_num_neighbors: float = 1.0,
        radial_hidden: tuple[int, ...] = DEFAULT_RADIAL_HIDDEN,
        precision: Precision = "float64",
        learned_density: bool = False,
        envelope_outside: bool = False,
        edge_members: tuple[Descriptor, ...] | None = None,
    ) -> None:
        super().__init__()
        self.body = _Convolution(
            backend,
            irreps_node,
            irreps_edge,
            irreps_target,
            num_radial,
            num_features,
            avg_num_neighbors,
            tuple(radial_hidden),
            precision,
            learned_density,
            envelope_outside,
            edge_members,
        )
        self.skip = backend.make_fully_connected_tp(
            FullyConnectedTPDescriptor(
                irreps_in1=self.body.irreps_out,
                irreps_in2=f"{num_elements}x0e",
                irreps_out=self.body.irreps_out,
                precision=precision,
            )
        )

    def forward(
        self,
        node_features: Tensor,
        edges: EdgeInputs,
        node_attrs: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> tuple[Tensor, Tensor | None]:
        """The message, grouped, and no skip to carry."""
        message = self.body(node_features, edges, sender, receiver, num_nodes)
        return self.skip(message, node_attrs), None


class ResidualInteractionBlock(nn.Module):
    """The residual block: the skip is taken from the input and carried onward.

    The product basis adds it after its contraction, which is where a trained
    model puts it, so the skip's output declaration is the product basis's.

    Args:
        irreps_skip_out: One channel's declaration of what the skip produces,
            the product basis's output. The remaining arguments are
            :class:`InteractionBlock`'s.
    """

    def __init__(
        self,
        backend,
        irreps_node: str,
        irreps_edge: str,
        irreps_target: str,
        irreps_skip_out: str,
        num_radial: int,
        num_features: int,
        num_elements: int,
        avg_num_neighbors: float = 1.0,
        radial_hidden: tuple[int, ...] = DEFAULT_RADIAL_HIDDEN,
        precision: Precision = "float64",
        learned_density: bool = False,
        envelope_outside: bool = False,
        edge_members: tuple[Descriptor, ...] | None = None,
    ) -> None:
        super().__init__()
        self.body = _Convolution(
            backend,
            irreps_node,
            irreps_edge,
            irreps_target,
            num_radial,
            num_features,
            avg_num_neighbors,
            tuple(radial_hidden),
            precision,
            learned_density,
            envelope_outside,
            edge_members,
        )
        self.skip = backend.make_fully_connected_tp(
            FullyConnectedTPDescriptor(
                irreps_in1=expanded_irreps(irreps_node, num_features),
                irreps_in2=f"{num_elements}x0e",
                irreps_out=expanded_irreps(irreps_skip_out, num_features),
                precision=precision,
            )
        )

    def forward(
        self,
        node_features: Tensor,
        edges: EdgeInputs,
        node_attrs: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> tuple[Tensor, Tensor]:
        """The message and the carried skip, both grouped."""
        carried = self.skip(node_features, node_attrs)
        message = self.body(node_features, edges, sender, receiver, num_nodes)
        return message, carried


def _even_scalars(irreps: str) -> int:
    return sum(
        mul
        for mul, ir in Irreps.parse(irreps).terms
        if ir.degree == 0 and ir.parity == 1
    )


class NonLinearInteractionBlock(nn.Module):
    """A convolution conditioned on both elements, with a gated output.

    Four things set it apart from the standard blocks, each a set of weights a
    trained model carries:

    * **It convolves at its own width.** The node features are mapped to
      ``irreps_up`` at ``num_up_features`` channels, convolved there, and the
      linear after the convolution maps back.
    * **Its radial networks see both elements.** Their input is the radial
      embedding with an embedding of the sender's element and one of the
      receiver's appended, and they are layer-normalized networks.
    * **Its normalization is learned**: the message is divided by
      ``alpha + beta * density``, the density learned per atom as in the
      density blocks and ``alpha``, ``beta`` two trained scalars.
    * **It ends in a nonlinearity**: a residual from the up-projection is
      added, the gate applied, and a last linear taken.

    The skip is a linear map of the input, carried to the product basis as
    the residual block carries its own, in the first layer too.

    Args:
        backend: The kernel backend. Consulted at construction only.
        irreps_node: One channel's node features.
        irreps_up: One channel's declaration of what is convolved.
        irreps_edge: The edge attributes.
        irreps_target: One channel's message.
        irreps_skip_out: One channel's skip output, the product basis's.
        num_radial: The radial embedding's width.
        num_features: The node features' channel count.
        num_up_features: The channel count of what is convolved.
        num_elements: How many elements.
        radial_hidden: The radial network's hidden widths.
        precision: The dtype name every op is built at.
        envelope_outside: As :class:`InteractionBlock`'s.
        edge_members: As :class:`InteractionBlock`'s.
    """

    alpha: nn.Parameter
    beta: nn.Parameter

    def __init__(
        self,
        backend,
        irreps_node: str,
        irreps_up: str,
        irreps_edge: str,
        irreps_target: str,
        irreps_skip_out: str,
        num_radial: int,
        num_features: int,
        num_up_features: int,
        num_elements: int,
        radial_hidden: tuple[int, ...] = DEFAULT_RADIAL_HIDDEN,
        precision: Precision = "float64",
        envelope_outside: bool = False,
        edge_members: tuple[Descriptor, ...] | None = None,
    ) -> None:
        super().__init__()
        node_flat = expanded_irreps(irreps_node, num_features)
        up_flat = expanded_irreps(irreps_up, num_up_features)
        target_flat = expanded_irreps(irreps_target, num_features)
        nonlinear_flat = gate_input_irreps(target_flat)
        element_scalars = _even_scalars(node_flat)
        convolution = ChannelwiseTPConvDescriptor(
            irreps_node=irreps_up,
            irreps_edge=irreps_edge,
            irreps_out=irreps_target,
            num_features=num_up_features,
            precision=precision,
        )
        self.num_up_features = num_up_features
        self.num_paths = len(convolution.paths)
        self.envelope_outside = envelope_outside
        self.irreps_out = target_flat

        def linear(irreps_in: str, irreps_out: str) -> LinearDescriptor:
            return LinearDescriptor(
                irreps_in=irreps_in, irreps_out=irreps_out, precision=precision
            )

        embedding = linear(f"{num_elements}x0e", f"{element_scalars}x0e")
        self.skip = backend.make_linear(
            linear(node_flat, expanded_irreps(irreps_skip_out, num_features))
        )
        self.linear_up = backend.make_linear(linear(node_flat, up_flat))
        self.linear_residual = backend.make_linear(linear(up_flat, nonlinear_flat))
        self.sender_embedding = backend.make_linear(embedding)
        self.receiver_embedding = backend.make_linear(embedding)
        self.convolution = backend.make_channelwise_tp_conv(convolution)
        conditioned = num_radial + 2 * element_scalars
        kind: RadialMLPKind = "layer_norm_silu"
        weights = RadialMLPDescriptor(
            channels=(conditioned, *radial_hidden, self.num_paths * num_up_features),
            kind=kind,
            precision=precision,
        )
        density = RadialMLPDescriptor(
            channels=(conditioned, *_NONLINEAR_DENSITY_HIDDEN, 1),
            kind=kind,
            precision=precision,
        )
        self.radial_mlp = _radial_network(backend, weights)
        self.density_mlp = _radial_network(backend, density)
        self.reduce = backend.make_segment_reduce(
            SegmentReduceDescriptor(
                reduction="sum", num_features=1, precision=precision
            )
        )
        self.linear_mix = backend.make_linear(
            linear(
                expanded_irreps(convolution.irreps_paths, num_up_features),
                nonlinear_flat,
            )
        )
        self.gate = Gate(target_flat)
        self.linear_out = backend.make_linear(linear(target_flat, target_flat))
        dtype = TORCH_DTYPES[precision]
        self.alpha = nn.Parameter(torch.tensor(_INITIAL_ALPHA, dtype=dtype))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        members = (
            None
            if edge_members is None
            else (*edge_members, embedding, embedding, weights, density, convolution)
        )
        self.span_members = members
        self.span = _ask_for_span(backend, members)

    def forward(
        self,
        node_features: Tensor,
        edges: EdgeInputs,
        node_attrs: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> tuple[Tensor, Tensor]:
        """The message and the carried skip, both grouped."""
        carried = self.skip(node_features)
        up = self.linear_up(node_features)
        residual = self.linear_residual(up)
        conditioned = torch.cat(
            [
                edges.radial,
                self.sender_embedding(node_attrs).index_select(0, sender),
                self.receiver_embedding(node_attrs).index_select(0, receiver),
            ],
            dim=-1,
        )
        weights = self.radial_mlp(conditioned)
        edge_density = torch.tanh(self.density_mlp(conditioned) ** 2)
        if self.envelope_outside:
            weights = weights * edges.envelope
            edge_density = edge_density * edges.envelope
        density = self.reduce(edge_density, receiver, num_nodes)
        message = self.convolution(
            up,
            edges.attributes,
            weights.reshape(-1, self.num_paths, self.num_up_features),
            sender,
            receiver,
            num_nodes,
        )
        message = self.linear_mix(message) / (density * self.beta + self.alpha)
        return self.linear_out(self.gate(message + residual)), carried

    def initialize_weights(self, seed: int) -> None:
        """The frozen tree's start for what is the block's own.

        ``alpha`` at 20 and ``beta`` at 0, and the two element embeddings
        uniform on ``[-1e-3, 1e-3]`` rather than at the standard normal their
        ops draw. Called after the ops have drawn, so this overrides them.
        """
        with torch.no_grad():
            self.alpha.fill_(_INITIAL_ALPHA)
            self.beta.zero_()
        for offset, embedding in enumerate(
            (self.sender_embedding, self.receiver_embedding)
        ):
            state = dict(embedding.to_canonical())
            weight = state["weight"]
            generator = torch.Generator().manual_seed(seed + offset)
            unit = torch.rand(weight.shape, generator=generator, dtype=torch.float64)
            state["weight"] = ((2.0 * unit - 1.0) * _ELEMENT_EMBEDDING_BOUND).to(
                device=weight.device, dtype=weight.dtype
            )
            embedding.load_canonical(state)

    def to_canonical(self) -> dict[str, Tensor]:
        """The two normalization scalars. The ops hold their own weights."""
        return {"alpha": self.alpha.detach(), "beta": self.beta.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.alpha.copy_(state["alpha"])
            self.beta.copy_(state["beta"])
