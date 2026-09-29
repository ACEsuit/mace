"""The blocks of a backbone that reads a magnetic moment on every atom.

The moment enters twice, and neither is a sum onto the node features. Its
length, squashed by a per-element saturation, is expanded in Chebyshev
polynomials and read by the radial networks beside the edge length. Its
direction, as solid harmonics, is coupled into the messages and into the
product basis by tensor products of their own. That is why this is not the
declared node-input stream the standard backbone mixes in: a linear map of the
moment onto the features computes a different function.

Three things here are properties of the trained models rather than choices:

**The moment is a polar vector.** Its harmonics are ``0e + 1o + 2e``, as the
edge directions' are, so a structure inverted with its moments turns into
itself. Physically a moment is axial, and a model built that way is a different
model.

**The harmonics are solid, not normalized.** ``|m|^l Y_lm(m / |m|)`` with each
``l`` block divided by ``sqrt(4 pi)``, which is what the frozen tree's
sphericart call computes. It is a polynomial, so it is finite and smooth at a
zero moment, and so is its derivative. The squashed length is smooth there too,
because the length enters only squared.

**The messages are divided by a learned density**, ``1 + sum_j tanh(d(r_ij)^2)``,
and not by an average neighbour count. There is no neighbour count in this
backbone at all.

The tensor products run on the backend's channelwise convolution. A message
couples twice, first with the edge direction and then with the sender's
moment, and only then is summed onto the receiver, so the first coupling is a
convolution onto the edges themselves and the second one sums the edges onto
the nodes. The product basis couples each node with its own moment, which is a
convolution whose edges are the nodes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    RadialBasisDescriptor,
    SphericalHarmonicsDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.paths import channelwise_paths
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

from mace_torch.kernels import segment_sum
from mace_torch.nn.layout import (
    channel_layout_index,
    expanded_irreps,
    inverse_layout_index,
    path_layout_index,
)
from mace_torch.nn.radial import ChebyshevBasis
from mace_torch.nn.radial_mlp import RadialMLP

__all__ = [
    "MagneticBackbone",
    "MagneticInteractionBlock",
    "MagneticProductBlock",
    "MomentFeatures",
    "OneBodyMomentEnergy",
]

#: The radial networks' hidden widths, as the trained models carry them. The
#: product basis has its own and they are the same numbers.
RADIAL_HIDDEN = (64, 64, 64)


def _degrees(lmax: int) -> str:
    return "+".join(
        f"{degree}{'e' if degree % 2 == 0 else 'o'}" for degree in range(lmax + 1)
    )


class MomentFeatures(nn.Module):
    """What the blocks read off each atom's moment.

    Args:
        backend: The kernel backend. Consulted at construction only.
        saturation: One saturation per element of the table, in muB. A moment
            at or beyond it reads as saturated.
        num_basis: How many Chebyshev polynomials of the squashed length.
        lmax: The highest degree of the moment's harmonics.
        precision: The dtype the constants are held at.
    """

    saturation: Tensor

    def __init__(
        self,
        backend,
        saturation: Sequence[float],
        num_basis: int,
        lmax: int,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.irreps = _degrees(lmax)
        self.register_buffer(
            "saturation",
            torch.tensor(list(saturation), dtype=getattr(torch, precision)),
        )
        self.basis = ChebyshevBasis(num_basis=num_basis, include_constant=False)
        self.harmonics = backend.make_spherical_harmonics(
            SphericalHarmonicsDescriptor(
                lmax=lmax, normalize=False, precision=precision
            )
        )

    def squashed_length(self, moments: Tensor, element: Tensor) -> Tensor:
        """``1 - 2 min(|m| / m_max, 1)^2``, as ``[n_atoms, 1]``.

        One at a zero moment and minus one at saturation. The clamp's upper
        bound is where the derivative against a moment's length stops.
        """
        lengths = torch.linalg.vector_norm(moments, dim=-1, keepdim=True)
        scaled = lengths / self.saturation[element].unsqueeze(-1)
        return 1 - 2 * torch.clamp(scaled, min=0.0, max=1.0) ** 2

    def forward(
        self, moments: Tensor, element: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """The squashed length, its expansion, and the solid harmonics.

        Returns:
            ``[n_atoms, 1]``, ``[n_atoms, num_basis]`` and
            ``[n_atoms, (lmax + 1)^2]``.
        """
        squashed = self.squashed_length(moments, element)
        harmonics = self.harmonics(moments) / math.sqrt(4 * math.pi)
        return squashed, self.basis(squashed), harmonics

    def to_canonical(self) -> dict[str, Tensor]:
        """The saturations, which are a constant of the trained model.

        Not a weight, but a model rebuilt with other values reads every moment
        at another length, so they travel with the weights.
        """
        return {"saturation": self.saturation.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.saturation.copy_(state["saturation"])


class _PairCoupling(nn.Module):
    """A channelwise coupling of features with an attribute, as a convolution.

    The backend's convolution computes, for each receiver ``r``,
    ``sum_{e -> r} TP(x[sender(e)], y[e]; w[e])``. Every use here is that with
    a choice of which rows are the senders and which the receivers, so the one
    op serves the edge coupling, the moment coupling and the product's.
    """

    def __init__(
        self,
        backend,
        irreps_node: str,
        irreps_attribute: str,
        irreps_target: str,
        num_features: int,
        precision: Precision,
    ) -> None:
        super().__init__()
        self.paths = channelwise_paths(irreps_node, irreps_attribute, irreps_target)
        self.num_paths = len(self.paths)
        self.num_features = num_features
        self.irreps_paths = "+".join(str(path.irrep) for path in self.paths)
        self.convolution = backend.make_channelwise_tp_conv(
            ChannelwiseTPConvDescriptor(
                irreps_node=irreps_node,
                irreps_edge=irreps_attribute,
                irreps_out=irreps_target,
                precision=precision,
            )
        )

    def forward(
        self,
        features: Tensor,
        attributes: Tensor,
        weights: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_receivers: int,
    ) -> Tensor:
        """``[rows, channel, width]`` in, ``[receivers, channel, paths]`` out."""
        return self.convolution(
            features,
            attributes,
            weights.reshape(-1, self.num_paths, self.num_features),
            sender,
            receiver,
            num_receivers,
        )


class MagneticInteractionBlock(nn.Module):
    """A message coupled with the edge direction and then with the moment.

    Args:
        backend: The kernel backend. Consulted at construction only.
        irreps_node: One channel's node-feature declaration.
        irreps_edge: The edge harmonics.
        irreps_moment: The moment harmonics.
        irreps_skip_out: What the residual skip produces, the product basis's
            output. ``None`` for the first layer, whose skip is applied to the
            message instead and replaces it.
        num_radial: Width of the radial embedding.
        num_moment_basis: Width of the moment's length expansion.
        num_features: The channel count.
        num_elements: How many species, the skip's second input.
        precision: The dtype every op is built at.
    """

    to_channels: Tensor
    from_paths: Tensor

    def __init__(
        self,
        backend,
        irreps_node: str,
        irreps_edge: str,
        irreps_moment: str,
        irreps_skip_out: str | None,
        num_radial: int,
        num_moment_basis: int,
        num_features: int,
        num_elements: int,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        node_flat = expanded_irreps(irreps_node, num_features)
        target_flat = expanded_irreps(irreps_edge, num_features)
        self.num_features = num_features
        self.node_width = Irreps.parse(irreps_node).dimension
        self.irreps_out = target_flat

        self.linear_up = backend.make_linear(
            LinearDescriptor(
                irreps_in=node_flat, irreps_out=node_flat, precision=precision
            )
        )
        self.edge_coupling = _PairCoupling(
            backend, irreps_node, irreps_edge, irreps_edge, num_features, precision
        )
        self.moment_coupling = _PairCoupling(
            backend,
            self.edge_coupling.irreps_paths,
            irreps_moment,
            irreps_edge,
            num_features,
            precision,
        )
        joint = num_radial + num_moment_basis
        self.edge_weights = RadialMLP(
            joint, RADIAL_HIDDEN, self.edge_coupling.num_paths * num_features, precision
        )
        self.moment_weights = RadialMLP(
            joint, (), self.moment_coupling.num_paths * num_features, precision
        )
        self.density = RadialMLP(num_radial, (), 1, precision)
        paths_flat = "+".join(
            f"{num_features}x{path.irrep}" for path in self.moment_coupling.paths
        )
        self.linear = backend.make_linear(
            LinearDescriptor(
                irreps_in=paths_flat, irreps_out=target_flat, precision=precision
            )
        )
        self.residual = irreps_skip_out is not None
        self.skip = backend.make_fully_connected_tp(
            FullyConnectedTPDescriptor(
                irreps_in1=node_flat if self.residual else target_flat,
                irreps_in2=f"{num_elements}x0e",
                irreps_out=(
                    expanded_irreps(irreps_skip_out, num_features)
                    if irreps_skip_out is not None
                    else target_flat
                ),
                precision=precision,
            )
        )
        self.register_buffer(
            "to_channels",
            torch.tensor(inverse_layout_index(irreps_node, num_features)),
            persistent=False,
        )
        self.register_buffer(
            "from_paths",
            torch.tensor(path_layout_index(self.moment_coupling.paths, num_features)),
            persistent=False,
        )

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        edge_radial: Tensor,
        moment_basis: Tensor,
        moment_harmonics: Tensor,
        element_attributes: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> tuple[Tensor, Tensor | None]:
        """The message, flat and grouped by irrep, and the carried skip."""
        carried = (
            self.skip(node_features, element_attributes) if self.residual else None
        )
        mapped = self.linear_up(node_features)[..., self.to_channels].reshape(
            -1, self.num_features, self.node_width
        )
        joint = torch.cat([edge_radial, moment_basis[sender]], dim=-1)
        num_edges = int(sender.shape[0])
        edges = torch.arange(num_edges, device=sender.device)

        on_edges = self.edge_coupling(
            mapped, edge_attributes, self.edge_weights(joint), sender, edges, num_edges
        )
        message = self.moment_coupling(
            on_edges,
            moment_harmonics[sender],
            self.moment_weights(joint),
            edges,
            receiver,
            num_nodes,
        )
        density = segment_sum(
            torch.tanh(self.density(edge_radial) ** 2), receiver, num_nodes
        )
        message = self.linear(message.reshape(num_nodes, -1)[..., self.from_paths])
        message = message / (density + 1)
        if not self.residual:
            message = self.skip(message, element_attributes)
        return message, carried


class MagneticProductBlock(nn.Module):
    """The many-body contraction, then a coupling with the atom's own moment.

    ``linear(TP(c, Y(m))) + linear_ori(c) + skip``, where ``c`` is the
    contraction. The skip is the residual interaction's, and the first layer
    has none.

    Args:
        backend: The kernel backend. Consulted at construction only.
        irreps_in: One channel's input declaration.
        irreps_out: One channel's output declaration.
        irreps_moment: The moment harmonics.
        num_moment_basis: Width of the moment's length expansion.
        correlation: The body order.
        num_elements: How many species, which the contraction weights index.
        num_features: The channel count.
        precision: The dtype every op is built at.
    """

    to_channels: Tensor
    from_channels: Tensor
    from_paths: Tensor

    def __init__(
        self,
        backend,
        irreps_in: str,
        irreps_out: str,
        irreps_moment: str,
        num_moment_basis: int,
        correlation: int,
        num_elements: int,
        num_features: int,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.width_in = Irreps.parse(irreps_in).dimension
        out_flat = expanded_irreps(irreps_out, num_features)
        self.contraction = backend.make_symmetric_contraction(
            SymmetricContractionDescriptor(
                irreps_in=irreps_in,
                irreps_out=irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                num_features=num_features,
                precision=precision,
            )
        )
        self.moment_coupling = _PairCoupling(
            backend, irreps_out, irreps_moment, irreps_out, num_features, precision
        )
        self.moment_weights = RadialMLP(
            num_moment_basis,
            RADIAL_HIDDEN,
            self.moment_coupling.num_paths * num_features,
            precision,
        )
        paths_flat = "+".join(
            f"{num_features}x{path.irrep}" for path in self.moment_coupling.paths
        )
        self.linear = backend.make_linear(
            LinearDescriptor(
                irreps_in=paths_flat, irreps_out=out_flat, precision=precision
            )
        )
        self.linear_ori = backend.make_linear(
            LinearDescriptor(
                irreps_in=out_flat, irreps_out=out_flat, precision=precision
            )
        )
        self.register_buffer(
            "to_channels",
            torch.tensor(inverse_layout_index(irreps_in, num_features)),
            persistent=False,
        )
        self.register_buffer(
            "from_channels",
            torch.tensor(channel_layout_index(irreps_out, num_features)),
            persistent=False,
        )
        self.register_buffer(
            "from_paths",
            torch.tensor(path_layout_index(self.moment_coupling.paths, num_features)),
            persistent=False,
        )

    def forward(
        self,
        message: Tensor,
        element: Tensor,
        moment_basis: Tensor,
        moment_harmonics: Tensor,
        skip: Tensor | None,
    ) -> Tensor:
        """Flat grouped features in, flat grouped features out."""
        nodes = message.shape[0]
        features = message[..., self.to_channels].reshape(
            nodes, self.num_features, self.width_in
        )
        contracted = self.contraction(features, element)
        atoms = torch.arange(nodes, device=message.device)
        coupled = self.moment_coupling(
            contracted,
            moment_harmonics,
            self.moment_weights(moment_basis),
            atoms,
            atoms,
            nodes,
        )
        grouped = contracted.reshape(nodes, -1)[..., self.from_channels]
        mapped = self.linear(coupled.reshape(nodes, -1)[..., self.from_paths])
        mapped = mapped + self.linear_ori(grouped)
        if skip is not None:
            mapped = mapped + skip
        return mapped


class OneBodyMomentEnergy(nn.Module):
    """A per-atom energy of the moment's length alone, per element and head.

    ``sum_k c[Z, k, h] T_k(s) - offset[Z, h]`` over Chebyshev polynomials of
    the squashed length ``s``, the constant one included. It goes into the
    scaled sum beside the readouts. The offset is zero unless a model set it to
    move the zero of the term onto its isolated-atom energies.

    Args:
        num_elements: The element table's size.
        num_basis: How many polynomials, the constant included.
        num_heads: How many heads.
        precision: The dtype the coefficients are held at.
    """

    offset: Tensor

    def __init__(
        self,
        num_elements: int,
        num_basis: int,
        num_heads: int = 1,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        dtype = getattr(torch, precision)
        self.basis = ChebyshevBasis(num_basis=num_basis, include_constant=True)
        self.coefficients = nn.Parameter(
            torch.randn(num_elements, num_basis, num_heads, dtype=dtype)
        )
        self.register_buffer(
            "offset", torch.zeros(num_elements, num_heads, dtype=dtype)
        )

    def forward(self, squashed: Tensor, element: Tensor, head: Tensor | None) -> Tensor:
        """``[n_atoms]``, each atom's term under its own head."""
        per_head = (
            self.basis(squashed).unsqueeze(-1) * self.coefficients[element]
        ).sum(dim=1) - self.offset[element]
        if head is None:
            return per_head[:, 0]
        return per_head.gather(1, head.unsqueeze(-1)).squeeze(-1)

    def initialize_weights(self, seed: int) -> None:
        """A standard normal draw, from the seed alone, as the frozen tree has."""
        with torch.no_grad():
            generator = torch.Generator().manual_seed(seed)
            self.coefficients.copy_(
                torch.randn(
                    self.coefficients.shape, generator=generator, dtype=torch.float64
                ).to(self.coefficients.dtype)
            )

    def to_canonical(self) -> dict[str, Tensor]:
        return {
            "coefficients": self.coefficients.detach(),
            "offset": self.offset.detach(),
        }

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.coefficients.copy_(state["coefficients"])
            self.offset.copy_(state["offset"])


class MagneticBackbone(nn.Module):
    """The message-passing backbone of a model that reads moments.

    Node features in, node features out, like the standard backbone, and with
    the same interface to the readouts above it: one grouped tensor per layer,
    declared by :attr:`layer_irreps`.

    Args:
        backend: The kernel backend. Consulted at construction only.
        atomic_numbers: The element table, ascending.
        saturation: One moment saturation per element, in muB.
        num_layers: How many interaction and product pairs.
        num_features: The channel width.
        lmax: The highest degree of the edge harmonics.
        moment_lmax: The highest degree of the moment harmonics.
        hidden_irreps: One channel's node-feature irreps.
        num_radial: Width of the radial embedding.
        num_moment_basis: Width of the moment's length expansion.
        cutoff: In Angstrom.
        cutoff_order: The order of the envelope at the cutoff.
        correlation: The body order of the product basis.
        precision: The dtype name every op is built at.
    """

    def __init__(
        self,
        backend,
        atomic_numbers: Sequence[int],
        saturation: Sequence[float],
        num_layers: int = 2,
        num_features: int = 16,
        lmax: int = 2,
        moment_lmax: int = 2,
        hidden_irreps: str = "0e+1o",
        num_radial: int = 8,
        num_moment_basis: int = 6,
        cutoff: float = 5.0,
        cutoff_order: int = 6,
        correlation: int = 3,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        if len(saturation) != len(atomic_numbers):
            raise ValueError(
                f"there are {len(saturation)} moment saturations for "
                f"{len(atomic_numbers)} elements {list(atomic_numbers)}. There "
                f"is one per element."
            )
        self.atomic_numbers = list(atomic_numbers)
        self.num_features = num_features
        self.cutoff = cutoff
        self.hidden_irreps = hidden_irreps
        edge_irreps = _degrees(lmax)
        self.edge_irreps = edge_irreps
        self.edge_attributes = backend.make_spherical_harmonics(
            SphericalHarmonicsDescriptor(lmax=lmax, precision=precision)
        )
        self.radial = backend.make_radial_basis(
            RadialBasisDescriptor(
                kind="bessel",
                num_basis=num_radial,
                cutoff=cutoff,
                cutoff_order=cutoff_order,
                precision=precision,
            )
        )
        self.moments = MomentFeatures(
            backend, saturation, num_moment_basis, moment_lmax, precision
        )
        self.node_embedding = backend.make_linear(
            LinearDescriptor(
                irreps_in=f"{len(self.atomic_numbers)}x0e",
                irreps_out=f"{num_features}x0e",
                precision=precision,
            )
        )
        self.layer_irreps = [
            "0e" if layer == num_layers - 1 else hidden_irreps
            for layer in range(num_layers)
        ]
        interactions, products = [], []
        for layer in range(num_layers):
            interactions.append(
                MagneticInteractionBlock(
                    backend,
                    irreps_node="0e" if layer == 0 else hidden_irreps,
                    irreps_edge=edge_irreps,
                    irreps_moment=self.moments.irreps,
                    irreps_skip_out=None if layer == 0 else self.layer_irreps[layer],
                    num_radial=num_radial,
                    num_moment_basis=num_moment_basis,
                    num_features=num_features,
                    num_elements=len(self.atomic_numbers),
                    precision=precision,
                )
            )
            products.append(
                MagneticProductBlock(
                    backend,
                    irreps_in=edge_irreps,
                    irreps_out=self.layer_irreps[layer],
                    irreps_moment=self.moments.irreps,
                    num_moment_basis=num_moment_basis,
                    correlation=correlation,
                    num_elements=len(self.atomic_numbers),
                    num_features=num_features,
                    precision=precision,
                )
            )
        self.interactions = nn.ModuleList(interactions)
        self.products = nn.ModuleList(products)

    def element_index(self, atomic_numbers: Tensor) -> Tensor:
        """Each node's position in the element table."""
        table = torch.as_tensor(
            self.atomic_numbers,
            dtype=atomic_numbers.dtype,
            device=atomic_numbers.device,
        )
        return (atomic_numbers[:, None] == table[None, :]).float().argmax(dim=-1)

    def forward(self, graph: Mapping[str, Any]) -> list[Tensor]:
        """Every layer's node features, in order.

        Args:
            graph: The flat dict, read and never written to. It carries the
                moments under ``magmom``, ``[n_atoms, 3]`` in muB.
        """
        positions = graph["positions"]
        sender, receiver = graph["edge_index"][0], graph["edge_index"][1]
        num_nodes = int(positions.shape[0])
        if "vectors" in graph:
            vectors = graph["vectors"]
        else:
            vectors = positions[receiver] - positions[sender] + graph["shifts"]
        edge_attributes = self.edge_attributes(vectors)
        radial = self.radial(vectors.norm(dim=-1, keepdim=True))

        element = self.element_index(graph["atomic_numbers"])
        one_hot = torch.zeros(
            num_nodes,
            len(self.atomic_numbers),
            dtype=positions.dtype,
            device=positions.device,
        )
        one_hot[torch.arange(num_nodes, device=positions.device), element] = 1.0
        _, moment_basis, moment_harmonics = self.moments(
            graph["magmom"].to(positions.dtype), element
        )

        features = self.node_embedding(one_hot)
        outputs = []
        for interaction, product in zip(self.interactions, self.products, strict=True):
            message, carried = interaction(
                features,
                edge_attributes,
                radial,
                moment_basis,
                moment_harmonics,
                one_hot,
                sender,
                receiver,
                num_nodes,
            )
            features = product(
                message, element, moment_basis, moment_harmonics, carried
            )
            outputs.append(features)
        return outputs
