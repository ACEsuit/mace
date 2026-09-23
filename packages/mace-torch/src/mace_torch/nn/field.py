"""The blocks a charge-aware model closes around the long-range solve.

A charge-aware model predicts a density of Gaussian multipoles per atom, reads
the potential that density creates back through a projection onto each atom,
and updates the density from what it read. These are the pieces of that loop
that hold weights: the maps that read a density off the node features, the
update that reads a new one off the potential, and the readout that turns the
converged density into a local energy.

Every tensor here is flat and grouped by irrep, the layout the backbone's node
features come in. A density is held **spin major**: the alpha channel's
multipoles, then the beta channel's, which is the declaration
``0e+1o+0e+1o`` for dipoles and why it is written unsimplified.

Two contractions recur and are written once each. :class:`InvariantProducts`
takes a pair of feature sets to one scalar per channel, and
:class:`ScalarModulation` scales each channel's irreps by a scalar of its own.
Both act channel by channel with one weight per pair of channels, and both carry
the Clebsch-Gordan factor of their coupling, ``1 / sqrt(2l + 1)``, as a
constant rather than in the weights.

Each module holding weights of its own writes them in a canonical form and
draws them from a seed, as the kernel ops do. A checkpoint carries only what a
module writes, so a weight without a canonical form would be silently left
behind.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import LinearDescriptor
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

from mace_torch.nn.radial_mlp import SECOND_MOMENT_SCALE

__all__ = [
    "SIGMOID_SECOND_MOMENT_SCALE",
    "BiasReadout",
    "ChargeUpdate",
    "ElectronEnergyReadout",
    "InvariantProducts",
    "LayerNormMLP",
    "ScalarModulation",
]

#: The multiplier trained gates carry on their sigmoid: the unit-second-moment
#: constant sampled from a standard normal, as :data:`SECOND_MOMENT_SCALE` is
#: for the SiLU. It is the sampled value and not the exact one, because trained
#: weights were fitted against the sampled one.
SIGMOID_SECOND_MOMENT_SCALE = 1.8467055342154763


def _normal(like: Tensor, seed: int, scale: float) -> Tensor:
    """A seeded standard normal shaped and typed like ``like``, times ``scale``."""
    generator = torch.Generator().manual_seed(seed)
    draw = torch.randn(like.shape, generator=generator, dtype=torch.float64)
    return (draw * scale).to(dtype=like.dtype, device=like.device)


def _scalar_channels(irreps: str) -> int:
    """How many ``0e`` channels a declaration carries."""
    return sum(
        mul
        for mul, ir in Irreps.parse(irreps).terms
        if ir.degree == 0 and ir.parity == 1
    )


class LayerNormMLP(nn.Module):
    """Linear, layer norm and SiLU, repeated, with a plain linear last.

    Args:
        widths: The width of every layer, input first and output last.
    """

    def __init__(self, widths: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index, (width_in, width_out) in enumerate(itertools.pairwise(widths)):
            layers.append(nn.Linear(width_in, width_out, bias=True))
            if index < len(widths) - 2:
                layers.append(nn.LayerNorm(width_out))
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)

    def to_canonical(self) -> dict[str, Tensor]:
        """Each layer's parameters, under ``<layer>.<name>``."""
        return {name: value.detach() for name, value in self.net.named_parameters()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            for name, value in self.net.named_parameters():
                value.copy_(state[name])

    def initialize_weights(self, seed: int) -> None:
        """The usual draw for a linear, uniform within ``1 / sqrt(fan_in)``,
        and a layer norm that starts as the identity."""
        with torch.no_grad():
            for index, layer in enumerate(self.net):
                if isinstance(layer, nn.Linear):
                    bound = 1.0 / math.sqrt(layer.in_features)
                    generator = torch.Generator().manual_seed(seed + index)
                    for value in (layer.weight, layer.bias):
                        draw = torch.rand(
                            value.shape, generator=generator, dtype=torch.float64
                        )
                        value.copy_(((2 * draw - 1) * bound).to(value.dtype))
                elif isinstance(layer, nn.LayerNorm):
                    layer.weight.fill_(1.0)
                    layer.bias.zero_()


class InvariantProducts(nn.Module):
    """Two feature sets in, one scalar per channel out.

    For every irrep the two sets share, each output channel ``u`` gathers
    ``sum_v w[u, v] <left_u, right_v> / sqrt(2l + 1)``, and the irreps' parts
    are added. Paper symbol: a channel-mixed dot product.

    Args:
        irreps: The declaration both inputs carry, grouped, every term with the
            same multiplicity: the channel count.
    """

    def __init__(self, irreps: str) -> None:
        super().__init__()
        terms = Irreps.parse(irreps).terms
        channels = {mul for mul, _ in terms}
        if len(channels) != 1:
            raise ValueError(
                f"{irreps!r} carries different multiplicities per irrep, and "
                f"the product pairs channels one to one."
            )
        self.channels = channels.pop()
        self.paths: list[tuple[int, int, int, int, int]] = []
        offsets, offset = [], 0
        for mul, ir in terms:
            offsets.append(offset)
            offset += mul * ir.dimension
        for left, (_, left_ir) in enumerate(terms):
            for right, (_, right_ir) in enumerate(terms):
                if left_ir == right_ir:
                    self.paths.append(
                        (left, right, offsets[left], offsets[right], left_ir.dimension)
                    )
        self.weight = nn.Parameter(
            torch.zeros(len(self.paths) * self.channels * self.channels)
        )

    def path_weight(self, index: int) -> Tensor:
        """``[channels, channels]``, path ``index``'s weights."""
        span = self.channels * self.channels
        return self.weight[index * span : (index + 1) * span].view(
            self.channels, self.channels
        )

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        """``[n, dim]`` twice in, ``[n, channels]`` out."""
        nodes = left.shape[0]
        out = left.new_zeros((nodes, self.channels))
        span = self.channels
        for index, (_, _, left_start, right_start, dimension) in enumerate(self.paths):
            size = span * dimension
            left_block = left[:, left_start : left_start + size].view(
                nodes, span, dimension
            )
            right_block = right[:, right_start : right_start + size].view(
                nodes, span, dimension
            )
            pair = torch.einsum("nud,nvd->nuv", left_block, right_block)
            out = out + torch.einsum(
                "nuv,uv->nu", pair, self.path_weight(index)
            ) / math.sqrt(dimension)
        return out

    def to_canonical(self) -> dict[str, Tensor]:
        """``[paths * channels * channels]``, path major, each ``[u, v]``."""
        return {"weight": self.weight.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])

    def initialize_weights(self, seed: int) -> None:
        """A standard normal over the fan-in: every path's partner channels."""
        with torch.no_grad():
            fan_in = len(self.paths) * self.channels
            self.weight.copy_(_normal(self.weight, seed, fan_in**-0.5))


class ScalarModulation(nn.Module):
    """Each channel's irreps scaled by a scalar mixed for that channel.

    ``out_u = features_u * (sum_v w_l[u, v] scalars_v) / sqrt(2l + 1)``, with
    one weight matrix per term of the declaration.

    Args:
        irreps: The features' declaration, grouped, every term with the channel
            count as its multiplicity.
        num_scalars: How many scalars mix into each channel.
    """

    def __init__(self, irreps: str, num_scalars: int) -> None:
        super().__init__()
        self.terms = Irreps.parse(irreps).terms
        self.num_scalars = num_scalars
        self.spans = [(mul, ir.dimension) for mul, ir in self.terms]
        self.weight = nn.Parameter(
            torch.zeros(sum(mul * num_scalars for mul, _ in self.spans))
        )

    def forward(self, features: Tensor, scalars: Tensor) -> Tensor:
        """``[n, dim]`` and ``[n, num_scalars]`` in, ``[n, dim]`` out."""
        nodes = features.shape[0]
        pieces, feature_offset, weight_offset = [], 0, 0
        for mul, dimension in self.spans:
            block = features[:, feature_offset : feature_offset + mul * dimension]
            weight = self.weight[
                weight_offset : weight_offset + mul * self.num_scalars
            ].view(mul, self.num_scalars)
            mixed = scalars @ weight.T
            pieces.append(
                (
                    block.view(nodes, mul, dimension)
                    * mixed.unsqueeze(-1)
                    / math.sqrt(dimension)
                ).reshape(nodes, -1)
            )
            feature_offset += mul * dimension
            weight_offset += mul * self.num_scalars
        return torch.cat(pieces, dim=-1)

    def to_canonical(self) -> dict[str, Tensor]:
        """Term major, each ``[channel, scalar]``."""
        return {"weight": self.weight.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])

    def initialize_weights(self, seed: int) -> None:
        """A standard normal over the fan-in: the scalars mixed in."""
        with torch.no_grad():
            self.weight.copy_(_normal(self.weight, seed, self.num_scalars**-0.5))


class _GatedActivation(nn.Module):
    """SiLU on the scalars, and each higher channel scaled by a sigmoid gate.

    Both activations carry their unit-second-moment multiplier, as trained
    charge-aware readouts do. The input is the scalars, then one gate per
    gated channel, then the gated channels; the gates are spent.
    """

    gate_repeat: Tensor

    def __init__(self, num_scalars: int, gated: str) -> None:
        super().__init__()
        self.num_scalars = num_scalars
        terms = Irreps.parse(gated).terms if gated else ()
        self.num_gates = sum(mul for mul, _ in terms)
        repeats = [ir.dimension for mul, ir in terms for _ in range(mul)]
        self.register_buffer(
            "gate_repeat", torch.tensor(repeats, dtype=torch.long), persistent=False
        )

    def forward(self, features: Tensor) -> Tensor:
        scalars = SECOND_MOMENT_SCALE * torch.nn.functional.silu(
            features[..., : self.num_scalars]
        )
        if not self.num_gates:
            return scalars
        split = self.num_scalars + self.num_gates
        gates = SIGMOID_SECOND_MOMENT_SCALE * torch.sigmoid(
            features[..., self.num_scalars : split]
        )
        gated = features[..., split:]
        return torch.cat(
            [scalars, gated * torch.repeat_interleave(gates, self.gate_repeat, dim=-1)],
            dim=-1,
        )


class BiasReadout(nn.Module):
    """Linear, gate, biased linear, gate, biased linear.

    The readout a charge-aware model reads its sources through. The middle is
    ``hidden``: its scalars, and each of its higher terms the output also
    carries, gated. Only the last two maps carry a bias, because the first one
    reads node features whose scalar offset is the backbone's.

    Args:
        backend: The kernel backend. Consulted at construction only.
        irreps_in: The input declaration, grouped.
        hidden: The middle's declaration, grouped.
        irreps_out: The output declaration, in the order it is returned.
        precision: The dtype every op is built at.
    """

    def __init__(
        self,
        backend,
        irreps_in: str,
        hidden: str,
        irreps_out: str,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        out_irreps = {ir for _, ir in Irreps.parse(irreps_out).terms}
        hidden_terms = Irreps.parse(hidden).terms
        num_scalars = sum(
            mul
            for mul, ir in hidden_terms
            if ir.degree == 0 and ir.parity == 1 and ir in out_irreps
        )
        gated = "+".join(
            f"{mul}x{ir}"
            for mul, ir in hidden_terms
            if ir.degree > 0 and ir in out_irreps
        )
        num_gates = sum(mul for mul, _ in Irreps.parse(gated).terms) if gated else 0
        self.activation = _GatedActivation(num_scalars, gated)
        gate_in = f"{num_scalars + num_gates}x0e" + (f"+{gated}" if gated else "")
        middle = f"{num_scalars}x0e" + (f"+{gated}" if gated else "")
        self.first = backend.make_linear(
            LinearDescriptor(
                irreps_in=irreps_in, irreps_out=gate_in, precision=precision
            )
        )
        self.middle = backend.make_linear(
            LinearDescriptor(
                irreps_in=middle, irreps_out=gate_in, has_bias=True, precision=precision
            )
        )
        self.last = backend.make_linear(
            LinearDescriptor(
                irreps_in=middle,
                irreps_out=irreps_out,
                has_bias=True,
                precision=precision,
            )
        )

    def forward(self, features: Tensor) -> Tensor:
        middle = self.activation(self.first(features))
        middle = self.activation(self.middle(middle))
        return self.last(middle)


class ChargeUpdate(nn.Module):
    """One step of the density recursion: potential in, density increment out.

    The potential, the node features and the current density are each mapped
    into the node features' declaration and added. That sum is contracted with
    the features to one invariant per channel, joined with an element
    embedding, passed through an MLP, and used to scale the features, which a
    readout turns into the increment of both spin channels' multipoles plus
    two unnormalized Fukui weights.

    Args:
        backend: The kernel backend. Consulted at construction only.
        node_irreps: The node features' declaration, grouped.
        potential_irreps: The potential features' declaration, both spins.
        density_irreps: The density's declaration, both spins.
        num_elements: The element table's size.
        precision: The dtype every op is built at.
    """

    def __init__(
        self,
        backend,
        node_irreps: str,
        potential_irreps: str,
        density_irreps: str,
        num_elements: int,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        channels = _scalar_channels(node_irreps)
        invariants = f"{channels}x0e"

        def linear(irreps_in: str, irreps_out: str):
            return backend.make_linear(
                LinearDescriptor(
                    irreps_in=irreps_in, irreps_out=irreps_out, precision=precision
                )
            )

        self.from_potential = linear(potential_irreps, node_irreps)
        self.from_features = linear(node_irreps, node_irreps)
        self.from_density = linear(density_irreps, node_irreps)
        self.element_embedding = linear(f"{num_elements}x0e", invariants)
        self.products = InvariantProducts(node_irreps)
        self.mlp = LayerNormMLP([2 * channels, 64, 64, 64, channels])
        self.modulation = ScalarModulation(node_irreps, channels)
        density_max_l = max(ir.degree for _, ir in Irreps.parse(density_irreps).terms)
        readout_hidden = "+".join(
            f"32x{degree}{'e' if degree % 2 == 0 else 'o'}"
            for degree in range(density_max_l + 1)
        )
        self.readout = BiasReadout(
            backend,
            node_irreps,
            readout_hidden,
            f"{density_irreps}+2x0e",
            precision,
        )

    def forward(
        self,
        one_hot: Tensor,
        node_features: Tensor,
        potential: Tensor,
        density: Tensor,
    ) -> Tensor:
        """``[n, density + 2]``: the increment of both spins, then two Fukui
        weights.

        Args:
            one_hot: ``[n, num_elements]``.
            node_features: ``[n, dim]``, the layer-mixed features.
            potential: ``[n, potential_dim]``, both spins' projected potential.
            density: ``[n, density_dim]``, the current density, both spins.
        """
        mixed = (
            self.from_potential(potential)
            + self.from_features(node_features)
            + self.from_density(density)
        )
        invariants = torch.cat(
            [
                self.products(node_features, mixed),
                self.element_embedding(one_hot),
            ],
            dim=-1,
        )
        scaled = self.modulation(node_features, self.mlp(invariants))
        return self.readout(scaled)


class ElectronEnergyReadout(nn.Module):
    """The local energy of the converged density, one number per atom.

    The total density and the potential are each mapped into the node
    features' declaration with a bias, contracted with the features to one
    invariant per channel each, and an MLP reads the energy off both.

    Args:
        backend: The kernel backend. Consulted at construction only.
        node_irreps: The node features' declaration, grouped.
        potential_irreps: The potential features' declaration, both spins.
        density_irreps: The density's declaration, both spins.
        precision: The dtype every op is built at.
    """

    def __init__(
        self,
        backend,
        node_irreps: str,
        potential_irreps: str,
        density_irreps: str,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        channels = _scalar_channels(node_irreps)

        def biased(irreps_in: str):
            return backend.make_linear(
                LinearDescriptor(
                    irreps_in=irreps_in,
                    irreps_out=node_irreps,
                    has_bias=True,
                    precision=precision,
                )
            )

        self.from_density = biased(density_irreps)
        self.from_potential = biased(potential_irreps)
        self.density_products = InvariantProducts(node_irreps)
        self.potential_products = InvariantProducts(node_irreps)
        self.mlp = LayerNormMLP([2 * channels, 128, 128, 128, 1])

    def forward(
        self, node_features: Tensor, potential: Tensor, density: Tensor
    ) -> Tensor:
        """``[n]``, in eV.

        Args:
            node_features: ``[n, dim]``, the last layer's features.
            potential: ``[n, potential_dim]``, the last step's potential.
            density: ``[n, density_dim]``, the converged density plus the
                density it started from, both spins.
        """
        invariants = torch.cat(
            [
                self.density_products(node_features, self.from_density(density)),
                self.potential_products(node_features, self.from_potential(potential)),
            ],
            dim=-1,
        )
        return self.mlp(invariants).squeeze(-1)
