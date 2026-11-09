"""Learned affine field response for FixedPointSCF.

Inspired by OneBodyVariableUpdate and its energy readouts in ACEsuit/mace-scf
(MIT), commit 784a510339aacc4412b61b6926ef1a92bca387c0. Node features are fixed
through the solve; the response is affine in the projected potential.
"""

import math

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.specs import FullyConnectedTPSpec, LinearSpec
from torch import Tensor, nn

from mace_torch.backends import layout
from mace_torch.nn.field import InvariantProducts, ScalarModulation


class ScalarBilinear(nn.Module):
    """Couple matching irreps to even scalars, ``[n,left/right] -> [n,out]``.

    Weights are path-major ``[out,left_channel,right_channel]``. The real
    Clebsch-Gordan scalar contraction is the dot product divided by sqrt(2l+1),
    with component normalization across all matching channel pairs.
    """

    def __init__(self, left: str, right: str, channels: int, feature_layout):
        super().__init__()
        self.layout = feature_layout
        self.left_terms, self.right_terms = layout.terms(left), layout.terms(right)
        self.paths = [
            (i, j, a, b, ir.dimension)
            for i, (a, ir) in enumerate(Irreps.parse(left))
            for j, (b, jr) in enumerate(Irreps.parse(right))
            if ir == jr
        ]
        self.channels = channels
        self.fan_in = sum(a * b for _, _, a, b, _ in self.paths)
        self.weight = nn.Parameter(torch.zeros(channels * self.fan_in))

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        first, second = (
            self.layout.blocks(left, self.left_terms),
            self.layout.blocks(right, self.right_terms),
        )
        out = left.new_zeros((left.shape[0], self.channels))
        offset = 0
        for i, j, a, b, dimension in self.paths:
            width = self.channels * a * b
            weights = self.weight[offset : offset + width].reshape(self.channels, a, b)
            out = out + torch.einsum(
                "nud,nvd,kuv->nk", first[i], second[j], weights
            ) / math.sqrt(dimension * self.fan_in)
            offset += width
        return out

    def to_canonical(self):
        return {"weight": self.weight.detach()}

    def load_canonical(self, state):
        with torch.no_grad():
            self.weight.copy_(state["weight"])

    def initialize_weights(self, seed):
        with torch.no_grad():
            self.weight.copy_(
                torch.randn(
                    self.weight.shape,
                    generator=torch.Generator().manual_seed(seed),
                    dtype=torch.float64,
                ).to(self.weight)
            )


class AffineFieldUpdate(nn.Module):
    """Element-conditioned potential times node features -> density multipoles."""

    def __init__(
        self,
        backend,
        node_irreps,
        potential_irreps,
        density_irreps,
        num_elements,
        precision,
        biased=True,
    ):
        super().__init__()
        self.layout = layout.layout_of(backend)
        self.potential_terms, self.density_terms = (
            layout.terms(potential_irreps),
            layout.terms(density_irreps),
        )
        attributes = f"{num_elements}x0e"
        self.potential = backend.make_fully_connected_tp(
            FullyConnectedTPSpec(
                irreps_in1=potential_irreps,
                irreps_in2=attributes,
                irreps_out=node_irreps,
                precision=precision,
            )
        )
        self.bias = (
            backend.make_linear(
                LinearSpec(
                    irreps_in=node_irreps, irreps_out=node_irreps, precision=precision
                )
            )
            if biased
            else None
        )
        channels = sum(mul for mul, ir in Irreps.parse(node_irreps) if ir.degree == 0)
        self.products = InvariantProducts(node_irreps, feature_layout=self.layout)
        self.modulation = ScalarModulation(
            node_irreps, channels, feature_layout=self.layout
        )
        self.readout = backend.make_fully_connected_tp(
            FullyConnectedTPSpec(
                irreps_in1=node_irreps,
                irreps_in2=attributes,
                irreps_out=density_irreps,
                precision=precision,
            )
        )

    def _read(self, node_features, one_hot, embedded):
        invariants = self.products(node_features, embedded)
        output = self.readout(self.modulation(node_features, invariants), one_hot)
        return self.layout.to_canonical(output, self.density_terms)

    def response(self, node_features, one_hot, potential):
        embedded = self.potential(
            self.layout.from_canonical(potential, self.potential_terms), one_hot
        )
        return self._read(node_features, one_hot, embedded)

    def forward(self, node_features, one_hot, potential):
        embedded = self.potential(
            self.layout.from_canonical(potential, self.potential_terms), one_hot
        )
        if self.bias is not None:
            embedded = embedded + self.bias(node_features)
        return self._read(node_features, one_hot, embedded)


class FixedPointEnergy(nn.Module):
    """Local electron energy, in eV per atom, at the solved density."""

    def __init__(
        self, backend, kind, node_irreps, potential_irreps, density_irreps, precision
    ):
        super().__init__()
        self.kind = kind
        self.layout = layout.layout_of(backend)
        self.potential_terms, self.density_terms = (
            layout.terms(potential_irreps),
            layout.terms(density_irreps),
        )
        if kind == "quadratic_field":
            hidden = "32x0e+32x1o"
            self.charge = backend.make_linear(
                LinearSpec(
                    irreps_in=density_irreps, irreps_out=hidden, precision=precision
                )
            )
            self.field = backend.make_linear(
                LinearSpec(
                    irreps_in=potential_irreps, irreps_out=hidden, precision=precision
                )
            )
            self.products = ScalarBilinear(hidden, hidden, 32, self.layout)
            self.down = ScalarBilinear("32x0e", node_irreps, 1, self.layout)
        else:
            self.down = ScalarBilinear(density_irreps, node_irreps, 1, self.layout)

    def forward(self, node_features, field, local, density):
        if self.kind == "quadratic_field":
            charge = self.charge(
                self.layout.from_canonical(density - local, self.density_terms)
            )
            potential = self.field(
                self.layout.from_canonical(0.01 * field, self.potential_terms)
            )
            return self.down(self.products(charge, potential), node_features).squeeze(
                -1
            )
        return self.down(
            self.layout.from_canonical(density, self.density_terms), node_features
        ).squeeze(-1)
