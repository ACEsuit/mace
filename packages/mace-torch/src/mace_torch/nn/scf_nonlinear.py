"""Nonlinear response and electron readouts for the reference SCF solver.

Architectures inspired by the field blocks in ACEsuit/mace-scf (MIT), revision
784a510339aacc4412b61b6926ef1a92bca387c0. Algebra uses V1's canonical irreps,
reduced product basis and existing interaction blocks, without an e3nn runtime.
"""

import math
from typing import NamedTuple

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.real_basis import wigner_3j_real
from mace_core.kernels.descriptors import (
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    RadialMLPDescriptor,
)
from torch import Tensor, nn

from mace_torch.backends import layout
from mace_torch.nn.field import InvariantProducts, ScalarModulation
from mace_torch.nn.fixed_point_scf import ScalarBilinear
from mace_torch.nn.interaction import ResidualInteractionBlock, _radial_network
from mace_torch.nn.product_basis import EquivariantProductBasisBlock


def linear(backend, source, target, precision, bias=False):
    return backend.make_linear(
        LinearDescriptor(
            irreps_in=source, irreps_out=target, precision=precision, has_bias=bias
        )
    )


class InvariantActivation(nn.Module):
    """Scalar activation; the historical 'shallow' variant is just linear."""

    def __init__(self, backend, channels, kind, precision):
        super().__init__()
        self.kind = kind
        self.mlp = (
            None
            if kind == "none"
            else _radial_network(
                backend,
                RadialMLPDescriptor(
                    channels=(channels, channels)
                    if kind == "shallow"
                    else (channels, 64, 64, channels),
                    kind="layer_norm_silu" if kind == "skip_mlp" else "normalized_silu",
                    precision=precision,
                ),
            )
        )

    def forward(self, values):
        if self.mlp is None:
            return values
        result = self.mlp(values)
        return values + result if self.kind == "skip_mlp" else result


def norm_tanh(features: Tensor, terms: layout.Terms) -> Tensor:
    """Saturate each irrep's norm without rotating its components.

    Dimensionless input [...,components]. Scalars use tanh; vectors use
    v*tanh(|v|)/|v|. A series in |v|² avoids undefined derivatives at zero,
    including the second derivatives required for force training.
    """
    pieces = []
    offset = 0
    for multiplicity, dimension in terms:
        block = features[..., offset : offset + multiplicity * dimension]
        offset += multiplicity * dimension
        if dimension == 1:
            pieces.append(torch.tanh(block))
        else:
            block = block.reshape(*block.shape[:-1], multiplicity, dimension)
            squared = block.square().sum(-1, keepdim=True)
            small = squared < 1e-4
            radius = torch.sqrt(torch.where(small, torch.ones_like(squared), squared))
            series_input = torch.where(small, squared, torch.zeros_like(squared))
            series = (
                1
                - series_input / 3
                + 2 * series_input.square() / 15
                - 17 * series_input.pow(3) / 315
            )
            factor = torch.where(small, series, torch.tanh(radius) / radius)
            pieces.append((block * factor).flatten(-2))
    return torch.cat(pieces, dim=-1)


class PotentialEmbedding(nn.Module):
    """Projected potential [atoms,components] to canonical node features."""

    widths: Tensor

    def __init__(self, backend, source, target, elements, kind, precision):
        super().__init__()
        self.kind = kind
        self.layout = layout.layout_of(backend)
        self.terms = layout.terms(source)
        if kind in ("tanh", "norm_tanh"):
            # Five complete copies, not five copies of each irrep term.
            self.potential = linear(
                backend, "+".join([source] * 5), target, precision, True
            )
            self.register_buffer(
                "widths",
                torch.logspace(0, 1, 5, dtype=getattr(torch, precision)),
                persistent=False,
            )
        else:
            self.potential = backend.make_fully_connected_tp(
                FullyConnectedTPDescriptor(
                    irreps_in1=source,
                    irreps_in2=f"{elements}x0e",
                    irreps_out=target,
                    precision=precision,
                )
            )
        self.bias = (
            linear(backend, target, target, precision) if kind != "linear" else None
        )

    def forward(self, potential, nodes, attributes):
        if self.kind in ("tanh", "norm_tanh"):
            scaled = potential[:, None, :] / self.widths[None, :, None]
            if self.kind == "norm_tanh":
                expanded = norm_tanh(scaled, self.terms).flatten(1)
            else:
                expanded = torch.tanh(scaled).flatten(1)
            result = self.potential(
                self.layout.from_canonical(expanded, self.terms * 5)
            )
        else:
            result = self.potential(
                self.layout.from_canonical(potential, self.terms), attributes
            )
        return result if self.bias is None else result + self.bias(nodes)


class BilinearTensor(nn.Module):
    """Learned real CG products with explicit path-major effective weights.

    ``uvw`` stores [out,left,right], ``uvu`` stores [left,right]. Coefficients
    use unit-Frobenius CG tables. Path normalization is in the canonical weight,
    as in the SCF invariant products; conversion absorbs the source convention.
    Inputs/outputs have shape [atoms,irrep dimensions] in the selected layout.
    """

    def __init__(self, left, right, output, paths, mode, feature_layout):
        super().__init__()
        self.layout = feature_layout
        self.left_terms, self.right_terms, self.out_terms = map(
            layout.terms, (left, right, output)
        )
        self.paths = tuple(paths)
        self.mode = mode
        first, second, target = map(
            lambda v: tuple(Irreps.parse(v)), (left, right, output)
        )
        weights = []
        for index, (i, j, k) in enumerate(self.paths):
            a, ir = first[i]
            b, jr = second[j]
            c, kr = target[k]
            self.register_buffer(
                f"cg_{index}",
                torch.tensor(
                    wigner_3j_real(ir.degree, jr.degree, kr.degree).copy(),
                    dtype=torch.get_default_dtype(),
                ),
                persistent=False,
            )
            weights.append(
                nn.Parameter(torch.zeros((c, a, b) if mode == "uvw" else (a, b)))
            )
        self.weights = nn.ParameterList(weights)

    def forward(self, left, right):
        first = self.layout.blocks(left, self.left_terms)
        second = self.layout.blocks(right, self.right_terms)
        result = [
            left.new_zeros((left.shape[0], mul, dimension))
            for mul, dimension in self.out_terms
        ]
        for index, ((i, j, k), weight) in enumerate(
            zip(self.paths, self.weights, strict=True)
        ):
            cg = getattr(self, f"cg_{index}")
            if self.mode == "uvw":
                value = torch.einsum(
                    "nui,nvj,ijk,wuv->nwk", first[i], second[j], cg, weight
                )
            else:
                value = torch.einsum(
                    "nui,nvj,ijk,uv->nuk", first[i], second[j], cg, weight
                )
            result[k] = result[k] + value
        return self.layout.join(result)

    def to_canonical(self):
        return {f"path_{i}": w.detach() for i, w in enumerate(self.weights)}

    def load_canonical(self, state):
        with torch.no_grad():
            for i, w in enumerate(self.weights):
                w.copy_(state[f"path_{i}"])

    def initialize_weights(self, seed):
        with torch.no_grad():
            for i, weight in enumerate(self.weights):
                target = self.paths[i][2]
                fan = sum(
                    w.shape[-1] * (w.shape[-2] if self.mode == "uvw" else 1)
                    for path, w in zip(self.paths, self.weights, strict=True)
                    if path[2] == target
                )
                std = math.sqrt(self.out_terms[target][1] / fan)
                weight.copy_(
                    torch.randn(
                        weight.shape,
                        generator=torch.Generator().manual_seed(seed + i),
                        dtype=torch.float64,
                    ).to(weight)
                    * std
                )


class FeatureMixer(nn.Module):
    def __init__(self, backend, target, source, kind, precision):
        super().__init__()
        self.kind = kind
        self.embedding = (
            None if kind == "none" else linear(backend, source, target, precision, True)
        )
        if kind == "product":
            terms = tuple(Irreps.parse(target))
            paths = []
            for i, (_, ir) in enumerate(terms):
                for j, (_, jr) in enumerate(terms):
                    for k, (_, kr) in enumerate(terms):
                        if (
                            ir.parity * jr.parity == kr.parity
                            and abs(ir.degree - jr.degree)
                            <= kr.degree
                            <= ir.degree + jr.degree
                        ):
                            paths.append((i, j, k))
            paths.sort(
                key=lambda path: (terms[path[2]][1].degree, terms[path[2]][1].parity)
            )
            intermediate = "+".join(f"{terms[i][0]}x{terms[k][1]}" for i, j, k in paths)
            self.product = BilinearTensor(
                target,
                target,
                intermediate,
                [(i, j, n) for n, (i, j, k) in enumerate(paths)],
                "uvu",
                layout.layout_of(backend),
            )
            # Linear descriptors retain repeated terms; the converter folds the
            # reference's simplified input into this equivalent declaration.
            self.down = linear(backend, intermediate, target, precision)

    def forward(self, first, second):
        if self.embedding is None:
            return first
        embedded = self.embedding(second)
        return (
            first + embedded
            if self.kind == "additive"
            else self.down(self.product(first, embedded))
        )


class PreparedEdges(NamedTuple):
    attributes: Tensor
    radial: Tensor
    differentiated_twice: bool = True


class ManyBodyResponse(nn.Module):
    """Existing density-normalized interaction and reduced product basis."""

    def __init__(
        self,
        backend,
        nodes,
        elements,
        radial_size,
        precision,
        correlation=1,
        use_skip=True,
        mixer="additive",
        node_mixer="none",
        interaction="density",
        radial_hidden=(64, 64, 64),
    ):
        super().__init__()
        terms = tuple(Irreps.parse(nodes))
        channels = terms[0][0]
        single = "+".join(str(ir) for _, ir in terms)
        target = "0e+1o+2e"
        self.interaction = ResidualInteractionBlock(
            backend,
            single,
            target,
            target,
            single,
            radial_size,
            channels,
            elements,
            radial_hidden=radial_hidden,
            precision=precision,
            learned_density=True,
        )
        self.central = (
            linear(backend, nodes, nodes, precision)
            if interaction == "augmented"
            else None
        )
        self.mixer = FeatureMixer(
            backend, layout.expanded_irreps(target, channels), nodes, mixer, precision
        )
        self.node_mixer = FeatureMixer(
            backend,
            layout.expanded_irreps(target, channels),
            nodes,
            node_mixer,
            precision,
        )
        self.product = EquivariantProductBasisBlock(
            backend,
            target,
            single,
            correlation,
            elements,
            channels,
            precision=precision,
        )
        self.use_skip = use_skip
        if not use_skip:
            # A disabled residual must not leave trainable, unused parameters.
            self.interaction.skip = None

    def forward(
        self, nodes, embedded, attributes, edge_attributes, edge_radial, edge_index
    ):
        sender, receiver = edge_index
        count = nodes.shape[0]
        edges = PreparedEdges(edge_attributes, edge_radial)
        if self.use_skip:
            message, skip = self.interaction(
                embedded, edges, attributes, sender, receiver, count
            )
        else:
            message = self.interaction.body(embedded, edges, sender, receiver, count)
            skip = None
        if self.central is not None:
            body = self.interaction.body
            mapped = self.central(body.linear_up(embedded))
            weights = body.radial_mlp(edge_radial).reshape(
                -1, body.num_paths, body.num_features
            )
            extra = body.convolution(
                mapped, edge_attributes, weights, receiver, receiver, count
            )
            assert body.density_mlp is not None
            density = body.reduce(
                torch.tanh(body.density_mlp(edge_radial) ** 2), receiver, count
            )
            message = message + body.linear(extra) / (density + 1)
        message = self.node_mixer(self.mixer(message, embedded), nodes)
        return self.product(
            message, attributes.argmax(-1), skip if self.use_skip else None
        )


class NonlinearFieldUpdate(nn.Module):
    """Nonlinear local/many-body response; returns density in e Å**l."""

    def __init__(
        self,
        backend,
        nodes,
        potential,
        density,
        elements,
        radial_size,
        options,
        precision,
    ):
        super().__init__()
        self.kind = options.update
        self.layout = layout.layout_of(backend)
        self.density_terms = layout.terms(density)
        self.channels = sum(mul for mul, ir in Irreps.parse(nodes) if ir.degree == 0)
        self.embedding = PotentialEmbedding(
            backend, potential, nodes, elements, options.potential_embedding, precision
        )
        self.products = (
            None
            if self.kind == "scaled"
            else InvariantProducts(nodes, feature_layout=self.layout)
        )
        self.activation = InvariantActivation(
            backend, self.channels, options.update_nonlinearity, precision
        )
        self.modulation = ScalarModulation(
            nodes, self.channels, feature_layout=self.layout
        )
        self.skip = None
        if self.kind in ("scaled", "scaled_dot"):
            self.skip = BilinearTensor(
                nodes,
                nodes,
                nodes,
                ((0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)),
                "uvu",
                self.layout,
            )
        self.many_body = (
            ManyBodyResponse(
                backend,
                nodes,
                elements,
                radial_size,
                precision,
                options.update_correlation,
                options.update_use_skip,
                options.central_mixer,
                options.node_mixer,
                options.update_interaction,
                options.update_radial_hidden,
            )
            if self.kind == "many_body"
            else None
        )
        self.readout = backend.make_fully_connected_tp(
            FullyConnectedTPDescriptor(
                irreps_in1=nodes,
                irreps_in2=f"{elements}x0e",
                irreps_out=density,
                precision=precision,
            )
        )
        self.register_buffer(
            "scales",
            torch.tensor(
                options.update_scales or (1.0,) * elements,
                dtype=getattr(torch, precision),
            ),
            persistent=False,
        )

    def forward(
        self, nodes, attributes, field, edge_attributes, edge_radial, edge_index
    ):
        embedded = self.embedding(field, nodes, attributes)
        transformed = (
            embedded
            if self.many_body is None
            else self.many_body(
                nodes, embedded, attributes, edge_attributes, edge_radial, edge_index
            )
        )
        invariants = (
            transformed[:, : self.channels]
            if self.products is None
            else self.products(nodes, transformed)
        )
        features = self.modulation(nodes, self.activation(invariants))
        if self.skip is not None:
            features = features + self.skip(nodes, embedded)
        result = self.layout.to_canonical(
            self.readout(features, attributes), self.density_terms
        )
        return (
            result * (attributes @ self.scales)[:, None]
            if self.skip is not None
            else result
        )


class NonlinearElectronEnergy(nn.Module):
    """Post-SCF local electron energy [atoms], in eV."""

    def __init__(
        self, backend, kind, nodes, potential, density, elements, radial_size, precision
    ):
        super().__init__()
        self.kind = kind
        self.layout = layout.layout_of(backend)
        self.potential_terms, self.density_terms = (
            layout.terms(potential),
            layout.terms(density),
        )
        channels = sum(m for m, ir in Irreps.parse(nodes) if ir.degree == 0)
        if kind == "mlp_field":
            self.charge = linear(backend, density, nodes, precision, True)
            self.field = linear(backend, potential, nodes, precision, True)
            self.charge_products = InvariantProducts(nodes, feature_layout=self.layout)
            self.field_products = InvariantProducts(nodes, feature_layout=self.layout)
            self.mlp = _radial_network(
                backend,
                RadialMLPDescriptor(
                    channels=(channels, 128, 128, 128, 1),
                    kind="normalized_silu",
                    precision=precision,
                ),
            )
        elif kind == "quadratic_charges":
            hidden = "32x0e+32x1o"
            self.charge = linear(backend, density, hidden, precision, True)
            self.local = linear(backend, density, hidden, precision, True)
            self.products = BilinearTensor(
                hidden,
                hidden,
                hidden,
                ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)),
                "uvw",
                self.layout,
            )
            self.down = ScalarBilinear(hidden, nodes, 1, self.layout)
        else:
            self.charge = PotentialEmbedding(
                backend, density, nodes, elements, "biased_linear", precision
            )
            self.field = (
                PotentialEmbedding(
                    backend, potential, nodes, elements, "biased_linear", precision
                )
                if kind == "many_body_charges_field"
                else None
            )
            self.many_body = ManyBodyResponse(
                backend, nodes, elements, radial_size, precision, correlation=2
            )
            self.products = InvariantProducts(nodes, feature_layout=self.layout)
            self.activation = InvariantActivation(backend, channels, "mlp", precision)
            self.modulation = ScalarModulation(
                nodes, channels, feature_layout=self.layout
            )
            self.down = backend.make_fully_connected_tp(
                FullyConnectedTPDescriptor(
                    irreps_in1=nodes,
                    irreps_in2=f"{elements}x0e",
                    irreps_out="0e",
                    precision=precision,
                )
            )

    def forward(
        self,
        nodes,
        field,
        local,
        density,
        attributes,
        edge_attributes,
        edge_radial,
        edge_index,
    ):
        if self.kind == "mlp_field":
            assert self.field is not None
            charges = self.charge(
                self.layout.from_canonical(density + local, self.density_terms)
            )
            potential = self.field(
                self.layout.from_canonical(field, self.potential_terms)
            )
            return self.mlp(
                self.charge_products(nodes, charges)
                + self.field_products(nodes, potential)
            ).squeeze(-1)
        if self.kind == "quadratic_charges":
            charges = self.charge(
                self.layout.from_canonical(density, self.density_terms)
            ) + self.local(self.layout.from_canonical(local, self.density_terms))
            return self.down(self.products(charges, charges), nodes).squeeze(-1)
        embedded = self.charge(density + local, nodes, attributes)
        if self.field is not None:
            embedded = embedded + self.field(field, nodes, attributes)
        product = self.many_body(
            nodes, embedded, attributes, edge_attributes, edge_radial, edge_index
        )
        return self.down(
            self.modulation(nodes, self.activation(self.products(nodes, product))),
            attributes,
        ).squeeze(-1)
