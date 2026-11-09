"""Oxidation embeddings and direct local polarizability.

Architectures follow ACEsuit/mace-scf's bonded_blocks and
LinearPolarizabilityReadoutBlock (MIT), revision
784a510339aacc4412b61b6926ef1a92bca387c0. Tensor algebra uses V1's canonical
operators and CG tables. Formal charge is an input in e, not a fitted target.
"""

import math

import torch
from mace_core.clebsch_gordan import symmetric_matrix_basis
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.real_basis import wigner_3j_real
from mace_core.kernels.specs import FullyConnectedTPSpec, LinearSpec
from torch import Tensor, nn

from mace_torch.backends import layout
from mace_torch.backends.radial import ChebyshevBasis


class OxidationEmbedding(nn.Module):
    """Eight Chebyshev modes of formal charge → [atoms, scalar channels].

    The configured range maps to [-1,1]. Values outside it extrapolate through
    the same polynomials; clamping would change the trained function.
    """

    def __init__(self, backend, channels, charge_range, precision):
        super().__init__()
        self.lower, upper = charge_range
        self.factor = 2.0 / (upper - self.lower)
        self.basis = ChebyshevBasis(8)
        self.linear = backend.make_linear(
            LinearSpec(
                irreps_in="8x0e",
                irreps_out=f"{channels}x0e",
                precision=precision,
                has_bias=True,
            )
        )

    def forward(self, formal: Tensor) -> Tensor:
        return self.linear(self.basis((formal[:, None] - self.lower) * self.factor - 1))


class OxidationGraphEmbedding(nn.Module):
    """Apply the LSC oxidation mixer after the declared scalar embeddings."""

    def __init__(self, backend, config, atomic_numbers, precision, previous=None):
        super().__init__()
        options = config.local_sources
        self.previous = previous
        self.input_name = options.formal_charge_input
        self.register_buffer(
            "formal_charges",
            torch.tensor(
                [(options.formal_charges or {}).get(z, 0.0) for z in atomic_numbers],
                dtype=getattr(torch, precision),
            ),
            persistent=False,
        )
        channels = config.num_channels
        self.oxidation = OxidationEmbedding(
            backend, channels, options.oxidation_range, precision
        )
        self.product = (
            backend.make_fully_connected_tp(
                FullyConnectedTPSpec(
                    irreps_in1=f"{channels}x0e",
                    irreps_in2=f"{channels}x0e",
                    irreps_out=f"{channels}x0e",
                    precision=precision,
                )
            )
            if options.oxidation_mixer == "product"
            else None
        )

    def forward(self, graph, features):
        if self.previous is not None:
            features = self.previous(graph, features)
        formal = (
            graph[self.input_name].reshape(-1)
            if self.input_name
            else graph["node_attrs"] @ self.formal_charges
        ).to(features.dtype)
        embedded = self.oxidation(formal)
        return (
            features + embedded
            if self.product is None
            else self.product(features, embedded)
        )


class LocalPolarizability(nn.Module):
    """A symmetric [atoms,3,3] tensor in Å³, independent of the energy.

    Linear skip plus per-channel self-products after an equivariant linear.
    Only degrees 0 and 2 can contribute to the symmetric tensor. No cross-channel
    products or field-dependent energy terms are introduced. Input features use
    the charge-aware backbone's (y, z, x) edge convention; outputs are Cartesian.
    """

    def __init__(self, backend, irreps, precision):
        super().__init__()
        self.layout = layout.layout_of(backend)
        self.terms = layout.terms(irreps)
        self.linear = backend.make_linear(
            LinearSpec(
                irreps_in=irreps,
                irreps_out=irreps,
                precision=precision,
            )
        )
        paths, terms = [], []
        for index, (channels, irrep) in enumerate(Irreps.parse(irreps)):
            for degree in (0, 2):
                if degree > 2 * irrep.degree:
                    continue
                paths.append((index, degree))
                terms.append(f"{channels}x{degree}e")
                self.register_buffer(
                    f"cg_{len(paths) - 1}",
                    torch.tensor(
                        wigner_3j_real(irrep.degree, irrep.degree, degree).copy(),
                        dtype=getattr(torch, precision),
                    )
                    * math.sqrt(2 * degree + 1),
                    persistent=False,
                )
        self.paths = tuple(paths)
        self.products_irreps = "+".join(terms)
        self.readout = backend.make_linear(
            LinearSpec(
                irreps_in=self.products_irreps,
                irreps_out="0e+2e",
                precision=precision,
            )
        )
        self.skip = backend.make_linear(
            LinearSpec(
                irreps_in=irreps,
                irreps_out="0e+2e",
                precision=precision,
            )
        )
        basis = torch.tensor(symmetric_matrix_basis(), dtype=getattr(torch, precision))
        self.register_buffer(
            "cartesian_basis", basis[:, [2, 0, 1]][:, :, [2, 0, 1]], persistent=False
        )

    def forward(self, features: Tensor) -> Tensor:
        blocks = self.layout.blocks(self.linear(features), self.terms)
        products = [
            torch.einsum(
                "nui,nuj,ijk->nuk",
                blocks[index],
                blocks[index],
                getattr(self, f"cg_{path}"),
            )
            for path, (index, _) in enumerate(self.paths)
        ]
        spherical = self.layout.to_canonical(
            self.readout(self.layout.join(products)) + self.skip(features),
            ((1, 1), (1, 5)),
        )
        return torch.einsum("nc,cij->nij", spherical, self.cartesian_basis)
