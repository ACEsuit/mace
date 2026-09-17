"""The v1 radial stack against the frozen legacy one, in one process.

Legacy is the oracle for the numbers, never for the structure: the two sides
take different inputs (legacy reads a one-hot plus a species table, v1 reads
per-node atomic numbers), and what is asserted is the output alone.
"""

import pytest
import torch
from mace_torch.nn.embedding import LinearNodeEmbeddingBlock, RadialEmbeddingBlock
from mace_torch.nn.radial import ChebyshevBasis, ZBLBasis

from mace.modules.blocks import LinearNodeEmbeddingBlock as LegacyNodeEmbedding
from mace.modules.blocks import RadialEmbeddingBlock as LegacyRadialEmbeddingBlock
from mace.modules.extensions import ChebyshevBasisGeneral as LegacyChebyshevBasisGeneral
from mace.modules.radial import ChebychevBasis as LegacyChebychevBasis
from mace.modules.radial import ZBLBasis as LegacyZBLBasis
from tests.golden.harness import tolerance

#: Pure math in one process at fp64: the two sides differ only in the order of
#: the arithmetic, and the row exists so no test invents a number.
CLOSED_FORM = tolerance("closed_form_fp64")

R_MAX = 3.0
SPECIES = torch.tensor([1, 6, 8])


def assert_parity(v1: torch.Tensor, legacy: torch.Tensor, what: str) -> None:
    torch.testing.assert_close(
        v1,
        legacy,
        atol=CLOSED_FORM.atol,
        rtol=CLOSED_FORM.rtol,
        msg=lambda m: f"{what}: {m}",
    )


def _graph(seed: int = 0, num_nodes: int = 12, num_edges: int = 40):
    """A random graph over three species, edge lengths spanning the cutoff and
    beyond so that the zero-beyond-r_max branch is exercised too."""
    generator = torch.Generator().manual_seed(seed)
    element_index = torch.randint(0, len(SPECIES), (num_nodes,), generator=generator)
    node_attrs = torch.eye(len(SPECIES))[element_index]
    node_atomic_numbers = SPECIES[element_index]
    edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=generator)
    lengths = torch.rand(num_edges, 1, generator=generator) * (1.2 * R_MAX) + 0.3
    return lengths, node_attrs, node_atomic_numbers, edge_index


LEGACY_TRANSFORM_NAME = {"none": "None", "agnesi": "Agnesi", "soft": "Soft"}


@pytest.mark.parametrize("radial_basis", ["bessel", "gaussian", "chebyshev"])
@pytest.mark.parametrize("distance_transform", ["none", "agnesi", "soft"])
@pytest.mark.parametrize("apply_cutoff", [True, False])
def test_radial_embedding_block_parity(
    radial_basis, distance_transform, apply_cutoff, fp64
):
    """`apply_cutoff` here is the *legacy* flag: it selects which of legacy's
    two return shapes the two v1 tensors are compared against."""
    lengths, node_attrs, node_atomic_numbers, edge_index = _graph()
    legacy = LegacyRadialEmbeddingBlock(
        r_max=R_MAX,
        num_bessel=6,
        num_polynomial_cutoff=5,
        radial_type=radial_basis,
        distance_transform=LEGACY_TRANSFORM_NAME[distance_transform],
        apply_cutoff=apply_cutoff,
    )
    v1 = RadialEmbeddingBlock(
        r_max=R_MAX,
        num_basis=6,
        num_polynomial_cutoff=5,
        radial_basis=radial_basis,
        distance_transform=distance_transform,
    )
    legacy_radial, legacy_cutoff = legacy(lengths, node_attrs, edge_index, SPECIES)
    v1_radial, v1_cutoff = v1(lengths, node_atomic_numbers, edge_index)
    # v1 always returns the bare basis and the envelope; legacy pre-multiplies
    # them under --apply_cutoff and returns the envelope only when it does not.
    if apply_cutoff:
        assert legacy_cutoff is None
        assert_parity(v1_radial * v1_cutoff, legacy_radial, "basis x cutoff")
    else:
        assert_parity(v1_radial, legacy_radial, "bare basis")
        assert_parity(v1_cutoff, legacy_cutoff, "cutoff")


def test_zbl_parity(fp64):
    lengths, node_attrs, node_atomic_numbers, edge_index = _graph(seed=1)
    lengths = lengths * 0.5  # inside the covalent radii, where the term is live
    legacy = LegacyZBLBasis(p=6)(lengths, node_attrs, edge_index, SPECIES)
    v1 = ZBLBasis(polynomial_order=6)(lengths, node_atomic_numbers, edge_index)
    assert (v1 != 0).any(), "the test graph has no edge inside the repulsion radius"
    assert_parity(v1, legacy, "ZBL per-node energies")


@pytest.mark.parametrize("include_constant", [True, False])
def test_chebyshev_basis_parity_at_the_magnetic_configuration(include_constant, fp64):
    """Legacy `ChebyshevBasisGeneral` at `r_max=0.0`, the configuration the
    magnetic family constructs; the input is a transformed magnetic-moment
    length on [-1, 1]. v1 has one Chebyshev class, so the switch is tested on it."""
    generator = torch.Generator().manual_seed(2)
    x = torch.rand(20, 1, generator=generator) * 2.0 - 1.0
    legacy = LegacyChebyshevBasisGeneral(
        r_max=0.0, num_basis=7, include_constant=include_constant
    )
    v1 = ChebyshevBasis(num_basis=7, include_constant=include_constant)
    assert_parity(
        v1(x), legacy(x), f"ChebyshevBasis include_constant={include_constant}"
    )


def test_chebyshev_basis_parity_with_the_radial_type_class(fp64):
    """Legacy `ChebychevBasis`, the `--radial_type chebyshev` class, is the same
    function as the general one without the constant; v1 keeps a single class.
    Distances beyond 1 Angstrom exercise the divergent branch legacy evaluates."""
    lengths, _, _, _ = _graph(seed=3)
    legacy = LegacyChebychevBasis(r_max=R_MAX, num_basis=6)
    v1 = ChebyshevBasis(num_basis=6)
    assert_parity(
        v1(lengths), legacy(lengths), "ChebyshevBasis vs legacy ChebychevBasis"
    )


def test_linear_node_embedding_parity_with_the_legacy_weight_rescaled(fp64):
    """Legacy's equivariant linear layer on `0e` inputs is `x @ W / sqrt(fan_in)`;
    the v1 weight is that product, so a converter divides by `sqrt(num_elements)`."""
    from e3nn import o3

    num_elements, num_channels = 3, 8
    legacy = LegacyNodeEmbedding(
        irreps_in=o3.Irreps(f"{num_elements}x0e"),
        irreps_out=o3.Irreps(f"{num_channels}x0e"),
    )
    v1 = LinearNodeEmbeddingBlock(num_elements=num_elements, num_channels=num_channels)
    legacy_weight = legacy.linear.weight.detach().reshape(num_elements, num_channels)
    with torch.no_grad():
        v1.weight.copy_(legacy_weight / num_elements**0.5)
    node_attrs = torch.eye(num_elements)[[0, 2, 1, 2]]
    assert_parity(v1(node_attrs), legacy(node_attrs), "node embedding")
