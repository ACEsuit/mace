"""The native real spherical harmonics against the legacy e3nn convention.

e3nn is not importable here (it is removed from the v1 stack), so the
convention is pinned two ways: committed decimal literals produced by
`e3nn.o3.SphericalHarmonics(3, normalize=True, normalization="component")` on
three fixed vectors, and the algebraic properties that single the convention
out (component norm, the `(x, y, z)` order of the `l = 1` block, the parity of
each block, rotation equivariance). The live comparison against the frozen
legacy stack is `tests/parity/test_spherical_harmonics_parity.py`.
"""

import math

import pytest
import torch
from conftest import assert_close, fp64_only
from mace_torch.backends.reference.spherical_harmonics import (
    E3NN_AXIS_ORDER,
    SphericalHarmonics,
    spherical_harmonics,
)

#: The three fixed input vectors: a generic direction, the polar axis of the
#: e3nn convention, and a non-unit vector.
REFERENCE_VECTORS = [[0.3, -1.2, 0.8], [0.0, 2.0, 0.0], [-2.5, 0.5, 1.0]]

#: `e3nn.o3.SphericalHarmonics(o3.Irreps.spherical_harmonics(3), normalize=True,
#: normalization="component")` on REFERENCE_VECTORS, e3nn 0.4.4, fp64.
REFERENCE_LMAX = 3
REFERENCE_VALUES = [
    [
        1.0, 0.35273781075132915, -1.4109512430053166,
        0.9406341620035445, 0.4283483885206359, -0.6425225827809539,
        1.1077295280240893, -1.7133935540825436, 0.4908158618465621,
        0.35922931682216425, -0.9232037003151735, 0.7648289051862515,
        -0.3426571199957823, 2.0395437471633375, -1.0578375732778034,
        0.19368283748517423,
    ],
    [
        1.0, 0.0, 1.7320508075688772,
        0.0, 0.0, 0.0,
        2.23606797749979, 0.0, 0.0,
        0.0, 0.0, 0.0,
        2.6457513110645907, 0.0, 0.0,
        0.0,
    ],
    [
        1.0, -1.5811388300841898, 0.3162277660168379,
        0.6324555320336758, -1.2909944487358056, -0.6454972243679028,
        -1.0062305898749055, 0.25819888974716104, -1.355544171172596,
        0.8274095004781382, -0.6236095644623235, 1.2325166214790868,
        -0.6843150130145012, -0.4930066485916346, -0.6547900426854397,
        -1.8075715241214705,
    ],
]  # fmt: skip


def _block(harmonics: torch.Tensor, degree: int) -> torch.Tensor:
    return harmonics[..., degree * degree : (degree + 1) ** 2]


def test_reference_values_are_the_e3nn_convention():
    """The literals themselves obey the convention they claim: the l=1 block is
    sqrt(3) times the unit vector, and each block has squared norm 2l+1."""
    for vector, values in zip(REFERENCE_VECTORS, REFERENCE_VALUES, strict=True):
        unit = torch.tensor(vector, dtype=torch.float64)
        unit = unit / unit.norm()
        values_tensor = torch.tensor(values, dtype=torch.float64)
        assert_close(_block(values_tensor, 1), math.sqrt(3.0) * unit, "l=1 block")
        for degree in range(REFERENCE_LMAX + 1):
            assert_close(
                _block(values_tensor, degree).norm() ** 2,
                2 * degree + 1,
                f"norm l={degree}",
            )


def test_native_matches_the_committed_e3nn_values():
    vectors = torch.tensor(REFERENCE_VECTORS)
    assert_close(
        spherical_harmonics(vectors, REFERENCE_LMAX),
        REFERENCE_VALUES,
        "vs e3nn literals",
    )
    assert_close(
        SphericalHarmonics(REFERENCE_LMAX)(vectors),
        REFERENCE_VALUES,
        "module vs literals",
    )


def test_lower_lmax_is_a_prefix_of_higher_lmax():
    vectors = torch.tensor(REFERENCE_VECTORS)
    full = spherical_harmonics(vectors, 6)
    for lmax in range(7):
        assert torch.equal(
            spherical_harmonics(vectors, lmax), full[:, : (lmax + 1) ** 2]
        )


@pytest.mark.parametrize("lmax", [0, 1, 2, 3, 5, 8])
def test_component_normalisation_and_shape(lmax):
    torch.manual_seed(0)
    vectors = torch.randn(64, 3) * 2.0
    harmonics = spherical_harmonics(vectors, lmax)
    assert harmonics.shape == (64, (lmax + 1) ** 2)
    assert SphericalHarmonics(lmax).output_dim == (lmax + 1) ** 2
    for degree in range(lmax + 1):
        assert_close(
            _block(harmonics, degree).norm(dim=-1) ** 2,
            torch.full((64,), 2.0 * degree + 1.0),
            f"||Y^{degree}||^2 = 2l+1",
        )


def test_the_l1_block_is_the_unit_vector_in_xyz_order():
    """This is the e3nn convention in one line: y is the polar axis, and the
    l=1 block reads (x, y, z), not (y, z, x)."""
    torch.manual_seed(1)
    vectors = torch.randn(16, 3)
    unit = torch.nn.functional.normalize(vectors, dim=-1)
    assert_close(
        _block(spherical_harmonics(vectors, 1), 1), math.sqrt(3.0) * unit, "l=1"
    )


def test_the_axis_order_is_the_permutation_develop_uses_for_sphericart():
    """(x, y, z) -> (z, x, y): the polar axis of the textbook harmonics is y."""
    assert E3NN_AXIS_ORDER == (2, 0, 1)


def test_input_is_normalised_so_scale_does_not_matter():
    torch.manual_seed(2)
    vectors = torch.randn(32, 3)
    scales = torch.rand(32, 1) * 5 + 0.1
    assert_close(
        spherical_harmonics(vectors * scales, 4),
        spherical_harmonics(vectors, 4),
        "scale",
    )


def test_without_normalisation_the_result_is_the_solid_harmonic():
    """`normalize=False` returns r^l Y^l of the direction: homogeneous of degree l."""
    torch.manual_seed(3)
    vectors = torch.randn(32, 3) * 2.0
    radius = vectors.norm(dim=-1, keepdim=True)
    solid = spherical_harmonics(vectors, 4, normalize=False)
    spherical = spherical_harmonics(vectors, 4)
    for degree in range(5):
        assert_close(
            _block(solid, degree),
            radius**degree * _block(spherical, degree),
            f"r^l l={degree}",
        )


def test_a_zero_vector_maps_to_the_scalar_channel_alone():
    """e3nn forwards zeros (not nan) through the normalisation; so does this."""
    harmonics = spherical_harmonics(torch.zeros(2, 3), 3)
    expected = torch.zeros(2, 16)
    expected[:, 0] = 1.0
    assert torch.equal(harmonics, expected)
    assert torch.isfinite(harmonics).all()


def test_blocks_have_parity_minus_one_to_the_l():
    torch.manual_seed(4)
    vectors = torch.randn(16, 3)
    forward = spherical_harmonics(vectors, 5)
    inverted = spherical_harmonics(-vectors, 5)
    for degree in range(6):
        assert_close(
            _block(inverted, degree),
            (-1.0) ** degree * _block(forward, degree),
            f"P l={degree}",
        )


def test_rotation_equivariance_of_each_block():
    """Rotating the input by R rotates every l block by the same matrix D^l(R):
    checked through invariants, without importing a Wigner-D implementation.
    For a rotation R, the block norms are unchanged and the Gram matrix between
    two rotated directions equals the Gram matrix between the originals."""
    torch.manual_seed(5)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(rotation) < 0:
        rotation = -rotation
    first, second = torch.randn(8, 3), torch.randn(8, 3)
    for degree in range(6):
        original = (
            _block(spherical_harmonics(first, 5), degree)
            * _block(spherical_harmonics(second, 5), degree)
        ).sum(-1)
        rotated = (
            _block(spherical_harmonics(first @ rotation.T, 5), degree)
            * _block(spherical_harmonics(second @ rotation.T, 5), degree)
        ).sum(-1)
        assert_close(rotated, original, f"Gram invariance l={degree}")


@fp64_only
@pytest.mark.parametrize("lmax", [1, 3, 5])
def test_gradcheck_and_gradgradcheck(lmax):
    """Force and stress training differentiate twice through the harmonics."""
    torch.manual_seed(6)
    vectors = (torch.randn(5, 3) * 1.5).requires_grad_(True)

    def function(vectors_):
        return spherical_harmonics(vectors_, lmax)

    assert torch.autograd.gradcheck(function, (vectors,))
    assert torch.autograd.gradgradcheck(function, (vectors,))


def test_dtype_follows_the_input():
    for dtype in (torch.float32, torch.float64):
        out = spherical_harmonics(torch.randn(4, 3, dtype=dtype), 3)
        assert out.dtype == dtype


def test_bad_inputs_are_errors_naming_the_problem():
    with pytest.raises(ValueError, match="lmax"):
        spherical_harmonics(torch.zeros(1, 3), -1)
    with pytest.raises(ValueError, match="3 Cartesian components"):
        spherical_harmonics(torch.zeros(1, 4), 2)
    with pytest.raises(ValueError, match="lmax"):
        SphericalHarmonics(-2)
