"""The native spherical harmonics against the legacy e3nn call, in one process.

Legacy builds `o3.SphericalHarmonics(sh_irreps, normalize=True,
normalization="component")` in every model (`mace/modules/models.py:163`); a
silent mismatch in ordering or normalisation here would poison every downstream
parity test, so the comparison runs up to well beyond any model's `lmax`, on
unit and non-unit vectors, and on the derivatives force training takes.
"""

import pytest
import torch
from e3nn import o3
from mace_torch.backends.reference.spherical_harmonics import spherical_harmonics

from tests.golden.harness import tolerance

CLOSED_FORM = tolerance("closed_form_fp64")


def assert_parity(v1: torch.Tensor, legacy: torch.Tensor, what: str) -> None:
    torch.testing.assert_close(
        v1,
        legacy,
        atol=CLOSED_FORM.atol,
        rtol=CLOSED_FORM.rtol,
        msg=lambda m: f"{what}: {m}",
    )


def legacy_spherical_harmonics(lmax: int) -> torch.nn.Module:
    return o3.SphericalHarmonics(
        o3.Irreps.spherical_harmonics(lmax), normalize=True, normalization="component"
    )


def _vectors(seed: int, count: int = 200) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, 3, generator=generator) * 2.0


@pytest.mark.parametrize("lmax", list(range(0, 8)))
def test_values_match_on_non_unit_vectors(lmax, fp64):
    vectors = _vectors(seed=lmax)
    assert_parity(
        spherical_harmonics(vectors, lmax),
        legacy_spherical_harmonics(lmax)(vectors),
        f"lmax={lmax}",
    )


@pytest.mark.parametrize("lmax", [1, 3, 5])
def test_values_match_on_unit_vectors(lmax, fp64):
    vectors = torch.nn.functional.normalize(_vectors(seed=10 + lmax), dim=-1)
    assert_parity(
        spherical_harmonics(vectors, lmax),
        legacy_spherical_harmonics(lmax)(vectors),
        f"lmax={lmax}",
    )


@pytest.mark.parametrize("lmax", [1, 3])
def test_first_and_second_derivatives_match(lmax, fp64):
    """The path force training takes: grad, then grad of the grad."""
    vectors = _vectors(seed=20 + lmax, count=16).requires_grad_(True)
    weights = _vectors(seed=30 + lmax, count=16)[:, :1].repeat(1, (lmax + 1) ** 2)
    cotangent = torch.randn(16, 3, generator=torch.Generator().manual_seed(40 + lmax))

    def derivatives(function):
        (gradient,) = torch.autograd.grad(
            (function(vectors) * weights).sum(), vectors, create_graph=True
        )
        (second,) = torch.autograd.grad((gradient * cotangent).sum(), vectors)
        return gradient, second

    v1_gradient, v1_second = derivatives(lambda v: spherical_harmonics(v, lmax))
    legacy_gradient, legacy_second = derivatives(legacy_spherical_harmonics(lmax))
    assert_parity(v1_gradient, legacy_gradient, f"first derivative lmax={lmax}")
    assert_parity(v1_second, legacy_second, f"second derivative lmax={lmax}")


def test_zero_vector_matches(fp64):
    zero = torch.zeros(3, 3)
    assert torch.equal(
        spherical_harmonics(zero, 3), legacy_spherical_harmonics(3)(zero)
    )
