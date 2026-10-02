"""The radial embedding sections of the model schema: what a config writes,
what it defaults to, and what it is refused.

How a tagged union loads, exports and reports errors is the config base's and
is tested there (`test_mace_core_config_kinds.py`); this file pins only what
these sections add: the defaults, the kind names and the size bounds."""

import pytest
from mace_core.config import (
    AgnesiTransformConfig,
    BaseConfig,
    BesselBasisConfig,
    ChebyshevBasisConfig,
    DistanceTransformConfig,
    GaussianBasisConfig,
    NoDistanceTransformConfig,
    PolynomialCutoffConfig,
    RadialBasisConfig,
    SoftTransformConfig,
)
from pydantic import ValidationError

#: A warning the test did not ask for is a failure.
pytestmark = pytest.mark.filterwarnings("error")


class RadialRoot(BaseConfig):
    """The three sections as a model config will hold them."""

    radial_basis: RadialBasisConfig = BesselBasisConfig()
    distance_transform: DistanceTransformConfig = NoDistanceTransformConfig()
    cutoff: PolynomialCutoffConfig = PolynomialCutoffConfig()


def error_locations(excinfo):
    return [".".join(map(str, error["loc"])) for error in excinfo.value.errors()]


def test_the_defaults_are_the_legacy_command_line_defaults():
    """`--radial_type bessel --num_radial_basis 8 --distance_transform None
    --num_cutoff_basis 5`. The flag sized every kind of basis, so each one
    defaults to 8; the Gaussian class default of 128 is not the CLI default."""
    config = RadialRoot.from_dict({})
    assert config.radial_basis == BesselBasisConfig(num_basis=8, trainable=False)
    assert config.distance_transform == NoDistanceTransformConfig()
    assert config.cutoff == PolynomialCutoffConfig(polynomial_order=5)
    for section in (GaussianBasisConfig, ChebyshevBasisConfig):
        assert section().num_basis == 8


@pytest.mark.parametrize(
    ("field", "written", "expected"),
    [
        (
            "radial_basis",
            {"kind": "bessel", "trainable": True},
            BesselBasisConfig(trainable=True),
        ),
        (
            "radial_basis",
            {"kind": "gaussian", "num_basis": 16},
            GaussianBasisConfig(num_basis=16),
        ),
        (
            "radial_basis",
            {"kind": "chebyshev", "include_constant": True},
            ChebyshevBasisConfig(include_constant=True),
        ),
        ("distance_transform", {"kind": "none"}, NoDistanceTransformConfig()),
        (
            "distance_transform",
            {"kind": "agnesi", "amplitude": 1.2},
            AgnesiTransformConfig(amplitude=1.2),
        ),
        (
            "distance_transform",
            {"kind": "soft", "steepness": 2.0},
            SoftTransformConfig(steepness=2.0),
        ),
    ],
    ids=["bessel", "gaussian", "chebyshev", "none", "agnesi", "soft"],
)
def test_the_kind_picks_the_section(field, written, expected):
    assert getattr(RadialRoot.from_dict({field: written}), field) == expected


@pytest.mark.parametrize(
    ("written", "location"),
    [
        ({"radial_basis": {"kind": "bessel", "num_basis": 0}}, "radial_basis.bessel"),
        (
            {"radial_basis": {"kind": "gaussian", "num_basis": 1}},
            "radial_basis.gaussian",
        ),
        ({"cutoff": {"polynomial_order": 0}}, "cutoff"),
    ],
    ids=["bessel-empty", "gaussian-one-centre", "cutoff-order-zero"],
)
def test_sizes_the_module_cannot_build_are_errors(written, location):
    """A Gaussian basis needs two centres: its width divides by
    `num_basis - 1`."""
    with pytest.raises(ValidationError) as excinfo:
        RadialRoot.from_dict(written)
    assert all(where.startswith(location) for where in error_locations(excinfo))
