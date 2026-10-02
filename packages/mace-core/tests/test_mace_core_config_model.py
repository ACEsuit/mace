"""The radial embedding sections of the model schema: what a config writes,
what it defaults to, and what it is refused."""

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


def test_an_empty_config_is_the_legacy_command_line_default():
    """`--radial_type bessel --num_radial_basis 8 --distance_transform None
    --num_cutoff_basis 5`."""
    config = RadialRoot.from_dict({})
    assert config.radial_basis == BesselBasisConfig(num_basis=8, trainable=False)
    assert config.distance_transform == NoDistanceTransformConfig()
    assert config.cutoff == PolynomialCutoffConfig(polynomial_order=5)


def test_every_basis_defaults_to_eight_functions():
    """The legacy flag sized every kind; the Gaussian class default of 128 is
    not the command-line default."""
    for section in (BesselBasisConfig, GaussianBasisConfig, ChebyshevBasisConfig):
        assert section().num_basis == 8


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ({"kind": "bessel", "trainable": True}, BesselBasisConfig(trainable=True)),
        ({"kind": "gaussian", "num_basis": 16}, GaussianBasisConfig(num_basis=16)),
        (
            {"kind": "chebyshev", "include_constant": True},
            ChebyshevBasisConfig(include_constant=True),
        ),
    ],
    ids=["bessel", "gaussian", "chebyshev"],
)
def test_the_kind_picks_the_basis(written, expected):
    assert RadialRoot.from_dict({"radial_basis": written}).radial_basis == expected


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ({"kind": "none"}, NoDistanceTransformConfig()),
        ({"kind": "agnesi", "amplitude": 1.2}, AgnesiTransformConfig(amplitude=1.2)),
        ({"kind": "soft", "steepness": 2.0}, SoftTransformConfig(steepness=2.0)),
    ],
    ids=["none", "agnesi", "soft"],
)
def test_the_kind_picks_the_transform(written, expected):
    config = RadialRoot.from_dict({"distance_transform": written})
    assert config.distance_transform == expected


def test_both_exports_load_back_to_the_same_config():
    config = RadialRoot.from_dict(
        {
            "radial_basis": {"kind": "gaussian", "num_basis": 16},
            "distance_transform": {"kind": "agnesi"},
            "cutoff": {"polynomial_order": 6},
        }
    )
    assert RadialRoot.from_dict(config.to_resolved_dict()) == config
    assert RadialRoot.from_dict(config.to_user_dict()) == config
    assert config.to_user_dict()["radial_basis"] == {
        "kind": "gaussian",
        "num_basis": 16,
    }


def test_an_unknown_kind_is_an_error_at_its_field():
    with pytest.raises(ValidationError, match="fourier") as excinfo:
        RadialRoot.from_dict({"radial_basis": {"kind": "fourier"}})
    assert error_locations(excinfo) == ["radial_basis"]
    with pytest.raises(ValidationError, match="Agnesi"):
        RadialRoot.from_dict({"distance_transform": {"kind": "Agnesi"}})


def test_a_basis_without_its_kind_is_an_error():
    """The tag says which basis the other fields belong to; there is no
    fallback to the default kind."""
    with pytest.raises(ValidationError) as excinfo:
        RadialRoot.from_dict({"radial_basis": {"num_basis": 16}})
    assert error_locations(excinfo) == ["radial_basis"]


def test_a_field_of_another_kind_is_an_error_under_the_chosen_one():
    """`steepness` is the soft transform's; under agnesi it is an unknown key."""
    with pytest.raises(ValidationError) as excinfo:
        RadialRoot.from_dict(
            {"distance_transform": {"kind": "agnesi", "steepness": 2.0}}
        )
    assert error_locations(excinfo) == ["distance_transform.agnesi.steepness"]


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


def test_no_section_holds_the_cutoff_radius():
    """`r_max` is one model-level field, shared by the basis, the envelope and
    the neighbour list."""
    for section in (
        BesselBasisConfig,
        GaussianBasisConfig,
        ChebyshevBasisConfig,
        AgnesiTransformConfig,
        SoftTransformConfig,
        PolynomialCutoffConfig,
    ):
        assert "r_max" not in section.model_fields
