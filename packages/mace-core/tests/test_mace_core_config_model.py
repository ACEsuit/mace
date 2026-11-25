"""The sections of the model schema, the radial embedding's and the model's
own: what a config writes, what it defaults to, and what it is refused.

How a tagged union loads, exports and reports errors is the config base's and
is tested there (`test_mace_core_config_kinds.py`); this file pins only what
these sections add: the defaults, the kind names and the size bounds."""

import re

import pytest
from mace_core.config import (
    AgnesiTransformConfig,
    BaseConfig,
    BesselBasisConfig,
    ChebyshevBasisConfig,
    DistanceTransformConfig,
    GaussianBasisConfig,
    ModelConfig,
    NoDistanceTransformConfig,
    PolynomialCutoffConfig,
    RadialBasisConfig,
    ScaleShiftSpec,
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


# ---------------------------------------------------------------------------
# The model section
# ---------------------------------------------------------------------------


def test_the_model_section_defaults_to_the_reduced_basis_and_round_trips():
    config = ModelConfig()
    assert config.clebsch_gordan_basis == "reduced"
    assert (config.interaction_first, config.interaction) == (None, "residual")
    dumped = config.model_dump(mode="json")
    assert ModelConfig.model_validate(dumped) == config
    full = ModelConfig(clebsch_gordan_basis="full")
    assert ModelConfig.model_validate(full.model_dump(mode="json")) == full


def test_interaction_names_every_layer_unless_the_first_is_set_apart():
    every = ModelConfig(num_interactions=3, interaction="density")
    assert [every.interaction_of(layer) for layer in range(3)] == ["density"] * 3
    first = ModelConfig(num_interactions=3, interaction_first="plain")
    assert [first.interaction_of(layer) for layer in range(3)] == [
        "plain",
        "residual",
        "residual",
    ]


@pytest.mark.parametrize("layer", [True, 0.0])
def test_a_layer_that_is_not_an_integer_is_refused(layer):
    with pytest.raises(TypeError, match="layer is an integer"):
        ModelConfig().interaction_of(layer)


@pytest.mark.parametrize("layer", [-1, 2])
def test_a_layer_the_model_does_not_have_is_refused(layer):
    with pytest.raises(ValueError, match=f"layer {layer} is not one of the 2"):
        ModelConfig(num_interactions=2).interaction_of(layer)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clebsch_gordan_basis", "cueq"),
        ("interaction", "attention_residual"),
        ("num_interactions", 0),
        # Chosen by whoever builds the model, as the device is.
        ("backend", "cueq"),
    ],
)
def test_the_model_section_refuses_what_it_does_not_build(field, value):
    with pytest.raises(ValidationError):
        ModelConfig(**{field: value})


@pytest.mark.parametrize(
    "options",
    [
        {"edge_irreps": "8x0e+8x1o"},
        {"edge_irreps": "8x0e+8x1o", "interaction_first": "nonlinear"},
        {"interaction_first": "nonlinear", "use_edge_irreps_first": True},
        {
            "edge_irreps": "8x0e+8x1o",
            "interaction": "nonlinear",
            "interaction_first": "residual",
            "use_edge_irreps_first": True,
        },
    ],
)
def test_an_edge_irreps_setting_no_layer_reads_is_refused(options):
    with pytest.raises(ValidationError, match="edge_irreps"):
        ModelConfig(**options)


@pytest.mark.parametrize("edge_irreps", ["8x0e+4x1o", "8x0e+8x1o+4x2e"])
def test_edge_irreps_with_mixed_multiplicities_are_refused(edge_irreps):
    with pytest.raises(ValidationError, match=r"multiplicities \[4, 8\]"):
        ModelConfig(edge_irreps=edge_irreps, interaction="nonlinear")


@pytest.mark.parametrize("edge_irreps", ["8x0e+8xq", ""])
def test_edge_irreps_that_do_not_parse_are_refused_by_name(edge_irreps):
    with pytest.raises(
        ValidationError, match=re.escape(f"edge_irreps={edge_irreps!r}: ")
    ):
        ModelConfig(edge_irreps=edge_irreps, interaction="nonlinear")


def test_a_first_layer_convolving_scalars_needs_a_scalar_term():
    with pytest.raises(ValidationError, match="no '0e' term"):
        ModelConfig(
            edge_irreps="8x1o",
            interaction="nonlinear",
            use_edge_irreps_first=True,
        )
    ModelConfig(edge_irreps="8x1o", interaction="nonlinear")


def test_edge_irreps_are_accepted_where_a_nonlinear_layer_reads_them():
    ModelConfig(edge_irreps="8x0e+8x1o", interaction="nonlinear")
    ModelConfig(
        edge_irreps="8x0e+8x1o", interaction="nonlinear", use_edge_irreps_first=True
    )
    ModelConfig(
        edge_irreps="8x0e+8x1o",
        interaction_first="nonlinear",
        use_edge_irreps_first=True,
    )


# ---------------------------------------------------------------------------
# Scale and shift
# ---------------------------------------------------------------------------

STATISTICS = {
    "energy_mean": -0.25,
    "energy_std": 0.5,
    "rms_forces": 1.5,
    "rms_dipoles": 0.75,
}


@pytest.mark.parametrize(
    ("scale", "shift", "expected"),
    [
        ("std", "mean", (0.5, -0.25)),
        ("std", "none", (0.5, 0.0)),
        ("rms_forces", "mean", (1.5, -0.25)),
        ("rms_forces", "none", (1.5, 0.0)),
        ("rms_dipoles", "none", (0.75, 0.0)),
        ("none", "mean", (1.0, -0.25)),
        ("none", "none", (1.0, 0.0)),
    ],
)
def test_each_method_takes_its_value_from_its_statistic(scale, shift, expected):
    spec = ScaleShiftSpec(scale=scale, shift=shift)
    assert spec.scale_and_shift(**STATISTICS) == expected


@pytest.mark.parametrize(
    ("scale", "shift", "absent"),
    [
        ("std", "none", "energy_std"),
        ("std", "mean", "energy_mean"),
        ("rms_forces", "none", "rms_forces"),
        ("rms_dipoles", "none", "rms_dipoles"),
        ("none", "mean", "energy_mean"),
    ],
)
def test_a_statistic_a_method_needs_and_the_data_lacks_is_an_error(
    scale, shift, absent
):
    statistics = {**STATISTICS, absent: None}
    spec = ScaleShiftSpec(scale=scale, shift=shift)
    with pytest.raises(ValueError, match=re.escape(f"needs [{absent!r}]")):
        spec.scale_and_shift(**statistics)


def test_no_scale_and_no_shift_need_no_statistics():
    spec = ScaleShiftSpec(scale="none", shift="none")
    assert spec.scale_and_shift() == (1.0, 0.0)


def test_a_dipole_scale_with_an_energy_shift_is_refused():
    with pytest.raises(ValidationError, match="cannot shift"):
        ScaleShiftSpec(scale="rms_dipoles", shift="mean")


def test_a_zero_scale_becomes_one_and_says_so(caplog):
    statistics = {**STATISTICS, "rms_forces": 0.0}
    with caplog.at_level("WARNING"):
        spec = ScaleShiftSpec(scale="rms_forces", shift="mean")
        scale_shift = spec.scale_and_shift(**statistics)
    assert scale_shift == (1.0, -0.25)
    assert "measured zero" in caplog.text


def test_scale_shift_is_one_entry_per_named_head():
    config = ModelConfig.model_validate(
        {
            "scale_shift": {
                "pbe": {"scale": "std", "shift": "mean"},
                "mp2": {"scale": "none"},
            }
        }
    )
    assert config.scale_shift == {
        "pbe": ScaleShiftSpec(scale="std", shift="mean"),
        "mp2": ScaleShiftSpec(scale="none", shift="none"),
    }
    assert ModelConfig.model_validate(config.model_dump(mode="json")) == config


@pytest.mark.parametrize(
    ("scale_shift", "message"),
    [
        ({}, "at least 1 item"),
        ({" ": {"scale": "std"}}, "pattern"),
        ({"pbe": {"scale": "rms"}}, "rms"),
        ({"pbe": {"shift": "median"}}, "median"),
    ],
)
def test_scale_shift_refuses_what_names_no_head_or_no_method(scale_shift, message):
    with pytest.raises(ValidationError, match=message):
        ModelConfig.model_validate({"scale_shift": scale_shift})
