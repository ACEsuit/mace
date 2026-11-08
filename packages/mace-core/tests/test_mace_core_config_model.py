"""The sections of the model schema, the radial embedding's and the model's
own: what a config writes, what it defaults to, and what it is refused.

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
    ModelConfig,
    NoDistanceTransformConfig,
    PolynomialCutoffConfig,
    RadialBasisConfig,
    SoftTransformConfig,
)
from pydantic import ValidationError


@pytest.mark.parametrize("gate", ["silu", "tanh", "abs", "None"])
def test_readout_activation_round_trips(gate):
    config = ModelConfig(readout={"gate": gate})
    assert (
        ModelConfig.model_validate_json(config.model_dump_json()).readout.gate == gate
    )


def test_unknown_readout_activation_is_refused():
    with pytest.raises(ValidationError, match="gate"):
        ModelConfig(readout={"gate": "tahn"})


@pytest.mark.parametrize("name", ["field_update", "field_readout"])
def test_inert_polar_selectors_are_not_configuration_fields(name):
    with pytest.raises(ValidationError, match=name):
        ModelConfig(polar={name: "unused"})


def test_polar_feature_norms_match_the_projection():
    with pytest.raises(ValidationError, match="feature_norms"):
        ModelConfig(
            polar={
                "feature_max_l": 1,
                "feature_widths": (1.0, 2.0),
                "feature_norms": (1.0,),
            }
        )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"name": "positions"}, "schema"),
        ({"name": "energy"}, "schema"),
        ({"name": "conditioning_present"}, "schema"),
        ({"name": "bad.name"}, "name"),
        ({"name": ""}, "name"),
        ({"name": "total_charge", "per": "atom"}, "per graph"),
        ({"name": "external_field", "input_dim": 1}, "3 components"),
        ({"kind": "categorical"}, "num_classes"),
        ({"input_dim": 0}, "input_dim"),
    ],
)
def test_feature_config_refuses_reserved_names_and_invalid_shapes(changes, match):
    from mace_core.config.model import GraphFeatureConfig

    with pytest.raises(ValueError, match=match):
        GraphFeatureConfig.model_validate(
            {
                "name": "conditioning",
                "kind": "continuous",
                "embedding_dim": 2,
                **changes,
            }
        )


def test_custom_feature_storage_survives_model_serialization_without_schema_rows():
    model = ModelConfig.model_validate(
        {
            "graph_features": [
                {
                    "name": "site_state",
                    "per": "atom",
                    "kind": "continuous",
                    "embedding_dim": 2,
                },
                {
                    "name": "temperature",
                    "per": "graph",
                    "kind": "continuous",
                    "embedding_dim": 2,
                },
                {
                    "name": "total_charge",
                    "kind": "categorical",
                    "embedding_dim": 2,
                    "num_classes": 3,
                },
            ]
        }
    )
    assert ModelConfig.model_validate_json(
        model.model_dump_json()
    ).feature_storages == {"site_state": "atom", "temperature": "graph"}
    with pytest.raises(ValueError, match="repeats"):
        ModelConfig(graph_features=(model.graph_features[0], model.graph_features[0]))


@pytest.mark.parametrize(
    "conflict", ["storage", "dimension", "keys", "label", "categorical"]
)
def test_model_data_and_catalogue_agree_on_feature_meaning(conflict):
    from mace_core.config.resolved import ResolvedConfig

    input_spec = {
        "name": "conditioning",
        "per_atom": False,
        "irreps": "0e",
        "units": "unit",
        "differentiable": True,
    }
    feature = {"name": "conditioning", "kind": "continuous", "embedding_dim": 2}
    document = {
        "catalogue": {"inputs": [input_spec], "observables": []},
        "model": {"graph_features": [feature]},
    }
    if conflict == "storage":
        feature["per"] = "atom"
    elif conflict == "dimension":
        feature["input_dim"] = 2
    elif conflict == "keys":
        document["data"] = {"embedding_specs": {"conditioning": {"per": "atom"}}}
    elif conflict == "label":
        document.pop("catalogue")
        feature["name"] = "dipole"
    else:
        feature.update(kind="categorical", num_classes=2)
    with pytest.raises(ValueError, match=r"conditioning|dipole"):
        ResolvedConfig.model_validate(document)


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
    assert (config.interaction_first, config.interaction) == ("residual", "residual")
    dumped = config.model_dump(mode="json")
    assert ModelConfig.model_validate(dumped) == config
    full = ModelConfig(clebsch_gordan_basis="full")
    assert ModelConfig.model_validate(full.model_dump(mode="json")) == full


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clebsch_gordan_basis", "cueq"),
        ("interaction", "attention_residual"),
        ("num_interactions", 0),
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
            "use_edge_irreps_first": True,
        },
    ],
)
def test_an_edge_irreps_setting_no_layer_reads_is_refused(options):
    with pytest.raises(ValidationError, match="edge_irreps"):
        ModelConfig(**options)


def test_edge_irreps_are_accepted_where_a_nonlinear_layer_reads_them():
    ModelConfig(edge_irreps="8x0e+8x1o", interaction="nonlinear")
    ModelConfig(
        edge_irreps="8x0e+8x1o",
        interaction_first="nonlinear",
        use_edge_irreps_first=True,
    )
