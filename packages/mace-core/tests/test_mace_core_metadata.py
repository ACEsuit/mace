"""`ModelMetadata`: JSON round trip, schema versioning, the training group,
citation rendering."""

import json
import subprocess
import sys

import mace_core
import mace_core.metadata
import pytest
from mace_core.config import BaseConfig, ConfigSection
from mace_core.metadata import (
    SCHEMA_VERSION,
    Citation,
    ConfigRecord,
    DataSourceSummary,
    DataSummary,
    HeadSummary,
    MetadataSchemaError,
    ModelMetadata,
    ObservableErrors,
    Provenance,
    StageSummary,
    TrainingSummary,
    format_citations,
)
from pydantic import ValidationError

MACE_PAPER = Citation(
    title="MACE: Higher Order Equivariant Message Passing Neural Networks "
    "for Fast and Accurate Force Fields",
    authors=["I. Batatia", "D. P. Kovacs", "G. N. C. Simm", "C. Ortner", "G. Csanyi"],
    venue="Advances in Neural Information Processing Systems",
    year=2022,
    url="https://arxiv.org/abs/2206.07697",
)


def full_record() -> ModelMetadata:
    """Every field set, so the round trip is tested on all of them."""
    return ModelMetadata(
        config=ConfigRecord(
            user={"model": {"num_interactions": 3}},
            resolved={"name": "mace", "model": {"num_interactions": 3, "cutoff": 5.0}},
        ),
        provenance=Provenance(
            versions={"mace-core": "1.0.2", "mace-torch": "1.1.0"},
            git_commit="a" * 40,
        ),
        data=DataSummary(
            sources=[
                DataSourceSummary(
                    name="water",
                    num_configurations=1200,
                    num_atoms=64_000,
                    elements=[1, 8],
                    reference_keys=["pbe_energy", "pbe_forces"],
                ),
                DataSourceSummary(name="ice", elements=[1, 8]),
            ]
        ),
        heads={
            "pbe": HeadSummary(sources=["water", "ice"]),
            "r2scan": HeadSummary(sources=["ice"]),
        },
        doi="10.5281/zenodo.0000000",
        citations=[MACE_PAPER, Citation(title="A dataset paper", doi="10.1000/xyz")],
        notes="Trained for the round-trip test.",
        training=training_summary(),
    )


def training_summary() -> TrainingSummary:
    return TrainingSummary(
        stages=[
            StageSummary(name="main", epochs=180, stop_reason="patience"),
            StageSummary(name="stage_two", epochs=20, stop_reason="max_epochs"),
        ],
        validation_errors={
            "pbe": {
                "energy": ObservableErrors(rmse=1.2e-3, mae=0.9e-3),
                "forces": ObservableErrors(rmse=4.1e-2, mae=2.8e-2),
            },
            "r2scan": {"energy": ObservableErrors(rmse=2.0e-3, mae=1.5e-3)},
        },
    )


def minimal_record(**fields) -> ModelMetadata:
    return ModelMetadata(
        config=ConfigRecord(),
        provenance=Provenance(versions={"mace-core": "0.0.0"}),
        **fields,
    )


def as_document(record: ModelMetadata) -> dict:
    return json.loads(record.to_json())


# ---------------------------------------------------------------------------
# Round trip and schema version


def test_json_round_trip_is_lossless():
    record = full_record()
    assert ModelMetadata.from_json(record.to_json()) == record


def test_minimal_record_round_trips_too():
    record = ModelMetadata(
        config=ConfigRecord(), provenance=Provenance(versions={"mace-core": "0.0.0"})
    )
    assert ModelMetadata.from_json(record.to_json()) == record
    assert record.heads == {}


def test_heads_must_name_summarised_sources():
    record = full_record()
    record.heads["pbe"].sources.append("vapour")
    with pytest.raises(ValidationError, match=r"heads\.pbe\.sources names 'vapour'"):
        ModelMetadata.model_validate(record.model_dump())
    record = full_record()
    record.data.sources.append(DataSourceSummary(name="ice"))
    with pytest.raises(ValidationError, match="names a source twice"):
        ModelMetadata.model_validate(record.model_dump())


def test_config_and_provenance_are_mandatory():
    with pytest.raises(ValidationError, match="config"):
        ModelMetadata.model_validate({"provenance": {"versions": {"mace-core": "0"}}})


def test_lossy_value_is_refused_rather_than_stored():
    record = full_record()
    record.config.user["shape"] = (2, 3)  # JSON brings it back as a list
    with pytest.raises(MetadataSchemaError, match="does not survive a JSON round trip"):
        record.to_json()


def test_schema_version_is_written():
    assert SCHEMA_VERSION == "1.2"
    assert as_document(full_record())["schema_version"] == SCHEMA_VERSION


#: Every way a record is read: a dict, a JSON string, and keyword construction.
READ_ROUTES = pytest.mark.parametrize(
    "read",
    [
        ModelMetadata.model_validate,
        lambda d: ModelMetadata.from_json(json.dumps(d)),
        lambda d: ModelMetadata(**d),
    ],
    ids=["model_validate", "from_json", "construction"],
)


@READ_ROUTES
@pytest.mark.parametrize(
    "version",
    ["1.0", "1.1", "1.3", "2.0", "0.9", 1, "1", "1.0.0", "01.2", "v1.2", None],
)
def test_an_unreadable_schema_version_is_refused_on_every_route(read, version):
    document = as_document(minimal_record())
    document["schema_version"] = version
    document["a_field_of_another_version"] = 1
    with pytest.raises(ValidationError) as excinfo:
        read(document)
    assert excinfo.value.error_count() == 1
    message = str(excinfo.value)
    assert f"schema_version {version!r}" in message
    assert f"reads schema_version {SCHEMA_VERSION}" in message
    assert f"mace-core {mace_core.__version__}" in message


def test_a_record_lacking_the_optional_fields_takes_their_defaults():
    document = {
        "schema_version": SCHEMA_VERSION,
        "config": {},
        "provenance": {"versions": {"mace-core": "0"}},
    }
    record = ModelMetadata.model_validate(document)
    assert record.training is None
    assert record.notes == ""


def test_reading_json_requires_the_schema_version():
    document = as_document(minimal_record())
    del document["schema_version"]
    with pytest.raises(ValidationError) as excinfo:
        ModelMetadata.from_json(json.dumps(document))
    message = str(excinfo.value)
    assert "has no schema_version" in message
    assert f"reads schema_version {SCHEMA_VERSION}" in message
    assert f"mace-core {mace_core.__version__}" in message


@pytest.mark.parametrize(
    "build",
    [ModelMetadata.model_validate, lambda d: ModelMetadata(**d)],
    ids=["model_validate", "construction"],
)
def test_building_a_record_takes_the_current_schema_version(build):
    document = as_document(minimal_record())
    del document["schema_version"]
    assert build(document).schema_version == SCHEMA_VERSION


#: Where a name or a version sits in a full record, and how to empty it.
EMPTY_NAMES = {
    "provenance version": ("provenance", "versions", "mace-core", ""),
    "provenance distribution": ("provenance", "versions", "", "1.0.0"),
    "data source": ("data", "sources", 0, "name", ""),
    "stage": ("training", "stages", 0, "name", ""),
    "head with errors": ("training", "validation_errors", "", {}),
    "observable with errors": (
        "training",
        "validation_errors",
        "pbe",
        "",
        {"rmse": 0.0, "mae": 0.0},
    ),
    "head": ("heads", "", {"sources": []}),
}


@pytest.mark.parametrize("where", EMPTY_NAMES.values(), ids=EMPTY_NAMES.keys())
def test_an_empty_name_or_version_is_refused(where):
    *path, key, value = where
    document = as_document(full_record())
    node = document
    for step in path:
        node = node[step]
    node[key] = value
    with pytest.raises(ValidationError, match="string_too_short"):
        ModelMetadata.from_json(json.dumps(document))


def test_infinity_survives_and_nan_is_refused():
    # pydantic's default writes inf/nan as null, which would silently turn a
    # config value into a different one. NaN is never equal to itself, so it
    # cannot pass the round-trip check; a config value that is NaN is a bug
    # upstream, not something to store.
    record = full_record()
    record.config.resolved["model"]["cutoff"] = float("inf")
    back = ModelMetadata.from_json(record.to_json())
    assert back.config.resolved["model"]["cutoff"] == float("inf")
    record.config.resolved["cutoff"] = float("nan")
    with pytest.raises(MetadataSchemaError, match="does not survive"):
        record.to_json()


@pytest.mark.parametrize(
    "where",
    [[], ["training"], ["training", "stages", 0]],
    ids=["top", "training", "stage"],
)
def test_unknown_fields_are_rejected(where):
    document = as_document(full_record())
    assert document["schema_version"] == SCHEMA_VERSION
    node = document
    for key in where:
        node = node[key]
    node["note"] = "x"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelMetadata.model_validate(document)


def test_config_record_is_built_from_a_config():
    class Section(ConfigSection):
        cutoff: float = 5.0

    class Config(BaseConfig):
        seed: int = 1
        model: Section = Section()

    record = ConfigRecord.from_config(Config.model_validate({"model": {"cutoff": 4.0}}))
    assert record.user == {"model": {"cutoff": 4.0}}
    assert record.resolved == {"seed": 1, "model": {"cutoff": 4.0}}
    # The embedded form is the fixed point: resolving it again changes nothing.
    assert Config.model_validate(record.resolved).to_resolved_dict() == record.resolved


# ---------------------------------------------------------------------------
# The training group


@pytest.mark.parametrize("set_", [True, False], ids=["set", "unset"])
def test_the_training_group_round_trips_set_or_unset(set_):
    record = full_record().model_copy(
        update={"training": training_summary() if set_ else None}
    )
    back = ModelMetadata.from_json(record.to_json())
    assert back == record
    assert (back.training is None) == (not set_)


def test_a_training_summary_names_each_stage_once_and_in_order():
    stages = [
        StageSummary(name="main", epochs=3, stop_reason="converged"),
        StageSummary(name="main", epochs=1, stop_reason="interrupted"),
    ]
    with pytest.raises(ValidationError, match="names a stage twice"):
        TrainingSummary(stages=stages)
    with pytest.raises(ValidationError, match="too_short"):
        TrainingSummary(stages=[])
    with pytest.raises(ValidationError, match="literal_error"):
        StageSummary.model_validate(
            {"name": "main", "epochs": 3, "stop_reason": "timeout"}
        )
    with pytest.raises(ValidationError, match="greater_than_equal"):
        StageSummary(name="main", epochs=-1, stop_reason="max_epochs")
    summary = training_summary()
    assert [stage.name for stage in summary.stages] == ["main", "stage_two"]


def test_validation_errors_are_magnitudes():
    with pytest.raises(ValidationError, match="greater_than_equal"):
        ObservableErrors(rmse=-1.0, mae=0.0)
    with pytest.raises(ValidationError, match="finite_number"):
        ObservableErrors(rmse=float("inf"), mae=0.0)


def test_training_errors_name_known_heads():
    document = as_document(full_record())
    document["training"]["validation_errors"]["pbe0"] = {
        "energy": {"rmse": 1.0, "mae": 1.0}
    }
    with pytest.raises(
        ValidationError, match="names head 'pbe0', which is not in heads"
    ):
        ModelMetadata.model_validate(document)


# ---------------------------------------------------------------------------
# Citations


def test_citations_render_to_a_numbered_block():
    block = format_citations(full_record().citations)
    assert block.splitlines() == [
        "[1] I. Batatia, D. P. Kovacs, G. N. C. Simm, C. Ortner, G. Csanyi. "
        "MACE: Higher Order Equivariant Message Passing Neural Networks for "
        "Fast and Accurate Force Fields. "
        "Advances in Neural Information Processing Systems (2022). "
        "https://arxiv.org/abs/2206.07697",
        "[2] A dataset paper. https://doi.org/10.1000/xyz",
    ]


def test_no_citations_render_to_nothing():
    assert format_citations([]) == ""


def test_metadata_module_imports_neither_torch_nor_jax():
    """In a fresh interpreter, so another test's imports cannot mask a leak."""
    code = (
        "import sys, mace_core.metadata; "
        "leaked = {'torch', 'jax', 'e3nn'} & set(sys.modules); "
        "assert not leaked, leaked"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
