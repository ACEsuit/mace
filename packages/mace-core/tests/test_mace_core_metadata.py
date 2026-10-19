"""`ModelMetadata`: JSON round trip, schema versioning, the conversion and
training groups, citation rendering."""

import itertools
import json
import re
import subprocess
import sys
from typing import get_args, get_origin

import mace_core
import mace_core.metadata
import pytest
from mace_core.config import BaseConfig, ConfigSection
from mace_core.metadata import (
    SCHEMA_VERSION,
    Citation,
    ClebschGordanConversion,
    ConfigRecord,
    ConversionRecord,
    DataSourceSummary,
    DataSummary,
    HeadSummary,
    MetadataSchemaError,
    ModelMetadata,
    ObservableErrors,
    ParityGate,
    Provenance,
    StageSummary,
    TrainingSummary,
    format_citations,
)
from pydantic import BaseModel, ValidationError

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
        conversion=conversion_record(),
        training=training_summary(),
    )


def conversion_record() -> ConversionRecord:
    return ConversionRecord(
        source_format="mace-torch-pickle",
        source_sha256="0123456789abcdef" * 4,
        source_versions={"mace-torch": "0.3.16", "torch": "2.5.1"},
        converter_versions={"mace-core": "1.0.0", "mace-torch-v1": "1.0.0"},
        headless=True,
        clebsch_gordan_conversion=ClebschGordanConversion(
            read="full", written="reduced"
        ),
        parity_gate=ParityGate(
            tolerance_row="fp64_cpu_reference",
            max_absolute_deviation={"energy": 2.5e-13, "forces": 0.0},
        ),
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
    assert SCHEMA_VERSION == "1.0"
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
    ("version", "reason"),
    [
        ("1.1", "newer than this code"),
        ("2.0", "newer than this code"),
        ("0.9", "older major"),
        ("0.0", "older major"),
        (1, "not a '<major>.<minor>' string"),
        ("1", "not a '<major>.<minor>' string"),
        ("1.0.0", "not a '<major>.<minor>' string"),
        ("01.0", "not a '<major>.<minor>' string"),
        ("v1.0", "not a '<major>.<minor>' string"),
    ],
)
def test_an_unreadable_schema_version_is_refused_on_every_route(read, version, reason):
    document = as_document(minimal_record())
    document["schema_version"] = version
    document["a_field_of_another_version"] = 1
    with pytest.raises(ValidationError) as excinfo:
        read(document)
    assert excinfo.value.error_count() == 1  # the version, not the unknown field
    message = str(excinfo.value)
    assert reason in message
    assert f"schema_version {version!r}" in message  # the record's
    assert f"reads schema_version {SCHEMA_VERSION}" in message  # the reader's
    assert f"mace-core {mace_core.__version__}" in message


@READ_ROUTES
@pytest.mark.parametrize(
    ("version", "advice"),
    [
        ("1.1", "Upgrade mace-core to a release that reads schema_version 1.1"),
        ("2.0", "Upgrade mace-core to a release that reads schema_version 2.0"),
        ("0.9", "reconvert the model"),
    ],
)
def test_a_refusal_says_what_to_do(read, version, advice):
    document = as_document(minimal_record())
    document["schema_version"] = version
    with pytest.raises(ValidationError, match=re.escape(advice)):
        read(document)


class NextMinorMetadata(ModelMetadata):
    """What `ModelMetadata` would be after a minor bump that adds one field."""

    added_in_the_next_minor: str = "its default"


@pytest.fixture
def next_minor_reader(monkeypatch):
    """The schema one minor on, with a field the current one lacks."""
    monkeypatch.setattr(mace_core.metadata, "SCHEMA_VERSION", "1.1")
    return NextMinorMetadata


@pytest.mark.parametrize(
    "read",
    ["model_validate", "from_json", "construction"],
)
def test_an_older_minor_loads_with_defaults_for_what_it_lacks(next_minor_reader, read):
    # What a 1.0 writer produced: the 1.0 fields, under its own version.
    document = {**as_document(full_record()), "schema_version": "1.0"}
    readers = {
        "model_validate": next_minor_reader.model_validate,
        "from_json": lambda d: next_minor_reader.from_json(json.dumps(d)),
        "construction": lambda d: next_minor_reader(**d),
    }
    record = readers[read](document)
    assert record.added_in_the_next_minor == "its default"
    # Read into the newer schema, so it is written back as the newer version.
    assert record.schema_version == "1.1"
    assert json.loads(record.to_json())["schema_version"] == "1.1"
    assert record.conversion == full_record().conversion


def test_the_newer_reader_still_refuses_a_newer_minor(next_minor_reader):
    document = as_document(minimal_record())
    document["schema_version"] = "1.2"
    with pytest.raises(ValidationError, match=re.escape("reads schema_version 1.1")):
        next_minor_reader.model_validate(document)


def test_an_older_minor_with_an_unknown_key_is_still_refused(next_minor_reader):
    document = as_document(minimal_record())
    document["notse"] = "a typo of notes"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        next_minor_reader.model_validate(document)


def test_a_record_lacking_the_optional_fields_takes_their_defaults():
    document = {
        "schema_version": SCHEMA_VERSION,
        "config": {},
        "provenance": {"versions": {"mace-core": "0"}},
    }
    record = ModelMetadata.model_validate(document)
    assert record.conversion is None
    assert record.training is None
    assert record.notes == ""


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
    [[], ["conversion"], ["conversion", "parity_gate"], ["training"]],
    ids=["top", "conversion", "parity_gate", "training"],
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
# Fields per schema version

#: Every record class and its fields, per released schema version. A minor bump
#: adds an entry; the newest one is what the code must hold.
SCHEMA_FIELDS = {
    "1.0": {
        "Citation": {"authors", "doi", "title", "url", "venue", "year"},
        "ClebschGordanConversion": {"read", "written"},
        "ConfigRecord": {"resolved", "user"},
        "ConversionRecord": {
            "clebsch_gordan_conversion",
            "converter_versions",
            "headless",
            "parity_gate",
            "source_format",
            "source_sha256",
            "source_versions",
        },
        "DataSourceSummary": {
            "elements",
            "name",
            "num_atoms",
            "num_configurations",
            "reference_keys",
        },
        "DataSummary": {"sources"},
        "HeadSummary": {"sources"},
        "ModelMetadata": {
            "citations",
            "config",
            "conversion",
            "data",
            "doi",
            "heads",
            "notes",
            "provenance",
            "schema_version",
            "training",
        },
        "ObservableErrors": {"mae", "rmse"},
        "ParityGate": {"max_absolute_deviation", "tolerance_row"},
        "Provenance": {"git_commit", "versions"},
        "StageSummary": {"epochs", "name", "stop_reason"},
        "TrainingSummary": {"stages", "validation_errors"},
    },
}


def _version_key(version: str) -> tuple[int, int]:
    major, minor = version.split(".")
    return int(major), int(minor)


def _record_classes(root: type[BaseModel]) -> dict[str, type[BaseModel]]:
    """Every record class reachable from `root`, by name."""

    def classes_in(annotation):
        outer = get_origin(annotation) or annotation
        if isinstance(outer, type) and issubclass(outer, BaseModel):
            yield outer
        for argument in get_args(annotation):
            yield from classes_in(argument)

    found, to_visit = {}, [root]
    while to_visit:
        record = to_visit.pop()
        if record.__name__ not in found:
            found[record.__name__] = record
            for field in record.model_fields.values():
                to_visit.extend(classes_in(field.annotation))
    return found


def _fields_added_without_default(root: type[BaseModel], baseline: dict) -> list[str]:
    """Fields a record of `baseline` lacks and that have no default to fill
    them with.

    A class the baseline does not have is reached only through a field it
    lacks, so that class's own fields are free to be required."""
    missing = []
    for name, record in _record_classes(root).items():
        for field_name, field in record.model_fields.items():
            added = name in baseline and field_name not in baseline[name]
            if added and field.is_required():
                missing.append(f"{name}.{field_name}")
    return missing


def test_the_newest_field_list_is_the_code():
    """Adding a field without a minor bump fails here, since a reader of the
    old minor would refuse the new field as unknown rather than name a version.
    """
    fields = {
        name: set(record.model_fields)
        for name, record in _record_classes(ModelMetadata).items()
    }
    assert max(SCHEMA_FIELDS, key=_version_key) == SCHEMA_VERSION
    assert fields == SCHEMA_FIELDS[SCHEMA_VERSION]


def test_a_minor_only_adds_fields():
    versions = sorted(SCHEMA_FIELDS, key=_version_key)
    assert all(version.startswith("1.") for version in versions)
    for older, newer in itertools.pairwise(versions):
        for name, fields in SCHEMA_FIELDS[older].items():
            assert name in SCHEMA_FIELDS[newer], f"{name} removed in {newer}"
            removed = fields - SCHEMA_FIELDS[newer][name]
            assert not removed, f"{name} lost {removed} in {newer}"


def test_every_field_added_after_1_0_has_a_default():
    assert _fields_added_without_default(ModelMetadata, SCHEMA_FIELDS["1.0"]) == []


class WithRequiredAddition(ModelMetadata):
    added_without_default: int


class WithOptionalAddition(ModelMetadata):
    added_with_default: int = 0


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (WithRequiredAddition, ["WithRequiredAddition.added_without_default"]),
        (WithOptionalAddition, []),
    ],
)
def test_the_default_check_catches_a_required_addition(record, expected):
    # Each subclass stands in for a later ModelMetadata, so it is compared
    # against ModelMetadata's 1.0 fields.
    baseline = {
        **SCHEMA_FIELDS["1.0"],
        record.__name__: SCHEMA_FIELDS["1.0"]["ModelMetadata"],
    }
    assert _fields_added_without_default(record, baseline) == expected


# ---------------------------------------------------------------------------
# The conversion and training groups


@pytest.mark.parametrize(
    "groups",
    [
        {},
        {"conversion": True},
        {"training": True},
        {"conversion": True, "training": True},
    ],
    ids=["neither", "conversion", "training", "both"],
)
def test_the_groups_round_trip_set_or_unset(groups):
    record = full_record().model_copy(
        update={
            "conversion": conversion_record() if "conversion" in groups else None,
            "training": training_summary() if "training" in groups else None,
        }
    )
    back = ModelMetadata.from_json(record.to_json())
    assert back == record
    assert (back.conversion is None) == ("conversion" not in groups)
    assert (back.training is None) == ("training" not in groups)


def test_nothing_was_converted_means_no_clebsch_gordan_entry():
    record = conversion_record().model_copy(update={"clebsch_gordan_conversion": None})
    assert ConversionRecord.model_validate(record.model_dump()) == record
    with pytest.raises(ValidationError, match="converts nothing"):
        ClebschGordanConversion(read="reduced", written="reduced")
    with pytest.raises(ValidationError, match="literal_error"):
        ClebschGordanConversion.model_validate(
            {"read": "full", "written": "compressed"}
        )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"source_sha256": "0123"}, "string_pattern_mismatch"),
        ({"source_sha256": "0123456789ABCDEF" * 4}, "string_pattern_mismatch"),
        ({"source_versions": {"torch": "2.5.1"}}, "no 'mace-torch' entry"),
        ({"converter_versions": {}}, "too_short"),
        ({"source_format": ""}, "string_too_short"),
    ],
)
def test_a_conversion_record_refuses_what_a_converter_cannot_have_seen(change, match):
    document = {**conversion_record().model_dump(), **change}
    with pytest.raises(ValidationError, match=match):
        ConversionRecord.model_validate(document)


@pytest.mark.parametrize(
    "deviations",
    [{}, {"energy": -1e-12}, {"energy": float("inf")}, {"energy": float("nan")}],
    ids=["none", "negative", "infinite", "nan"],
)
def test_a_parity_gate_holds_finite_magnitudes(deviations):
    with pytest.raises(ValidationError):
        ParityGate(
            tolerance_row="fp64_cpu_reference", max_absolute_deviation=deviations
        )


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
