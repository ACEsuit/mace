"""`ModelMetadata`: JSON round trip, schema versioning, citation rendering."""

import json
import subprocess
import sys

import pytest
from mace_core.config import ConfigSection, ReforgeBaseConfig
from mace_core.metadata import (
    SCHEMA_VERSION,
    Citation,
    ConfigRecord,
    DataSourceSummary,
    DataSummary,
    E0Details,
    MetadataSchemaError,
    ModelMetadata,
    ParentModel,
    Provenance,
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
        provenance=Provenance(code_version="1.0.0", git_commit="a" * 40),
        data=DataSummary(
            sources=[
                DataSourceSummary(
                    name="water",
                    num_configurations=1200,
                    num_atoms=64_000,
                    elements=["H", "O"],
                    reference_keys=["pbe_energy", "pbe_forces"],
                ),
                DataSourceSummary(name="ice", elements=["H", "O"]),
            ]
        ),
        e0={
            "pbe": E0Details(
                source="estimated",
                method="least_squares",
                parameters={"reference_key": "pbe_energy"},
                values={"H": -13.6, "O": -430.2},
            ),
            "r2scan": E0Details(source="explicit", values={"H": -13.7, "O": -431.0}),
        },
        doi="10.5281/zenodo.0000000",
        citations=[MACE_PAPER, Citation(title="A dataset paper", doi="10.1000/xyz")],
        notes="Trained for the round-trip test.",
    )


# ---------------------------------------------------------------------------
# Round trip and schema version


def test_json_round_trip_is_lossless():
    record = full_record()
    assert ModelMetadata.from_json(record.to_json()) == record


def test_minimal_record_round_trips_too():
    record = ModelMetadata(
        config=ConfigRecord(), provenance=Provenance(code_version="0.0.0")
    )
    assert ModelMetadata.from_json(record.to_json()) == record
    assert record.e0 == {}


def test_config_and_provenance_are_mandatory():
    with pytest.raises(ValidationError, match="config"):
        ModelMetadata.model_validate({"provenance": {"code_version": "0"}})


def test_lossy_value_is_refused_rather_than_stored():
    record = full_record()
    record.e0["pbe"].parameters["shape"] = (2, 3)  # JSON brings it back as a list
    with pytest.raises(MetadataSchemaError, match="does not survive a JSON round trip"):
        record.to_json()


def test_lineage_round_trips_through_two_levels():
    foundation = ParentModel(role="initial_weights", name="mace-mp-0b3")  # no record
    distilled = full_record()
    distilled.parents = [
        foundation,
        ParentModel(role="teacher", name="teacher.model", metadata=full_record()),
    ]
    fine_tuned = full_record()
    fine_tuned.parents = [
        ParentModel(role="initial_weights", name="distilled.model", metadata=distilled)
    ]
    back = ModelMetadata.from_json(fine_tuned.to_json())
    assert back == fine_tuned
    assert back.parents[0].metadata is not None
    grandparents = back.parents[0].metadata.parents
    assert [p.role for p in grandparents] == ["initial_weights", "teacher"]
    assert grandparents[0].metadata is None


def test_schema_version_is_written():
    assert json.loads(full_record().to_json())["schema_version"] == SCHEMA_VERSION


def test_future_schema_version_is_rejected_clearly():
    document = json.loads(full_record().to_json())
    document["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(MetadataSchemaError) as excinfo:
        ModelMetadata.from_json(json.dumps(document))
    message = str(excinfo.value)
    assert f"schema_version {SCHEMA_VERSION + 1}" in message
    assert f"reads schema_version {SCHEMA_VERSION}" in message
    assert "upgrade" in message


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda d: d.pop("schema_version"),
            "schema_version None; expected the integer 1",
        ),
        (lambda d: d.update(schema_version="1"), "schema_version '1'; expected"),
        (lambda d: d.update(schema_version=1.0), "schema_version 1.0; expected"),
    ],
)
def test_missing_or_non_integer_schema_version_is_rejected(mutate, message):
    document = json.loads(full_record().to_json())
    mutate(document)
    with pytest.raises(MetadataSchemaError, match=message):
        ModelMetadata.from_json(json.dumps(document))


@pytest.mark.parametrize(
    ("text", "message"),
    [("[1]", "must be a JSON object, not list"), ("{", "is not valid JSON")],
)
def test_non_record_json_is_rejected_with_context(text, message):
    with pytest.raises(MetadataSchemaError, match=message):
        ModelMetadata.from_json(text)


def test_infinity_survives_and_nan_is_refused():
    # pydantic's default writes inf/nan as null, which would silently turn an
    # E0 into a different value. NaN is never equal to itself, so it cannot
    # pass the round-trip check; an E0 or a config value that is NaN is a bug
    # upstream, not something to store.
    record = full_record()
    record.e0["pbe"].values["H"] = float("inf")
    back = ModelMetadata.from_json(record.to_json())
    assert back.e0["pbe"].values["H"] == float("inf")
    record.config.resolved["cutoff"] = float("nan")
    with pytest.raises(MetadataSchemaError, match="does not survive"):
        record.to_json()


def test_schema_version_is_pinned_on_direct_validation_as_well():
    document = json.loads(full_record().to_json())
    document["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValidationError, match="schema_version"):
        ModelMetadata.model_validate(document)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelMetadata.model_validate(
            {"config": {}, "provenance": {"code_version": "0"}, "note": "x"}
        )


def test_e0_source_is_one_of_two_values():
    with pytest.raises(ValidationError, match="source"):
        E0Details.model_validate({"source": "guessed"})


def test_config_record_is_built_from_a_config():
    class Section(ConfigSection):
        cutoff: float = 5.0

    class Config(ReforgeBaseConfig):
        seed: int = 1
        model: Section = Section()

    record = ConfigRecord.from_config(Config.model_validate({"model": {"cutoff": 4.0}}))
    assert record.user == {"model": {"cutoff": 4.0}}
    assert record.resolved == {"seed": 1, "model": {"cutoff": 4.0}}
    # The embedded form is the fixed point: resolving it again changes nothing.
    assert Config.model_validate(record.resolved).to_resolved_dict() == record.resolved


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
