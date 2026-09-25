"""The resolved configuration, as a model records it.

Two different statements meet here. The config says where a head's E0s should
come from, and the record says where they came from; `average` is a request
that can fail and a table is already an answer. What is asserted is that the
whole schema survives the trip, and that the two halves do not get confused.
"""

from mace_core.cli import set_value
from mace_core.config import read_config_file
from mace_core.config.e0s import (
    E0sAverage,
    E0sEstimated,
    E0sFromFoundation,
    E0sIsolatedAtoms,
    E0sTable,
)
from mace_core.config.provenance import E0_METHODS, e0_details
from mace_core.config.resolved import ResolvedConfig
from mace_core.metadata import ConfigRecord, ModelMetadata, Provenance


def loaded(schema, path=None, settings=None):
    """A config the way a command line builds one: a file, then the values its
    flags set, each at its dotted path."""
    document = read_config_file(path) if path is not None else {}
    for dotted_path, value in (settings or {}).items():
        set_value(document, dotted_path, value)
    return schema.from_dict(document)


SETTINGS = {
    "runtime.name": "run",
    "model.r_max": "4.5",
    "model.observables": ["energy", "forces", "stress"],
    "training.optimizer": {"kind": "schedulefree", "warmup_steps": 100},
    "loss.kind": {"kind": "huber", "delta": 0.05},
    "data.heads": {"pbe": {"train_file": "train.xyz"}},
}


def record(config: ResolvedConfig) -> ModelMetadata:
    return ModelMetadata(
        config=ConfigRecord.from_config(config),
        provenance=Provenance(code_version="0.0.0"),
    )


def test_the_whole_schema_survives_the_record():
    """Every default filled, written out, read back, and still the same object."""
    config = loaded(ResolvedConfig, settings=SETTINGS)
    metadata = record(config)
    restored = ResolvedConfig.model_validate(metadata.config.resolved)
    assert restored == config


def test_the_record_survives_json():
    """A checkpoint's sidecar is text, so the trip that matters goes through it."""
    metadata = record(loaded(ResolvedConfig, settings=SETTINGS))
    restored = ModelMetadata.model_validate_json(metadata.model_dump_json())
    assert restored == metadata
    assert ResolvedConfig.model_validate(restored.config.resolved) == loaded(
        ResolvedConfig, settings=SETTINGS
    )


def test_recording_twice_gives_the_same_record():
    """The fixed point, over the full schema rather than one section."""
    config = loaded(ResolvedConfig, settings=SETTINGS)
    once = record(config).config.resolved
    twice = record(ResolvedConfig.model_validate(once)).config.resolved
    assert once == twice


def test_the_record_keeps_what_the_user_wrote_apart_from_what_was_filled_in():
    """Both halves, because a reader asking 'what did they set' and a reader
    asking 'what did it run with' want different answers."""
    config = loaded(ResolvedConfig, settings=SETTINGS)
    written = record(config)
    assert written.config.user["model"]["r_max"] == 4.5
    assert "num_channels" not in written.config.user["model"]
    assert written.config.resolved["model"]["num_channels"] == 128


def test_every_e0_kind_has_a_recorded_method():
    """The guard, and the reason `e0_details` also raises on an unknown kind.

    A kind added to the union and not to the method table would record how its
    energies were obtained as a blank, which reads as "nobody wrote it down"
    rather than as a bug. Asserted as a set so the failure names the kind.
    """
    kinds = {
        E0sTable(values={1: -13.6}).kind,
        E0sIsolatedAtoms().kind,
        E0sAverage().kind,
        E0sFromFoundation().kind,
        E0sEstimated().kind,
    }
    assert kinds == set(E0_METHODS)


def test_a_table_is_recorded_as_an_answer_and_the_rest_as_methods():
    assert e0_details(E0sTable(values={1: -13.6}), {"H": -13.6}).source == "explicit"
    for spec in (E0sIsolatedAtoms(), E0sAverage(), E0sFromFoundation(), E0sEstimated()):
        assert e0_details(spec, {"H": -13.6}).source == "estimated"
        assert e0_details(spec, {"H": -13.6}).method


def test_the_kind_settings_tell_two_runs_of_one_method_apart():
    """Without them, two fine-tunes off different foundation heads read alike."""
    first = e0_details(E0sFromFoundation(head="mp"), {"H": -13.6})
    second = e0_details(E0sFromFoundation(head="spice"), {"H": -13.6})
    assert first.method == second.method
    assert first.parameters != second.parameters


def test_the_request_is_not_recorded_as_the_result():
    """A table's own values are the request, keyed by atomic number.

    Recording them would say the energies were obtained by being asked for.
    The caller passes what was resolved, and the record carries that.
    """
    details = e0_details(E0sTable(values={1: -13.6}), {"H": -12.0})
    assert details.values == {"H": -12.0}
    assert "values" not in details.parameters
