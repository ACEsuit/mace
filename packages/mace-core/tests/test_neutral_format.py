"""The neutral weights format: what it accepts, and everything it refuses.

The writer of these files runs against another package and imports nothing
from this one, so the reader is the only place the contract is enforced. Each
refusal here is a way a file could be misread instead.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from mace_core.weights.neutral_format import (
    NeutralFormatError,
    NeutralSidecar,
    read_neutral,
    write_neutral,
)


def sidecar(**changes) -> NeutralSidecar:
    document = {
        "format": "mace-neutral",
        "version": 1,
        "provenance": {
            "source_file": "model.model",
            "source_sha256": "0" * 64,
            "source_class": "Example",
            "source_version": "0.3.17",
            "converter_version": "1",
            "headless": False,
        },
        "family": "scale_shift",
        "config": {"r_max": 5.0},
        "heads": ["default"],
        "dtype": "float64",
        "ops": {
            "node_embedding": {
                "op_kind": "linear",
                "schema_version": "1.0",
                "descriptor": {"irreps_in": "2x0e", "irreps_out": "4x0e"},
                "tensors": ["weight", "bias"],
            }
        },
        "derived": {"node_embedding.linear.output_mask": "rebuilt from the irreps"},
    }
    document.update(changes)
    return NeutralSidecar.model_validate(document)


TENSORS = {
    "node_embedding::weight": np.arange(8, dtype=np.float64),
    "node_embedding::bias": np.zeros(0),
}


def test_an_artifact_round_trips(tmp_path):
    written = write_neutral(tmp_path / "model", sidecar(), TENSORS)
    artifact = read_neutral(written)
    assert artifact.sidecar == sidecar()
    assert np.array_equal(artifact.tensor("node_embedding", "weight"), np.arange(8))


def test_either_half_or_the_bare_name_reads_it(tmp_path):
    write_neutral(tmp_path / "model", sidecar(), TENSORS)
    for name in ("model", "model.json", "model.safetensors"):
        assert read_neutral(tmp_path / name).sidecar.heads == ("default",)


def test_the_tensors_are_not_a_pickle(tmp_path):
    """The file opens as safetensors, whose header is JSON and whose body is
    raw numbers, and the sidecar is JSON: there is nothing to execute."""
    write_neutral(tmp_path / "model", sidecar(), TENSORS)
    raw = (tmp_path / "model.safetensors").read_bytes()
    header = int.from_bytes(raw[:8], "little")
    assert set(json.loads(raw[8 : 8 + header])) >= set(TENSORS)
    json.loads((tmp_path / "model.json").read_text())
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "model.json",
        "model.safetensors",
    ]


def test_a_tensor_no_op_declares_is_refused(tmp_path):
    with pytest.raises(NeutralFormatError, match="declared by no op"):
        write_neutral(
            tmp_path / "model", sidecar(), {**TENSORS, "stray::x": np.zeros(1)}
        )


def test_a_declared_tensor_missing_from_the_file_is_refused(tmp_path):
    write_neutral(tmp_path / "model", sidecar(), TENSORS)
    document = json.loads((tmp_path / "model.json").read_text())
    document["ops"]["node_embedding"]["tensors"].append("scale")
    (tmp_path / "model.json").write_text(json.dumps(document))
    with pytest.raises(NeutralFormatError, match="node_embedding::scale"):
        read_neutral(tmp_path / "model")


@pytest.mark.parametrize(("field", "value"), [("format", "other"), ("version", 2)])
def test_another_format_or_version_is_refused(tmp_path, field, value):
    write_neutral(tmp_path / "model", sidecar(), TENSORS)
    document = json.loads((tmp_path / "model.json").read_text())
    document[field] = value
    (tmp_path / "model.json").write_text(json.dumps(document))
    with pytest.raises(NeutralFormatError, match=field):
        read_neutral(tmp_path / "model")


def test_an_unknown_field_is_refused(tmp_path):
    """A field this reader does not know is a field it would ignore."""
    write_neutral(tmp_path / "model", sidecar(), TENSORS)
    document = json.loads((tmp_path / "model.json").read_text())
    document["provenance"]["trained_on"] = "something"
    (tmp_path / "model.json").write_text(json.dumps(document))
    with pytest.raises(NeutralFormatError, match="does not validate"):
        read_neutral(tmp_path / "model")


def test_a_missing_half_is_refused(tmp_path):
    write_neutral(tmp_path / "model", sidecar(), TENSORS)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(NeutralFormatError, match="two files"):
        read_neutral(tmp_path / "model")


def test_an_absent_op_or_tensor_is_named(tmp_path):
    artifact = read_neutral(write_neutral(tmp_path / "model", sidecar(), TENSORS))
    with pytest.raises(NeutralFormatError, match=r"no op 'readouts\.0'"):
        artifact.tensor("readouts.0", "weight")
    with pytest.raises(NeutralFormatError, match="not 'scale'"):
        artifact.tensor("node_embedding", "scale")
