"""The exported representative reproduces both committed energy anchors."""

import json

import torch
from mace_torch.cli.model import _rebuild
from mace_torch.deploy import load_deployed
from mace_torch.deploy.full_basis import export_full_basis_weights
from mace_torch.deploy.neutral_io import import_neutral
from mace_torch.deploy.reference import verify_against_reference
from mace_torch.serialization import canonical_state, load_canonical_state
from mace_torch.train import write_model
from safetensors.torch import load_file

from tests.parity.test_fm_convert_legacy import (
    CATALOGUE,
    FIXTURES,
    REFERENCE,
    fixture_converted as fixture_converted,
    reference,
)


def test_full_basis_export_reproduces_frozen_goldens(converted, tmp_path):
    name, neutral = converted
    imported = import_neutral(neutral, CATALOGUE)
    checkpoint = write_model(tmp_path / "v1", imported.engine, imported.metadata)
    artifact = export_full_basis_weights(checkpoint)
    record = json.loads(artifact.read_text())
    tensors = load_file(artifact.with_suffix(".safetensors"))
    deployed = load_deployed(checkpoint)
    full_config = deployed.config.model_copy(
        update={
            "model": deployed.config.model.model_copy(
                update={"clebsch_gordan_basis": "full"}
            )
        }
    )
    consumer = _rebuild(deployed, full_config, deployed.heads)
    state = canonical_state(deployed.engine)
    for block in record["blocks"]:
        state[block["component"]] = {
            "weight": tensors[block["tensor"]],
            "path_counts": torch.tensor([group["count"] for group in block["groups"]]),
        }
    load_canonical_state(consumer, state)
    report = verify_against_reference(
        consumer,
        reference(name),
        FIXTURES,
        z_table=deployed.z_table,
        cutoff=deployed.r_max,
        atol=REFERENCE.atol,
        rtol=REFERENCE.rtol,
    )
    assert report.passed, report.describe()
    assert {difference.quantity for difference in report.deviations} == {
        "energy", "forces", "stress"
    }
