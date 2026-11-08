"""FM-5: production conversion against the independent dielectric oracle."""

import hashlib
import json
import sys

import numpy as np
import pytest
import torch
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.data.inputs import graph_inputs_of
from mace_torch.deploy.legacy import convert_legacy
from mace_torch.deploy.loader import load_deployed
from mace_torch.deploy.reference import VerificationError, verify_against_reference
from mace_torch.serialization import canonical_state

from tests.golden import harness
from tests.golden.build_dielectric_anchor import MODEL_PATH
from tests.golden.targets.dielectric import FIXTURE_NAMES, QUANTITIES, REFERENCE_PATH
from tests.parity.test_dipole_parity import legacy_batch, v1_graph
from tests.parity.test_fm_convert_families import close, converted
from tests.parity.test_fm_convert_legacy import EXTRACTOR, FIXTURES, extract_here


def test_fm_convert_dielectric_reference_and_exact_reload(fp64, isolated, tmp_path):
    source_digest = hashlib.sha256(MODEL_PATH.read_bytes()).digest()
    result = convert_legacy(
        MODEL_PATH,
        tmp_path / "converted",
        DEFAULT_CATALOGUE,
        python=sys.executable,
        reference=REFERENCE_PATH,
        fixtures=FIXTURES,
    )
    assert result.report.passed, result.report.describe()
    expected_pairs = {
        (fixture, name) for fixture in FIXTURE_NAMES for name in QUANTITIES
    }
    assert {(d.fixture, d.quantity) for d in result.report.deviations} == expected_pairs
    assert not result.report.skipped
    neutral_digest = {
        path: hashlib.sha256(path.read_bytes()).digest()
        for path in (result.neutral, result.neutral.with_suffix(".safetensors"))
    }
    deployed = load_deployed(result.checkpoint)
    assert deployed.metadata.conversion.golden_gate_passed
    before, after = (
        canonical_state(result.model.engine),
        canonical_state(deployed.engine),
    )
    assert before.keys() == after.keys()
    for op, tensors in before.items():
        assert tensors.keys() == after[op].keys()
        for name, value in tensors.items():
            assert torch.equal(value, after[op][name]), (op, name)
    report = verify_against_reference(
        deployed.engine,
        REFERENCE_PATH,
        FIXTURES,
        z_table=deployed.z_table,
        cutoff=deployed.config.model.r_max,
        graph_inputs=graph_inputs_of(deployed.config.model),
    )
    assert report.passed, report.describe()
    assert {(d.fixture, d.quantity) for d in report.deviations} == expected_pairs
    assert hashlib.sha256(MODEL_PATH.read_bytes()).digest() == source_digest
    for path, digest in neutral_digest.items():
        assert hashlib.sha256(path.read_bytes()).digest() == digest


@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
@pytest.mark.parametrize("total_charge", [0.0, -1.0])
def test_fm_convert_dielectric_live_full_jacobians(
    fp64, isolated, tmp_path, fixture, total_charge
):
    legacy = torch.load(MODEL_PATH, weights_only=False, map_location="cpu").eval()
    imported, _ = converted(legacy, tmp_path)
    atoms = harness.load_fixtures(names=[fixture])[fixture]
    charges = np.zeros(len(atoms))
    expected = legacy(
        legacy_batch(legacy, atoms, charges, total_charge),
        compute_dielectric_derivatives=True,
    )
    graph = v1_graph(
        atoms,
        legacy.atomic_numbers.tolist(),
        float(legacy.r_max),
        charges,
        total_charge,
    )
    actual = imported.engine(graph, compute=("dmu_dr", "dalpha_dr"))
    for name in QUANTITIES:
        close(actual.get(name), expected[name])
    close(actual.get("charges").sum(), torch.tensor(total_charge))


@pytest.mark.parametrize("quantity", ["dmu_dr", "dalpha_dr"])
def test_fm_convert_dielectric_refuses_wrong_derivative_reference(
    fp64, isolated, tmp_path, quantity
):
    document = json.loads(REFERENCE_PATH.read_text())
    entry = document["fixtures"]["water_cluster"]["outputs"][quantity]
    value = np.asarray(entry["value"])
    value.flat[0] += 1.0
    entry["value"] = value.tolist()
    reference = tmp_path / "wrong-reference.json"
    reference.write_text(json.dumps(document))
    output = tmp_path / "refused"
    with pytest.raises(VerificationError, match=quantity) as error:
        convert_legacy(
            MODEL_PATH,
            output,
            DEFAULT_CATALOGUE,
            python=sys.executable,
            reference=reference,
            fixtures=FIXTURES,
        )
    assert [
        (d.fixture, d.quantity) for d in error.value.report.deviations if not d.passed
    ] == [("water_cluster", quantity)]
    assert not output.with_suffix(".json").exists()


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.float64])
def test_fm_convert_dielectric_basis_validation_uses_checkpoint_precision(
    isolated, tmp_path, source_dtype
):
    model = torch.load(MODEL_PATH, weights_only=False, map_location="cpu").to(
        source_dtype
    )
    # Deliberately use the other precision in the extraction process.
    torch.set_default_dtype(
        torch.float64 if source_dtype == torch.float32 else torch.float32
    )
    source = tmp_path / "source.model"
    torch.save(model, source)
    extract_here(source, tmp_path / "valid")
    model.change_of_basis[0, 0, 0] += 0.1
    torch.save(model, source)
    with pytest.raises(EXTRACTOR["ExtractionError"], match="change of basis"):
        extract_here(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid.json").exists()
