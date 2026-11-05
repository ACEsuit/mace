"""Artifact-specific legacy references for the rest of the published roster."""

import hashlib

import pytest

from tests.golden import foundation_artifacts as fa
from tests.golden import harness
from tests.golden.targets.foundation_roster import MODELS, reference_path, specification

pytestmark = pytest.mark.network


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name, marks=[pytest.mark.polar] if name.startswith("polar-") else []
        )
        for name in MODELS
    ],
)
def test_published_roster_artifact_reproduces_its_reference(name):
    spec = specification(name)
    calculator, source = fa.load_calculator(spec)
    reference = harness.load_reference(reference_path(name))
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == reference["provenance"]["sha256"]
    )
    assert spec.origin == reference["provenance"]["source"]
    assert set(reference["fixtures"]) == set(spec.fixture_names)
    snapshot = harness.snapshot_outputs(
        calculator,
        harness.load_fixtures(names=spec.fixture_names),
        dtype="float64",
        device="cpu",
        backend="e3nn",
    )
    harness.compare_to_reference(
        snapshot, reference, row=harness.FP64_CPU_REFERENCE.name, strict_channels=True
    )
