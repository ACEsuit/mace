"""The supplemental OFF23 stress reference remains tied to its frozen source."""

import hashlib

import numpy as np
import pytest

from tests.golden import foundation_artifacts as fa
from tests.golden import harness
from tests.golden.targets.off_medium_periodic import FIXTURES, REFERENCE


def test_periodic_reference_pins_source_and_nonzero_stress():
    reference = harness.load_reference(harness.REFERENCES_DIR / REFERENCE)
    assert reference["provenance"]["sha256"] == fa.ARTIFACTS["off_medium"].sha256
    assert set(reference["fixtures"]) == set(FIXTURES)
    for fixture in reference["fixtures"].values():
        stress = np.asarray(fixture["outputs"]["stress"]["value"])
        assert np.isfinite(stress).all() and np.max(np.abs(stress)) > 0


@pytest.mark.network
def test_frozen_off_medium_matches_periodic_reference():
    import torch

    previous = torch.get_default_dtype()
    try:
        spec = fa.ARTIFACTS["off_medium"]
        calculator, source = fa.load_calculator(spec)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == spec.sha256
        snapshot = harness.snapshot_outputs(
            calculator, harness.load_fixtures(names=FIXTURES),
            dtype="float64", device="cpu", backend="e3nn",
        )
        harness.compare_to_reference(
            snapshot, harness.load_reference(harness.REFERENCES_DIR / REFERENCE),
            row=harness.FP64_CPU_REFERENCE.name,
        )
    finally:
        torch.set_default_dtype(previous)
