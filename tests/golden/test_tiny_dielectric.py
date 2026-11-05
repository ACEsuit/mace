"""Offline dielectric oracle, built and evaluated only by frozen legacy."""

import hashlib
import json

import numpy as np
import torch

from tests.golden import harness
from tests.golden.build_dielectric_anchor import MODEL_PATH, build_model
from tests.golden.targets.dielectric import (
    FIXTURE_NAMES,
    QUANTITIES,
    REFERENCE_PATH,
    snapshot,
)


def test_tiny_dielectric_reproduces_its_reference():
    model = torch.load(MODEL_PATH, weights_only=False, map_location="cpu").eval()
    reference = harness.load_reference(REFERENCE_PATH)
    assert set(reference["fixtures"]) == set(FIXTURE_NAMES)
    for entry in reference["fixtures"].values():
        assert set(entry["outputs"]) == set(QUANTITIES)
    harness.compare_to_reference(
        snapshot(model),
        reference,
        row=harness.FP64_CPU_REFERENCE.name,
        strict_channels=True,
    )


def test_tiny_dielectric_recipe_and_nontrivial_response():
    sidecar = json.loads(MODEL_PATH.with_suffix(".build.json").read_text())
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == sidecar["sha256"]
    model = build_model()
    assert type(model).__name__ == "AtomicDielectricMACE"
    assert next(model.parameters()).dtype == torch.float64
    assert sum(p.numel() for p in model.parameters()) == sidecar["num_parameters"]
    reference = harness.load_reference(REFERENCE_PATH)
    harness.compare_to_reference(
        snapshot(model),
        reference,
        row=harness.FP64_CPU_REFERENCE.name,
        strict_channels=True,
    )
    water = reference["fixtures"]["water_cluster"]["outputs"]
    for name in QUANTITIES:
        assert np.any(np.abs(water[name]["value"]) > harness.FP64_CPU_REFERENCE.atol), (
            name
        )
    alpha = np.asarray(water["polarizability"]["value"])
    assert np.any(
        np.abs(alpha - np.eye(3) * np.trace(alpha) / 3)
        > harness.FP64_CPU_REFERENCE.atol
    )
