"""The committed numbers, both precisions; no legacy import or fresh oracle."""

import json
from pathlib import Path

import jax
import pytest
from jax_conformance import run_jax_model_conformance
from mace_jax.models.registry import SUPPORTED_MODELS


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_tiny_anchor_conformance(dtype):
    report = run_jax_model_conformance("tiny", dtypes=(dtype,))
    assert len(report) == 1 and report[0]["equivariance"] == "passed"


@pytest.mark.network
@pytest.mark.parametrize("model_id", SUPPORTED_MODELS)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_published_model_conformance(request, record_property, model_id, dtype, device):
    manifest = request.config.getoption("--jax-model-artifacts")
    artifacts = {} if manifest is None else json.loads(Path(manifest).read_text())
    report = run_jax_model_conformance(
        model_id,
        dtypes=(dtype,),
        devices=(device,),
        artifact_path=artifacts.get(model_id),
    )
    record_property("conformance", json.dumps(report))
    assert len(report) == 1
    jax.clear_caches()
