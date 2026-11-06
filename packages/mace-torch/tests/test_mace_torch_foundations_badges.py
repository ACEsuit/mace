"""A numerical regression must turn a previously supported row red."""

import copy
from types import SimpleNamespace

import pytest
import torch
from conftest import fp64_only
from mace_core.artifacts import ArtifactRef
from mace_core.model_registry import get_model
from mace_torch.calculators import foundations

from tests.foundations import registry_badges as badges
from tests.golden import harness


@fp64_only
@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("fault", [None, "energy", "default_dtype"])
def test_badge_compares_committed_values_and_requires_publication(
    monkeypatch, tmp_path, local, fault
):
    entry = get_model("mace-mp-0-small").model_copy(
        update={"artifact_ref": ArtifactRef(url="https://models.example/native.json")}
    )
    frozen = harness.load_reference(harness.REFERENCES_DIR / entry.golden_refs[0])
    actual = copy.deepcopy(frozen)
    if fault == "energy":
        fixture = next(iter(actual["fixtures"].values()))
        fixture["outputs"]["energy"]["value"] += 1.0
    model = SimpleNamespace(
        config=SimpleNamespace(model=entry.hyperparameters()),
        heads=tuple(entry.metadata.heads),
    )
    seen = []

    def loader(reference, **kwargs):
        seen.append(reference)
        default = "float64" if fault == "default_dtype" else entry.default_dtype
        model.engine = torch.nn.Linear(
            1, 1, dtype=getattr(torch, kwargs.get("default_dtype", default))
        )
        return SimpleNamespace(models=[model])

    monkeypatch.setattr(foundations, "mace_mp", loader)
    monkeypatch.setattr(harness, "snapshot_outputs", lambda *args, **kwargs: actual)
    row = badges.evaluate_entry(
        entry, framework="torch", local_artifact="candidate.json" if local else None
    )
    assert row["status"] == (
        "failed" if fault else "validated_local" if local else "supported"
    )
    assert row["supported"] == (not fault and not local)
    if fault == "default_dtype":
        assert row["checks"][-1]["interface_dtype"] == "float64"
        assert row["checks"][-1]["dtype"] == "float32"
        assert row["checks"][-1]["status"] == "failed"
        assert row["checks"][0]["status"] == "passed"
        assert row["hardware"]["framework_version"] == torch.__version__
    assert row["goldens"][0]["expected"] == frozen
    assert seen[0] == ("candidate.json" if local else entry.model_id)
    assert badges.write_row(tmp_path, row).name == entry.badge_ref


@fp64_only
def test_missing_publication_and_missing_golden_are_distinct():
    entry = get_model("mace-mp-0-small").model_copy(update={"artifact_ref": None})
    row = badges.evaluate_entry(entry, framework="torch")
    assert row["status"] == "pending_publication" and not row["supported"]
    entry = entry.model_copy(update={"golden_refs": ("does-not-exist.json",)})
    row = badges.evaluate_entry(entry, framework="torch")
    assert row["status"] == "failed" and not row["supported"]
