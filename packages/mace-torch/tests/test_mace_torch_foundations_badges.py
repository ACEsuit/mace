"""A numerical regression must turn a previously supported row red."""

import copy
from types import SimpleNamespace

import pytest
from conftest import fp64_only
from mace_core.artifacts import ArtifactRef
from mace_core.model_registry import get_model
from mace_torch.calculators import foundations

from tests.foundations import registry_badges as badges
from tests.golden import harness


@fp64_only
@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("wrong", [True, False])
def test_badge_compares_committed_values_and_requires_publication(
    monkeypatch, tmp_path, local, wrong
):
    entry = get_model("mace-mp-0-small").model_copy(
        update={"artifact_ref": ArtifactRef(url="https://models.example/native.json")}
    )
    frozen = harness.load_reference(harness.REFERENCES_DIR / entry.golden_refs[0])
    actual = copy.deepcopy(frozen)
    if wrong:
        fixture = next(iter(actual["fixtures"].values()))
        fixture["outputs"]["energy"]["value"] += 1.0
    model = SimpleNamespace(
        config=SimpleNamespace(model=entry.hyperparameters()),
        heads=tuple(entry.metadata.heads),
    )
    seen = []

    def loader(reference, **kwargs):
        seen.append(reference)
        return SimpleNamespace(models=[model])

    monkeypatch.setattr(foundations, "mace_mp", loader)
    monkeypatch.setattr(harness, "snapshot_outputs", lambda *args, **kwargs: actual)
    row = badges.evaluate_entry(
        entry, framework="torch", local_artifact="candidate.json" if local else None
    )
    assert row["status"] == (
        "failed" if wrong else "validated_local" if local else "supported"
    )
    assert row["supported"] == (not wrong and not local)
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
