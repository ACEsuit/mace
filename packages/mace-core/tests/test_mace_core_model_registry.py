"""Registry identities cover the frozen roster and remain framework-free."""

import json
import logging
import subprocess
import sys

import pytest
from mace_core.foundation_roster import ROSTER
from mace_core.model_registry import (
    ASL_NOTICE,
    RegistryEntry,
    RetiredModelError,
    UnpublishedModelError,
    export_registry,
    family_default_dtype,
    get_model,
    list_models,
    published_reference,
    resolve_model,
)


@pytest.mark.parametrize(
    "source",
    [entry for entry in ROSTER if entry.family],
    ids=lambda entry: f"{entry.loader}-{entry.name}",
)
def test_every_retained_alias_warns_and_has_one_canonical_target(source):
    with pytest.warns(RuntimeWarning, match="deprecated; use") as warnings:
        entry = resolve_model(source.name, family=source.loader)
    assert entry.model_id in str(warnings[0].message)
    assert get_model(entry.model_id) == entry
    assert resolve_model(entry.model_id, family=source.loader) == entry
    assert entry.metadata.citations and entry.license.source
    assert entry.hyperparameters().clebsch_gordan_basis == "reduced"


@pytest.mark.parametrize(
    "family,expected,dtype",
    [
        ("mace_mp", "mace-mpa-0-medium", "float32"),
        ("mace_off", "mace-off23-medium", "float64"),
        ("mace_omol", "mace-omol-0-extra-large", "float64"),
        ("mace_mdp", "mace-mdp", "float64"),
    ],
)
def test_defaults_come_from_registry_and_log_the_selected_id(
    family, expected, dtype, caplog
):
    with caplog.at_level(logging.INFO):
        entry = resolve_model(None, family=family)
    assert entry.model_id == expected and expected in caplog.text
    assert entry.default and family_default_dtype(family) == dtype


def test_polar_has_no_implicit_model_and_unknown_names_do_not_fall_back():
    with pytest.raises(ValueError, match="explicit model"):
        resolve_model(None, family="mace_polar")
    with pytest.raises(KeyError):
        resolve_model("misspelled", family="mace_mp")
    with pytest.raises(KeyError):
        resolve_model("mace-off23-small", family="mace_mp")
    with pytest.raises(RetiredModelError, match="mace-off23"):
        get_model("mace_anicc")


def test_export_and_queries_share_labels_and_cannot_mutate_the_catalogue(tmp_path):
    rows = list_models()
    assert len(rows) == 24
    assert {row.status for row in rows} == {"recommended"}
    data = json.loads(export_registry(tmp_path / "registry.json").read_text())
    assert [RegistryEntry.model_validate(row) for row in data["models"]] == list(rows)
    assert data["retired"]["mace_anicc"] == "mace-off23-medium"
    row = rows[0]
    row.metadata.notes = "changed by caller"
    assert get_model(row.model_id).metadata.notes != "changed by caller"
    assert all(
        head.e0 is None for entry in rows for head in entry.metadata.heads.values()
    )


def test_unpublished_models_have_no_artifact_doi_or_supported_badge():
    for entry in list_models():
        entry = entry.model_copy(update={"artifact_ref": None})
        with pytest.raises(UnpublishedModelError, match="publication is pending"):
            published_reference(entry)
        assert entry.badge_ref == entry.model_id + ".json"
        assert (entry.license.notice() == ASL_NOTICE) == (
            entry.license.identifier == "ASL"
        )
    assert get_model("mace-mh-1").license.identifier == "ASL"
    assert get_model("mace-mdp").license.identifier == "ASL"
    assert get_model("mace-polar-1-small").license.identifier == "ASL"
    assert get_model("mace-mpa-0-medium").license.notice() is None


def test_registry_import_and_query_load_no_framework():
    code = (
        "from mace_core.model_registry import list_models; import sys; "
        "assert len(list_models())==24; "
        'assert not set(sys.modules)&{"torch","jax","e3nn","mace"}'
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
