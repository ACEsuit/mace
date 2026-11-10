"""The retired source surface stays absent and every removal has provenance."""

import ast
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_and_launcher_cannot_return():
    assert not (ROOT / "mace").exists()
    assert not (ROOT / "packages/mace-launcher").exists()
    assert not (ROOT / "setup.cfg").exists()
    assert not (ROOT / "mace_torch.egg-info").exists(), (
        "Remove stale legacy editable metadata"
    )
    projects = [
        tomllib.loads(path.read_text())["project"]
        for path in (ROOT / "packages").glob("*/pyproject.toml")
    ]
    owners = [
        project["name"] for project in projects if "mace" in project.get("scripts", {})
    ]
    assert owners == ["mace-torch"]
    assert all(project["name"] != "mace-torch-v1" for project in projects)


def test_deleted_test_inventory_and_successors_are_real():
    ledger = json.loads(Path(__file__).with_name("retired_tests.json").read_text())
    # The captured source inventory also works in shallow and filtered clones.
    # source_revision records provenance without making that commit a runtime input.
    old = ledger["source_files"]
    assert old == sorted(set(old))
    assert all(name.startswith("tests/") and name.endswith(".py") for name in old)
    deleted = {name for name in old if not (ROOT / name).exists()}
    assert set(ledger["retired_files"]) == deleted
    assert {item["path"] for item in ledger["files"]} == deleted
    for item in ledger["files"]:
        assert item["native_successors"], item["path"]
        for successor in item["native_successors"]:
            path = ROOT / successor.split("::")[0]
            assert path.is_file(), successor
            ast.parse(path.read_text())
