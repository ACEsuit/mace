"""REL-1: migration instructions cover the source surface and run as published."""

from __future__ import annotations

import ast
import importlib
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.cli import build_parser, discover
from mace_core.config.legacy import LEGACY_TRAIN_DESTS, Kept
from mace_core.config.resolved import ResolvedConfig
from mace_core.elements import DefaultKeys
from mace_core.foundation_roster import ROSTER
from mace_core.model_registry import list_models

from tests.golden.check_inventory import (
    read_rows,
    source_entry_points,
    source_train_dests,
)

pytestmark = fp64_only  # Documentation has no dtype-dependent behavior.
ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / ".claude/skills/mace-migrate"
REFS = SKILL / "references"
GUIDE = ROOT / "docs/reforge/migration.md"
GUIDE_URL = (
    "https://github.com/ACEsuit/mace/blob/mace-reforge/docs/reforge/migration.md"
)


def table(path: Path, header: str) -> list[list[str]]:
    lines = path.read_text().splitlines()
    result = []
    active = False
    for line in lines:
        if line.startswith(f"| {header} |"):
            active = True
            continue
        if not line.startswith("|"):
            active = False
        if active and not line.startswith("| ---"):
            result.append(
                [cell.strip().strip("`") for cell in line.strip("|").split("|")]
            )
    assert result, f"missing table {header} in {path}"
    return result


def unique_rows(rows: list[list[str]], column: int = 0) -> dict[str, list[str]]:
    keys = [row[column] for row in rows]
    assert len(keys) == len(set(keys)), "duplicate documented identifiers"
    return dict(zip(keys, rows, strict=True))


def check_change_coverage(documented: list[list[str]]) -> None:
    inventory, errors = read_rows()
    assert not errors
    expected = {
        row.ident: row
        for row in inventory
        if row.verdict in {"DROP", "MERGE"} or row.ident.startswith("train.")
    }
    actual = unique_rows(documented)
    assert actual.keys() == expected.keys(), (
        f"missing: {expected.keys() - actual.keys()}; "
        f"stale: {actual.keys() - expected.keys()}"
    )
    for ident, row in actual.items():
        assert row[1] == expected[ident].verdict, ident
        assert row[3].strip() and row[3].lower() not in {"todo", "idem"}, ident
    # Derive destinations from argparse source, not the adapter under test.
    assert {
        key.removeprefix("train.") for key in actual if key.startswith("train.")
    } == set(source_train_dests())
    for dest, disposition in LEGACY_TRAIN_DESTS.items():
        if isinstance(disposition, Kept):
            assert disposition.path in actual[f"train.{dest}"][3], dest
    for dest, declaration in source_train_dests().items():
        for spelling in re.findall(r"--[A-Za-z0-9_]+", declaration.detail):
            assert spelling in actual[f"train.{dest}"][2], (dest, spelling)


def test_inventory_and_training_destinations_are_documented():
    check_change_coverage(table(REFS / "changes.md", "Inventory ID"))


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong_decision", "stale"])
def test_coverage_rejects_incomplete_or_misleading_map(damage):
    rows = table(REFS / "changes.md", "Inventory ID")
    if damage == "missing":
        rows = [row for row in rows if row[0] != "train.start_swa"]
    elif damage == "duplicate":
        rows.append(rows[0])
    elif damage == "wrong_decision":
        rows[0][1] = "DROP" if rows[0][1] != "DROP" else "KEEP"
    else:
        rows.append(["train.nonexistent", "KEEP", "--nonexistent", "Use a field."])
    with pytest.raises(AssertionError):
        check_change_coverage(rows)


def test_every_console_entry_and_legacy_module_has_a_route_or_retirement():
    rows = table(REFS / "commands.md", "Entry point")
    entries = unique_rows([row for row in rows if row[0] != "—"])
    assert entries.keys() == source_entry_points().keys()
    modules = unique_rows([row for row in rows if row[1] != "—"], column=1)
    # Includes the unregistered converter modules and package initializer.
    tracked = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            "fdef2fb577925cd8e7b66876403870128e69ec97",
            "mace/cli",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    assert modules.keys() == {Path(path).name for path in tracked}
    parser = build_parser(discover())
    for row in rows:
        if row[2] in {"removed", "package marker"}:
            assert row[3]
            continue
        tokens = shlex.split(row[2])
        assert tokens[:1] == ["mace"]
        with pytest.raises(SystemExit) as stopped:
            parser.parse_args([*tokens[1:], "--help"])
        assert stopped.value.code == 0, tokens


def test_foundation_roster_names_precision_and_pending_publication():
    rows = table(REFS / "foundations.md", "Loader")
    actual = {(row[0], row[1]): row for row in rows}
    assert len(actual) == len(rows)
    assert actual.keys() == {(item.loader, item.name) for item in ROSTER}
    registry = list_models()
    for entry in ROSTER:
        row = actual[entry.loader, entry.name]
        if entry.architecture is None:
            assert entry.replaced_by == row[2] and "Retired" in row[4]
            continue
        model = next(model for model in registry if model.model_id == row[2])
        assert model.family == entry.loader
        assert entry.name in model.aliases or entry.name == model.model_id
        assert model.default_dtype == row[3]
        if model.artifact_ref is None or model.metadata.doi is None:
            assert "publication pending" in row[4]
        if model.model_id in {"mace-polar-1-medium", "mace-polar-1-large"}:
            assert "Float32 parity open" in row[4]


def test_label_keys_match_the_data_contract():
    rows = unique_rows(table(REFS / "python-api.md", "Convention"))
    assert rows.keys() == set(DefaultKeys.convention_names())
    for member in DefaultKeys:
        row = rows[member.convention_name]
        assert row[1] == member.value
        assert row[2] == member.storage
        assert ast.literal_eval(row[3]) == member.value_shape


def test_retained_runtime_controls_are_not_omitted_from_the_migration():
    documented = unique_rows(table(REFS / "python-api.md", "Runtime control"))
    expected = {
        row.ident.removeprefix("env.")
        for row in read_rows()[0]
        if row.ident.startswith("env.") and row.verdict == "KEEP"
    }
    assert documented.keys() == expected
    assert all(row[1] == "KEEP" for row in documented.values())


@pytest.mark.parametrize(
    "document,required",
    [
        ("python-api.md", "`head` is no longer a default file key"),
        ("python-api.md", "Flat nine-component arrays are rejected"),
        ("python-api.md", "conflicting references fail"),
        ("python-api.md", "`include_constant`"),
        ("changes.md", "model.distance_transform.kind"),
        ("changes.md", "model.radial_basis.kind"),
    ],
)
def test_non_flag_contract_changes_have_explicit_migration_notes(document, required):
    assert required in (REFS / document).read_text()


def test_documented_python_targets_exist():
    for row in table(REFS / "python-api.md", "Legacy surface"):
        module, name = row[1].rsplit(".", 1)
        assert callable(getattr(importlib.import_module(module), name)), row[1]


def test_metadata_version_note_names_the_current_exact_reader_contract():
    from mace_core.metadata import SCHEMA_VERSION

    assert f"schema {SCHEMA_VERSION} exactly" in GUIDE.read_text()


def blocks(path: Path, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)\n```", path.read_text(), re.DOTALL)


def test_published_training_recipe_translates_and_validates(tmp_path):
    recipe, _ = blocks(REFS / "commands.md", "sh")
    for line in recipe.splitlines():
        tokens = shlex.split(line)
        assert tokens.pop(0) == "mace"
        # Invoke the installed launcher's real main, independent of shell PATH.
        checked = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from mace_core.cli import main; "
                "raise SystemExit(main(sys.argv[1:]))",
                *tokens,
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    config = ResolvedConfig.load(tmp_path / "migrated.yaml")
    assert config.model.r_max == 4 and config.model.num_channels == 8
    assert config.model.radial_basis.kind == "gaussian"
    assert config.model.distance_transform.kind == "soft"
    assert config.precision.model == "float64"
    assert config.training.optimizer.kind == "lbfgs"
    assert config.training.stage_two.start_epoch == 4
    assert config.loss.stage_two_weights["energy"] == 3
    assert not (tmp_path / "train.xyz").exists()


def test_published_converter_recipe_is_accepted_without_loading_pickle():
    from mace_torch.cli.convert_legacy import build_parser

    _, recipe = blocks(REFS / "commands.md", "sh")
    tokens = shlex.split(recipe)
    assert tokens[:3] == ["python", "-m", "mace_torch.cli.convert_legacy"]
    args = build_parser().parse_args(tokens[3:])
    assert args.reference.name == "reference.json"
    assert args.fixtures.name == "fixtures"


def test_published_custom_label_example_reads_the_named_property(tmp_path, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]])
    atoms.info["DFT_energy"] = -1.25
    write(tmp_path / "custom.xyz", atoms, format="extxyz")
    monkeypatch.chdir(tmp_path)
    _, example = blocks(REFS / "python-api.md", "python")
    namespace = {}
    exec(compile(example, "migration-custom-label-example", "exec"), namespace)
    configurations = namespace["configurations"]
    assert len(configurations) == 1
    assert configurations[0].head == "pbe"
    np.testing.assert_equal(configurations[0].properties["energy"], -1.25)


def test_published_native_load_example_evaluates_a_local_checkpoint(
    tmp_path, monkeypatch
):
    from mace_torch.train.checkpoint import write_model
    from test_mace_torch_calculator import water
    from test_mace_torch_full_batch import built_task

    _, built = built_task(tmp_path)
    write_model(tmp_path / "model", built.engine, built.metadata)
    example, _ = blocks(REFS / "python-api.md", "python")
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(compile(example, "migration-native-load-example", "exec"), namespace)
    atoms = water()
    atoms.calc = namespace["calculator"]
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()


def relative_links(path: Path):
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
        if "://" not in target and not target.startswith("#"):
            yield (path.parent / target.split("#", 1)[0]).resolve()


def test_portable_skill_keeps_all_references_inside_its_directory(tmp_path):
    copied = tmp_path / "mace-migrate"
    shutil.copytree(SKILL, copied)
    metadata = yaml.safe_load((copied / "SKILL.md").read_text().split("---", 2)[1])
    assert metadata["name"] == "mace-migrate" and metadata["description"]
    for document in copied.rglob("*.md"):
        for linked in relative_links(document):
            assert linked.is_relative_to(copied) and linked.is_file(), linked
    assert {path.name for path in relative_links(copied / "SKILL.md")} == {
        "commands.md",
        "changes.md",
        "python-api.md",
        "foundations.md",
    }


def test_guide_is_reachable_from_readmes_and_docs_with_valid_local_links():
    readmes = [ROOT / "README.md"]
    for package in ("mace-core", "mace-torch", "mace-jax"):
        directory = ROOT / "packages" / package
        project = tomllib.loads((directory / "pyproject.toml").read_text())["project"]
        readmes.append(directory / project["readme"])
    for path in readmes:
        assert GUIDE_URL in path.read_text() or GUIDE in set(relative_links(path)), path
    landing = ROOT / "docs/index.md"
    assert GUIDE in set(relative_links(landing))
    for path in (landing, GUIDE, *SKILL.rglob("*.md")):
        for linked in relative_links(path):
            assert linked.is_file(), (path, linked)
