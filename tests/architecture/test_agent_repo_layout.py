"""Public guidance and docs are complete, reachable repository artifacts."""

import json
import re
import subprocess
from pathlib import Path
import yaml

from tests.conftest import CAPABILITY_PROBES

ROOT = Path(__file__).resolve().parents[2]


def test_public_skills_have_valid_names_and_existing_references():
    manifest = json.loads((ROOT / "skills.json").read_text())
    entries = manifest["skills"]
    assert len({item["name"] for item in entries}) == len(entries)
    assert {item["path"] for item in entries} == {
        str(path.relative_to(ROOT))
        for path in (ROOT / ".claude/skills").glob("*/SKILL.md")
    }
    for entry in entries:
        path = ROOT / entry["path"]
        text = path.read_text()
        frontmatter = yaml.safe_load(text.split("---", 2)[1])
        assert frontmatter["name"] == entry["name"]
        assert frontmatter["description"].strip()
        assert not re.search(r"/Users/|/home/[^ /]+/", text), path
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#")[0]).exists(), (path, target)


def test_guidance_is_tracked_and_no_package_is_missing():
    tracked = set(
        subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    )
    required = {"CLAUDE.md", "CONTRIBUTING.md", "GOVERNANCE.md", "skills.json"}
    required.update(
        str(path.relative_to(ROOT))
        for path in (ROOT / ".claude/skills").rglob("*")
        if path.is_file()
    )
    required.update(
        str(path.relative_to(ROOT)) for path in (ROOT / "packages").glob("*/CLAUDE.md")
    )
    assert required <= tracked, sorted(required - tracked)
    for project in (ROOT / "packages").glob("*/pyproject.toml"):
        assert project.with_name("CLAUDE.md").is_file(), project


def test_every_capability_has_a_user_guide():
    pages = json.loads((ROOT / "docs/capability_pages.json").read_text())
    assert pages.keys() == CAPABILITY_PROBES.keys()
    for page in pages.values():
        assert (ROOT / "docs" / page).is_file()
