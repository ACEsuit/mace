"""What an artifact reference is, and the names it is cached under."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from mace_core.artifacts import (
    ArtifactRef,
    cache_dir,
    cache_name,
    legacy_cache_names,
)
from mace_core_artifact_urls import FAMILIES, MDP, MP_SMALL, OFF_MEDIUM, POLAR_S


def test_a_reference_is_an_https_url():
    assert ArtifactRef(OFF_MEDIUM).url == OFF_MEDIUM
    with pytest.raises(ValueError, match="https"):
        ArtifactRef("http://example.com/model.model")


def test_the_cache_follows_xdg_and_defaults_under_home(tmp_path):
    assert cache_dir({"XDG_CACHE_HOME": str(tmp_path)}) == tmp_path / "mace"
    assert cache_dir({}) == Path.home() / ".cache" / "mace"


@pytest.mark.parametrize(
    ("url", "stripped", "plain"),
    [
        (MP_SMALL, "20231210mace128L0_energy_epoch249model", None),
        (POLAR_S, "MACEPOLAR1Smodel", None),
        (OFF_MEDIUM, None, "MACE-OFF23_medium.model"),
        (MDP, None, "MACE-MDP.model"),
        (
            "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/"
            "MACE-OFF23_small.model?raw=true",
            "MACEOFF23_smallmodelrawtrue",
            "MACE-OFF23_small.model",
        ),
    ],
)
def test_the_frozen_trees_two_names_are_both_known(url, stripped, plain):
    """The names the frozen tree's loaders wrote, which the ticket quotes from
    real caches. A wrong entry here is a full re-download for every upgrading
    user, and a cold CI cache never notices."""
    names = legacy_cache_names(url)
    if stripped is not None:
        assert stripped in names
    if plain is not None:
        assert plain in names


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_cache_name_is_the_file_name_behind_a_digest(family):
    url = FAMILIES[family]
    name = cache_name(url)
    digest, _, file_name = name.partition("-")
    assert file_name == url.rsplit("/", 1)[1]
    assert len(digest) == 12
    assert name not in legacy_cache_names(url)


def test_two_artifacts_with_one_file_name_do_not_collide():
    first = "https://example.com/a/model.model"
    second = "https://example.com/b/model.model"
    assert cache_name(first) != cache_name(second)


def test_a_link_and_its_raw_form_share_a_cache_name():
    """The same file named two ways is one artifact."""
    blob = "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model"
    raw = (
        "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/"
        "MACE-OFF23_small.model"
    )
    assert cache_name(blob) == cache_name(raw)


def test_the_module_imports_no_framework():
    probe = (
        "import sys, mace_core.artifacts; "
        "print(sorted(m for m in ('torch', 'jax') if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "[]"
