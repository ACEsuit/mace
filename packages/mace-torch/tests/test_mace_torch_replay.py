"""The published replay datasets: the table, the cache, and a failed download.

No test here touches the network. The fetch takes what opens the URL, and every
test hands in one that answers with a replay file, so the real download runs:
where the file goes, that it is fetched once, and what happens when the answer
is not a dataset.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from mace_core.artifacts import cache_name
from mace_core.config.data import CURATED_DATASETS
from mace_torch.finetune.replay import (
    CURATED_URLS,
    ReplayDownloadError,
    cache_directory,
    fetch,
    legacy_cached_path,
    read_curated,
)


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    """Every test gets a cache of its own, never the user's."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def replay_file(path, count=3):
    """Labelled the way the published files are: ase's own property names."""
    frames = []
    for index in range(count):
        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
        atoms.calc = SinglePointCalculator(
            atoms, energy=-10.0 - index, forces=np.full((3, 3), 0.1 * index)
        )
        frames.append(atoms)
    write(path, frames, format="extxyz")


def replay_bytes(tmp_path) -> bytes:
    path = tmp_path / "replay.xyz"
    replay_file(path)
    return path.read_bytes()


class Answer:
    """What ``urlopen`` returns, reduced to what a download reads."""

    def __init__(self, body: bytes, content_type: str, fail: Exception | None):
        self.headers = {"Content-Type": content_type}
        self._body = io.BytesIO(body)
        self._fail = fail

    def read(self, size):
        if self._fail is not None:
            raise self._fail
        return self._body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Downloads:
    """Answers with a replay file, and counts how often it was asked."""

    def __init__(self, body=b"", content_type="application/octet-stream", fail=None):
        self.calls: list[str] = []
        self.body = body
        self.content_type = content_type
        self.fail = fail

    def __call__(self, url, timeout):
        self.calls.append(url)
        return Answer(self.body, self.content_type, self.fail)


@pytest.fixture(name="downloads")
def fixture_downloads(tmp_path):
    return Downloads(replay_bytes(tmp_path))


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_every_name_the_schema_accepts_has_a_source():
    assert set(CURATED_URLS) == set(CURATED_DATASETS)


def test_the_sources_are_the_frozen_trees():
    """`mace/tools/multihead_tools.py:154-165`, quoted."""
    base = "https://github.com/ACEsuit/mace-foundations/releases/download/"
    assert {
        "mp": base + "mace_mp_0b/mp_traj_combined.xyz",
        "omat": base + "mace_omat_0/mp_traj_combined_omat.xyz",
        "matpes_pbe": base + "mace_matpes_0/matpes-pbe-replay-data.xyz",
        "matpes_r2scan": base + "mace_matpes_0/matpes-r2scan-replay-data.extxyz",
    } == CURATED_URLS


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


def test_the_cache_follows_xdg(tmp_path):
    assert cache_directory() == tmp_path / "cache" / "mace"


def test_the_cache_falls_back_to_the_home_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cache_directory() == tmp_path / ".cache" / "mace"


def test_the_frozen_trees_name_is_known():
    """So a user who ran a legacy fine-tune does not download it again."""
    assert legacy_cached_path("mp").name == "mp_traj_combinedxyz"
    assert legacy_cached_path("matpes_r2scan").name == "matpesr2scanreplaydataextxyz"


def test_a_dataset_is_downloaded_once(downloads):
    first = fetch("mp", downloads)
    second = fetch("mp", downloads)
    assert first == second == cache_directory() / cache_name(CURATED_URLS["mp"])
    assert downloads.calls == [CURATED_URLS["mp"]]


def test_a_file_already_in_the_cache_is_not_downloaded(downloads):
    """What a legacy run leaves behind, or a login node prepared by hand."""
    path = legacy_cached_path("omat")
    path.parent.mkdir(parents=True)
    replay_file(path)
    assert fetch("omat", downloads) == path
    assert downloads.calls == []


# ---------------------------------------------------------------------------
# A failed download
# ---------------------------------------------------------------------------


def test_a_web_page_is_refused_and_leaves_nothing_behind():
    page = Downloads(b"<html></html>", content_type="text/html; charset=utf-8")
    with pytest.raises(ReplayDownloadError, match="HTML page"):
        fetch("mp", page)
    assert not list(cache_directory().glob("*"))


def test_an_interrupted_download_leaves_no_cache_to_trust(downloads):
    downloads.fail = ConnectionResetError("the connection dropped")
    with pytest.raises(ReplayDownloadError, match="could not be downloaded"):
        fetch("mp", downloads)
    assert not list(cache_directory().glob("*"))


def test_the_failure_says_where_to_put_the_file():
    """Compute nodes on the clusters this runs on have no network."""

    def offline(url, timeout):
        raise OSError("no route to host")

    where = cache_directory() / cache_name(CURATED_URLS["mp"])
    with pytest.raises(ReplayDownloadError, match=str(where)):
        fetch("mp", offline)


# ---------------------------------------------------------------------------
# Reading it
# ---------------------------------------------------------------------------


def test_the_structures_are_read_with_their_labels(downloads):
    """The published files label with ase's reserved names; the energies and
    forces come back out of the calculator ase moved them into."""
    configurations = read_curated("mp", head="replay", opener=downloads)
    assert len(configurations) == 3
    assert configurations[2].properties["energy"] == pytest.approx(-12.0)
    assert np.allclose(configurations[2].properties["forces"], 0.2)
    assert {item.head for item in configurations} == {"replay"}
