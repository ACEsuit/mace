"""The published replay datasets: the table, the cache, and a failed download.

No test here touches the network. The download is a function the fetch takes,
and every test hands in one that writes a file, so what is checked is what the
fetch does around it: where the file goes, that it is fetched once, and what
happens when the answer is not a dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from mace_core.config.data import CURATED_DATASETS
from mace_torch.finetune.replay import (
    CURATED_URLS,
    ReplayDownloadError,
    cache_directory,
    cached_path,
    fetch,
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


class Downloads:
    """A download that writes a replay file and counts how often it ran."""

    def __init__(self, headers="Content-Type: application/octet-stream"):
        self.calls: list[str] = []
        self.headers = headers

    def __call__(self, url: str, path: str) -> str:
        self.calls.append(url)
        replay_file(path)
        return self.headers


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


def test_the_file_keeps_the_frozen_trees_name():
    """So a user who ran a legacy fine-tune does not download it again."""
    assert cached_path("mp").name == "mp_traj_combinedxyz"
    assert cached_path("matpes_r2scan").name == "matpesr2scanreplaydataextxyz"


def test_a_dataset_is_downloaded_once():
    downloads = Downloads()
    first = fetch("mp", downloads)
    second = fetch("mp", downloads)
    assert first == second == cached_path("mp")
    assert downloads.calls == [CURATED_URLS["mp"]]


def test_a_file_already_in_the_cache_is_not_downloaded():
    """What a legacy run leaves behind, or a login node prepared by hand."""
    path = cached_path("omat")
    path.parent.mkdir(parents=True)
    replay_file(path)
    downloads = Downloads()
    fetch("omat", downloads)
    assert downloads.calls == []


# ---------------------------------------------------------------------------
# A failed download
# ---------------------------------------------------------------------------


def test_a_web_page_is_refused_and_leaves_nothing_behind():
    with pytest.raises(ReplayDownloadError, match="web page"):
        fetch("mp", Downloads(headers="Content-Type: text/html; charset=utf-8"))
    assert not cached_path("mp").exists()
    assert not list(cache_directory().glob("*.part"))


def test_an_interrupted_download_leaves_no_cache_to_trust():
    def interrupted(url, path):
        with open(path, "w") as handle:
            handle.write("1\n\nO 0 0")
        raise ConnectionResetError("the connection dropped")

    with pytest.raises(ReplayDownloadError, match="could not be downloaded"):
        fetch("mp", interrupted)
    assert not cached_path("mp").exists()


def test_the_failure_says_where_to_put_the_file():
    """Compute nodes on the clusters this runs on have no network."""

    def offline(url, path):
        raise OSError("no route to host")

    with pytest.raises(ReplayDownloadError, match=str(cached_path("mp"))):
        fetch("mp", offline)


# ---------------------------------------------------------------------------
# Reading it
# ---------------------------------------------------------------------------


def test_the_structures_are_read_with_their_labels():
    """The published files label with ase's reserved names; the energies and
    forces come back out of the calculator ase moved them into."""
    configurations = read_curated("mp", head="replay", download=Downloads())
    assert len(configurations) == 3
    assert configurations[2].properties["energy"] == pytest.approx(-12.0)
    assert np.allclose(configurations[2].properties["forces"], 0.2)
    assert {item.head for item in configurations} == {"replay"}
