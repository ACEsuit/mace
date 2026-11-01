"""The published replay datasets: where they live, and reading them once.

A replay head trains on a sample of the data the foundation model was fitted
to, so the fine-tune does not forget it. Four such datasets are published
beside the foundation models, and a head names one by the name the schema
accepts.

**Fetched as any other artifact is**, through
:func:`mace_core.artifacts.resolve_artifact`: downloaded once into the cache,
whole or not at all, and never an HTML page in its place. A file the frozen
tree cached is adopted where it is. Its name is the URL's file name with
every character that is not alphanumeric or an underscore dropped
(``mp_traj_combinedxyz``), and keeping it adoptable is the point: a user who
ran a legacy fine-tune has several hundred megabytes on disk already.

Nothing is written to the working directory: legacy subselects into
``mp_finetuning-<tag>.xyz`` there and reads it back, which is a file another
run in the same directory overwrites.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mace_core.artifacts import (
    ArtifactDownloadError,
    cache_dir,
    cache_name,
    download,
    resolve_artifact,
)
from mace_core.config.data import CuratedDataset
from mace_core.data import Configuration, KeySpecification, open_dataset

__all__ = [
    "CURATED_URLS",
    "ReplayDownloadError",
    "cache_directory",
    "fetch",
    "legacy_cached_path",
    "read_curated",
    "replay_key_specification",
]

logger = logging.getLogger(__name__)

_RELEASES = "https://github.com/ACEsuit/mace-foundations/releases/download"

#: Where each published replay dataset is downloaded from. The frozen tree's
#: table (`mace/tools/multihead_tools.py:154-165`), unchanged.
CURATED_URLS: dict[str, str] = {
    "mp": f"{_RELEASES}/mace_mp_0b/mp_traj_combined.xyz",
    "omat": f"{_RELEASES}/mace_omat_0/mp_traj_combined_omat.xyz",
    "matpes_pbe": f"{_RELEASES}/mace_matpes_0/matpes-pbe-replay-data.xyz",
    "matpes_r2scan": f"{_RELEASES}/mace_matpes_0/matpes-r2scan-replay-data.extxyz",
}


class ReplayDownloadError(RuntimeError):
    """A replay dataset that could not be fetched."""


def cache_directory() -> Path:
    """``$XDG_CACHE_HOME/mace``, or ``~/.cache/mace`` when it is unset."""
    return cache_dir()


def legacy_cached_path(name: CuratedDataset) -> Path:
    """Where the frozen tree kept the dataset, which is adopted when present."""
    basename = CURATED_URLS[name].rsplit("/", 1)[1]
    return cache_directory() / "".join(
        character for character in basename if character.isalnum() or character == "_"
    )


def fetch(name: CuratedDataset, opener: Any = None) -> Path:
    """The dataset's local path, downloading it the first time.

    Args:
        name: One of :data:`CURATED_URLS`.
        opener: What opens the URL, for a test to stand in for the network.

    Raises:
        ReplayDownloadError: If the download fails, or answers with an HTML
            page rather than a file, which is what a moved release asset
            returns with a success status.
    """
    url = CURATED_URLS[name]
    destination = cache_directory() / cache_name(url)

    def get(link: str, target: Path) -> Path:
        logger.info("Downloading the %r replay dataset from %s", name, link)
        return download(link, target, opener=opener)

    try:
        return resolve_artifact(url, fetch=get)
    except ArtifactDownloadError as failure:
        raise ReplayDownloadError(
            f"the {name!r} replay dataset could not be downloaded: {failure} "
            f"Compute nodes often have no network; download it on a login "
            f"node first, and it is kept at {destination}."
        ) from failure


def replay_key_specification() -> KeySpecification:
    """How the published replay datasets are labelled.

    With ase's own property names, ``energy``, ``forces`` and ``stress``, which
    ase reads back into the calculator rather than into the structure. The
    parser recovers them from there and says so, which is the right handling
    for a file this project did not write.
    """
    return KeySpecification.from_defaults().update(
        graph_keys={"energy": "energy", "stress": "stress"},
        atom_keys={"forces": "forces"},
    )


def read_curated(
    name: CuratedDataset, head: str, opener: Any = None
) -> list[Configuration]:
    """Every structure of a published replay dataset, filed under ``head``."""
    path = fetch(name, opener)
    # Named explicitly: a file adopted from the frozen tree's cache has no
    # suffix for a backend to be recognized by.
    backend = open_dataset(
        path,
        format="xyz",
        key_spec=replay_key_specification(),
        head=head,
        keep_isolated_atoms=True,
    )
    return list(backend.iter_range())
