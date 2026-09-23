"""The published replay datasets: where they live, and reading them once.

A replay head trains on a sample of the data the foundation model was fitted
to, so the fine-tune does not forget it. Four such datasets are published
beside the foundation models, and a head names one by the name the schema
accepts.

**Downloaded once, into the same cache the frozen tree uses.** The file is kept
under ``$XDG_CACHE_HOME/mace``, or ``~/.cache/mace``, under exactly the name the
frozen tree gives it (``mace/tools/multihead_tools.py:166-170``): the URL's file
name with every character that is not alphanumeric or an underscore dropped.
The name reads oddly, ``mp_traj_combinedxyz``, and keeping it is the point: a
user who ran a legacy fine-tune has the file already and does not download
several hundred megabytes a second time.

Two differences from the frozen tree. The download is written beside the
cache and moved into place only once it is complete, so an interrupted one
does not leave a truncated file that every later run then reads as the
dataset. And nothing is written to the working directory: legacy subselects
into ``mp_finetuning-<tag>.xyz`` there and reads it back, which is a file
another run in the same directory overwrites.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

from mace_core.config.data import CuratedDataset
from mace_core.data import Configuration, KeySpecification, open_dataset

__all__ = [
    "CURATED_URLS",
    "ReplayDownloadError",
    "cache_directory",
    "cached_path",
    "fetch",
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

#: Downloads a URL to a path and returns its response headers as text.
Downloader = Callable[[str, str], str]


class ReplayDownloadError(RuntimeError):
    """A replay dataset that could not be fetched."""


def cache_directory() -> Path:
    """``$XDG_CACHE_HOME/mace``, or ``~/.cache/mace`` when it is unset."""
    root = os.environ.get("XDG_CACHE_HOME")
    return (Path(root) if root else Path.home() / ".cache") / "mace"


def cached_path(name: CuratedDataset) -> Path:
    """Where the dataset is kept, under the frozen tree's own file name."""
    basename = os.path.basename(CURATED_URLS[name])
    return cache_directory() / "".join(
        character for character in basename if character.isalnum() or character == "_"
    )


def _urlretrieve(url: str, path: str) -> str:
    _, headers = urllib.request.urlretrieve(url, path)
    return str(headers)


def fetch(name: CuratedDataset, download: Downloader = _urlretrieve) -> Path:
    """The dataset's local path, downloading it the first time.

    Args:
        name: One of :data:`CURATED_URLS`.
        download: Fetches a URL to a path. The network by default; a test
            hands in a function that writes a file.

    Raises:
        ReplayDownloadError: If the download fails, or answers with a web page
            rather than a file, which is what a moved release asset returns
            with a success status.
    """
    path = cached_path(name)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    url = CURATED_URLS[name]
    logger.info("Downloading the %r replay dataset from %s", name, url)
    try:
        headers = download(url, str(partial))
    except Exception as failure:
        partial.unlink(missing_ok=True)
        raise ReplayDownloadError(
            f"the {name!r} replay dataset could not be downloaded from {url}: "
            f"{failure}. Compute nodes often have no network; download it on a "
            f"login node first, and it is kept at {path}."
        ) from failure
    if "text/html" in headers:
        partial.unlink(missing_ok=True)
        raise ReplayDownloadError(
            f"{url} answered with a web page rather than the {name!r} dataset. "
            f"The release asset may have moved."
        )
    partial.replace(path)
    return path


def replay_key_specification() -> KeySpecification:
    """How the published replay datasets are labelled.

    With ase's own property names, ``energy``, ``forces`` and ``stress``, which
    ase reads back into the calculator rather than into the structure. The
    parser recovers them from there and says so, which is the right handling
    for a file this project did not write.
    """
    return KeySpecification.from_defaults().update(
        info_keys={"energy": "energy", "stress": "stress"},
        arrays_keys={"forces": "forces"},
    )


def read_curated(
    name: CuratedDataset, head: str, download: Downloader = _urlretrieve
) -> list[Configuration]:
    """Every structure of a published replay dataset, filed under ``head``."""
    path = fetch(name, download)
    # Named explicitly: the cache file keeps the frozen tree's name, which has
    # no suffix for a backend to be recognized by.
    backend = open_dataset(
        path,
        format="xyz",
        key_spec=replay_key_specification(),
        head=head,
        keep_isolated_atoms=True,
    )
    return list(backend.iter_range())
