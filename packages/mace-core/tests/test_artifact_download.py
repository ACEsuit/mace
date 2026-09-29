"""Fetching an artifact: whole or not at all, never an error page, and once.

No network: the server is a stand-in handed to the download as its opener,
except in the one test that fetches a real artifact, which runs only when
``MACE_CI_ALLOW_NETWORK=1``.
"""

from __future__ import annotations

import os
import urllib.request

import pytest
from mace_core import artifacts
from mace_core.artifacts import (
    ArtifactDownloadError,
    ArtifactRef,
    cache_name,
    download,
    legacy_cache_names,
    normalize_download_url,
    resolve_artifact,
)
from mace_core_artifact_urls import FAMILIES, OFF_MEDIUM

BODY = b"a model, in several blocks " * 40000


class Response:
    """What ``urlopen`` returns, reduced to what a download reads."""

    def __init__(
        self, body=BODY, content_type="application/octet-stream", fail_after=None
    ):
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
        self._body = body
        self._offset = 0
        self._fail_after = fail_after

    def read(self, size):
        if self._fail_after is not None and self._offset >= self._fail_after:
            raise OSError("connection reset by peer")
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Server:
    """Answers every request with the next of ``responses``, and records it."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requested: list[str] = []

    def __call__(self, url, timeout):
        self.requested.append(url)
        return self.responses.pop(0)


def fetcher(server):
    def fetch(url, destination):
        return download(url, destination, opener=server, progress=None)

    return fetch


# ---------------------------------------------------------------------------
# The link
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model?raw=true",
            "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_small.model",
        ),
        (
            "https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_medium.model?raw=true",
            "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_medium.model",
        ),
        (
            "https://example.com/model.pt?download=1",
            "https://example.com/model.pt?download=1",
        ),
        (
            "https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-S.model",
            "https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-S.model",
        ),
    ],
    ids=["blob", "raw", "other-host", "release-asset"],
)
def test_a_github_file_link_is_fetched_from_the_raw_host(url, expected):
    assert normalize_download_url(url) == expected


@pytest.mark.parametrize(
    "size",
    ["small", "medium", "large", "medium-as-link"],
)
def test_each_off_model_is_fetched_once_from_the_raw_host(tmp_path, size):
    """The frozen tree's four MACE-OFF resolution cases, without its loader:
    which name means which URL is the registry's, and what is pinned here is
    that the URL is fetched from the raw host and cached where it says."""
    name = "medium" if size == "medium-as-link" else size
    raw = (
        "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/"
        f"MACE-OFF23_{name}.model"
    )
    url = raw
    if size == "medium-as-link":
        url = (
            "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/"
            "MACE-OFF23_medium.model?raw=true"
        )
    server = Server(Response())
    path = resolve_artifact(url, cache=tmp_path, fetch=fetcher(server))
    assert server.requested == [raw]
    assert path == tmp_path / cache_name(url)
    assert path.read_bytes() == BODY


# ---------------------------------------------------------------------------
# Whole or not at all
# ---------------------------------------------------------------------------


def test_an_interrupted_download_leaves_nothing_and_the_next_one_succeeds(tmp_path):
    destination = tmp_path / cache_name(OFF_MEDIUM)
    broken = Server(Response(fail_after=256 * 1024))
    with pytest.raises(ArtifactDownloadError, match="Nothing was cached"):
        download(OFF_MEDIUM, destination, opener=broken, progress=None)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []

    path = resolve_artifact(
        OFF_MEDIUM, cache=tmp_path, fetch=fetcher(Server(Response()))
    )
    assert path.read_bytes() == BODY
    assert sorted(p.name for p in tmp_path.iterdir()) == [cache_name(OFF_MEDIUM)]


def test_a_timeout_is_reported_as_a_failed_download(tmp_path):
    def silent(url, timeout):
        raise TimeoutError(f"no byte in {timeout} s")

    with pytest.raises(ArtifactDownloadError, match="no byte in"):
        download(OFF_MEDIUM, tmp_path / "x.model", opener=silent, progress=None)
    assert list(tmp_path.iterdir()) == []


def test_the_per_read_timeout_is_passed_to_the_connection(tmp_path):
    seen = {}

    def opener(url, timeout):
        seen["timeout"] = timeout
        return Response()

    download(OFF_MEDIUM, tmp_path / "x.model", opener=opener, progress=None)
    assert seen["timeout"] == artifacts.DOWNLOAD_TIMEOUT


def test_progress_is_reported_as_it_arrives(tmp_path):
    seen = []
    download(
        OFF_MEDIUM,
        tmp_path / "x.model",
        opener=Server(Response()),
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (len(BODY), len(BODY))
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)


# ---------------------------------------------------------------------------
# Never an error page
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("content_type", ["text/html", "text/html; charset=utf-8"])
def test_an_html_page_is_refused_for_every_family(tmp_path, family, content_type):
    url = FAMILIES[family]
    server = Server(Response(b"<html>Sign in</html>", content_type=content_type))
    with pytest.raises(ArtifactDownloadError, match="HTML") as refused:
        resolve_artifact(url, cache=tmp_path, fetch=fetcher(server))
    assert normalize_download_url(url) in str(refused.value)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# At most once
# ---------------------------------------------------------------------------


def test_a_cached_artifact_is_read_with_no_network(tmp_path, monkeypatch):
    first = resolve_artifact(
        OFF_MEDIUM, cache=tmp_path, fetch=fetcher(Server(Response()))
    )

    def no_network(*args, **kwargs):
        raise AssertionError("the network was touched for a cached artifact")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert resolve_artifact(OFF_MEDIUM, cache=tmp_path) == first
    assert resolve_artifact(ArtifactRef(OFF_MEDIUM), cache=tmp_path) == first


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_file_under_a_legacy_name_is_adopted_where_it_is(tmp_path, family):
    """Where it is, not moved: the frozen tree still looks there."""
    url = FAMILIES[family]
    for name in legacy_cache_names(url):
        cache = tmp_path / name.replace("/", "_")
        cache.mkdir()
        existing = cache / name
        existing.write_bytes(b"already here")

        def refuse(url, destination, existing=existing):
            raise AssertionError(f"{url} was fetched though {existing} exists")

        assert resolve_artifact(url, cache=cache, fetch=refuse) == existing
        assert sorted(p.name for p in cache.iterdir()) == [name]


def test_the_cache_follows_xdg_cache_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    path = resolve_artifact(OFF_MEDIUM, fetch=fetcher(Server(Response())))
    assert path == tmp_path / "mace" / cache_name(OFF_MEDIUM)


def test_a_local_path_is_used_as_it_is(tmp_path):
    local = tmp_path / "mine.model"
    local.write_bytes(b"x")
    assert resolve_artifact(local) == local
    assert resolve_artifact(str(local)) == local
    with pytest.raises(FileNotFoundError, match="no artifact at"):
        resolve_artifact(tmp_path / "absent.model")


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("MACE_CI_ALLOW_NETWORK") != "1",
    reason="downloads are opt-in: set MACE_CI_ALLOW_NETWORK=1",
)
def test_a_published_artifact_downloads_once(tmp_path, monkeypatch):
    url = (
        "https://github.com/ACEsuit/mace-off/blob/main/mace_off23/"
        "MACE-OFF23_small.model?raw=true"
    )
    path = resolve_artifact(url, cache=tmp_path)
    assert path.stat().st_size > 1_000_000
    with path.open("rb") as handle:
        assert handle.read(2) == b"PK", "a torch checkpoint is a zip archive"

    def no_network(*args, **kwargs):
        raise AssertionError("the network was touched for a cached artifact")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert resolve_artifact(url, cache=tmp_path) == path
