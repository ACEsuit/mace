"""Getting an artifact's bytes onto disk: once, whole, and in one place.

An artifact is a file a model is built from, named by the URL it is published
at. :func:`resolve_artifact` turns a reference to one into a local path: a
local path is returned as it is, and a URL is looked up in the cache and
downloaded into it only if it is not there. :func:`download` is the only code
that writes into the cache.

Three properties hold for every artifact, whichever family it belongs to:

* **Atomic.** The body is streamed into a ``.part`` sibling and renamed into
  place only once it is complete. An interrupted download leaves nothing at
  the cache path, so it is fetched again next time rather than failing later
  as a truncated archive.
* **Never an error page.** A response whose content type is HTML is refused
  before a byte is written, naming the URL. A login or error page cached as a
  model would be read back on every later call.
* **At most once.** A cached artifact is read without any network access.

**The cache names.** An artifact is cached under the URL's file name prefixed
by a short digest of the URL, so two artifacts with the same file name do not
collide. The frozen tree cached under two other names, both still present in
every existing cache: the file name with everything but letters, digits and
underscores stripped, and the file name with its query cut off. A file under
either is adopted where it is, not fetched again and not moved, so the frozen
tree keeps finding it too.

Nothing here imports a framework: both implementations resolve their artifacts
through it.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DOWNLOAD_TIMEOUT",
    "ArtifactDownloadError",
    "ArtifactRef",
    "cache_dir",
    "cache_name",
    "download",
    "legacy_cache_names",
    "normalize_download_url",
    "resolve_artifact",
]

#: Seconds without a byte after which a download is abandoned. Per read, not
#: in total, so a large artifact on a slow link still arrives.
DOWNLOAD_TIMEOUT = 120.0

#: What one read asks for.
_BLOCK = 256 * 1024


class ArtifactDownloadError(RuntimeError):
    """An artifact could not be fetched, or what came back is not one."""


@dataclass(frozen=True)
class ArtifactRef:
    """An artifact, by the URL it is published at.

    Attributes:
        url: An ``https`` URL.
    """

    url: str

    def __post_init__(self) -> None:
        if urllib.parse.urlsplit(self.url).scheme != "https":
            raise ValueError(
                f"{self.url!r} is not an https URL. An artifact is published at "
                f"one; a file already on disk is passed as a path instead."
            )


def cache_dir(environ: Mapping[str, str] | None = None) -> Path:
    """``$XDG_CACHE_HOME/mace``, or ``~/.cache/mace`` when it is unset."""
    environ = os.environ if environ is None else environ
    root = environ.get("XDG_CACHE_HOME")
    return (Path(root) if root else Path.home() / ".cache") / "mace"


def normalize_download_url(url: str) -> str:
    """A GitHub file link rewritten to ``raw.githubusercontent.com``.

    ``github.com/{org}/{repo}/{blob,raw}/{ref}/{path}`` serves an HTML page
    or a redirect; the raw host serves the file. Everything else passes
    through unchanged, a release asset (``.../releases/download/...``) and
    any other host included. A query other than ``raw=true`` is kept.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "github.com":
        return url
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) < 5 or segments[2] not in {"blob", "raw"}:
        return url
    org, repo, _, ref = segments[:4]
    path = "/".join(segments[4:])
    normalized = f"https://raw.githubusercontent.com/{org}/{repo}/{ref}/{path}"
    if parts.query and parts.query != "raw=true":
        normalized = f"{normalized}?{parts.query}"
    return normalized


def _file_name(url: str) -> str:
    return os.path.basename(urllib.parse.urlsplit(url).path)


def cache_name(url: str) -> str:
    """The name ``url`` is cached under: a digest of it, then its file name."""
    digest = hashlib.sha256(normalize_download_url(url).encode()).hexdigest()
    return f"{digest[:12]}-{_file_name(url)}"


def legacy_cache_names(url: str) -> tuple[str, ...]:
    """The names the frozen tree cached ``url`` under, in the order checked.

    Its MACE-MP and MACE-Polar loaders kept the letters, digits and
    underscores of the URL's last component, query included; its MACE-OFF,
    OMOL and MDP loaders cut the query off and kept the rest.
    """
    last = os.path.basename(url)
    stripped = "".join(c for c in last if c.isalnum() or c == "_")
    plain = last.split("?")[0]
    return tuple(dict.fromkeys(name for name in (stripped, plain) if name))


Opener = Callable[..., Any]


def _progress(done: int, total: int) -> None:
    if total > 0:
        sys.stderr.write(
            f"\rDownloading: {min(100.0, done * 100 / total):.1f}% "
            f"({done / 2**20:.1f} MB / {total / 2**20:.1f} MB)"
        )
        sys.stderr.flush()


def download(
    url: str,
    destination: Path,
    *,
    timeout: float = DOWNLOAD_TIMEOUT,
    opener: Opener | None = None,
    progress: Callable[[int, int], None] | None = _progress,
) -> Path:
    """Fetch ``url`` into ``destination``, whole or not at all.

    Args:
        url: What to fetch. A GitHub file link is rewritten first.
        destination: Where it ends up.
        timeout: Seconds a single read may wait.
        opener: What opens the URL; ``urllib.request.urlopen`` unless given.
        progress: Called with the bytes received and the total, when the
            server says the total. ``None`` reports nothing.

    Raises:
        ArtifactDownloadError: The response is an HTML page, or the transfer
            failed. Nothing is left at ``destination`` either way.
    """
    target = normalize_download_url(url)
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = opener or urllib.request.urlopen
    total = 0
    try:
        with opener(target, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "") or ""
            if content_type.split(";")[0].strip().lower() == "text/html":
                raise ArtifactDownloadError(
                    f"{target} answered with an HTML page rather than the "
                    f"artifact, which is what an error or login page looks "
                    f"like. Check the URL; nothing was cached."
                )
            total = int(response.headers.get("Content-Length", 0) or 0)
            done = 0
            with open(partial, "wb") as out:
                while block := response.read(_BLOCK):
                    out.write(block)
                    done += len(block)
                    if progress is not None:
                        progress(done, total)
        os.replace(partial, destination)
        if progress is not None and total > 0:
            sys.stderr.write("\n")
    except ArtifactDownloadError:
        partial.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as failure:
        partial.unlink(missing_ok=True)
        raise ArtifactDownloadError(
            f"downloading {target} failed: {failure}. Nothing was cached; "
            f"the next attempt starts again."
        ) from failure
    return destination


def resolve_artifact(
    reference: str | Path | ArtifactRef,
    *,
    cache: Path | None = None,
    fetch: Callable[[str, Path], Path] = download,
) -> Path:
    """The local path of an artifact, downloading it at most once.

    Args:
        reference: A path to a file on disk, an ``https`` URL, or an
            :class:`ArtifactRef`.
        cache: Where artifacts are kept. :func:`cache_dir` unless given.
        fetch: What downloads a missing one, :func:`download` unless given.

    Raises:
        FileNotFoundError: ``reference`` is a path and no file is there.
        ArtifactDownloadError: It had to be downloaded and could not be.
    """
    if isinstance(reference, Path) or (
        isinstance(reference, str) and not reference.startswith("https:")
    ):
        path = Path(reference)
        if not path.is_file():
            raise FileNotFoundError(
                f"no artifact at {path}. Pass a file that exists, or the https "
                f"URL it is published at."
            )
        return path
    url = reference.url if isinstance(reference, ArtifactRef) else reference
    ArtifactRef(url)
    root = cache if cache is not None else cache_dir()
    for name in (cache_name(url), *legacy_cache_names(url)):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return fetch(url, root / cache_name(url))
