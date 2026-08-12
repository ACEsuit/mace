"""The published foundation checkpoints this suite pins, and how it reaches them.

A golden for a foundation model is only worth something if the file it was
taken from is the file the test loads. Everything here exists to make that
identity checkable rather than assumed:

* the **loader call** is data, not prose. ``loader``/``loader_kwargs`` are
  what the reference was generated with and what the test replays, so the two
  cannot drift apart -- in particular the dtype and device, which matter
  because ``mace_mp`` defaults to ``float32`` and to CUDA-if-present;
* the **origin** is either a path in this repository or the URL the package
  downloads from, and for the downloaded tiers a test asserts it is still the
  URL ``foundations_models.py`` holds. An upstream re-pointing of an alias is
  then a failure that says so, instead of a reference silently describing a
  different model;
* the **digest** is recorded and checked. `sha256` pins the bytes, so a
  re-uploaded artifact at the same URL fails with "the artifact changed"
  rather than with an unexplained numeric drift;
* which file was read is **observed**, not reconstructed. ``load_calculator``
  watches ``torch.load`` while the loader runs, so the digest is taken from
  the file the calculator actually opened. Recomputing the cache path here
  would be a second copy of ``mace_off``'s naming rule, and a copy that goes
  stale is exactly the failure this module is supposed to catch.

The registry is imported by ``tests/golden/test_foundation_goldens.py`` and by
``tests/golden/regenerate.py``; both go through it so the reference and the
assertion cannot describe different artifacts.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

TRACKED_MPA0 = "mace/calculators/foundations_models/mace-mpa-0-medium.model"
_TRACKED_MPA0 = TRACKED_MPA0
_MP_SMALL_URL = (
    "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/"
    "2023-12-10-mace-128-L0_energy_epoch-249.model"
)
_OFF_SMALL_URL = (
    "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/"
    "MACE-OFF23_small.model"
)


@dataclass(frozen=True)
class FoundationArtifact:
    """One published checkpoint, its loader call, and its committed reference."""

    name: str
    #: the ``mace.calculators.foundations_models`` entry point to call
    loader: str
    #: the exact keyword arguments the reference was generated with
    loader_kwargs: Dict[str, object]
    #: file name under ``tests/golden/references/``
    reference: str
    #: sha256 of the checkpoint the reference was generated from
    sha256: str
    #: repository-relative path (tracked tier) or download URL (network tier)
    origin: str
    #: True when reaching the artifact requires a download
    network: bool
    #: manifest tags selecting the fixtures this model is evaluated on;
    #: empty means the whole set
    fixture_tags: Tuple[str, ...]
    description: str
    #: only for the network tier: the name of the dict in
    #: foundations_models.py that must still hold ``origin``, and the key
    url_table: Tuple[str, str] = field(default=("", ""))
    #: where the same bytes are published, when the artifact is a local file.
    #: Recorded so a consumer that reaches the model by URL (a converter, a
    #: GPU parity run) can tell it is looking at this reference's artifact.
    release_url: str = ""


ARTIFACTS: Dict[str, FoundationArtifact] = {
    # The only published artifact that can be pinned on every pull request:
    # it is in the tree, `download_mace_mp_checkpoint` short-circuits to it
    # for `model=None`, and it is therefore what an unqualified `mace_mp()`
    # hands a user -- the highest-traffic model in the project.
    "mpa0_medium": FoundationArtifact(
        name="mpa0_medium",
        loader="mace_mp",
        loader_kwargs={"default_dtype": "float64", "device": "cpu"},
        reference="mpa0_medium_e3nn_cpu_fp64.json",
        sha256="75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638",
        origin=_TRACKED_MPA0,
        network=False,
        fixture_tags=(),
        description=(
            "MACE-MPA-0 medium, the checkpoint tracked in this repository and "
            "the model an unqualified mace_mp() returns. No download; the "
            "release artifact at release_url has the identical digest, so a "
            "consumer that fetches it by URL is looking at this same file."
        ),
        release_url=(
            "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/"
            "mace-mpa-0-medium.model"
        ),
    ),
    "mp_small": FoundationArtifact(
        name="mp_small",
        loader="mace_mp",
        loader_kwargs={
            "model": "small",
            "default_dtype": "float64",
            "device": "cpu",
        },
        reference="mp_small_e3nn_cpu_fp64.json",
        sha256="2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736",
        origin=_MP_SMALL_URL,
        network=True,
        fixture_tags=(),
        description=(
            "MACE-MP-0 small (2023-12-10-mace-128-L0_energy_epoch-249), the "
            "pre-3.10 small tier, downloaded from the mace-mp release."
        ),
        url_table=("mace_mp_urls", "small"),
    ),
    "off_small": FoundationArtifact(
        name="off_small",
        loader="mace_off",
        loader_kwargs={
            "model": "small",
            "default_dtype": "float64",
            "device": "cpu",
        },
        reference="off_small_e3nn_cpu_fp64.json",
        sha256="165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f",
        origin=_OFF_SMALL_URL,
        network=True,
        # An organic-chemistry model has no business being asked for the
        # stress of a slab: the molecular tag is how the fixture set is
        # narrowed to what this model was fitted for.
        fixture_tags=("molecular",),
        description=(
            "MACE-OFF23 small, organic chemistry (H C N O F P S Cl Br I), "
            "downloaded from the mace-off repository. ASL licensed."
        ),
        url_table=("mace_off_urls", "small"),
    ),
}

#: The tracked ANI checkpoint is deliberately absent; see
#: test_foundation_goldens.py::test_the_tracked_anicc_checkpoint_cannot_be
#: _loaded_on_cpu for the measurement that says why.
ANICC_TRACKED_PATH = "mace/calculators/foundations_models/ani500k_large_CC.model"


def sha256_of(path: Path) -> str:
    """Digest of a checkpoint, streamed -- these files reach 80 MB."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def _watch_torch_load() -> Iterator[List[str]]:
    """Record every path ``torch.load`` is handed while the block runs."""
    import torch  # pylint: disable=import-outside-toplevel

    seen: List[str] = []
    real = torch.load

    def spy(*args, **kwargs):
        target = kwargs.get("f", args[0] if args else None)
        if isinstance(target, (str, Path)):
            seen.append(str(target))
        return real(*args, **kwargs)

    torch.load = spy
    try:
        yield seen
    finally:
        torch.load = real


class NetworkAccessRefused(AssertionError):
    """Raised by :func:`no_network` when something inside it tries to download."""


@contextlib.contextmanager
def no_network() -> Iterator[None]:
    """Make any download attempt inside the block raise instead of succeed.

    The tracked tier's claim is not "it happens to be cached here" but "it
    never reaches the network", and that is the difference between a golden
    that runs in a fresh offline clone and one that runs on the machine that
    downloaded it once. Asserting it costs two patches; assuming it costs a
    red nightly on the day the short-circuit in
    ``download_mace_mp_checkpoint`` stops firing.
    """
    import urllib.request  # pylint: disable=import-outside-toplevel

    from mace.calculators import (  # pylint: disable=import-outside-toplevel
        foundations_models,
    )

    def refuse(*args, **kwargs):  # pylint: disable=unused-argument
        raise NetworkAccessRefused(
            "this evaluation reached the network; the tracked tier must load "
            "the checkpoint committed in this repository and nothing else"
        )

    real_urlopen = urllib.request.urlopen
    real_retrieve = foundations_models._urlretrieve_with_timeout  # noqa: SLF001  # pylint: disable=protected-access
    urllib.request.urlopen = refuse
    foundations_models._urlretrieve_with_timeout = refuse  # noqa: SLF001  # pylint: disable=protected-access
    try:
        yield
    finally:
        urllib.request.urlopen = real_urlopen
        foundations_models._urlretrieve_with_timeout = real_retrieve  # noqa: SLF001  # pylint: disable=protected-access


@contextlib.contextmanager
def tracked_checkpoint_in_place() -> Iterator[None]:
    """Make the tracked checkpoint reachable where the package looks for it.

    The checkpoints under ``mace/calculators/foundations_models/`` are tracked
    in git but **not packaged**: ``setup.cfg`` declares no ``package_data``
    and ``MANIFEST.in`` carries only ``py.typed``, so the published wheel is
    ~300 KB and contains ``foundations_models.py`` without the directory of
    the same name. CI installs the wheel (``pip install ".[dev]"`` in
    ``.github/actions/setup-mace``), so in the very job this golden runs in,
    ``local_model_path`` points at a file that does not exist,
    ``download_mace_mp_checkpoint(None)`` stops short-circuiting, and an
    unqualified ``mace_mp()`` downloads MPA-0 from the release — on a job that
    has no network opt-in and must not need one.

    So the module's path is pointed at the copy in the checkout, which is the
    same artifact (the digest check proves it) in the place the package
    expects. This is a property of how MACE is packaged, not of this test: if
    the wheel ever ships the checkpoint, the branch below simply never fires.
    """
    from mace.calculators import (  # pylint: disable=import-outside-toplevel
        foundations_models,
    )

    installed = Path(foundations_models.local_model_path)
    in_checkout = REPO_ROOT / TRACKED_MPA0
    if installed.is_file() or not in_checkout.is_file():
        yield
        return
    foundations_models.local_model_path = str(in_checkout)
    try:
        yield
    finally:
        foundations_models.local_model_path = str(installed)


def load_calculator(spec: FoundationArtifact):
    """Build the calculator for ``spec`` and report the file it read.

    Returns ``(calculator, checkpoint_path)``. The path is observed from the
    loader's own ``torch.load`` call rather than recomputed, so the digest the
    caller then takes is of the bytes that produced the numbers.
    """
    from mace.calculators import (  # pylint: disable=import-outside-toplevel
        foundations_models,
    )

    loader = getattr(foundations_models, spec.loader)
    with tracked_checkpoint_in_place(), _watch_torch_load() as loaded:
        calc = loader(**spec.loader_kwargs)
    # `.pt` is e3nn's Wigner table and anything else torch may pull in on
    # first use; a checkpoint is whatever else the loader opened.
    checkpoints = {path for path in loaded if not path.endswith(".pt")}
    if len(checkpoints) != 1:
        raise AssertionError(
            f"{spec.name}: expected the loader to read exactly one checkpoint, "
            f"it read {sorted(checkpoints)}"
        )
    return calc, Path(checkpoints.pop())


def expected_origin_url(spec: FoundationArtifact) -> str:
    """The URL ``foundations_models.py`` currently holds for a network tier."""
    from mace.calculators import (  # pylint: disable=import-outside-toplevel
        foundations_models,
    )

    table_name, key = spec.url_table
    return getattr(foundations_models, table_name)[key]
