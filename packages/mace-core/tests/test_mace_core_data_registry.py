"""Resolving a dataset to its backend: by name, by sniffing, never by guessing.

Third-party backends are installed as real distributions in a temporary
directory on ``sys.path``, so discovery goes through entry point metadata
exactly as it does for a wheel.
"""

from __future__ import annotations

import sys
import textwrap

import ase.io
import pytest
from ase import Atoms
from mace_core.data import (
    DataBackendNotAvailableError,
    DuplicateDataBackendError,
    KeySpecification,
    available_data_backends,
    open_dataset,
)
from mace_core.data.backends.xyz import XYZBackend

KEY_SPEC = KeySpecification.from_defaults()


def _install(path, monkeypatch, name, module_source, entry_point):
    """A module and a distribution registering one dataset backend."""
    module = f"{name}_dataset"
    (path / f"{module}.py").write_text(textwrap.dedent(module_source))
    info = path / f"{module}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {module}\nVersion: 1.0\n"
    )
    (info / "entry_points.txt").write_text(
        f"[mace.data_backends]\n{entry_point.format(module=module)}\n"
    )
    monkeypatch.syspath_prepend(str(path))


@pytest.fixture(autouse=True)
def _forget_installed_modules():
    """Drop the modules a test installed, so no later test imports a stale one."""
    before = set(sys.modules)
    yield
    for module in set(sys.modules) - before:
        if module.endswith("_dataset"):
            del sys.modules[module]


DUMMY = """
    from mace_core.data import InMemoryBackend

    class DummyBackend(InMemoryBackend):
        @classmethod
        def sniff(cls, source):
            return source.endswith(".dummy")

        @classmethod
        def open(cls, source, *, key_spec, head="Default", **options):
            return cls([], source=source)
"""


def _xyz(tmp_path) -> str:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    atoms.info["REF_energy"] = -1.0
    path = tmp_path / "train.xyz"
    ase.io.write(path, [atoms])
    return str(path)


def test_a_path_nothing_claims_is_an_error_not_a_fall_through(tmp_path):
    with pytest.raises(DataBackendNotAvailableError, match="no dataset backend"):
        open_dataset(tmp_path / "typo.xzy", key_spec=KEY_SPEC)


def test_a_path_two_backends_claim_is_an_error_naming_both(tmp_path, monkeypatch):
    _install(
        tmp_path,
        monkeypatch,
        "greedy",
        DUMMY.replace('source.endswith(".dummy")', "True"),
        "greedy = {module}:DummyBackend",
    )
    with pytest.raises(DataBackendNotAvailableError, match=r"'greedy', 'xyz'"):
        open_dataset(_xyz(tmp_path), key_spec=KEY_SPEC)


def test_an_unknown_format_lists_the_registered_ones():
    with pytest.raises(
        DataBackendNotAvailableError, match=r"registered names are .*'xyz'"
    ):
        open_dataset("train.xyz", format="zarr", key_spec=KEY_SPEC)


def test_a_backend_installed_beside_mace_core_is_discovered_and_opened(
    tmp_path, monkeypatch
):
    _install(tmp_path, monkeypatch, "dummy", DUMMY, "dummy = {module}:DummyBackend")
    assert "dummy" in {d.name for d in available_data_backends()}
    backend = open_dataset("train.dummy", key_spec=KEY_SPEC)
    assert type(backend).__name__ == "DummyBackend" and len(backend) == 0


def test_a_backend_that_fails_to_import_breaks_nothing_else(tmp_path, monkeypatch):
    _install(
        tmp_path,
        monkeypatch,
        "broken",
        "raise ImportError('needs a library that is not installed')",
        "broken = {module}:Backend",
    )
    listed = {d.name: d for d in available_data_backends()}
    assert not listed["broken"].loaded
    assert "not installed" in listed["broken"].reason

    # By name: only the named backend is imported, so the broken one is never
    # touched. By sniffing: it is listed as unavailable, and xyz still wins.
    assert isinstance(
        open_dataset(_xyz(tmp_path), format="xyz", key_spec=KEY_SPEC), XYZBackend
    )
    assert isinstance(open_dataset(_xyz(tmp_path), key_spec=KEY_SPEC), XYZBackend)
    with pytest.raises(DataBackendNotAvailableError, match="not installed"):
        open_dataset("train.xyz", format="broken", key_spec=KEY_SPEC)
    with pytest.raises(DataBackendNotAvailableError, match=r"not usable: \[.broken"):
        open_dataset(tmp_path / "typo.xzy", key_spec=KEY_SPEC)


@pytest.mark.parametrize(
    ("module_source", "reason"),
    [
        ("def Backend():\n    pass\n", "is not a dataset backend"),
        (
            DUMMY.replace('return source.endswith(".dummy")', "raise OSError('stat')"),
            "sniff raised OSError",
        ),
    ],
    ids=["not-a-backend", "sniff-raises"],
)
def test_a_backend_that_cannot_answer_does_not_break_sniffing(
    tmp_path, monkeypatch, module_source, reason
):
    target = "Backend" if "def Backend" in module_source else "DummyBackend"
    _install(
        tmp_path, monkeypatch, "faulty", module_source, f"faulty = {{module}}:{target}"
    )
    assert isinstance(open_dataset(_xyz(tmp_path), key_spec=KEY_SPEC), XYZBackend)
    with pytest.raises(DataBackendNotAvailableError, match=reason):
        open_dataset(tmp_path / "typo.xzy", key_spec=KEY_SPEC)


def test_a_backend_whose_open_returns_no_backend_is_refused(tmp_path, monkeypatch):
    _install(
        tmp_path,
        monkeypatch,
        "hollow",
        DUMMY.replace("return cls([], source=source)", "return None"),
        "hollow = {module}:DummyBackend",
    )
    with pytest.raises(DataBackendNotAvailableError, match="returned NoneType"):
        open_dataset("train.dummy", key_spec=KEY_SPEC)


def test_auto_is_not_a_name_a_backend_can_take(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, "squatter", DUMMY, "auto = {module}:DummyBackend")
    listed = {d.name: d for d in available_data_backends()}
    assert not listed["auto"].loaded and "reserved" in listed["auto"].reason


def test_resolving_a_name_imports_no_other_backend(tmp_path, monkeypatch):
    _install(
        tmp_path,
        monkeypatch,
        "tripwire",
        "raise AssertionError('imported while resolving another backend')",
        "tripwire = {module}:Backend",
    )
    open_dataset(_xyz(tmp_path), format="xyz", key_spec=KEY_SPEC)
    assert "tripwire_dataset" not in sys.modules


def test_two_backends_under_one_name_are_refused(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, "rival", DUMMY, "xyz = {module}:DummyBackend")
    with pytest.raises(DuplicateDataBackendError, match="from rival_dataset"):
        available_data_backends()
    with pytest.raises(DuplicateDataBackendError, match="'xyz'"):
        open_dataset("train.xyz", format="xyz", key_spec=KEY_SPEC)


def test_the_in_memory_backend_is_not_registered():
    assert "memory" not in {d.name for d in available_data_backends()}
