"""A stock or broken GROMACS binary cannot satisfy the native capability."""

import subprocess
import pytest
from tests.conftest import _gromacs_available


@pytest.mark.parametrize(
    "code,banner,expected",
    [
        (0, "GROMACS version: 2026.3\nNN potential: disabled", False),
        (1, "MACE AOTInductor v1", False),
        (0, "GROMACS version: 2026.3\nNN potential: MACE AOTInductor v1", True),
    ],
)
def test_probe_requires_working_native_runtime(monkeypatch, code, banner, expected):
    monkeypatch.setenv("MACE_GROMACS_BINARY", "/test/gmx")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, code, "", banner),
    )
    assert _gromacs_available() is expected


def test_no_binary_is_optional(monkeypatch):
    monkeypatch.delenv("MACE_GROMACS_BINARY", raising=False)
    assert not _gromacs_available()
