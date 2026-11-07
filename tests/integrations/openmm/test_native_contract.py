"""The native OpenMM capability probes the actual linked executable."""

import subprocess
import pytest
from tests.conftest import _native_openmm_available


@pytest.mark.parametrize(
    "code, banner, expected",
    [
        (0, "mace-openmm-native-v1\n", True),
        (1, "mace-openmm-native-v1\n", False),
        (0, "other executable\n", False),
    ],
)
def test_native_probe(monkeypatch, code, banner, expected):
    monkeypatch.setenv("MACE_OPENMM_DRIVER", "/test/driver")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, code, banner, ""),
    )
    assert _native_openmm_available() == expected


def test_native_probe_without_binary(monkeypatch):
    monkeypatch.delenv("MACE_OPENMM_DRIVER", raising=False)
    assert not _native_openmm_available()
