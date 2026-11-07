"""Optional OpenMM loading cannot introduce a second model serialization path."""

import pickle
import sys
from pathlib import Path
from types import ModuleType

import pytest
from mace_torch.deploy.openmm import OpenMMEvaluator


def evaluator():
    root = Path(__file__).resolve().parents[3]
    return OpenMMEvaluator(root / "packages/mace-jax/tests/fixtures/tiny.json", [8, 1, 1])


def test_openmm_callback_requires_the_direct_interface(monkeypatch):
    monkeypatch.setitem(sys.modules, "openmmtorch", ModuleType("openmmtorch"))
    with pytest.raises(ImportError, match="PythonTorchForce"):
        evaluator().as_force()


def test_openmm_callback_is_rebuilt_from_the_neutral_bundle():
    with pytest.raises(TypeError, match="V1 bundle"):
        pickle.dumps(evaluator())
