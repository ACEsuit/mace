"""Network-only historical extraction stays in its own pinned interpreter."""

import os
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def legacy_python(tmp_path_factory):
    from mace_torch.deploy.legacy import pinned_environment

    configured = os.environ.get("MACE_LEGACY_PYTHON")
    if configured:
        interpreter = Path(configured).resolve()
        if not interpreter.is_file():
            pytest.fail(f"MACE_LEGACY_PYTHON does not exist: {interpreter}")
        return interpreter
    return pinned_environment(tmp_path_factory.mktemp("legacy-converter"))


@pytest.fixture(name="regression_set", scope="session")
def fixture_regression_set():
    path = Path(__file__).resolve().parents[1] / "golden/datasets/regression_train.xyz"
    assert path.is_file()
    return path
