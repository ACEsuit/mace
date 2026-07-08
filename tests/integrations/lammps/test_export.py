"""Export CLI contract: mace_create_lammps_model produces loadable artifacts.

Contract tier — no LAMMPS binary. The libtorch format is plain TorchScript;
the mliap format additionally converts the model to the cueq layout (hence the
cueq capability marker on that test).
"""

import shutil

import pytest
import torch

from tests.helpers import REPO_ROOT, run_mace_train
from tests.integrations.lammps._harness import lammps_style_cluster

CREATE_LAMMPS_MODEL = REPO_ROOT / "mace" / "cli" / "create_lammps_model.py"


@pytest.fixture(name="model_copy")
def fixture_model_copy(trained_tiny_model_path, tmp_path):
    # The CLI writes its artifact NEXT TO the input model; copy into tmp_path
    # so parallel tests do not race on the session-scoped model directory.
    dest = tmp_path / "tiny.model"
    shutil.copy(trained_tiny_model_path, dest)
    return dest


def test_export_libtorch(model_copy):
    run_mace_train({}, extra_argv=[str(model_copy)], script=CREATE_LAMMPS_MODEL)
    artifact = model_copy.parent / (model_copy.name + "-lammps.pt")
    assert artifact.exists(), "libtorch export artifact missing"

    torch.set_default_dtype(torch.float64)
    loaded = torch.jit.load(artifact, map_location="cpu")
    eager = torch.load(model_copy, map_location="cpu").double()
    batch, local_or_ghost, _, _ = lammps_style_cluster(eager, n_repeat=3)
    out = loaded(batch, local_or_ghost)
    assert out["total_energy_local"] is not None
    assert torch.isfinite(out["total_energy_local"]).all()
    assert torch.isfinite(out["forces"]).all()


def test_export_libtorch_float32(model_copy):
    run_mace_train(
        {"dtype": "float32"},
        extra_argv=[str(model_copy)],
        script=CREATE_LAMMPS_MODEL,
    )
    artifact = model_copy.parent / (model_copy.name + "-lammps.pt")
    assert artifact.exists()
    loaded = torch.jit.load(artifact, map_location="cpu")
    assert next(loaded.parameters()).dtype == torch.float32


@pytest.mark.cueq
def test_export_mliap(model_copy):
    run_mace_train(
        {"format": "mliap"},
        extra_argv=[str(model_copy)],
        script=CREATE_LAMMPS_MODEL,
    )
    artifact = model_copy.parent / (model_copy.name + "-mliap_lammps.pt")
    assert artifact.exists(), "mliap export artifact missing"
    # mliap artifacts are pickled modules (not TorchScript). LAMMPS_MLIAP_MACE
    # wraps the converted model as .model (MACEEdgeForcesWrapper) whose .model
    # carries the flag the ML-IAP runtime dispatches on.
    loaded = torch.load(artifact, map_location="cpu")
    assert loaded.element_types, "element_types must be populated for ML-IAP"
    assert getattr(loaded.model.model, "lammps_mliap", False) is True
