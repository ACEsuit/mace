"""LES training through the normal staged pipeline, including energy foundations."""

import os
from importlib.util import find_spec

import numpy as np
import pytest
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_torch.calculators import MACECalculator
from mace_torch.cli.run_train import run
from mace_torch.deploy.loader import load_deployed
from test_mace_torch_full_batch import configuration

pytestmark = pytest.mark.les


@pytest.fixture(autouse=True)
def solver():
    if find_spec("les") is None:
        if "les" in os.environ.get("MACE_REQUIRE_CAPS", "").split(","):
            pytest.fail("MACE_REQUIRE_CAPS guarantees les")
        pytest.skip("requires the optional LES solver")


@fp64_only
@pytest.mark.parametrize("foundation", [False, True])
def test_native_les_trains_and_reloads(foundation, tmp_path):
    from ase import Atoms
    from ase.io import read

    template = configuration(tmp_path, max_num_epochs=1)
    doc = template.model_dump(mode="json")
    if foundation:
        pretrained = run(template)
        doc["finetune"]["foundation_model"] = str(pretrained.checkpoint_path)
        doc["runtime"]["work_dir"] = str(tmp_path / "tune")
    doc["model"].update(
        model="les",
        les={"use_dipole": True, "use_induced_charge": True, "output_scale": 1.0},
    )
    result = run(ResolvedConfig.model_validate(doc))
    assert result.checkpoint_path is not None
    calc = MACECalculator(result.checkpoint_path, compute_bec=True)
    atoms = read(tmp_path / "train.xyz", index=-1)
    assert isinstance(atoms, Atoms)
    atoms.calc = calc
    assert (
        np.isfinite(atoms.get_potential_energy())
        and np.isfinite(atoms.get_forces()).all()
    )
    assert calc.results["bec"].shape == (len(atoms), 2, 3, 3)
    assert np.max(np.abs(calc.results["LES_kappas"])) > 0
    stored = load_deployed(result.checkpoint_path)
    if foundation:
        assert stored.metadata.parents[0].role == "initial_weights"
    assert stored.config.model.model == "les"


@pytest.mark.parametrize("nonlinear", [False, True])
def test_tensor_readout_transforms_as_a_cartesian_tensor(nonlinear):
    import torch
    from conftest import assert_close
    from mace_torch.backends import ReferenceBackend
    from mace_torch.nn.les import TensorReadout

    dtype = "float64" if torch.get_default_dtype() == torch.float64 else "float32"
    readout = TensorReadout(
        ReferenceBackend(), "4x0e+3x1o+2x1e", dtype, nonlinear=nonlinear, positive=True
    )
    features = torch.randn(5, 19)
    expected = readout(features).detach()
    assert_close(expected, expected.transpose(1, 2))
    # A rotation and an improper rotation distinguish polar and axial vectors.
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    for transform in (rotation, -rotation):
        polar = features[:, 4:13].reshape(-1, 3, 3) @ transform.T
        axial = features[:, 13:].reshape(-1, 2, 3) @ transform.T * transform.det()
        rotated = torch.cat(
            [features[:, :4], polar.flatten(1), axial.flatten(1)], dim=1
        )
        assert_close(readout(rotated), transform @ expected @ transform.T)


@fp64_only
def test_les_force_training_resumes_from_its_optimizer_state(tmp_path):
    import torch
    from mace_torch.serialization import canonical_state

    doc = configuration(tmp_path).model_dump(mode="json")
    doc["model"].update(
        model="les", les={"use_dipole": True, "use_induced_charge": True}
    )

    def fit(directory, epochs, resume=False):
        doc["runtime"].update(work_dir=str(tmp_path / directory), restart_latest=resume)
        doc["training"]["max_num_epochs"] = epochs
        result = run(ResolvedConfig.model_validate(doc))
        assert result.checkpoint_path is not None
        assert all(np.isfinite(epoch.train_loss) for epoch in result.history)
        return canonical_state(load_deployed(result.checkpoint_path).engine)

    whole = fit("whole", 2)
    first = fit("split", 1)
    restarted = fit("split", 2, resume=True)
    assert whole.keys() == restarted.keys()
    assert any(
        not torch.equal(tensor, first[path][key])
        for path, weights in whole.items()
        if "latent_charges" in path
        for key, tensor in weights.items()
    )
    for path, weights in whole.items():
        assert weights.keys() == restarted[path].keys()
        for key, tensor in weights.items():
            assert torch.equal(tensor, restarted[path][key]), (path, key)
