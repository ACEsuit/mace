"""Energy-only fitting through training, recovery and saved-model consumers."""

from __future__ import annotations

import copy

import ase.io
import pytest
import torch

from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.tools.checkpoint import CheckpointHandler, CheckpointState
from mace.tools.train import take_step
from tests.golden.harness import FP64_CPU_REFERENCE as TOL
from tests.helpers import run_mace_train


@pytest.fixture
def energy_data(tmp_path, regression_set):
    frames = [
        atoms.copy()
        for atoms in ase.io.read(regression_set, ":")
        if atoms.info.get("config_type") == "molecule"
    ][:8]
    for atoms in frames:
        atoms.arrays.pop("REF_forces", None)
    train = tmp_path / "train.xyz"
    valid = tmp_path / "valid.xyz"
    ase.io.write(train, frames[:6])
    ase.io.write(valid, frames[6:])
    return train, valid


def run_energy_training(work, anchor, dataset, loss, compute_forces, **overrides):
    work.mkdir(exist_ok=True)
    train, valid = dataset
    params = {
        "name": "energy",
        "train_file": str(train),
        "valid_file": str(valid),
        "foundation_model": str(anchor),
        "multiheads_finetuning": False,
        "E0s": "foundation",
        "loss": loss,
        "energy_weight": 1.0,
        "forces_weight": 0.0,
        "compute_forces": compute_forces,
        "scaling": "no_scaling",
        "lr": 0.001,
        "batch_size": 3,
        "valid_batch_size": 2,
        "max_num_epochs": 2,
        "eval_interval": 1,
        "device": "cpu",
        "default_dtype": "float64",
        "seed": 11,
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "error_table": "PerAtomRMSE",
        "plot": False,
        "num_workers": 0,
        "save_cpu": None,
    }
    for key in ("model_dir", "checkpoints_dir", "results_dir", "log_dir", "work_dir"):
        params[key] = str(work)
    params.update(overrides)
    result = run_mace_train(
        params, check=False, capture_output=True, text=True, cwd=work
    )
    (work / "last-run.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


@pytest.mark.timeout(300)
@pytest.mark.parametrize("loss", ["weighted", "l1l2energyforces"])
@pytest.mark.parametrize("stage_two", [False, True])
def test_energy_only_cli_training_and_resume(
    tmp_path, anchor_scaleshift, energy_data, loss, stage_two
):
    options = {}
    if stage_two:
        options = {
            "stage_two": None,
            "start_stage_two": 1,
            "stage_two_forces_weight": 0.0,
        }
    directories = [tmp_path / "control", tmp_path / "energy-only"]
    for resume in (False, True):
        if resume:
            options.update(restart_latest=None, max_num_epochs=3)
        for work, compute_forces in zip(directories, (True, False)):
            result = run_energy_training(
                work, anchor_scaleshift, energy_data, loss, compute_forces, **options
            )
            if result.returncode:
                pytest.fail(result.stdout + result.stderr)
        control_checkpoints = sorted(directories[0].glob("*.pt"))
        if not control_checkpoints:
            pytest.fail("Training produced no checkpoint")
        for checkpoint in control_checkpoints:
            control = torch.load(checkpoint, map_location="cpu", weights_only=False)
            actual = torch.load(
                directories[1] / checkpoint.name, map_location="cpu", weights_only=False
            )
            for key, expected in control["model"].items():
                if not torch.allclose(
                    actual["model"][key], expected, rtol=TOL.rtol, atol=TOL.atol
                ):
                    pytest.fail(f"Energy-only checkpoint differs for {key}")
        if resume and "Loading checkpoint:" not in result.stdout:
            pytest.fail("The training command did not restore a checkpoint")

    for work in directories:
        exported = work / (
            "energy_stagetwo_compiled.model" if stage_two else "energy_compiled.model"
        )
        if not exported.is_file():
            pytest.fail("The training command did not export its compiled model")
        with tools.torch_tools.default_dtype(torch.float64):
            compiled = torch.jit.load(str(exported), map_location="cpu")
            eager = torch.load(
                str(exported).replace("_compiled.model", ".model"),
                map_location="cpu",
                weights_only=False,
            )
            atoms = ase.io.read(energy_data[1], 0)
            graph = data.AtomicData.from_config(
                data.config_from_atoms(atoms),
                z_table=tools.AtomicNumberTable(eager.atomic_numbers.tolist()),
                cutoff=float(eager.r_max),
            )
            batch = torch_geometric.batch.Batch.from_data_list([graph])
            expected = eager(batch.to_dict(), compute_force=False)["energy"]
            actual = compiled(batch.to_dict(), compute_force=False)["energy"]
            if not torch.allclose(actual, expected, rtol=TOL.rtol, atol=TOL.atol):
                pytest.fail("Compiled model changed the saved energy prediction")


@pytest.mark.timeout(180)
@pytest.mark.parametrize("loss", ["weighted", "l1l2energyforces"])
@pytest.mark.parametrize("stage_two", [False, True])
def test_disabled_forces_require_zero_active_weight(
    tmp_path, anchor_scaleshift, energy_data, loss, stage_two
):
    options = {"forces_weight": 1.0}
    if stage_two:
        options = {
            "forces_weight": 0.0,
            "stage_two": None,
            "start_stage_two": 1,
            "stage_two_forces_weight": 1.0,
        }
    result = run_energy_training(
        tmp_path / "invalid", anchor_scaleshift, energy_data, loss, False, **options
    )
    message = "forces_weight must be zero when forces are not computed"
    if result.returncode == 0 or message not in result.stdout + result.stderr:
        pytest.fail(
            f"Expected an explicit force-weight error:\n{result.stdout}{result.stderr}"
        )
    if stage_two and "Changing loss based on Stage Two Weights" not in result.stdout:
        pytest.fail("The run failed before exercising the stage-two loss")


@pytest.mark.parametrize(
    "loss_class",
    [modules.WeightedEnergyForcesLoss, modules.WeightedEnergyForcesL1L2Loss],
)
def test_energy_gradients_and_reloaded_loss_weights(
    tmp_path, anchor_scaleshift, regression_set, loss_class
):
    with tools.torch_tools.default_dtype(torch.float64):
        anchor = torch.load(anchor_scaleshift, map_location="cpu", weights_only=False)
        frames = [
            a
            for a in ase.io.read(regression_set, ":")
            if a.info.get("config_type") == "molecule"
        ][:2]
        keys = data.KeySpecification.from_defaults().update(
            info_keys={"energy": "REF_energy"}, arrays_keys={"forces": "REF_forces"}
        )
        graphs = [
            data.AtomicData.from_config(
                data.config_from_atoms(a, key_specification=keys),
                z_table=tools.AtomicNumberTable(anchor.atomic_numbers.tolist()),
                cutoff=float(anchor.r_max),
            )
            for a in frames
        ]
        gradients, trained = [], []
        for compute_forces in (True, False):
            model = copy.deepcopy(anchor)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            loss = loss_class(energy_weight=2.0, forces_weight=1.0)
            weight_file = tmp_path / "loss-weights.pt"
            torch.save(
                loss_class(energy_weight=2.0, forces_weight=0.0).state_dict(),
                weight_file,
            )
            loss.load_state_dict(torch.load(weight_file, weights_only=True))
            batch = torch_geometric.batch.Batch.from_data_list(graphs)
            output_args = {"forces": compute_forces, "stress": False, "virials": False}
            take_step(
                model,
                loss,
                batch,
                optimizer,
                None,
                output_args,
                None,
                torch.device("cpu"),
            )
            gradient = torch.cat(
                [
                    (
                        p.grad.flatten()
                        if p.grad is not None
                        else torch.zeros_like(p).flatten()
                    )
                    for p in model.parameters()
                ]
            )
            if not torch.isfinite(gradient).all() or gradient.norm() == 0:
                pytest.fail("Energy-only training produced invalid parameter gradients")
            gradients.append(gradient)
            handler = CheckpointHandler(str(tmp_path / str(compute_forces)), "energy")
            handler.save(CheckpointState(model, optimizer, scheduler), epochs=1)
            restored = copy.deepcopy(anchor)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
            restored_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                restored_optimizer, gamma=0.99
            )
            handler.load_latest(
                CheckpointState(restored, restored_optimizer, restored_scheduler),
                device=torch.device("cpu"),
                strict=True,
            )
            take_step(
                restored,
                loss,
                batch,
                restored_optimizer,
                None,
                output_args,
                None,
                torch.device("cpu"),
            )
            trained.append(
                torch.cat([p.detach().flatten() for p in restored.parameters()])
            )
            if not compute_forces:
                loss.load_state_dict(
                    loss_class(energy_weight=2.0, forces_weight=1.0).state_dict()
                )
                with pytest.raises(ValueError, match="forces_weight must be zero"):
                    take_step(
                        restored,
                        loss,
                        batch,
                        restored_optimizer,
                        None,
                        output_args,
                        None,
                        torch.device("cpu"),
                    )
        if not torch.allclose(gradients[0], gradients[1], rtol=TOL.rtol, atol=TOL.atol):
            pytest.fail(
                "Computing unused forces changed the energy parameter gradients"
            )
        if not torch.allclose(trained[0], trained[1], rtol=TOL.rtol, atol=TOL.atol):
            pytest.fail("Energy-only checkpoint continuation changed the trained model")
