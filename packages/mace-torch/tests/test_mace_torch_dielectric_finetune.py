"""Fine-tuning a dipole and polarizability model from one.

A dielectric model is trained on waters and written as a checkpoint; a
fine-tune starts from it on relabelled waters, trains, and writes its own. The
fine-tune declares only what it reads out and names its foundation: there is no
mode for this family, since the element transfer every fine-tune goes through
already carries every angular channel.

What an energy fine-tune has and this one lacks is checked as well: no E0s, no
energy loss, and a foundation that reads out something else is refused before
any data is read.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from mace_core.config.resolved import ResolvedConfig
from mace_torch.calculators import MACECalculator
from mace_torch.cli.run_train import run
from mace_torch.deploy.loader import load_deployed
from mace_torch.finetune.stages import build
from mace_torch.serialization import canonical_state
from mace_torch.train import ModelStageError
from test_mace_torch_extend_elements import water_foundation

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
OBSERVABLES = ["dipole", "polarizability"]


def waters(path, count=10, seed=0, scale=1.0):
    generator = np.random.default_rng(seed)
    frames = []
    for _ in range(count):
        atoms = Atoms(
            "OH2", positions=WATER + generator.normal(scale=0.05, size=(3, 3))
        )
        atoms.info["dipole"] = scale * generator.normal(size=3)
        polarizability = generator.normal(size=(3, 3))
        atoms.info["polarizability"] = scale * (polarizability + polarizability.T)
        frames.append(atoms)
    write(path, frames)
    return path


def training(directory, train_file, **sections):
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(directory), "seed": 1},
            "data": {
                "heads": {"default": {"train_file": str(train_file)}},
                "valid_fraction": 0.2,
                "pin_memory": False,
            },
            "training": {"max_num_epochs": 2, "batch_size": 4, "lr": 0.01},
            **sections,
        }
    )


@pytest.fixture(scope="module", name="foundation")
def fixture_foundation(tmp_path_factory):
    directory = tmp_path_factory.mktemp("dielectric_foundation")
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        config = training(
            directory,
            waters(directory / "train.xyz"),
            model={
                "model": "dielectric",
                "observables": OBSERVABLES,
                "r_max": 3.0,
                "num_channels": 4,
                "max_ell": 2,
                "hidden_irreps": "0e+1o+2e",
                "readout": {"mlp_irreps": "4x0e+4x1o+4x2e"},
            },
        )
        return run(config).checkpoint_path
    finally:
        torch.set_default_dtype(previous)


def fine_tune(directory, foundation, observables=OBSERVABLES):
    directory.mkdir(parents=True, exist_ok=True)
    return training(
        directory,
        waters(directory / "new.xyz", seed=5, scale=1.5),
        finetune={"foundation_model": str(foundation)},
        model={"observables": observables},
    )


def weights(engine):
    return {
        (path, name): tensor
        for path, tensors in canonical_state(engine).items()
        for name, tensor in tensors.items()
    }


def water():
    atoms = Atoms("OH2", positions=WATER + 0.02)
    atoms.info["charge"] = 0.0
    return atoms


def test_the_record_holds_the_element_table(foundation):
    """A model with no energies records no E0s, so the record names its
    elements on their own."""
    deployed = load_deployed(foundation)
    assert list(deployed.z_table.zs) == [1, 8]


def test_a_dielectric_fine_tune_trains_from_the_foundation(foundation, tmp_path):
    config = fine_tune(tmp_path / "fine_tune", foundation)
    assert config.runtime.error_table == "DipolePolarRMSE"
    before = weights(build(config).model)
    parent = weights(load_deployed(foundation).engine)
    assert before.keys() == parent.keys()
    assert all(torch.equal(before[key], parent[key]) for key in before), (
        "the fine-tune starts from the foundation's weights"
    )

    trained = run(config)
    assert trained.checkpoint_path is not None
    after = weights(load_deployed(trained.checkpoint_path).engine)
    assert any(not torch.equal(before[key], after[key]) for key in before)

    tuned, original = (
        MACECalculator(trained.checkpoint_path),
        MACECalculator(foundation),
    )
    atoms = water()
    tuned.calculate(atoms)
    original.calculate(atoms)
    assert tuned.results["dipole"].shape == (3,)
    assert tuned.results["polarizability"].shape == (3, 3)
    assert not np.allclose(tuned.results["dipole"], original.results["dipole"])


def test_the_fine_tuned_checkpoint_reproduces_the_trained_model(foundation, tmp_path):
    """The calculator over the written checkpoint computes what the model the
    run trained computes, before anything was written."""
    trained = run(fine_tune(tmp_path / "fine_tune", foundation))
    assert trained.checkpoint_path is not None
    deployed = load_deployed(trained.checkpoint_path)
    in_memory = dataclasses.replace(deployed, engine=trained.model)
    atoms = water()
    results = {}
    for name, calculator in (
        ("checkpoint", MACECalculator(models=deployed)),
        ("trained", MACECalculator(models=in_memory)),
    ):
        calculator.calculate(atoms)
        results[name] = calculator.results
    for name in ("dipole", "polarizability", "charges"):
        np.testing.assert_array_equal(
            results["checkpoint"][name], results["trained"][name], err_msg=name
        )


def test_a_foundation_that_reads_out_no_dipole_is_refused(tmp_path):
    """The frozen tree's `--model AtomicDielectricMACE` fine-tune from an energy
    model exits with an error, and so does this."""
    energy_model = water_foundation(tmp_path)
    with pytest.raises(ModelStageError, match="dipole"):
        build(fine_tune(tmp_path / "fine_tune", energy_model))


def test_an_energy_fine_tune_of_a_dielectric_model_is_refused(foundation, tmp_path):
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(tmp_path)},
            "finetune": {"foundation_model": str(foundation)},
            "data": {"heads": {"default": {"train_file": str(tmp_path / "x.xyz")}}},
            "model": {"observables": ["energy", "forces"]},
        }
    )
    with pytest.raises(ModelStageError, match="energy"):
        build(config)
