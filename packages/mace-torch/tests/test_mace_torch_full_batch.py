"""The full-batch regime: one L-BFGS step per epoch over the whole set.

The property the regime exists for is that its step follows the gradient of
the loss over the whole training set. The set here mixes structures of three
and six atoms on purpose: on structures that all have one size, weighing each
batch by its share of the structures gives the same number as the whole set's
loss, and a test on them could not tell the two apart.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import load_default_catalogue
from mace_torch.train import run_data_stage, run_model_stage, run_train_stage
from mace_torch.train.checkpoint import latest_run_checkpoint, read_run_state
from mace_torch.train.ddp import DistributedContext
from mace_torch.train.full_batch import accumulate, full_batch_step, set_totals
from mace_torch.train.loss import TermwiseLoss, build_loss
from mace_torch.train.optimizers import build_optimizer

CATALOGUE = load_default_catalogue()

HYDROGEN, OXYGEN = -13.6, -2040.0


def write_mixed_dataset(path, count: int = 13):
    """Waters and hydrogen peroxide dimers, three atoms and six."""
    generator = np.random.default_rng(0)
    frames = []
    for number, energy in ((1, HYDROGEN), (8, OXYGEN)):
        atom = Atoms(numbers=[number], positions=[[0.0, 0.0, 0.0]])
        atom.info["REF_energy"] = energy
        atom.info["config_type"] = "IsolatedAtom"
        atom.arrays["REF_forces"] = np.zeros((1, 3))
        frames.append(atom)
    water = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    for index in range(count):
        if index % 2:
            symbols, positions = "OH2", water
        else:
            symbols, positions = (
                "O2H4",
                np.vstack([water, water + np.array([2.5, 0.0, 0.0])]),
            )
        molecule = Atoms(
            symbols,
            positions=positions + generator.normal(scale=0.05, size=positions.shape),
        )
        molecule.info["REF_energy"] = (
            molecule.get_chemical_symbols().count("H") * HYDROGEN
            + molecule.get_chemical_symbols().count("O") * OXYGEN
            + 0.05 * index
        )
        molecule.arrays["REF_forces"] = generator.normal(
            scale=0.1, size=positions.shape
        )
        frames.append(molecule)
    write(path, frames)
    return path


def configuration(tmp_path, **training):
    settings = {
        "max_num_epochs": 2,
        "batch_size": 3,
        "valid_batch_size": 3,
        "lr": 0.01,
        "scheduler": {"kind": {"kind": "constant"}},
        **training,
    }
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(tmp_path), "seed": 1},
            "data": {
                "heads": {
                    "default": {
                        "train_file": str(write_mixed_dataset(tmp_path / "train.xyz")),
                        "e0s": {"kind": "isolated_atoms"},
                    }
                },
                "valid_fraction": 0.2,
                "pin_memory": False,
            },
            "model": {
                "observables": ["energy", "forces"],
                "r_max": 5.0,
                "num_interactions": 2,
                "num_channels": 4,
                "hidden_irreps": "0e+1o",
                "max_ell": 1,
                "correlation": 2,
                "readout": {"mlp_irreps": "4x0e"},
            },
            "training": settings,
            "loss": {"weights": {"energy": 1.0, "forces": 10.0}},
        }
    )


LBFGS = {"kind": "lbfgs"}


def built_task(tmp_path, **training):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = configuration(tmp_path, **training)
    data = run_data_stage(config, CATALOGUE)
    return config, run_model_stage(config, data, CATALOGUE)


def termwise(loss: torch.nn.Module) -> TermwiseLoss:
    assert isinstance(loss, TermwiseLoss)
    return loss


def gradient(model) -> torch.Tensor:
    return torch.cat(
        [p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]
    )


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


@fp64_only
def test_the_parts_add_up_to_the_loss_of_the_whole_set(tmp_path):
    """Its loss and its gradient, both, against one batch holding everything."""
    config, built = built_task(tmp_path)
    model, loss = built.model, termwise(build_loss(built.outputs, config.loss))
    loader = built.data.train_loader
    parts = list(loader.batches(0, drop_last=False))
    sizes = {
        int(batch.graph["ptr"][-1]) // int(batch.graph["num_graphs"]) for batch in parts
    }
    assert len(parts) > 1 and len(sizes) > 1, "meant to be batches of mixed sizes"

    model.zero_grad(set_to_none=True)
    totals = set_totals(loss, parts, DistributedContext())
    assembled = accumulate(model, parts, loss, totals)
    assembled_gradient = gradient(model).clone()

    loader.batch_size = sum(int(batch.graph["num_graphs"]) for batch in parts)
    (whole,) = list(loader.batches(0, drop_last=False))
    model.zero_grad(set_to_none=True)
    output = model(whole.graph, compute=("forces",), training=True)
    at_once = loss(output, whole)
    at_once.backward()

    torch.testing.assert_close(assembled, at_once.detach(), rtol=1e-12, atol=0.0)
    torch.testing.assert_close(
        assembled_gradient, gradient(model), rtol=1e-12, atol=1e-15
    )


@fp64_only
def test_weighing_by_share_of_structures_is_not_the_loss_of_the_whole_set(tmp_path):
    """The frozen tree's weighting, which the arithmetic above replaces.

    It is right for the energy, whose term is a mean over structures, and
    wrong for the forces, whose term is a mean over atoms."""
    config, built = built_task(tmp_path)
    model, loss = built.model, termwise(build_loss(built.outputs, config.loss))
    parts = list(built.data.train_loader.batches(0, drop_last=False))
    structures = sum(int(batch.graph["num_graphs"]) for batch in parts)
    shared = 0.0
    for batch in parts:
        output = model(batch.graph, compute=("forces",), training=True)
        shared += (
            float(loss(output, batch)) * int(batch.graph["num_graphs"]) / structures
        )
    whole = float(
        accumulate(model, parts, loss, set_totals(loss, parts, DistributedContext()))
    )
    assert abs(shared - whole) > 1e-9 * abs(whole)


def test_a_loss_written_from_scratch_is_refused(tmp_path):
    config, built = built_task(tmp_path, optimizer=LBFGS)
    optimizer = build_optimizer(built.model, config.training)
    with pytest.raises(TypeError, match="TermwiseLoss"):
        full_batch_step(built.model, lambda: [], torch.nn.MSELoss(), optimizer)


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------


@fp64_only
def test_an_epoch_is_one_step_through_a_closure_over_everything(tmp_path, monkeypatch):
    config, built = built_task(tmp_path, optimizer=LBFGS)
    loss = build_loss(built.outputs, config.loss)
    optimizer = build_optimizer(built.model, config.training)
    steps, passes = [], []
    step = optimizer.step

    def counted(closure):
        steps.append(1)
        return step(closure)

    monkeypatch.setattr(optimizer, "step", counted)
    loader = built.data.train_loader

    def batches():
        passes.append(1)
        return loader.batches(0, drop_last=False)

    before = [p.detach().clone() for p in built.model.parameters()]
    full_batch_step(built.model, batches, loss, optimizer)
    assert len(steps) == 1
    # One pass to count the set, then one per evaluation: the line search takes
    # several, which is why a missing broadcast diverges ranks.
    assert len(passes) > 2
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, built.model.parameters(), strict=True)
    )


def test_lbfgs_is_built_over_what_trains_with_its_own_step_length(tmp_path):
    config, built = built_task(tmp_path, optimizer=LBFGS)
    optimizer = build_optimizer(built.model, config.training)
    assert isinstance(optimizer, torch.optim.LBFGS)
    (group,) = optimizer.param_groups
    trainable = [p for p in built.model.parameters() if p.requires_grad]
    assert {id(p) for p in group["params"]} == {id(p) for p in trainable}
    assert (group["lr"], group["history_size"], group["max_iter"]) == (1.0, 200, 20)
    assert group["line_search_fn"] == "strong_wolfe"


def test_a_partial_learning_rate_factor_is_refused_under_lbfgs(tmp_path):
    config, built = built_task(
        tmp_path,
        optimizer=LBFGS,
        scheduler={"kind": {"kind": "constant"}, "group_factors": {"readouts": 0.5}},
    )
    with pytest.raises(ValueError, match="readouts"):
        build_optimizer(built.model, config.training)


# ---------------------------------------------------------------------------
# In a run
# ---------------------------------------------------------------------------


ADAM_THEN_LBFGS = [
    {"name": "main"},
    {"name": "polish", "start_epoch": 2, "optimizer": LBFGS},
]


def structures_seen(built, monkeypatch) -> dict[int, int]:
    """How many structures each epoch's training passes held, by epoch."""
    loader = built.data.train_loader
    batches = loader.batches
    seen: dict[int, int] = {}

    def counting(epoch, *, drop_last):
        for batch in batches(epoch, drop_last=drop_last):
            seen[epoch] = seen.get(epoch, 0) + int(batch.graph["num_graphs"])
            yield batch

    monkeypatch.setattr(loader, "batches", counting)
    return seen


@fp64_only
def test_a_full_batch_stage_sees_every_structure_and_a_mini_batch_one_drops_the_tail(
    tmp_path, monkeypatch
):
    config, built = built_task(tmp_path, max_num_epochs=3, stages=ADAM_THEN_LBFGS)
    structures = len(built.data.train_loader.indices("default", 0))
    assert structures % config.training.batch_size, "meant to leave a ragged tail"
    seen = structures_seen(built, monkeypatch)
    run_train_stage(config, built)
    whole = structures // config.training.batch_size * config.training.batch_size
    assert seen[0] == seen[1] == whole
    # The full-batch epoch passes over the set once to count it and once per
    # closure evaluation, each time all of it.
    assert seen[2] % structures == 0 and seen[2] >= 2 * structures


@fp64_only
def test_adam_then_lbfgs_trains_and_leaves_a_checkpoint_to_resume_from(tmp_path):
    config, built = built_task(tmp_path, max_num_epochs=4, stages=ADAM_THEN_LBFGS)
    trained = run_train_stage(config, built, checkpoint_path=tmp_path / "model")
    assert [record.stage for record in trained.history] == ["main"] * 2 + ["polish"] * 2
    losses = [record.train_loss for record in trained.history]
    assert losses[-1] < losses[2] < losses[0]
    latest = latest_run_checkpoint(tmp_path, "model")
    assert latest is not None
    assert (read_run_state(latest).epoch, read_run_state(latest).stage) == (4, "polish")


@fp64_only
def test_a_resume_inside_the_lbfgs_stage_continues_its_curvature(tmp_path):
    """L-BFGS carries its history as lists of tensors, and a resume that lost
    them would restart the approximation and take a different step."""
    stages = [{"name": "main", "optimizer": LBFGS}]
    config, built = built_task(tmp_path / "whole", max_num_epochs=3, stages=stages)
    whole = run_train_stage(config, built)

    config, built = built_task(tmp_path / "split", max_num_epochs=2, stages=stages)
    run_train_stage(config, built, checkpoint_path=tmp_path / "split" / "model")
    config, built = built_task(tmp_path / "split", max_num_epochs=3, stages=stages)
    continued = run_train_stage(
        config, built, checkpoint_path=tmp_path / "split" / "model", resume=True
    )
    assert continued.history[-1].train_loss == whole.history[-1].train_loss


@fp64_only
def test_resuming_a_mini_batch_run_under_lbfgs_starts_the_optimizer_afresh(
    tmp_path, caplog
):
    """Declared from what the checkpoint records rather than inferred from a
    failed load, since Adam's moments can load into L-BFGS and mean nothing."""
    config, built = built_task(tmp_path, max_num_epochs=2)
    run_train_stage(config, built, checkpoint_path=tmp_path / "model")

    config, built = built_task(tmp_path, max_num_epochs=3, optimizer=LBFGS)
    with caplog.at_level(logging.WARNING):
        continued = run_train_stage(
            config, built, checkpoint_path=tmp_path / "model", resume=True
        )
    assert "written by Adam and the run resumes with LBFGS" in caplog.text
    assert [record.epoch for record in continued.history] == [2]
