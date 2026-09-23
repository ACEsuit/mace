"""Run checkpoints: complete, found by what they say, and never a pickle.

The strongest check is the plainest one. A run interrupted and resumed has to
end with exactly the weights, average and schedule of the same run left alone,
across a stage that brings its own optimizer. Anything a resume forgot, the
optimizer's moments, the schedule's step, the average's shadow, the patience
count, shows up there as a difference.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch
from conftest import fp64_only
from mace_core.observables import load_default_catalogue
from mace_torch.serialization import CheckpointError, canonical_state
from mace_torch.train import (
    RunState,
    latest_run_checkpoint,
    read_run_checkpoint,
    retain_run_checkpoints,
    run_data_stage,
    run_model_stage,
    run_train_stage,
    write_run_checkpoint,
)
from mace_torch.train.ema import ExponentialMovingAverage
from mace_torch.train.optimizers import build_optimizer, build_scheduler
from test_mace_torch_training_pipeline import configuration

CATALOGUE = load_default_catalogue()
SOURCES = Path(__file__).resolve().parents[1] / "src" / "mace_torch"


def two_stage(tmp_path, **training):
    """EMA, a plateau schedule, and a second stage with an optimizer and a
    schedule of its own: what a resume can lose without failing. The second
    stage's schedule is the one that shows a resume restoring into the first
    stage's objects, since the two schedules step the rate differently."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = {
        "max_num_epochs": 6,
        "ema": {"enabled": True, "decay": 0.9},
        "scheduler": {"kind": {"plateau": {"factor": 0.5, "patience": 1}}},
        "stages": [
            {"name": "main"},
            {
                "name": "late",
                "start_epoch": 3,
                "lr": 0.005,
                "optimizer": {"adamw": {}},
                "scheduler": {"kind": {"exponential": {"gamma": 0.5}}},
            },
        ],
    }
    return configuration(tmp_path, **{**settings, **training})


def pipeline(config):
    data = run_data_stage(config, CATALOGUE)
    return run_model_stage(config, data, CATALOGUE)


def weights(model):
    return {
        f"{module}::{name}": value.clone()
        for module, values in canonical_state(model).items()
        for name, value in values.items()
    }


def assert_identical(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        assert torch.equal(value, right[key]), key


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize("split", [2, 4])
def test_a_resumed_run_ends_where_the_uninterrupted_one_does(tmp_path, split):
    """Split in the first stage, and inside the second, whose own optimizer a
    resume has to rebuild before it can restore it."""
    whole = tmp_path / "whole"
    config = two_stage(whole)
    uninterrupted = run_train_stage(
        config, pipeline(config), checkpoint_path=whole / "m"
    )

    directory = tmp_path / "split"
    first = two_stage(directory, max_num_epochs=split)
    run_train_stage(first, pipeline(first), checkpoint_path=directory / "m")
    rest = two_stage(directory)
    resumed = run_train_stage(
        rest, pipeline(rest), checkpoint_path=directory / "m", resume=True
    )

    assert [r.epoch for r in resumed.history] == list(range(split, 6))
    tail = uninterrupted.history[split:]
    assert [r.stage for r in resumed.history] == [r.stage for r in tail]
    assert [r.train_loss for r in resumed.history] == [r.train_loss for r in tail]
    assert [r.learning_rate for r in resumed.history] == [r.learning_rate for r in tail]
    assert_identical(weights(resumed.model), weights(uninterrupted.model))


@fp64_only
def test_a_resumed_run_keeps_the_patience_it_had_used(tmp_path):
    """At a rate of zero nothing improves after the first evaluation, so a run
    with a patience of three stops after its fourth epoch. Resumed after two,
    it must stop there too, not grant itself three more."""
    (tmp_path / "whole").mkdir()
    whole = configuration(tmp_path / "whole", lr=0.0, patience=3, max_num_epochs=10)
    uninterrupted = run_train_stage(
        whole, pipeline(whole), checkpoint_path=tmp_path / "whole" / "m"
    )
    stopped = [r.epoch for r in uninterrupted.history][-1]

    directory = tmp_path / "split"
    directory.mkdir()
    first = configuration(directory, lr=0.0, patience=3, max_num_epochs=2)
    run_train_stage(first, pipeline(first), checkpoint_path=directory / "m")
    rest = configuration(directory, lr=0.0, patience=3, max_num_epochs=10)
    resumed = run_train_stage(
        rest, pipeline(rest), checkpoint_path=directory / "m", resume=True
    )
    assert stopped < 9
    assert [r.epoch for r in resumed.history][-1] == stopped


@fp64_only
def test_the_average_continues_from_its_shadow(tmp_path):
    """Legacy wrote its checkpoints inside the average, so a resume restarted
    the average from the stepped weights. Here the shadow is its own state."""
    config = two_stage(tmp_path, max_num_epochs=2)
    run_train_stage(config, pipeline(config), checkpoint_path=tmp_path / "m")
    latest = latest_run_checkpoint(tmp_path, "m")
    assert latest is not None

    built = pipeline(config)
    optimizer = build_optimizer(built.model, config.training)
    ema = ExponentialMovingAverage(built.model.parameters(), 0.9)
    read_run_checkpoint(latest, model=built.model, optimizer=optimizer, ema=ema)

    document = json.loads(latest.read_text())
    saved_updates = dict((key, value) for key, value in document["ema"]["mapping"])[
        "num_updates"
    ]
    assert ema.num_updates == saved_updates > 0
    raw = [p.detach() for p in built.model.parameters() if p.requires_grad]
    assert any(not torch.equal(s, r) for s, r in zip(ema.shadow, raw, strict=True))


@fp64_only
def test_a_round_trip_restores_every_piece_exactly(tmp_path):
    config = two_stage(tmp_path, max_num_epochs=2)
    built = pipeline(config)
    trained = run_train_stage(config, built, checkpoint_path=tmp_path / "m")
    model = trained.model
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training.scheduler)
    assert scheduler is not None
    ema = ExponentialMovingAverage(model.parameters(), 0.9)
    for _ in range(3):
        loss = torch.stack(
            [(p**2).sum() for p in model.parameters() if p.requires_grad]
        ).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update()
    scheduler.step(1.0)  # ty: ignore[invalid-argument-type]
    state = RunState(7, 0.25, 5, 2, "main", False)
    sidecar = write_run_checkpoint(
        tmp_path / "rt",
        "m",
        state,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        metadata=built.metadata,
    )

    fresh = pipeline(config).model
    fresh_optimizer = build_optimizer(fresh, config.training)
    fresh_scheduler = build_scheduler(fresh_optimizer, config.training.scheduler)
    assert fresh_scheduler is not None
    fresh_ema = ExponentialMovingAverage(fresh.parameters(), 0.9)
    result = read_run_checkpoint(
        sidecar,
        model=fresh,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        ema=fresh_ema,
    )

    assert result.state == state
    assert result.optimizer_state == "restored"
    assert_identical(weights(fresh), weights(model))
    saved, loaded = optimizer.state_dict(), fresh_optimizer.state_dict()
    assert saved["param_groups"] == loaded["param_groups"]
    for index, entries in saved["state"].items():
        for name, value in entries.items():
            other = loaded["state"][index][name]
            assert torch.equal(torch.as_tensor(value), torch.as_tensor(other)), name
    assert scheduler.state_dict() == fresh_scheduler.state_dict()
    assert fresh_ema.num_updates == ema.num_updates
    for mine, theirs in zip(ema.shadow, fresh_ema.shadow, strict=True):
        assert torch.equal(mine, theirs)
    assert json.loads(sidecar.read_text())["config"]["config"]["resolved"]


# ---------------------------------------------------------------------------
# The reported partial resume, and the refusals
# ---------------------------------------------------------------------------


@fp64_only
def test_changed_parameter_groups_load_the_weights_and_say_why(tmp_path):
    config = two_stage(tmp_path, max_num_epochs=1)
    trained = run_train_stage(config, pipeline(config), checkpoint_path=tmp_path / "m")
    latest = latest_run_checkpoint(tmp_path, "m")
    assert latest is not None

    fresh = pipeline(config).model
    halves = [p for p in fresh.parameters() if p.requires_grad]
    regrouped = torch.optim.Adam(
        [{"params": halves[::2]}, {"params": halves[1::2]}], lr=0.01
    )
    ema = ExponentialMovingAverage(fresh.parameters(), 0.9)
    result = read_run_checkpoint(latest, model=fresh, optimizer=regrouped, ema=ema)
    assert result.optimizer_state == "reinitialized"
    assert result.reason is not None and "parameter groups changed" in result.reason
    del trained


@fp64_only
def test_a_corrupt_newest_checkpoint_is_refused_not_skipped(tmp_path):
    config = two_stage(tmp_path, max_num_epochs=3)
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(update={"save_all_checkpoints": True})
        }
    )
    run_train_stage(config, pipeline(config), checkpoint_path=tmp_path / "m")
    latest = latest_run_checkpoint(tmp_path, "m")
    assert latest is not None
    latest.write_text("{not json")
    with pytest.raises(CheckpointError, match="refused rather than skipped"):
        latest_run_checkpoint(tmp_path, "m")


@fp64_only
def test_truncated_tensors_are_refused(tmp_path):
    config = two_stage(tmp_path, max_num_epochs=1)
    run_train_stage(config, pipeline(config), checkpoint_path=tmp_path / "m")
    latest = latest_run_checkpoint(tmp_path, "m")
    assert latest is not None
    tensors = latest.with_name(latest.name.removesuffix(".json") + ".safetensors")
    tensors.write_bytes(tensors.read_bytes()[:100])
    built = pipeline(config)
    with pytest.raises(CheckpointError, match="cannot be read"):
        read_run_checkpoint(
            latest,
            model=built.model,
            optimizer=build_optimizer(built.model, config.training),
            ema=ExponentialMovingAverage(built.model.parameters(), 0.9),
        )


def test_a_function_in_the_state_is_refused_rather_than_pickled(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.5**epoch)

    class Stateful(torch.optim.lr_scheduler.LRScheduler):
        def state_dict(self):
            return {"rule": print}

    with pytest.raises(CheckpointError, match="builtin_function_or_method"):
        write_run_checkpoint(
            tmp_path,
            "m",
            RunState(1),
            model=model,
            optimizer=optimizer,
            scheduler=Stateful.__new__(Stateful),
        )
    del scheduler


# ---------------------------------------------------------------------------
# Finding and keeping
# ---------------------------------------------------------------------------


def linear_checkpoints(tmp_path, count, improving=()):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(1, count + 1):
        write_run_checkpoint(
            tmp_path,
            "m",
            RunState(epoch, improved=epoch in improving),
            model=model,
            optimizer=optimizer,
        )


def epochs_on_disk(tmp_path):
    return sorted(
        json.loads(path.read_text())["state"]["epoch"]
        for path in tmp_path.glob("m.run-*.json")
    )


def test_the_newest_is_found_by_its_recorded_epoch(tmp_path):
    """Swapping two files' names does not change which one is newest: the
    epoch comes from inside, never from a name."""
    linear_checkpoints(tmp_path, 3)
    first, third = tmp_path / "m.run-000001", tmp_path / "m.run-000003"
    for suffix in (".json", ".safetensors"):
        (tmp_path / f"swap{suffix}").write_bytes(Path(f"{first}{suffix}").read_bytes())
        Path(f"{first}{suffix}").write_bytes(Path(f"{third}{suffix}").read_bytes())
        Path(f"{third}{suffix}").write_bytes((tmp_path / f"swap{suffix}").read_bytes())
    latest = latest_run_checkpoint(tmp_path, "m")
    assert latest is not None
    assert json.loads(latest.read_text())["state"]["epoch"] == 3
    assert latest.name == "m.run-000001.json"


def test_an_interrupted_write_is_invisible(tmp_path):
    linear_checkpoints(tmp_path, 2)
    (tmp_path / "m.run-000003.partial.json").write_text("{}")
    latest = latest_run_checkpoint(tmp_path, "m")
    assert latest is not None and latest.name == "m.run-000002.json"


def test_by_default_only_the_newest_is_kept(tmp_path):
    linear_checkpoints(tmp_path, 4, improving=(1, 3))
    retain_run_checkpoints(tmp_path, "m", keep_improving=False, keep_all=False)
    assert epochs_on_disk(tmp_path) == [4]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "m.run-000004.json",
        "m.run-000004.safetensors",
    ]


def test_keep_checkpoints_keeps_every_improvement(tmp_path):
    linear_checkpoints(tmp_path, 4, improving=(1, 3))
    retain_run_checkpoints(tmp_path, "m", keep_improving=True, keep_all=False)
    assert epochs_on_disk(tmp_path) == [1, 3, 4]


def test_save_all_checkpoints_keeps_every_epoch(tmp_path):
    linear_checkpoints(tmp_path, 4, improving=(1, 3))
    retain_run_checkpoints(tmp_path, "m", keep_improving=False, keep_all=True)
    assert epochs_on_disk(tmp_path) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# No pickle
# ---------------------------------------------------------------------------


def test_no_v1_source_writes_or_reads_a_pickle():
    """No torch.save, no torch.load, no pickle file read or written, and not
    the flag the frozen tree needs to unpickle its own checkpoints. An
    in-memory round trip is not a file: the data backends are checked to
    survive one because that is how a loader worker receives them."""
    offenders = []
    roots = [SOURCES, SOURCES.parents[2] / "mace-core" / "src" / "mace_core"]
    for root in roots:
        for path in root.rglob("*.py"):
            # A file in a directory that is not an identifier is not v1 code:
            # nothing can import it, and the one there is the legacy side.
            if not all(p.isidentifier() for p in path.relative_to(root).parts[:-1]):
                continue
            source = path.read_text()
            if "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in source:
                offenders.append(f"{path.name}: the unpickling flag")
            for node in ast.walk(ast.parse(source)):
                if not (
                    isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                ):
                    continue
                if node.value.id == "torch" and node.attr in {"save", "load"}:
                    offenders.append(f"{path.name}:{node.lineno}: torch.{node.attr}")
                if node.value.id == "pickle" and node.attr in {"dump", "load"}:
                    offenders.append(f"{path.name}:{node.lineno}: pickle.{node.attr}")
    assert not offenders, offenders
