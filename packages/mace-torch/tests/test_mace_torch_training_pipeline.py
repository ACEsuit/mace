"""The three stages, end to end, on a task small enough to run in a test.

What is checked is not that the numbers are right, which the parity harness
owns, but that the run does the things whose absence is silent: the E0s are
resolved once and recorded, the model is built with weights it can move, the
loss goes down, and the checkpoint carries enough to rebuild what wrote it.
"""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import load_default_catalogue
from mace_torch.data import GraphDataset
from mace_torch.models import MACEOutputs, ObservableHead
from mace_torch.train import (
    RunState,
    run_data_stage,
    run_model_stage,
    run_train_stage,
)

CATALOGUE = load_default_catalogue()

#: The two references the isolated-atom structures carry.
HYDROGEN, OXYGEN = -13.6, -2040.0

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def write_dataset(path, count: int = 12):
    """A handful of waters, plus the two isolated atoms they reference."""
    generator = np.random.default_rng(0)
    frames = []
    for number, energy in ((1, HYDROGEN), (8, OXYGEN)):
        atom = Atoms(numbers=[number], positions=[[0.0, 0.0, 0.0]])
        atom.info["REF_energy"] = energy
        atom.info["config_type"] = "IsolatedAtom"
        atom.arrays["REF_forces"] = np.zeros((1, 3))
        frames.append(atom)
    for index in range(count):
        molecule = Atoms(
            "OH2", positions=WATER + generator.normal(scale=0.03, size=(3, 3))
        )
        molecule.info["REF_energy"] = 2 * HYDROGEN + OXYGEN + 0.05 * index
        molecule.arrays["REF_forces"] = generator.normal(scale=0.1, size=(3, 3))
        frames.append(molecule)
    write(path, frames)
    return path


def configuration(tmp_path, **training):
    settings = {
        "max_num_epochs": 6,
        "batch_size": 4,
        "valid_batch_size": 4,
        "lr": 0.02,
        "scheduler": {"kind": {"kind": "constant"}},
        **training,
    }
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(tmp_path), "seed": 1},
            "data": {
                "heads": {
                    "default": {
                        "train_file": str(write_dataset(tmp_path / "train.xyz")),
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


@pytest.fixture(name="pipeline")
def fixture_pipeline(tmp_path):
    """The two stages before training, which every test below needs."""
    config = configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    return config, data, run_model_stage(config, data, CATALOGUE)


# ---------------------------------------------------------------------------
# The data stage
# ---------------------------------------------------------------------------


@fp64_only
def test_the_isolated_atom_energies_are_read_once_and_recorded(pipeline):
    """Read off the reference structures, which the stage keeps until it has."""
    _, data, _ = pipeline
    assert data.e0s.values["default"] == {1: HYDROGEN, 8: OXYGEN}
    assert data.e0_provenance["default"].kind == "isolated_atoms"


@fp64_only
def test_the_reference_structures_leave_the_training_set(pipeline):
    """They are references, not structures to fit. One left in is a molecule
    with no neighbours whose energy the model is asked to reproduce on top of
    the reference it just became."""
    _, data, _ = pipeline
    kept = data.train_loader.datasets["default"].configurations
    assert all(len(item.atomic_numbers) > 1 for item in kept)


@fp64_only
def test_the_statistics_are_taken_against_those_energies(pipeline):
    _, data, _ = pipeline
    assert data.statistics.atomic_energies == {1: HYDROGEN, 8: OXYGEN}
    assert data.statistics.avg_num_neighbors == pytest.approx(2.0)


@fp64_only
def test_the_model_reads_the_bundle_s_statistics_rather_than_its_own(pipeline):
    """One measurement per run. Two would let the model be scaled by numbers
    the E0s were not taken against."""
    _, data, built = pipeline
    assert built.statistics is data.statistics


# ---------------------------------------------------------------------------
# The model stage
# ---------------------------------------------------------------------------


@fp64_only
def test_a_built_model_has_weights_it_can_move(pipeline):
    """The defect this guards is a model that trains and never changes: the
    ops allocate zeros, and a product of zeros has a zero gradient."""
    _, _, built = pipeline
    weighted = [
        name
        for name, parameter in built.model.named_parameters()
        if parameter.numel() and float(parameter.detach().abs().max()) > 0.0
    ]
    assert len(weighted) > 10


@fp64_only
def test_the_same_seed_builds_the_same_model(tmp_path):
    config = configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    first = run_model_stage(config, data, CATALOGUE)
    second = run_model_stage(config, data, CATALOGUE)
    for (_, left), (_, right) in zip(
        first.model.named_parameters(), second.model.named_parameters(), strict=True
    ):
        assert torch.equal(left, right)


@fp64_only
def test_the_model_produces_the_observables_that_were_requested(pipeline):
    _, data, built = pipeline
    assert built.outputs.derivatives == ("forces",)
    batch = next(iter(data.valid_loaders["default"]))
    output = built.model(batch.graph, compute=("forces",))
    assert output.total_energy is not None
    assert output.forces is not None
    assert output.stress is None


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


@fp64_only
def test_the_tiny_task_trains(pipeline):
    """The whole point: a model built from a configuration gets better."""
    config, _, built = pipeline
    trained = run_train_stage(config, built)
    losses = [record.train_loss for record in trained.history]
    assert len(losses) == config.training.max_num_epochs
    assert losses[-1] < losses[0]


@fp64_only
def test_the_checkpoint_carries_what_wrote_it(tmp_path):
    config = configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    trained = run_train_stage(config, built, checkpoint_path=tmp_path / "model")
    assert trained.checkpoint_path is not None
    sidecar = json.loads(trained.checkpoint_path.read_text())
    recorded = sidecar["config"]["config"]["resolved"]
    assert recorded["model"]["r_max"] == 5.0
    assert recorded["training"]["batch_size"] == 4


@fp64_only
def test_a_resume_continues_rather_than_restarting(tmp_path):
    """A resume that silently started over would report a first epoch as if it
    carried on from somewhere."""
    config = configuration(tmp_path, max_num_epochs=2)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    run_train_stage(config, built, checkpoint_path=tmp_path / "model")

    later = configuration(tmp_path, max_num_epochs=4)
    continued = run_train_stage(
        later, built, checkpoint_path=tmp_path / "model", resume=True
    )
    assert [record.epoch for record in continued.history] == [2, 3]


@fp64_only
def test_a_resume_with_nothing_to_resume_from_says_so(tmp_path):
    config = configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    with pytest.raises(FileNotFoundError, match="no run to continue"):
        run_train_stage(config, built, checkpoint_path=tmp_path / "absent", resume=True)


@fp64_only
def test_a_dry_run_stops_before_any_epoch_and_writes_nothing(tmp_path):
    """It is exercised on the frozen tree only as `does not crash`."""
    config = configuration(tmp_path, dry_run=True)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    trained = run_train_stage(config, built, checkpoint_path=tmp_path / "model")
    assert trained.history == ()
    assert trained.checkpoint_path is None
    assert not (tmp_path / "model.safetensors").exists()


@fp64_only
def test_the_average_is_what_the_run_is_judged_and_left_with(tmp_path):
    """With EMA on, the reported validation number comes from parameters the
    optimizer never saw, and so does the model the run returns. A run where it
    did nothing would be indistinguishable from one without it."""
    config = configuration(tmp_path, ema={"enabled": True, "decay": 0.9})
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    trained = run_train_stage(config, built)
    assert all(
        record.evaluated_with_ema
        for record in trained.history
        if record.valid_loss is not None
    )

    plain_config = configuration(tmp_path)
    plain_data = run_data_stage(plain_config, CATALOGUE)
    plain = run_train_stage(
        plain_config, run_model_stage(plain_config, plain_data, CATALOGUE)
    )
    averaged = [record.valid_loss for record in trained.history]
    stepped = [record.valid_loss for record in plain.history]
    assert averaged != stepped


@fp64_only
def test_a_run_state_knows_where_it_stopped():
    assert RunState(epoch=3, best_valid_loss=0.5, best_epoch=2).epoch == 3


# ---------------------------------------------------------------------------
# Two heads, which is what the balancing is for
# ---------------------------------------------------------------------------


def two_head_configuration(tmp_path, **training):
    """A small head and a large one, from two files of very different sizes."""
    settings = {
        "max_num_epochs": 2,
        "batch_size": 2,
        "valid_batch_size": 2,
        "lr": 0.02,
        "scheduler": {"kind": {"kind": "constant"}},
        **training,
    }
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(tmp_path), "seed": 1},
            "data": {
                "heads": {
                    "small": {
                        "train_file": str(
                            write_dataset(tmp_path / "small.xyz", count=4)
                        ),
                        "e0s": {"kind": "isolated_atoms"},
                    },
                    "large": {
                        "train_file": str(
                            write_dataset(tmp_path / "large.xyz", count=20)
                        ),
                        "e0s": {"kind": "isolated_atoms"},
                    },
                },
                "valid_fraction": 0.25,
                "pin_memory": False,
            },
            "model": {
                "observables": ["energy", "forces"],
                "r_max": 3.0,
                "num_channels": 4,
                "max_ell": 1,
                "num_interactions": 1,
                "correlation": 2,
            },
            "training": settings,
        }
    )


def head_counts(loader, epoch: int = 0) -> dict[int, int]:
    counts: dict[int, int] = {}
    for batch in loader.batches(epoch, drop_last=False):
        for head in batch.graph["head"].reshape(-1).tolist():
            counts[head] = counts.get(head, 0) + 1
    return counts


@fp64_only
def test_a_balanced_run_visits_the_two_heads_equally(tmp_path):
    """The small head is a fifth of the large one in the file and an equal
    share of the epoch."""
    config = two_head_configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    counts = head_counts(data.train_loader)
    assert len(counts) == 2
    assert len(set(counts.values())) == 1


@fp64_only
def test_a_proportional_run_visits_them_in_proportion(tmp_path):
    """The frozen tree's schedule, reachable as a mode."""
    config = two_head_configuration(tmp_path, head_balancing="proportional")
    data = run_data_stage(config, CATALOGUE)
    counts = head_counts(data.train_loader)
    assert counts[0] * 3 < counts[1]


@fp64_only
def test_each_head_keeps_its_own_validation_loader(tmp_path):
    """Averaged into one, a run cannot say which head got worse."""
    config = two_head_configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    assert set(data.valid_loaders) == {"small", "large"}
    assert set(data.reported_loaders()) == {
        "train_small",
        "train_large",
        "valid_small",
        "valid_large",
    }


@fp64_only
def test_a_two_head_run_trains(tmp_path):
    config = two_head_configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    trained = run_train_stage(config, built)
    assert len(trained.history) == 2
    assert trained.best_epoch is not None


@fp64_only
def test_a_finished_run_prints_an_error_table(tmp_path, caplog):
    """The end-to-end contract: a run says how well it did, per loader."""
    config = two_head_configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    with caplog.at_level("INFO"):
        run_train_stage(config, built)
    printed = "\n".join(record.message for record in caplog.records)
    assert "RMSE E / meV / atom" in printed
    for row in ("train_small", "train_large", "valid_small", "valid_large"):
        assert row in printed


@fp64_only
def test_a_skipped_head_is_left_out_of_the_table(tmp_path, caplog):
    config = two_head_configuration(tmp_path)
    config = config.model_copy(
        update={
            "data": config.data.model_copy(update={"skip_evaluate_heads": ("small",)})
        }
    )
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    with caplog.at_level("INFO"):
        run_train_stage(config, built)
    printed = "\n".join(record.message for record in caplog.records)
    assert "valid_large" in printed
    assert "valid_small" not in printed


@fp64_only
def test_every_validation_log_line_names_its_head(tmp_path, caplog):
    """Named on every line, because the lines are read next to the other
    heads' and next to the next epoch's."""
    config = two_head_configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    with caplog.at_level("INFO"):
        run_train_stage(config, built)
    lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("epoch ")
    ]
    assert lines
    assert all("head small" in line or "head large" in line for line in lines)


def test_an_unknown_error_table_is_refused_when_the_run_is_configured():
    """Not when the table is rendered, which is after the training it reports
    on."""
    with pytest.raises(ValueError, match="is not an error table"):
        ResolvedConfig.model_validate(
            {
                "runtime": {"error_table": "PerAtomQ95"},
                "data": {"heads": {"a": {"train_file": "x.xyz"}}},
                "model": {"observables": ["energy"]},
            }
        )


@fp64_only
def test_a_model_built_for_two_heads_gives_each_its_own_readout(tmp_path):
    """Two levels of theory share the backbone and nothing after it, so a
    head that is a different level of theory has weights of its own to fit."""
    config = configuration(tmp_path)
    head = config.data.heads["default"]
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={"heads": {"first": head, "second": head}}
            )
        }
    )
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    outputs = built.model.get_submodule("backbone.outputs")
    assert isinstance(outputs, MACEOutputs)
    assert cast(ObservableHead, outputs.heads["energy"]).num_heads == 2
    assert outputs.energy_head is not None
    assert outputs.energy_head.e0_table.shape[0] == 2


@fp64_only
def test_the_record_carries_each_heads_energies_and_how_they_were_obtained(tmp_path):
    """What a fine-tune reads a foundation model's energies from, and what
    tells `average` apart from a table that happens to hold the same numbers."""
    config = configuration(tmp_path)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    record = built.metadata.heads["default"].e0
    assert record.values == {"H": HYDROGEN, "O": OXYGEN}
    assert record.source == "estimated"
    assert record.method == "isolated_atoms"


@fp64_only
def test_the_run_ends_on_the_model_it_wrote(tmp_path):
    """The model a run returns is the checkpoint beside it, tensor for tensor.

    The checkpoint is written on an improvement, so it holds the best epoch.
    A run that returned its last epoch instead would hand back a model that is
    not the one on disk, and report errors for the wrong one. The frozen tree
    ends by loading that checkpoint back.
    """
    from mace_torch.serialization import canonical_state, read_canonical_state

    config = configuration(tmp_path, max_num_epochs=8, lr=0.05)
    data = run_data_stage(config, CATALOGUE)
    built = run_model_stage(config, data, CATALOGUE)
    trained = run_train_stage(config, built, checkpoint_path=tmp_path / "model")
    assert trained.best_epoch is not None
    written = read_canonical_state(tmp_path / "model")
    held = canonical_state(trained.model)
    assert set(written) == set(held)
    for path in held:
        for name, value in held[path].items():
            assert torch.equal(value, written[path][name]), f"{path}:{name}"
    assert trained.best_epoch != len(trained.history) - 1, (
        "the best epoch is the last one, so this run cannot tell the two apart; "
        "change the settings until it is not"
    )


@fp64_only
def test_the_same_configuration_trains_the_same_run(tmp_path):
    """The shuffle is seeded by the run, not drawn from the global generator,
    so two runs of one configuration step on the same batches in the same
    order and report the same losses to the last bit."""

    def history(directory):
        config = configuration(directory, max_num_epochs=3)
        data = run_data_stage(config, CATALOGUE)
        built = run_model_stage(config, data, CATALOGUE)
        return [record.valid_loss for record in run_train_stage(config, built).history]

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    assert history(first) == history(second)


@fp64_only
@pytest.mark.parametrize("seed", range(10))
def test_the_isolated_atoms_stay_out_of_the_split(tmp_path, seed):
    """They are references. Split with them in, one lands in the validation set
    for four seeds in ten on this file, and the isolated-atom E0s then have no
    energy for that element. The frozen tree splits without them too, so the
    split is over the same eight waters it splits."""
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(tmp_path), "seed": seed},
            "data": {
                "heads": {
                    "a": {
                        "train_file": str(write_dataset(tmp_path / "t.xyz", count=8)),
                        "e0s": {"isolated_atoms": {}},
                    }
                },
                "valid_fraction": 0.25,
                "pin_memory": False,
            },
            "model": {"observables": ["energy", "forces"], "r_max": 3.0},
        }
    )
    data = run_data_stage(config, CATALOGUE)
    dataset_ = data.valid_loaders["a"].dataset
    assert isinstance(dataset_, GraphDataset)
    validation = dataset_.configurations
    assert not any(item.config_type == "IsolatedAtom" for item in validation)
    assert len(validation) == 2
