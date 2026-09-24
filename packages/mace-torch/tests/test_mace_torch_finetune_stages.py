"""A fine-tune run step by step, against the same fine-tune run in one go.

The foundation is trained on waters. The fine-tune keeps a replay head that
subselects six of those waters at random, and a new head on relabelled ones.
Run in one go, the subselection happens inside the data stage; run step by
step, it is written to a file first, the replay head is pointed at the file,
and the model is built and trained from there. The two have to end with the
same weights.
"""

from __future__ import annotations

import pytest
import torch
from ase.io import read, write
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_torch.cli.run_train import run
from mace_torch.finetune.stages import (
    build,
    reading_selected,
    select_structures,
    train,
)
from mace_torch.serialization import canonical_state
from mace_torch.train import DataStageError
from test_mace_torch_extend_elements import water_foundation, waters


@pytest.fixture(scope="module", name="foundation")
def fixture_foundation(tmp_path_factory):
    return water_foundation(tmp_path_factory.mktemp("water_foundation"))


def relabelled(path):
    frames = []
    for atoms in waters(10, seed=4):
        if atoms.info.get("config_type") != "IsolatedAtom":
            atoms.info["REF_energy"] += 0.3
            atoms.arrays["REF_forces"] = 1.1 * atoms.arrays["REF_forces"]
        frames.append(atoms)
    write(path, frames)
    return path


def fine_tune(directory, foundation):
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / "replay.xyz", waters(14, seed=2))
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(directory), "seed": 11},
            "finetune": {"foundation_model": str(foundation)},
            "data": {
                "heads": {
                    "replay": {
                        "train_file": str(directory / "replay.xyz"),
                        "e0s": {"foundation": {}},
                        "subselect": {"num_samples": 6, "method": "random"},
                        "weight": 0.5,
                    },
                    "new": {
                        "train_file": str(relabelled(directory / "new.xyz")),
                        "e0s": {"isolated_atoms": {}},
                    },
                },
                "valid_fraction": 0.25,
                "pin_memory": False,
            },
            "model": {"observables": ["energy", "forces"]},
            "training": {
                "max_num_epochs": 2,
                "batch_size": 4,
                "valid_batch_size": 4,
                "lr": 0.01,
                "scheduler": {"kind": {"kind": "constant"}},
            },
        }
    )


@fp64_only
def test_step_by_step_ends_with_the_one_go_run_s_weights(foundation, tmp_path):
    one_go = run(fine_tune(tmp_path / "one_go", foundation))

    config = fine_tune(tmp_path / "steps", foundation)
    selected = select_structures(
        config, "replay", tmp_path / "steps" / "replay_kept.xyz"
    )
    staged_config = reading_selected(config, "replay", selected)
    built = build(staged_config)
    staged = train(staged_config, built)

    first = canonical_state(one_go.model.get_submodule("backbone"))
    second = canonical_state(staged.model.get_submodule("backbone"))
    assert first.keys() == second.keys()
    for path, tensors in first.items():
        for name, value in tensors.items():
            assert torch.allclose(second[path][name], value, atol=1e-12, rtol=0), (
                f"{path}:{name}"
            )


@fp64_only
def test_the_selection_file_holds_what_the_subselection_keeps(foundation, tmp_path):
    config = fine_tune(tmp_path, foundation)
    selected = select_structures(config, "replay", tmp_path / "kept.xyz")
    kept = read(selected, ":")
    isolated = [atoms for atoms in kept if atoms.info["config_type"] == "IsolatedAtom"]
    assert len(kept) - len(isolated) == 6


@fp64_only
def test_the_head_reading_the_file_keeps_its_other_settings(foundation, tmp_path):
    config = fine_tune(tmp_path, foundation)
    staged = reading_selected(config, "replay", tmp_path / "kept.xyz")
    head = staged.data.heads["replay"]
    assert head.subselect is None
    assert head.weight == 0.5
    assert head.e0s == config.data.heads["replay"].e0s
    assert staged.data.heads["new"] == config.data.heads["new"]


def test_a_head_the_configuration_lacks_is_refused(foundation, tmp_path):
    config = fine_tune(tmp_path, foundation)
    with pytest.raises(DataStageError, match="no head 'pt_head'"):
        select_structures(config, "pt_head", tmp_path / "kept.xyz")
