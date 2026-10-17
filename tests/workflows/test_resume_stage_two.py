"""Resuming a run takes the stage-two learning rate from the command line.

`SWALR` stores `swa_lr` inside the optimizer's param groups, and restoring the
optimizer from a checkpoint replaces those groups wholesale. So a resumed run used
whatever `swa_lr` the checkpoint carried -- and a checkpoint written by a run
without stage two carries none, which crashed the first stage-two scheduler step.

Both tests read the checkpoints the CLI leaves behind, the same black-box surface
as `test_cli_contracts.py`.
"""

from pathlib import Path

import pytest
import torch

from tests.helpers import run_mace_train
from tests.workflows.test_cli_contracts import base_training_params


def latest_checkpoint(directory: Path, swa: bool) -> Path:
    suffix = "_swa.pt" if swa else ".pt"
    candidates = [
        path
        for path in directory.glob("contract_run-7_epoch-*.pt")
        if path.name.endswith("_swa.pt") == swa and path.name.endswith(suffix)
    ]
    assert (
        candidates
    ), f"no {'stage-two' if swa else 'stage-one'} checkpoint in {directory}"
    return max(
        candidates, key=lambda path: int(path.stem.split("epoch-")[1].split("_")[0])
    )


def swa_lrs(checkpoint: Path) -> list:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return [group.get("swa_lr") for group in state["optimizer"]["param_groups"]]


@pytest.mark.timeout(900)
def test_stage_two_can_be_added_when_resuming_a_stage_one_run(tmp_path, regression_set):
    """A run trained without stage two can be resumed into stage two."""
    run_mace_train(base_training_params(tmp_path, regression_set, max_num_epochs=2))
    assert all(lr is None for lr in swa_lrs(latest_checkpoint(tmp_path, swa=False)))

    resumed = base_training_params(
        tmp_path,
        regression_set,
        max_num_epochs=5,
        restart_latest=None,
        swa=None,
        start_swa=3,
        swa_lr=0.0005,
    )
    run_mace_train(resumed)

    assert swa_lrs(latest_checkpoint(tmp_path, swa=True)) == [0.0005] * len(
        swa_lrs(latest_checkpoint(tmp_path, swa=True))
    )


@pytest.mark.timeout(900)
def test_resume_uses_the_stage_two_lr_of_the_current_command_line(
    tmp_path, regression_set
):
    """`--swa_lr` on the resumed command line wins over the checkpoint's."""
    first = base_training_params(
        tmp_path,
        regression_set,
        max_num_epochs=4,
        swa=None,
        start_swa=2,
        swa_lr=0.001,
    )
    run_mace_train(first)
    assert set(swa_lrs(latest_checkpoint(tmp_path, swa=True))) == {0.001}

    resumed = base_training_params(
        tmp_path,
        regression_set,
        max_num_epochs=6,
        restart_latest=None,
        swa=None,
        start_swa=2,
        swa_lr=0.0002,
    )
    run_mace_train(resumed)

    assert set(swa_lrs(latest_checkpoint(tmp_path, swa=True))) == {0.0002}
