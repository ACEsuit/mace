"""A completed checkpoint epoch must not receive more optimizer steps."""

from pathlib import Path

import pytest
import torch

from tests.helpers import run_mace_train
from tests.workflows.test_cli_contracts import base_training_params, training_records


@pytest.mark.parametrize("stage_two", [False, True])
@pytest.mark.parametrize("completed", [0, 2])
@pytest.mark.parametrize("extra_epochs", [0, 2])
def test_restart_skips_completed_epoch(
    tmp_path, regression_set, completed, extra_epochs, stage_two
):
    params = base_training_params(
        tmp_path,
        regression_set,
        hidden_irreps="8x0e",
        num_interactions=1,
        max_num_epochs=completed + 1,
        scheduler="ExponentialLR",
        lr_scheduler_gamma=0.5,
        save_all_checkpoints=None,
        keep_checkpoints=None,
    )
    if stage_two:
        params.update(swa=None, start_swa=1, swa_lr=0.005)
    env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    run_mace_train(params, env_extra=env)
    before = training_records(tmp_path, "contract", 7)
    params.update(restart_latest=None, max_num_epochs=completed + 1 + extra_epochs)
    run_mace_train(params, env_extra=env)
    after = training_records(tmp_path, "contract", 7)
    new_steps = [r for r in after[len(before) :] if r.get("mode") == "opt"]
    assert {r["epoch"] for r in new_steps} == set(
        range(completed + 1, completed + 1 + extra_epochs)
    )
    # A restart must advance the restored LR before its first new epoch,
    # just as an uninterrupted run does. Check the actual saved optimizer.
    if stage_two:
        return  # SWA scheduler/averaging state is not checkpointed today.
    for epoch in range(completed + 1, completed + 1 + extra_epochs):
        path = Path(tmp_path) / f"contract_run-7_epoch-{epoch}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for group in checkpoint["optimizer"]["param_groups"]:
            assert group["lr"] == pytest.approx(0.01 * 0.5**epoch)
