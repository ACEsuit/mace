"""Regression tests for #1843: DDP rank sync when patience triggers Stage Two."""

import importlib
from datetime import timedelta
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.optim.swa_utils import SWALR, AveragedModel

from mace.tools.checkpoint import CheckpointHandler
from mace.tools.utils import MetricsLogger
from tests.helpers import destroy_process_group


def _ddp_stage_two_worker(rank, rendezvous, directory):
    train_mod = importlib.import_module("mace.tools.train")
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        stage1_loss = torch.nn.MSELoss()
        stage2_loss = torch.nn.L1Loss()
        swa = train_mod.SWAContainer(
            model=AveragedModel(model),
            scheduler=SWALR(optimizer, swa_lr=1e-3),
            start=4,
            loss_fn=stage2_loss,
        )

        seen = []

        class RecordingSampler:
            def set_epoch(self, epoch):
                seen.append(("sampler", epoch))

        def fake_train_one_epoch(
            model, loss_fn, data_loader, optimizer, epoch, **kwargs
        ):  # pylint: disable=unused-argument
            seen.append(("train", epoch, loss_fn is stage2_loss))
            # Force all ranks to synchronize inside each epoch just like DDP.
            torch.distributed.barrier()

        eval_calls = {"count": 0}

        def fake_evaluate(
            model, loss_fn, data_loader, output_args, device
        ):  # pylint: disable=unused-argument
            idx = eval_calls["count"]
            eval_calls["count"] += 1
            # Initial eval -> 1.0; epoch 0 -> 0.5 (best); epoch 1 -> 1.0 (triggers patience=1);
            # epoch 5 -> 0.2.
            val = {0: 1.0, 1: 0.5, 2: 1.0}.get(idx, 0.2)
            return val, {"rmse_e_per_atom": val, "rmse_f": val}

        orig_train_one_epoch = train_mod.train_one_epoch
        orig_evaluate = train_mod.evaluate
        train_mod.train_one_epoch = fake_train_one_epoch
        train_mod.evaluate = fake_evaluate
        try:
            train_mod.train(
                model=model,
                loss_fn=stage1_loss,
                train_loader=[],
                valid_loaders={"Default": []},
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                start_epoch=0,
                max_num_epochs=6,
                patience=1,
                checkpoint_handler=CheckpointHandler(
                    directory=str(Path(directory) / "ckpt"),
                    tag="run",
                    swa_start=4,
                ),
                logger=MetricsLogger(directory=str(Path(directory) / "log"), tag="run"),
                eval_interval=1,
                output_args={"forces": False, "virials": False, "stress": False},
                device=torch.device("cpu"),
                log_errors="PerAtomRMSE",
                swa=swa,
                distributed=True,
                train_sampler=RecordingSampler(),
                rank=rank,
            )
        finally:
            train_mod.train_one_epoch = orig_train_one_epoch
            train_mod.evaluate = orig_evaluate

        # After patience triggers at epoch 1 -> swa.start (4), all ranks advance together
        # to epoch 5 with stage2_loss=True and exit before max_num_epochs (6).
        expected = [
            ("sampler", 0),
            ("train", 0, False),
            ("sampler", 1),
            ("train", 1, False),
            ("sampler", 5),
            ("train", 5, True),
        ]
        assert seen == expected, f"rank {rank}: {seen} != {expected}"
        (Path(directory) / f"passed_{rank}").write_text("ok")
    finally:
        destroy_process_group()


def test_ddp_patience_stage_two_syncs_epoch_across_ranks(tmp_path):
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "log").mkdir()
    mp.spawn(
        _ddp_stage_two_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    assert all((tmp_path / f"passed_{rank}").read_text() == "ok" for rank in range(2))
