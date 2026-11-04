"""The same launcher environment places native and frozen ranks identically."""

from argparse import Namespace

import pytest
import torch
from mace_torch.train.ddp import init_distributed

pytestmark = pytest.mark.usefixtures("isolated")

CASES = [
    ("torchrun", {"RANK": "5", "LOCAL_RANK": "1", "WORLD_SIZE": "8"}),
    (
        "slurm",
        {
            "SLURM_PROCID": "5",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS": "8",
            "SLURM_NTASKS_PER_NODE": "4",
            "SLURM_NNODES": "2",
            "SLURM_JOB_NODELIST": "node[01-02]",
        },
    ),
    (
        "mpi",
        {
            "OMPI_COMM_WORLD_RANK": "5",
            "OMPI_COMM_WORLD_SIZE": "8",
            "OMPI_COMM_WORLD_LOCAL_SIZE": "4",
        },
    ),
    ("mpi", {"PMI_RANK": "5", "PMI_SIZE": "8", "MPI_LOCALRANKID": "1"}),
    (
        "mpi",
        {
            "PALS_RANKID": "5",
            "PALS_NTASKS": "8",
            "PALS_LOCAL_SIZE": "4",
            "PALS_LOCAL_RANKID": "1",
        },
    ),
]


@pytest.mark.parametrize("launcher,variables", CASES)
def test_rank_and_rendezvous_environment_match_legacy(monkeypatch, launcher, variables):
    import os

    from mace.tools.distributed_tools import init_distributed as legacy_init

    outputs = []
    for initialize in (legacy_init, init_distributed):
        with monkeypatch.context() as patch:
            for key in list(os.environ):
                if key in {
                    "RANK",
                    "WORLD_SIZE",
                    "LOCAL_RANK",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                } or key.startswith(("SLURM_", "OMPI_", "PMI_", "PALS_", "MPI_LOCAL")):
                    patch.delenv(key)
            for key, value in {
                "MASTER_ADDR": "node01",
                "MASTER_PORT": "45679",
                **variables,
            }.items():
                patch.setenv(key, value)
            calls = []
            patch.setattr(torch.distributed, "is_initialized", lambda: False)
            patch.setattr(
                torch.distributed,
                "init_process_group",
                lambda calls=calls, **kwargs: calls.append(kwargs),
            )
            if initialize is legacy_init:
                rank = initialize(
                    Namespace(distributed=True, launcher=launcher, device="cpu")
                )
            else:
                result = initialize(True, launcher, "cpu")
                rank = (result.rank, result.local_rank, result.world_size)
            assert len(calls) == 1
            assert calls[0]["backend"] == "gloo" and calls[0]["init_method"] == "env://"
            outputs.append(
                (
                    rank,
                    {
                        name: os.environ[name]
                        for name in (
                            "RANK",
                            "LOCAL_RANK",
                            "WORLD_SIZE",
                            "MASTER_ADDR",
                            "MASTER_PORT",
                        )
                    },
                )
            )
    assert outputs[0] == outputs[1]
    assert outputs[0][0] == (5, 1, 8)
    assert outputs[0][1]["MASTER_PORT"] == "45679"
