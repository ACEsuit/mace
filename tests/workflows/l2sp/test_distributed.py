"""Two CPU ranks exercise frozen parameters through stage-two evaluation."""

import os
import socket
import subprocess
import sys
from contextlib import ExitStack

import pytest
import torch

from tests.golden.anchors import load_anchor
from tests.helpers import REPO_ROOT, run_train
from tests.workflows.l2sp.test_training import latest_checkpoint, training_args


@pytest.mark.timeout(180)
def test_distributed_l2sp_preserves_frozen_parameters(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args = training_args(
        tmp_path, distributed=None, launcher="torchrun", start_swa=1, valid_batch_size=1
    )
    argv = [
        f"--{key}" if value is None else f"--{key}={value}"
        for key, value in args.items()
    ]
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE="2",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo",
    )
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    processes = []
    with ExitStack() as stack:
        try:
            for rank in range(2):
                log = stack.enter_context((tmp_path / f"rank{rank}.log").open("w"))
                processes.append(
                    # Both ranks must start before waiting; cleanup is in finally.
                    subprocess.Popen(  # pylint: disable=consider-using-with
                        [sys.executable, str(run_train), *argv],
                        env=dict(env, RANK=str(rank), LOCAL_RANK=str(rank)),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            for rank, process in enumerate(processes):
                assert process.wait(timeout=120) == 0, (
                    tmp_path / f"rank{rank}.log"
                ).read_text()[-6000:]
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()
    checkpoint = latest_checkpoint(tmp_path)
    initial = load_anchor("tiny_scaleshift").state_dict()
    assert not torch.equal(
        checkpoint["model"]["node_embedding.linear.weight"],
        initial["node_embedding.linear.weight"],
    )
    for name, tensor in checkpoint["model"].items():
        if name.startswith(("products.", "readouts.", "interactions.1.")):
            torch.testing.assert_close(tensor, initial[name], atol=0, rtol=0)
    exported = torch.load(tmp_path / "l2sp_stagetwo.model", weights_only=False)
    for name, param in exported.named_parameters():
        assert param.requires_grad == name.startswith(
            ("node_embedding.", "interactions.0.")
        )
