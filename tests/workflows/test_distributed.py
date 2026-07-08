"""DDP smoke test: 2-process CPU training (gloo backend).

First CI coverage of mace/tools/distributed_tools.py: an actual multi-process
run of run_train --distributed --launcher torchrun. The two ranks are spawned
directly with a static rendezvous on 127.0.0.1 (instead of the torchrun
launcher) so the test does not depend on the host's DNS/hostname resolution.
GPU DDP (nccl) stays out of scope — it needs a multi-GPU runner.
"""

import os
import socket
import subprocess
import sys

import ase.io

from tests.helpers import REPO_ROOT, base_mace_params, run_train


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_run_train_distributed_cpu(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    params = base_mace_params()
    params.update(
        {
            "name": "ddp",
            "hidden_irreps": "16x0e",
            "checkpoints_dir": str(tmp_path),
            "model_dir": str(tmp_path),
            "train_file": str(tmp_path / "fit.xyz"),
            "max_num_epochs": 4,
            "start_swa": 2,
            "distributed": None,
            "launcher": "torchrun",  # reads RANK/WORLD_SIZE from the env
            "device": "cpu",
        }
    )
    argv = [
        (f"--{k}={v}" if v is not None else f"--{k}") for k, v in params.items()
    ]

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(REPO_ROOT) + ":" + base_env.get("PYTHONPATH", "")
    base_env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_free_port()),
            "WORLD_SIZE": "2",
            # gloo must not try to resolve the host's external interface.
            "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
        }
    )

    procs = []
    for rank in range(2):
        env = dict(base_env, RANK=str(rank), LOCAL_RANK=str(rank))
        procs.append(
            subprocess.Popen(
                [sys.executable, str(run_train)] + argv,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    outputs = []
    for proc in procs:
        out, _ = proc.communicate(timeout=800)
        outputs.append(out)

    for rank, (proc, out) in enumerate(zip(procs, outputs)):
        assert proc.returncode == 0, (
            f"rank {rank} failed (rc={proc.returncode})\n{out[-4000:]}"
        )
    assert (tmp_path / "ddp.model").exists()
