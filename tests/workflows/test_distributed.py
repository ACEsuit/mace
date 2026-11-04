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
import pytest

from tests.helpers import REPO_ROOT, base_mace_params, run_train


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("compile_training", [False, True])
def test_run_train_distributed_cpu(tmp_path, fitting_configs, compile_training):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    params = base_mace_params()
    params.update(
        {
            "name": "ddp_compiled" if compile_training else "ddp_eager",
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
    if compile_training:
        params["compile_training"] = None
        params.update(
            {
                "model": "ScaleShiftMACE",
                "hidden_irreps": "8x0e + 8x1o",
                "correlation": 2,
                "use_reduced_cg": True,
            }
        )
    argv = [(f"--{k}={v}" if v is not None else f"--{k}") for k, v in params.items()]

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

    failures = [
        f"rank {rank} failed (rc={proc.returncode})\n{out[-12000:]}"
        for rank, (proc, out) in enumerate(zip(procs, outputs))
        if proc.returncode != 0
    ]
    assert not failures, "\n\n".join(failures)
    model_name = "ddp_compiled" if compile_training else "ddp_eager"
    assert (tmp_path / f"{model_name}.model").exists()
    assert (tmp_path / f"{model_name}_stagetwo.model").exists()
    combined_output = "\n".join(outputs)
    if compile_training:
        assert "Compiled training DDP active" in combined_output
        assert "Compiler-friendly distributed training fallback" not in combined_output
