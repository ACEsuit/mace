"""A training set smaller than one batch fails where the cause is, or trains.

`drop_last` throws away an incomplete final batch. When the set is smaller than
`--batch_size` that is the only batch, so the loader comes out empty and the run
cannot train. It used to say so several frames away, in
`compute_avg_num_neighbors`, on a `torch.cat()` of an empty list naming neither
the batch size nor the set -- and a set of fewer than ten configurations reaches
this on default flags, because `--batch_size` defaults to 10.

The other half matters as much: when the partial batch is *kept*, which is what
`--lbfgs` does, nothing is wrong and the run must still work.

A distributed run reaches the same empty loader by a different route, and the
advice has to differ with it. There the loader keeps its partial batch and the
*sampler* holds the `drop_last`, splitting the set across ranks and discarding
the remainder, so a set smaller than the world size leaves a rank with nothing.
Lowering the batch size cannot fix that, and a message that blames it can be
plainly false -- two configurations over four ranks with `--batch_size 1` would
otherwise be reported as a set "fewer than --batch_size (1)".
"""

import os
import socket
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from tests.helpers import REPO_ROOT, base_mace_params, run_mace_train, run_train

#: fewer than the CLI's default `--batch_size` of 10, which is what makes this
#: reachable without asking for anything unusual. `base_mace_params` sets its own
#: batch size, so the cases below state one explicitly rather than inheriting it.
SMALL = 8
OVERSIZED_BATCH = 10


@pytest.fixture(name="small_set")
def fixture_small_set(tmp_path):
    rng = np.random.default_rng(0)
    frames = []
    for index in range(SMALL):
        atoms = Atoms(
            "H2O",
            positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]],
            cell=[8, 8, 8],
            pbc=True,
        )
        atoms.positions += rng.normal(0, 0.05, (3, 3))
        atoms.info["REF_energy"] = float(-10 + 0.1 * index)
        atoms.arrays["REF_forces"] = rng.normal(0, 0.1, (3, 3))
        frames.append(atoms)
    path = tmp_path / "small.xyz"
    write(path, frames)
    return path


def _params(tmp_path, small_set, **overrides):
    params = base_mace_params()
    params.update(
        {
            "train_file": str(small_set),
            "valid_fraction": 0.1,
            "E0s": "{1:-1.0, 8:-5.0}",
            "max_num_epochs": 1,
            "device": "cpu",
            "default_dtype": "float64",
            "model_dir": str(tmp_path),
            "checkpoints_dir": str(tmp_path),
            "results_dir": str(tmp_path),
            "log_dir": str(tmp_path),
        }
    )
    params.pop("swa", None)
    params.pop("start_swa", None)
    params.update(overrides)
    return params


def test_a_set_smaller_than_the_batch_fails_naming_the_batch_size(
    tmp_path, small_set
):
    """The default regime drops the only batch, so it cannot train. It has to
    fail on that, not on an empty-tensor error from a statistics helper.

    `OVERSIZED_BATCH` is the CLI's own default, so this is what a user gets from
    `mace_run_train` on a set of eight configurations with no tuning at all.
    """
    result = run_mace_train(
        _params(tmp_path, small_set, name="small", batch_size=OVERSIZED_BATCH),
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0, "a run with an empty training loader succeeded"
    assert "--batch_size" in output, (
        "the failure does not name the flag the user has to change:\n"
        + output[-2000:]
    )
    assert "non-empty list of Tensors" not in output, (
        "the run still dies in compute_avg_num_neighbors rather than at the "
        "check:\n" + output[-2000:]
    )


def test_keeping_the_partial_batch_still_trains(tmp_path, small_set):
    """`--lbfgs` sets `drop_last=False`, so the same set is fine. The guard must
    key on the loader being empty and not on the set being small."""
    run_mace_train(
        _params(
            tmp_path, small_set, name="kept", lbfgs=None, batch_size=OVERSIZED_BATCH
        )
    )


def test_a_batch_that_fits_is_untouched(tmp_path, small_set):
    """The ordinary case, so the guard cannot fire on a set it should accept."""
    run_mace_train(_params(tmp_path, small_set, name="fits", batch_size=4))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.timeout(300)
def test_a_rank_with_no_data_blames_the_sampler_and_not_the_batch_size(
    tmp_path, small_set
):
    """Four ranks over a set of seven: the sampler drops the remainder, so with
    `--batch_size 1` the batch size is provably not the cause and must not be
    what the message names."""
    params = _params(tmp_path, small_set, name="ddp", batch_size=1)
    params.update({"distributed": None, "launcher": "torchrun"})
    argv = [(f"--{k}={v}" if v is not None else f"--{k}") for k, v in params.items()]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_free_port()),
            "WORLD_SIZE": "8",  # more ranks than configurations, so a rank gets none
            "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
        }
    )

    procs = [
        subprocess.Popen(
            [sys.executable, str(run_train)] + argv,
            env=dict(env, RANK=str(rank), LOCAL_RANK=str(rank)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in range(8)
    ]
    outputs = [proc.communicate()[0] for proc in procs]

    assert all(proc.returncode != 0 for proc in procs), "a rank with no data trained"
    combined = "\n".join(outputs)
    assert "no training data" in combined, combined[-2000:]
    assert "Lower --batch_size" not in combined, (
        "the distributed case is still being blamed on the batch size:\n"
        + combined[-2000:]
    )
