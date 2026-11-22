"""The start method DataLoader workers are launched with.

Python 3.14 made forkserver the Linux default. A forkserver worker is sent the
dataset over a socket, and under torch's file_descriptor sharing strategy each
tensor goes as its own descriptor, past the 256 that forkserver accepts. The
in-memory dataset `run_train` builds has far more tensors than that, so every
run with `--num_workers` died on `ValueError: too many fds` before training.
"""

import sys

import pytest
import torch
from torch.utils.data import DataLoader

from mace.tools import worker_start_method


def test_no_workers_keeps_the_default():
    """DataLoader rejects a context when it starts no worker processes."""
    assert worker_start_method(0) is None


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS keeps spawn")
def test_workers_fork_off_macos():
    assert worker_start_method(2) == "fork"


@pytest.mark.skipif(sys.platform != "darwin", reason="only macOS keeps spawn")
def test_macos_keeps_its_default():
    assert worker_start_method(2) is None


def test_a_dataset_of_many_tensors_reaches_the_workers():
    """More tensors than forkserver can pass as descriptors."""
    dataset = [torch.full((3,), float(i)) for i in range(400)]
    loader = DataLoader(
        dataset,
        batch_size=50,
        num_workers=2,
        multiprocessing_context=worker_start_method(2),
    )

    total = sum(batch.sum().item() for batch in loader)

    assert total == 3 * sum(range(400))
