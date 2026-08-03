import logging
import tempfile

import numpy as np
import torch
import torch.nn.functional
from torch import nn, optim

from mace.tools import (
    AtomicNumberTable,
    CheckpointHandler,
    CheckpointState,
    atomic_numbers_to_indices,
)


def test_atomic_number_table():
    table = AtomicNumberTable(zs=[1, 8])
    array = np.array([8, 8, 1])
    indices = atomic_numbers_to_indices(array, z_table=table)
    expected = np.array([1, 1, 0], dtype=int)
    assert np.allclose(expected, indices)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x):
        return torch.nn.functional.relu(self.linear(x))


def test_save_load():
    model = MyModel()
    initial_lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    with tempfile.TemporaryDirectory() as directory:
        handler = CheckpointHandler(directory=directory, tag="test", keep=True)
        handler.save(state=CheckpointState(model, optimizer, scheduler), epochs=50)

        optimizer.step()
        scheduler.step()
        assert not np.isclose(optimizer.param_groups[0]["lr"], initial_lr)

        handler.load_latest(state=CheckpointState(model, optimizer, scheduler))
        assert np.isclose(optimizer.param_groups[0]["lr"], initial_lr)


def test_load_warns_and_resumes_on_optimizer_group_mismatch(caplog):
    """A checkpoint whose optimizer has a different number of parameter groups
    (e.g. get_params_options changed between runs) doesnt abort the restart:
    the model weights load, a warning is logged, and the epoch is preserved so
    training resumes with a freshly initialized optimizer."""
    model = MyModel()
    saved_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    saved_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=saved_optimizer, gamma=0.99
    )

    with tempfile.TemporaryDirectory() as directory:
        handler = CheckpointHandler(directory=directory, tag="test", keep=True)
        handler.save(
            state=CheckpointState(model, saved_optimizer, saved_scheduler), epochs=50
        )

        # Rebuild the optimizer with two parameter groups instead of one.
        parameters = list(model.parameters())
        new_optimizer = optim.SGD(
            [{"params": parameters[:1]}, {"params": parameters[1:]}],
            lr=0.001,
            momentum=0.9,
        )
        new_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=new_optimizer, gamma=0.99
        )

        with caplog.at_level(logging.WARNING):
            epoch = handler.load_latest(
                state=CheckpointState(model, new_optimizer, new_scheduler)
            )

    assert epoch == 50
    assert "could not restore optimizer" in caplog.text.lower()
