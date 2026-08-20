"""`--keep_checkpoints`, `--save_all_checkpoints` and `--clip_grad`.

Three flags whose effect is a file that is or is not deleted, and a gradient that
is or is not shortened. All three are invisible in a passing run, which is why
none of them had a test: the training finishes either way.

`--keep_checkpoints` is the one with teeth. `CheckpointIO.save` deletes the
previous file unless it is told to keep it, so a run with the flag off leaves one
checkpoint behind and a run with it on leaves every epoch's. Getting that backwards
either fills a disk or throws away the history a restart needs.
"""

import pytest
import torch

from mace.tools.checkpoint import CheckpointIO


def state(value):
    """The smallest thing `torch.save` will take and `torch.load` will return."""
    return {"weights": torch.tensor([float(value)])}


def saved(directory):
    return sorted(p.name for p in directory.iterdir() if p.suffix == ".pt")


# ---------------------------------------------------------------------------
# --keep_checkpoints
# ---------------------------------------------------------------------------


def test_without_keep_only_the_latest_checkpoint_survives(tmp_path):
    io = CheckpointIO(directory=str(tmp_path), tag="run", keep=False)

    io.save(state(1), epochs=1)
    io.save(state(2), epochs=2)
    io.save(state(3), epochs=3)

    assert len(saved(tmp_path)) == 1, saved(tmp_path)
    assert "epoch-3" in saved(tmp_path)[0]


def test_with_keep_every_epoch_is_kept(tmp_path):
    io = CheckpointIO(directory=str(tmp_path), tag="run", keep=True)

    io.save(state(1), epochs=1)
    io.save(state(2), epochs=2)
    io.save(state(3), epochs=3)

    assert len(saved(tmp_path)) == 3, saved(tmp_path)


def test_keep_last_spares_one_file_even_without_keep(tmp_path):
    """The third state: `keep_last` is how a caller protects one checkpoint,
    which is what the stage-two swap uses so the pre-swap model is not deleted."""
    io = CheckpointIO(directory=str(tmp_path), tag="run", keep=False)

    io.save(state(1), epochs=1)
    io.save(state(2), epochs=2, keep_last=True)
    io.save(state(3), epochs=3)

    assert len(saved(tmp_path)) == 2, saved(tmp_path)


def test_the_saved_checkpoint_is_what_comes_back(tmp_path):
    """Retention is only useful if the file is loadable, and the epoch in the
    name has to be the epoch of the contents."""
    io = CheckpointIO(directory=str(tmp_path), tag="run", keep=True)
    io.save(state(7), epochs=4)

    path = tmp_path / saved(tmp_path)[0]
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    assert "epoch-4" in path.name
    assert loaded["weights"].item() == 7.0


# ---------------------------------------------------------------------------
# --clip_grad
# ---------------------------------------------------------------------------


def test_clipping_shortens_a_gradient_that_is_too_long():
    """`take_step` calls `clip_grad_norm_` with the flag as the maximum, so the
    property being relied on is that the norm afterwards is the bound."""
    weight = torch.nn.Parameter(torch.zeros(3))
    weight.grad = torch.tensor([3.0, 4.0, 0.0])  # norm 5

    torch.nn.utils.clip_grad_norm_([weight], max_norm=1.0)

    assert torch.linalg.vector_norm(weight.grad).item() == pytest.approx(1.0)


def test_clipping_leaves_a_short_gradient_alone():
    weight = torch.nn.Parameter(torch.zeros(3))
    weight.grad = torch.tensor([0.3, 0.4, 0.0])  # norm 0.5

    torch.nn.utils.clip_grad_norm_([weight], max_norm=1.0)

    assert torch.linalg.vector_norm(weight.grad).item() == pytest.approx(0.5)


def test_clipping_keeps_the_direction():
    """A shorter gradient pointing somewhere else would be a different bug from
    no clipping at all, and both look like "training is worse"."""
    weight = torch.nn.Parameter(torch.zeros(2))
    weight.grad = torch.tensor([3.0, 4.0])
    before = weight.grad / torch.linalg.vector_norm(weight.grad)

    torch.nn.utils.clip_grad_norm_([weight], max_norm=0.5)
    after = weight.grad / torch.linalg.vector_norm(weight.grad)

    assert torch.allclose(before, after)
