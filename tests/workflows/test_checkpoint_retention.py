"""`--save_all_checkpoints` and `--keep_checkpoints`, through a real run.

The retention mechanism is pinned in
`tests/unit/test_checkpoint_retention_and_clipping.py`: `CheckpointIO.save`
deletes the previous file unless told to keep it. What that cannot say is whether
the flags reach it, and the two flags reach it differently. `--keep_checkpoints`
sets `keep` on the handler once; `--save_all_checkpoints` leaves `keep` alone and
instead saves again at every evaluation with `keep_last=True`.

So a run with neither leaves one checkpoint, and a run with either leaves one per
evaluation. Getting that backwards either fills a disk or throws away the history
a restart needs, and a training run reports neither.
"""

from pathlib import Path

import ase.io
import pytest

from tests.helpers import base_mace_params, make_fitting_configs, run_mace_train

#: Long enough that the validation loss improves more than once: a checkpoint is
#: written when it improves, so with too few epochs `--keep_checkpoints` has
#: nothing to accumulate and the test would be about the schedule, not the flag.
EPOCHS = 8


def train(tmp_path, name, **extra):
    ase.io.write(tmp_path / "fit.xyz", make_fitting_configs())
    checkpoints = tmp_path / f"ckpt_{name}"
    params = base_mace_params()
    params.update(
        {
            "name": name,
            "hidden_irreps": "8x0e",
            "checkpoints_dir": str(checkpoints),
            "model_dir": str(tmp_path / "model"),
            "results_dir": str(tmp_path / "results"),
            "log_dir": str(tmp_path / "logs"),
            "train_file": str(tmp_path / "fit.xyz"),
            "max_num_epochs": EPOCHS,
            "eval_interval": 1,
        }
    )
    params.pop("swa", None)
    params.pop("start_swa", None)
    params.update(extra)
    result = run_mace_train(params)
    assert result.returncode == 0
    return sorted(p.name for p in checkpoints.glob("*.pt"))


def test_a_plain_run_leaves_one_checkpoint(tmp_path):
    kept = train(tmp_path, "plain")

    assert len(kept) == 1, kept


def test_save_all_checkpoints_leaves_one_per_evaluation(tmp_path):
    """The flag's whole effect, and the reason it exists."""
    kept = train(tmp_path, "all", save_all_checkpoints=None)

    assert len(kept) > 1, kept
    epochs = {name.split("epoch-")[1].split(".")[0] for name in kept if "epoch-" in name}
    assert len(epochs) > 1, f"several files but one epoch: {kept}"


def test_keep_checkpoints_accumulates_what_a_plain_run_deletes(tmp_path):
    """A different route to the same outcome: `keep` on the handler rather than
    `keep_last` per save. Compared against a plain run of the same length rather
    than against a number, because how many checkpoints exist depends on how often
    the validation loss improved, and only the difference between the two runs is
    the flag's doing.
    """
    plain = train(tmp_path, "plain_ref")
    kept = train(tmp_path, "keep", keep_checkpoints=None)

    assert len(plain) == 1, plain
    assert len(kept) > len(plain), (plain, kept)
