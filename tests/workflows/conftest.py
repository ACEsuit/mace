"""Fixtures shared by the end-to-end contract suites.

These live here rather than in each file because two of the contract suites
need the same artefacts -- the committed anchors, the committed regression
dataset, and one multi-head model that only a fine-tuning run can produce --
and building them twice would double the cost of the slowest directory in the
tree.

Everything here is offline: the anchors and the dataset are committed, and the
fine-tuning fixture uses an anchor as its foundation model. Nothing in this
file downloads.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import ase.io
import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = TESTS_ROOT / "golden"

#: The two committed P0-1 parity anchors. `tiny_scaleshift` is what the
#: training CLI actually emits for `--model MACE`; `tiny_mace` is the plain
#: class, and the only one of the two whose forward returns the per-body-order
#: `contributions` the eval CLI can be asked for.
ANCHOR_SCALESHIFT = GOLDEN_ROOT / "models" / "tiny_scaleshift.model"
ANCHOR_MACE = GOLDEN_ROOT / "models" / "tiny_mace.model"

#: The committed end-to-end training set: surfaces, molecules, bulk, dipoles
#: and partial charges, all labelled by one closed-form potential so that
#: "the loss went down" is a statement about the optimiser. See
#: tests/golden/make_regression_set.py.
REGRESSION_SET = GOLDEN_ROOT / "datasets" / "regression_train.xyz"


@pytest.fixture(name="anchor_scaleshift", scope="session")
def fixture_anchor_scaleshift() -> Path:
    assert ANCHOR_SCALESHIFT.exists(), ANCHOR_SCALESHIFT
    return ANCHOR_SCALESHIFT


@pytest.fixture(name="anchor_mace", scope="session")
def fixture_anchor_mace() -> Path:
    assert ANCHOR_MACE.exists(), ANCHOR_MACE
    return ANCHOR_MACE


@pytest.fixture(name="regression_set", scope="session")
def fixture_regression_set() -> Path:
    assert REGRESSION_SET.exists(), REGRESSION_SET
    return REGRESSION_SET


@pytest.fixture(name="anchor_copy")
def fixture_anchor_copy(anchor_scaleshift, tmp_path) -> Path:
    """A per-test copy of the ScaleShiftMACE anchor.

    Several CLIs write their artefact *next to* the model they were given, so
    a test that pointed them at the committed file would write into the
    repository -- and two of them running in parallel would race.
    """
    dest = tmp_path / "anchor.model"
    shutil.copy(anchor_scaleshift, dest)
    return dest


@pytest.fixture(name="finetuned_multihead_model", scope="session")
def fixture_finetuned_multihead_model(tmp_path_factory) -> Path:
    """One offline multi-head fine-tuning run, shared by everything that needs
    a model with more than one head.

    Two contract suites need such a model and neither can build one cheaply:
    ``mace_select_head`` has nothing to select from without it, and the
    fine-tuning contracts assert on the heads it carries. It comes from an
    anchor rather than from a downloaded foundation model, so it costs a few
    seconds and no network.
    """
    from tests.helpers import run_mace_train  # noqa: PLC0415

    work = tmp_path_factory.mktemp("finetuned_multihead")
    finetune, replay = split_regression_set(work)
    run_mace_train(
        {
            "name": "mh",
            "train_file": str(finetune),
            "valid_fraction": 0.25,
            "E0s": "isolated",
            "loss": "weighted",
            "batch_size": 4,
            "valid_batch_size": 4,
            "max_num_epochs": 2,
            "eval_interval": 1,
            "device": "cpu",
            "default_dtype": "float64",
            "seed": 11,
            "foundation_model": str(ANCHOR_SCALESHIFT),
            "multiheads_finetuning": True,
            "pt_train_file": str(replay),
            "force_mh_ft_lr": True,
            "lr": 0.005,
            "error_table": "PerAtomRMSE",
            "save_cpu": None,
            "model_dir": str(work),
            "checkpoints_dir": str(work),
            "results_dir": str(work),
            "log_dir": str(work),
        }
    )
    model = work / "mh.model"
    assert model.exists()
    return model


def split_regression_set(destination: Path):
    """Split the committed regression set into a fine-tuning and a replay half.

    The isolated atoms go to both halves, because each head needs its own E0
    table. The fine-tuning half is the molecules with their labels scaled, so
    it is a genuinely different level of theory from what the anchor was
    trained on and adapting to it is measurable; the replay half is the
    condensed-phase configurations, which is the shape real replay data has.
    """
    configs = ase.io.read(REGRESSION_SET, index=":")
    isolated = [a for a in configs if a.info.get("config_type") == "IsolatedAtom"]

    finetune = [a.copy() for a in isolated]
    for atoms in (a for a in configs if a.info.get("config_type") == "molecule"):
        shifted = atoms.copy()
        shifted.info["REF_energy"] = 1.3 * atoms.info["REF_energy"] + 0.05 * len(atoms)
        shifted.arrays["REF_forces"] = 1.3 * atoms.arrays["REF_forces"]
        finetune.append(shifted)

    replay = [a.copy() for a in isolated]
    replay += [
        a.copy()
        for a in configs
        if a.info.get("config_type") in ("bulk", "bulk_triclinic", "surface")
    ]

    destination.mkdir(parents=True, exist_ok=True)
    finetune_path = destination / "finetune.xyz"
    replay_path = destination / "replay.xyz"
    ase.io.write(finetune_path, finetune)
    ase.io.write(replay_path, replay)
    return finetune_path, replay_path
