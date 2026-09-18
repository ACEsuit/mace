"""`--multi_processed_test`, which is a statement about a directory layout.

Both branches read `--test_dir`, and they disagree about what is in it: without
the flag the directory holds files whose names end `_test.h5`, one dataset each;
with it the directory holds one *folder* per test set, each holding the shards a
multi-process preprocessing wrote. Nothing covered either branch, and nothing
covered what happens when the flag and the layout disagree -- which is the part
worth knowing, because it is not an error.

The shards come from a real `mace_prepare_data` run, so the layouts are the ones
the tool actually produces rather than a test's idea of them.
"""

import shutil

import ase.io
import numpy as np
import pytest
from ase import Atoms

from tests.helpers import base_mace_params, preprocess_data, run_mace_train


@pytest.fixture(name="prepared", scope="module")
def fixture_prepared(tmp_path_factory):
    """One preprocessing run, giving two shards of one named test set."""
    tmp = tmp_path_factory.mktemp("prepared")
    rng = np.random.default_rng(0)
    configs = []
    for numbers in ([8], [1]):
        isolated = Atoms(numbers=numbers, positions=[[0, 0, 0]], cell=[6] * 3, pbc=True)
        isolated.info.update({"REF_energy": -1.0, "config_type": "IsolatedAtom"})
        configs.append(isolated)
    for _ in range(12):
        atoms = Atoms(
            "H2O",
            positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]],
            cell=[6] * 3,
            pbc=True,
        )
        atoms.positions += rng.normal(0, 0.05, size=atoms.positions.shape)
        atoms.info["REF_energy"] = float(rng.normal())
        atoms.new_array("REF_forces", rng.normal(size=(3, 3)))
        configs.append(atoms)
    ase.io.write(tmp / "train.xyz", configs)

    held_out = [atoms.copy() for atoms in configs[2:6]]
    for atoms in held_out:
        atoms.info["config_type"] = "held_out"
    ase.io.write(tmp / "test.xyz", configs[:2] + held_out)

    done = run_mace_train(
        {
            "train_file": str(tmp / "train.xyz"),
            "test_file": str(tmp / "test.xyz"),
            "r_max": 4.0,
            "num_process": 2,
            "valid_fraction": 0.2,
            "h5_prefix": str(tmp / "pre_"),
            "seed": 1,
            "energy_key": "REF_energy",
            "forces_key": "REF_forces",
        },
        script=preprocess_data,
    )
    assert done.returncode == 0
    shards = sorted((tmp / "pre_test").glob("held_out*.h5"))
    assert len(shards) == 2, sorted(p.name for p in (tmp / "pre_test").iterdir())
    return tmp, shards


@pytest.fixture(name="flat")
def fixture_flat(prepared, tmp_path):
    """The layout the flag-off branch looks for: `<name>_test.h5` files."""
    _, shards = prepared
    directory = tmp_path / "flat"
    directory.mkdir()
    shutil.copy(shards[0], directory / "held_out_test.h5")
    return directory


@pytest.fixture(name="sharded")
def fixture_sharded(prepared, tmp_path):
    """The layout the flag-on branch looks for: one folder of shards per set."""
    _, shards = prepared
    directory = tmp_path / "sharded" / "held_out"
    directory.mkdir(parents=True)
    for shard in shards:
        shutil.copy(shard, directory / shard.name)
    return directory.parent


def train(prepared, tmp_path, test_dir, multi_processed, check=True):
    source, _ = prepared
    params = base_mace_params()
    params.update(
        {
            "name": "mpt",
            "hidden_irreps": "8x0e",
            "r_max": 4.0,
            "checkpoints_dir": str(tmp_path / "ckpt"),
            "model_dir": str(tmp_path / "model"),
            "results_dir": str(tmp_path / "results"),
            "log_dir": str(tmp_path / "logs"),
            "train_file": str(source / "train.xyz"),
            "test_dir": str(test_dir),
            "multi_processed_test": multi_processed,
            "max_num_epochs": 1,
            "seed": 5,
            "loss": "weighted",
            "E0s": "isolated",
        }
    )
    params.pop("swa", None)
    params.pop("start_swa", None)
    params.pop("stress_key", None)
    return run_mace_train(params, check=check, capture_output=True, text=True)


def table_rows(stdout):
    """The `config_type` cells of every error table the run printed."""
    return [
        line.split("|")[1].strip()
        for line in stdout.splitlines()
        if line.startswith("|") and "config_type" not in line and "---" not in line
    ]


# ---------------------------------------------------------------------------
# each branch against the layout it is for
# ---------------------------------------------------------------------------


def test_flat_files_are_found_without_the_flag(prepared, tmp_path, flat):
    """`get_files_with_suffix(test_dir, "_test.h5")`, one dataset per file, named
    after the file and prefixed with the head."""
    done = train(prepared, tmp_path, flat, "False")

    assert "Default_held_out_test" in table_rows(done.stdout), done.stdout[-2000:]


def test_folders_of_shards_are_found_with_the_flag(prepared, tmp_path, sharded):
    """`glob(test_dir + "/*")`, one dataset per folder, named after the folder.
    This is the layout `mace_prepare_data --num_process N` is for."""
    done = train(prepared, tmp_path, sharded, "True", check=False)
    assert done.returncode == 0, done.stderr[-3000:]

    assert "Default_held_out" in table_rows(done.stdout), done.stdout[-2000:]


# ---------------------------------------------------------------------------
# and against the other one
# ---------------------------------------------------------------------------


def test_a_sharded_directory_read_as_flat_files_yields_no_test_set(
    prepared, tmp_path, sharded
):
    """Recorded. The suffix filter matches nothing, so the run trains and reports
    exactly as if `--test_dir` had not been given: no error, no warning, no test
    row. Getting the flag wrong costs the evaluation silently."""
    done = train(prepared, tmp_path, sharded, "False")

    assert done.returncode == 0
    assert not [row for row in table_rows(done.stdout) if "held_out" in row]


def test_flat_files_read_as_folders_of_shards_stop_the_run(prepared, tmp_path, flat):
    """The mismatch the other way round, and the loud one. Each `.h5` file is
    treated as a folder, the shard glob inside it returns nothing, and torch
    refuses to build a `ConcatDataset` of no datasets. So this pairing costs the
    run rather than the evaluation -- the opposite of the case above, and the
    reason both are worth having: the same wrong flag fails differently
    depending on which way it is wrong."""
    done = train(prepared, tmp_path, flat, "True", check=False)

    assert done.returncode != 0
    assert "datasets should not be an empty iterable" in done.stderr, done.stderr[-2000:]
