"""`mace_prepare_data` reads the property keys it is told to read.

The preprocessing CLI takes its own `--virials_key`, `--dipole_key`,
`--charges_key` and `--polarizability_key`, and nothing passed them. The training
side of the same convention is covered, but preprocessing is a separate parser
writing a separate artifact: a key that stops being read there produces shards
with the property missing, and training then fails, or worse trains on zeros, a
long way from the flag that caused it.

Each key is checked by writing the property under a NON-default name and requiring
it to arrive in the shard. Using the default name would pass whether the flag was
read or ignored.
"""

import json

import ase.io
import numpy as np
import pytest
from ase import Atoms

from tests.helpers import preprocess_data, run_mace_train

pytest.importorskip("h5py")
import h5py  # noqa: E402  # pylint: disable=wrong-import-position,wrong-import-order


@pytest.fixture(name="configs")
def fixture_configs():
    """Water plus the two isolated atoms preprocessing needs for E0s, with every
    property under a deliberately unusual key."""
    isolated = [
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3),
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3),
    ]
    for atoms in isolated:
        atoms.info["MY_energy"] = 0.0
        atoms.info["config_type"] = "IsolatedAtom"

    rng = np.random.default_rng(5)
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        cell=[6] * 3,
        pbc=True,
    )
    frames = []
    for _ in range(10):
        frame = water.copy()
        frame.positions += rng.normal(0.1, size=frame.positions.shape)
        frame.info["MY_energy"] = float(rng.normal(0.1))
        frame.new_array("MY_forces", rng.normal(0.1, size=frame.positions.shape))
        frame.info["MY_virials"] = rng.normal(0.1, size=(3, 3))
        frame.info["MY_dipole"] = rng.normal(0.1, size=3)
        frame.new_array("MY_charges", rng.normal(0.1, size=len(frame)))
        frames.append(frame)
    return isolated + frames


def preprocess(tmp_path, configs, **extra):
    ase.io.write(tmp_path / "sample.xyz", configs)
    params = {
        "train_file": tmp_path / "sample.xyz",
        "r_max": 5.0,
        "config_type_weights": "{'Default':1.0}",
        "num_process": 1,
        "valid_fraction": 0.1,
        "h5_prefix": tmp_path / "pre_",
        "compute_statistics": None,
        "seed": 42,
        "energy_key": "MY_energy",
        "forces_key": "MY_forces",
    }
    params.update(extra)
    result = run_mace_train(params, script=preprocess_data)
    assert result.returncode == 0
    return tmp_path


def shard_shapes(tmp_path, name):
    """The shapes one property takes across every configuration in the shards.

    A property that was read arrives with its real shape; one that was not is
    written as a 0-d placeholder. That distinction is the discriminator here,
    rather than the values, which are random.
    """
    shapes = set()
    for path in sorted((tmp_path / "pre_train").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            for batch in handle.values():
                for config in batch.values():
                    properties = config["properties"]
                    if name in properties:
                        shapes.add(np.asarray(properties[name]).shape)
    return shapes


def test_the_energy_and_forces_keys_reach_the_shards(tmp_path, configs):
    """The baseline: the two keys the existing test already passes, so a failure
    below is about the flag under test and not about the fixture."""
    preprocess(tmp_path, configs)

    assert (tmp_path / "pre_statistics.json").is_file()
    statistics = json.loads((tmp_path / "pre_statistics.json").read_text())
    assert "atomic_energies" in statistics


@pytest.mark.parametrize(
    "flag,info_key,shard_name,shape",
    [
        ("virials_key", "MY_virials", "virials", (3, 3)),
        ("dipole_key", "MY_dipole", "dipole", (3,)),
        ("charges_key", "MY_charges", "charges", (3,)),
    ],
)
def test_a_property_key_is_read_under_the_name_it_is_given(
    tmp_path, configs, flag, info_key, shard_name, shape
):
    preprocess(tmp_path, configs, **{flag: info_key})

    assert shard_shapes(tmp_path, shard_name) == {shape}


@pytest.mark.parametrize(
    "shard_name", ["virials", "dipole", "charges"]
)
def test_the_property_is_a_placeholder_when_no_key_is_given(
    tmp_path, configs, shard_name
):
    """The other half. The property is present in the shard either way, so a test
    that only checked presence would pass on a CLI that ignored the flag."""
    preprocess(tmp_path, configs)

    assert shard_shapes(tmp_path, shard_name) == {()}
