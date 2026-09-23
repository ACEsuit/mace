"""A fine-tune's data: the replay head, the foundation's table, and the guard.

The claim that runs through this file is that a replay head is not a kind of
its own. It is a head whose structures come from a published dataset, and every
test that exercises one exercises the same steps a file head goes through.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.elements import AtomicNumberTable
from mace_core.observables import load_default_catalogue
from mace_torch.data import GraphDataset
from mace_torch.finetune.foundation import FoundationContext
from mace_torch.finetune.replay import cached_path
from mace_torch.train import DataStageError, run_data_stage

CATALOGUE = load_default_catalogue()
HYDROGEN, OXYGEN = -13.6, -2040.0
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def waters(count: int, seed: int = 0, isolated: bool = True, reserved: bool = False):
    """Waters, optionally with their isolated atoms, labelled one of two ways.

    ``reserved`` labels with ase's own names through a calculator, which is
    how the published replay datasets are written.
    """
    generator = np.random.default_rng(seed)
    frames = []
    if isolated:
        for number, energy in ((1, HYDROGEN), (8, OXYGEN)):
            atom = Atoms(numbers=[number], positions=[[0.0, 0.0, 0.0]])
            atom.info["REF_energy"] = energy
            atom.info["config_type"] = "IsolatedAtom"
            atom.arrays["REF_forces"] = np.zeros((1, 3))
            frames.append(atom)
    for index in range(count):
        molecule = Atoms(
            "OH2", positions=WATER + generator.normal(scale=0.03, size=(3, 3))
        )
        energy = 2 * HYDROGEN + OXYGEN + 0.05 * index
        forces = generator.normal(scale=0.1, size=(3, 3))
        if reserved:
            molecule.calc = SinglePointCalculator(
                molecule, energy=energy, forces=forces
            )
        else:
            molecule.info["REF_energy"] = energy
            molecule.arrays["REF_forces"] = forces
        frames.append(molecule)
    return frames


def publish_replay(count: int):
    """Put a replay dataset in the cache, as a login node or a legacy run does."""
    path = cached_path("mp")
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, waters(count, seed=5, isolated=False, reserved=True), format="extxyz")


def configuration(tmp_path, heads, **data):
    """A fine-tune's configuration: it names the foundation model it starts
    from, which is what lets a head copy that model's energies at all."""
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(tmp_path), "seed": 1},
            "finetune": {"foundation_model": str(tmp_path / "foundation")},
            "data": {
                "heads": heads,
                "valid_fraction": 0.2,
                "pin_memory": False,
                **data,
            },
            "model": {"observables": ["energy", "forces"], "r_max": 3.0},
            "training": {"batch_size": 2, "valid_batch_size": 2},
        }
    )


def target_head(tmp_path, count=10, **settings):
    path = tmp_path / "target.xyz"
    write(path, waters(count))
    return {"train_file": str(path), "e0s": {"isolated_atoms": {}}, **settings}


def foundation(e0s=None, heads=("pt",)):
    return FoundationContext(
        z_table=AtomicNumberTable([1, 6, 8]),
        heads=tuple(heads),
        e0s=e0s or {name: {1: -13.0, 6: -1030.0, 8: -2041.0} for name in heads},
    )


def train_structures(data, head):
    return data.train_loader.datasets[head].configurations


# ---------------------------------------------------------------------------
# The replay head is a head
# ---------------------------------------------------------------------------


@fp64_only
def test_a_curated_head_reads_the_published_dataset(tmp_path):
    publish_replay(20)
    config = configuration(
        tmp_path,
        {
            "replay": {"curated": "mp", "e0s": {"foundation": {}}},
            "target": target_head(tmp_path),
        },
    )
    data = run_data_stage(config, CATALOGUE, foundation=foundation())
    assert set(data.heads) == {"replay", "target"}
    assert train_structures(data, "replay")
    assert {item.head for item in train_structures(data, "replay")} == {"replay"}


@fp64_only
def test_the_replay_head_copies_the_foundations_energies(tmp_path):
    publish_replay(10)
    config = configuration(
        tmp_path,
        {
            "replay": {"curated": "mp", "e0s": {"foundation": {}}},
            "target": target_head(tmp_path),
        },
    )
    data = run_data_stage(config, CATALOGUE, foundation=foundation())
    assert data.e0s.values["replay"][8] == -2041.0
    assert data.e0s.values["target"][8] == OXYGEN


@fp64_only
def test_several_foundation_heads_need_one_named(tmp_path):
    publish_replay(10)
    config = configuration(
        tmp_path,
        {
            "replay": {"curated": "mp", "e0s": {"foundation": {}}},
            "target": target_head(tmp_path),
        },
    )
    with pytest.raises(DataStageError, match="needs one named"):
        run_data_stage(
            config, CATALOGUE, foundation=foundation(heads=("pbe", "r2scan"))
        )


@fp64_only
def test_an_element_the_foundation_lacks_an_energy_for_is_refused(tmp_path):
    """No zero pad. The frozen tree gives it `head_energies.get(z, 0.0)`."""
    publish_replay(10)
    config = configuration(
        tmp_path,
        {
            "replay": {"curated": "mp", "e0s": {"foundation": {}}},
            "target": target_head(tmp_path),
        },
    )
    partial = {"pt": {1: -13.0, 6: -1030.0}}
    with pytest.raises(Exception, match="8"):
        run_data_stage(config, CATALOGUE, foundation=foundation(e0s=partial))


def test_a_head_naming_two_sources_is_refused(tmp_path):
    with pytest.raises(ValueError, match="both train_file"):
        configuration(
            tmp_path,
            {"replay": {"curated": "mp", "train_file": "x.xyz"}},
        )


# ---------------------------------------------------------------------------
# The foundation model's element table
# ---------------------------------------------------------------------------


@fp64_only
def test_a_fine_tune_is_built_over_the_elements_its_data_holds(tmp_path):
    """The frozen tree's default. The foundation also knows carbon, and a head
    here would otherwise need an energy for an element it never sees."""
    config = configuration(tmp_path, {"target": target_head(tmp_path)})
    data = run_data_stage(config, CATALOGUE, foundation=foundation())
    assert list(data.z_table.zs) == [1, 8]


@fp64_only
def test_an_element_outside_the_foundation_is_refused(tmp_path):
    config = configuration(tmp_path, {"target": target_head(tmp_path)})
    narrow = FoundationContext(
        z_table=AtomicNumberTable([1, 6]), heads=("pt",), e0s={"pt": {1: 0.0, 6: 0.0}}
    )
    with pytest.raises(DataStageError, match=r"elements \[8\]"):
        run_data_stage(config, CATALOGUE, foundation=narrow)


# ---------------------------------------------------------------------------
# Weight and subselection
# ---------------------------------------------------------------------------


@fp64_only
def test_a_heads_weight_multiplies_every_structures(tmp_path):
    config = configuration(tmp_path, {"target": target_head(tmp_path, weight=0.25)})
    data = run_data_stage(config, CATALOGUE)
    assert {item.weight for item in train_structures(data, "target")} == {0.25}


@fp64_only
def test_a_subselection_keeps_its_count_before_the_split(tmp_path):
    """Twenty published, twelve kept, and the twelve are then divided into
    training and validation like any other head's structures."""
    publish_replay(20)
    config = configuration(
        tmp_path,
        {
            "replay": {
                "curated": "mp",
                "e0s": {"foundation": {}},
                "subselect": {"num_samples": 12},
            },
            "target": target_head(tmp_path),
        },
    )
    data = run_data_stage(config, CATALOGUE, foundation=foundation())
    validation = data.valid_loaders["replay"].dataset
    assert isinstance(validation, GraphDataset)
    assert len(train_structures(data, "replay")) + len(validation) == 12


@fp64_only
def test_farthest_points_without_descriptors_are_refused(tmp_path):
    publish_replay(10)
    config = configuration(
        tmp_path,
        {
            "replay": {
                "curated": "mp",
                "e0s": {"foundation": {}},
                "subselect": {"num_samples": 5, "method": "fps"},
            },
            "target": target_head(tmp_path),
        },
    )
    with pytest.raises(DataStageError, match="descriptors"):
        run_data_stage(config, CATALOGUE, foundation=foundation())


# ---------------------------------------------------------------------------
# The ratio guard
# ---------------------------------------------------------------------------


@fp64_only
def test_the_guard_repeats_the_heads_the_reference_outnumbers(tmp_path):
    """Forty replay structures kept for training against four target ones is
    a ratio of 0.1, which is not below 0.1; against three it is, and the
    target head is repeated `1 + int(0.1 / (3 / 40))` = 2 times."""
    publish_replay(50)
    heads = {
        "replay": {"curated": "mp", "e0s": {"foundation": {}}},
        "target": target_head(tmp_path, count=4),
    }
    data = run_data_stage(
        configuration(tmp_path, heads, ratio_guard={"reference": "replay"}),
        CATALOGUE,
        foundation=foundation(),
    )
    replay = len(train_structures(data, "replay"))
    target = len(train_structures(data, "target"))
    unguarded = run_data_stage(
        configuration(tmp_path, heads), CATALOGUE, foundation=foundation()
    )
    before = len(train_structures(unguarded, "target"))
    ratio = before / replay
    expected = 1 + int(0.1 / ratio) if ratio < 0.1 else 1
    assert target == before * expected
    assert expected > 1


@fp64_only
def test_the_guard_never_repeats_the_reference(tmp_path):
    publish_replay(50)
    heads = {
        "replay": {"curated": "mp", "e0s": {"foundation": {}}},
        "target": target_head(tmp_path, count=4),
    }
    guarded = run_data_stage(
        configuration(tmp_path, heads, ratio_guard={"reference": "replay"}),
        CATALOGUE,
        foundation=foundation(),
    )
    plain = run_data_stage(
        configuration(tmp_path, heads), CATALOGUE, foundation=foundation()
    )
    assert len(train_structures(guarded, "replay")) == len(
        train_structures(plain, "replay")
    )


def test_a_guard_naming_no_head_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a head"):
        configuration(
            tmp_path, {"target": target_head(tmp_path)}, ratio_guard={"reference": "pt"}
        )


@fp64_only
def test_a_streamed_head_skips_the_guard_and_says_so(tmp_path, caplog):
    """The frozen tree skips it for LMDB and HDF5 heads without a word."""
    from mace_torch.train.data_stage import _guard_ratio

    config = configuration(
        tmp_path,
        {
            "target": target_head(tmp_path),
            "big": {"train_file": str(tmp_path / "shards.h5")},
        },
        ratio_guard={"reference": "big"},
    )
    with caplog.at_level(logging.INFO):
        kept = _guard_ratio([], config, dict(config.data.heads))
    assert kept == []
    assert any("streamed" in record.getMessage() for record in caplog.records)
