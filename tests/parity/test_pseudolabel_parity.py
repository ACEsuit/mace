"""Replay labels from the committed anchor, against the frozen tree's.

The anchor is written as a v1 checkpoint and relabels periodic structures
through the rewrite's stage; the frozen tree's
``generate_pseudolabels_for_configs`` relabels the same structures with the
anchor itself. The energies, forces and stresses agree, and so do the weights,
which carry the rule both trees share: a label keeps a weight its file gave
it, and a label the file had no weight for is weighted one.

The one difference is a deliberate one. The frozen tree writes the virials of
every structure whatever it was asked for; the rewrite writes what the
configuration names, the energy and the forces by default.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from mace_core.data.keys import KeySpecification
from mace_core.data.xyz import configuration_from_atoms
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.finetune.foundation import read_foundation
from mace_torch.finetune.pseudolabels import generate_pseudolabels

from tests.parity.test_anchor_as_foundation import ANCHOR, write_anchor_checkpoint
from tests.parity.test_fm00_training_step import load_anchor

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def structures():
    generator = np.random.default_rng(8)
    frames = []
    for index in range(5):
        atoms = Atoms(
            "OH2",
            positions=WATER + generator.normal(scale=0.05, size=(3, 3)),
            cell=np.eye(3) * 3.5 + generator.normal(scale=0.05, size=(3, 3)),
            pbc=True,
        )
        atoms.info["REF_energy"] = 99.0
        atoms.arrays["REF_forces"] = np.full((3, 3), 7.0)
        if index % 2 == 0:
            atoms.info["REF_stress"] = np.full(6, 0.5)
            atoms.info["config_stress_weight"] = 0.25
        frames.append(atoms)
    return frames


@pytest.fixture(name="anchor")
def fixture_anchor(fp64, tmp_path):
    legacy = load_anchor(ANCHOR)
    return legacy, write_anchor_checkpoint(legacy, tmp_path)


def test_the_labels_and_their_weights_are_the_frozen_tree_s(anchor, isolated):
    from mace.data import KeySpecification as LegacyKeys
    from mace.data import config_from_atoms
    from mace.tools import AtomicNumberTable as LegacyTable
    from mace.tools.multihead_tools import generate_pseudolabels_for_configs

    legacy, path = anchor
    frames = structures()
    foundation = read_foundation(path, DEFAULT_CATALOGUE)
    spec = foundation.config.finetune.pseudolabels.model_copy(
        update={"properties": ("energy", "forces", "stress")}
    )
    keys = KeySpecification.from_defaults()
    ours = generate_pseudolabels(
        foundation.engine,
        [configuration_from_atoms(atoms, keys) for atoms in frames],
        spec,
        z_table=foundation.z_table,
        cutoff=foundation.config.model.r_max,
        head_index=0,
        batch_size=2,
    )
    theirs = generate_pseudolabels_for_configs(
        legacy,
        [
            config_from_atoms(atoms, key_specification=LegacyKeys.from_defaults())
            for atoms in frames
        ],
        LegacyTable([int(z) for z in legacy.atomic_numbers.tolist()]),
        float(legacy.r_max),
        torch.device("cpu"),
        batch_size=2,
    )
    assert len(ours) == len(theirs)
    for mine, frozen in zip(ours, theirs, strict=True):
        assert abs(mine.properties["energy"] - frozen.properties["energy"]) < 1e-10
        np.testing.assert_allclose(
            mine.properties["forces"], frozen.properties["forces"], atol=1e-10
        )
        for name in ("energy", "forces", "stress"):
            assert mine.property_weights.get(name, 0.0) == pytest.approx(
                frozen.property_weights.get(name, 0.0)
            ), name
        if (
            frozen.properties.get("stress") is not None
            and mine.properties.get("stress") is not None
        ):
            np.testing.assert_allclose(
                np.asarray(mine.properties["stress"]).reshape(-1),
                np.asarray(frozen.properties["stress"]).reshape(-1),
                atol=1e-10,
            )
        assert mine.properties["energy"] != 99.0
    stressed = [item.properties.get("stress") is not None for item in ours]
    assert stressed == [True, False, True, False, True]
    assert all(frozen.properties.get("virials") is not None for frozen in theirs)
    assert all(mine.properties.get("virials") is None for mine in ours)
