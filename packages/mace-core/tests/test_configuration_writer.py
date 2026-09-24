"""Writing configurations back to extended XYZ, and reading them again.

A structure read from a file and written back reads as exactly the same
configuration: positions, labels, weights, config type. That is what lets a
fine-tune stop after selecting a head's structures and start again from the
file.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.io import write
from mace_core.data.keys import KeySpecification
from mace_core.data.xyz import read_configurations, write_configurations

KEYS = KeySpecification.from_defaults()


def structures():
    generator = np.random.default_rng(4)
    frames = []
    for index in range(3):
        atoms = Atoms(
            "OH2",
            positions=generator.normal(size=(3, 3)),
            cell=np.eye(3) * 6.0,
            pbc=index == 0,
        )
        atoms.info["REF_energy"] = float(generator.normal())
        atoms.arrays["REF_forces"] = generator.normal(size=(3, 3))
        atoms.info["config_type"] = "water"
        if index == 1:
            atoms.info["config_weight"] = 0.5
            atoms.info["config_forces_weight"] = 2.0
        if index == 2:
            del atoms.arrays["REF_forces"]
        frames.append(atoms)
    return frames


def test_a_read_structure_is_written_back_exactly(tmp_path):
    source = tmp_path / "source.xyz"
    write(source, structures())
    first = read_configurations(source, KEYS, keep_isolated_atoms=True).configurations
    written = write_configurations(tmp_path / "again.xyz", first, KEYS)
    again = read_configurations(written, KEYS, keep_isolated_atoms=True).configurations
    assert len(again) == len(first)
    for before, after in zip(first, again, strict=True):
        assert np.array_equal(before.atomic_numbers, after.atomic_numbers)
        assert np.array_equal(before.positions, after.positions)
        assert np.array_equal(np.asarray(before.cell), np.asarray(after.cell))
        assert before.pbc == after.pbc
        assert before.weight == after.weight
        assert before.config_type == after.config_type
        assert before.property_weights == after.property_weights
        for name, value in before.properties.items():
            other = after.properties[name]
            if value is None:
                assert other is None, name
            else:
                assert np.array_equal(np.asarray(value), np.asarray(other)), name


def test_an_absent_label_stays_absent(tmp_path):
    source = tmp_path / "source.xyz"
    write(source, structures())
    first = read_configurations(source, KEYS, keep_isolated_atoms=True).configurations
    written = write_configurations(tmp_path / "again.xyz", first, KEYS)
    third = read_configurations(written, KEYS, keep_isolated_atoms=True).configurations[
        2
    ]
    assert not third.is_labelled("forces")
    assert third.property_weights["forces"] == 0.0
