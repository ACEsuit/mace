"""A configuration for a model that reads out a dipole and no energy.

Three things follow from the observables alone: the error table the run prints,
the losses it may use, and reading the dipole from a file under the key
convention's own default name.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.keys import KeySpecification
from mace_core.data.xyz import read_configurations


def config(observables, **sections) -> ResolvedConfig:
    return ResolvedConfig.model_validate(
        {"model": {"observables": observables}, **sections}
    )


@pytest.mark.parametrize(
    ("observables", "table"),
    [
        (["dipole"], "DipoleRMSE"),
        (["dipole", "polarizability"], "DipolePolarRMSE"),
        (["energy", "forces"], "PerAtomRMSE"),
    ],
)
def test_the_error_table_follows_what_is_read_out(observables, table):
    assert config(observables).runtime.error_table == table


def test_a_written_error_table_is_kept():
    written = config(["dipole"], runtime={"error_table": "DipolePolarRMSE"})
    assert written.runtime.error_table == "DipolePolarRMSE"


def test_the_universal_loss_is_refused_without_an_energy():
    with pytest.raises(ValueError, match="universal"):
        config(["dipole", "polarizability"], loss={"kind": {"kind": "universal"}})


def test_the_resolved_configuration_reads_back_as_itself():
    """A resolved configuration writes every field, the default E0s of each head
    included, and a model with no energy has to accept its own record."""
    resolved = config(
        ["dipole", "polarizability"],
        data={"heads": {"default": {"train_file": "train.xyz"}}},
    )
    assert ResolvedConfig.model_validate(resolved.model_dump(mode="json")) == resolved


def test_written_e0s_are_still_refused_without_an_energy():
    with pytest.raises(ValueError, match="e0s"):
        config(
            ["dipole"],
            data={
                "heads": {
                    "default": {"train_file": "train.xyz", "e0s": {"kind": "average"}}
                }
            },
        )


def test_the_default_dipole_key_reads_the_dipole(tmp_path, caplog):
    """The key convention reads the dipole from ``dipole``, which is ase's name
    for the calculator's dipole, so ase puts it in the calculator on reading."""
    generator = np.random.default_rng(1)
    frames = []
    for _ in range(2):
        atoms = Atoms("OH2", positions=generator.normal(size=(3, 3)))
        atoms.info["dipole"] = generator.normal(size=3)
        frames.append(atoms)
    path = tmp_path / "dipoles.xyz"
    write(path, frames)
    with caplog.at_level(logging.WARNING, logger="mace_core.data.xyz"):
        read = read_configurations(
            path, KeySpecification.from_defaults(), keep_isolated_atoms=True
        ).configurations
    for atoms, configuration in zip(frames, read, strict=True):
        assert np.array_equal(configuration.properties["dipole"], atoms.info["dipole"])
    assert "not safe" not in caplog.text


def test_a_file_without_dipoles_reads_without_them(tmp_path):
    atoms = Atoms("OH2", positions=np.eye(3))
    atoms.info["REF_energy"] = -1.0
    path = tmp_path / "energies.xyz"
    write(path, [atoms])
    (configuration,) = read_configurations(
        path, KeySpecification.from_defaults(), keep_isolated_atoms=True
    ).configurations
    assert not configuration.is_labelled("dipole")
