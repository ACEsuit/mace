"""A declared observable the data does not carry is an error before training."""

import numpy as np
import pytest
from ase import Atoms
from mace_core.data import (
    KeySpecification,
    check_observables_labelled,
    configuration_from_atoms,
)
from mace_core.observables import (
    DEFAULT_CATALOGUE,
    ObservableCatalogue,
    ObservableSpec,
)

KEYS = KeySpecification.from_defaults()
DIPOLE = ObservableSpec(name="dipole", irreps="1o", per_atom=False, units="e*A")


def _structure(energy=None, dipole=None):
    atoms = Atoms("H2O", positions=np.eye(3))
    if energy is not None:
        atoms.info["REF_energy"] = energy
    if dipole is not None:
        atoms.info["dipole"] = np.asarray(dipole)
    return configuration_from_atoms(atoms, KEYS)


def test_one_labelled_structure_is_enough():
    check_observables_labelled(
        DEFAULT_CATALOGUE, [_structure(), _structure(energy=-1.0)], KEYS
    )


def test_an_observable_no_structure_carries_names_its_file_key():
    catalogue = ObservableCatalogue(observables=(DIPOLE,))
    with pytest.raises(ValueError, match=r"'dipole' is declared.*file key 'dipole'"):
        check_observables_labelled(catalogue, [_structure(energy=-1.0)], KEYS)
    check_observables_labelled(catalogue, [_structure(dipole=[0.0, 0.1, 0.2])], KEYS)


def test_an_observable_the_keys_do_not_read_is_refused():
    catalogue = ObservableCatalogue(
        observables=(
            ObservableSpec(name="spin", irreps="0e", per_atom=True, units="1"),
        )
    )
    with pytest.raises(ValueError, match="reads no property of that name"):
        check_observables_labelled(catalogue, [_structure(energy=-1.0)], KEYS)
