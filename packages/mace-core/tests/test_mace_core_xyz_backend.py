"""The XYZ backend: the contract around ``read_configurations``, and its options.

The parsing itself (key variants, weights, the ase reserved keys) is pinned by
``test_mace_core_data_configuration.py``; these tests pin that the backend
passes its options through and keeps the contract.
"""

from __future__ import annotations

import ase.io
import numpy as np
import pytest
from ase import Atoms
from mace_core.data import DataBackendError, KeySpecification, open_dataset
from mace_core.data.backends.xyz import XYZBackend
from mace_core.data.conformance import check_data_backend


def _water(energy: float | None = -14.0, config_type: str = "Default") -> Atoms:
    atoms = Atoms(
        "OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
    )
    atoms.info["config_type"] = config_type
    if energy is not None:
        atoms.info["REF_energy"] = energy
        atoms.arrays["REF_forces"] = np.full((3, 3), 0.1)
    return atoms


def _isolated(number: int, energy: float) -> Atoms:
    atoms = Atoms(numbers=[number], positions=[[0.0, 0.0, 0.0]])
    atoms.info.update(config_type="IsolatedAtom", REF_energy=energy)
    return atoms


def _write(tmp_path, atoms_list, name="train.xyz") -> str:
    path = tmp_path / name
    ase.io.write(path, atoms_list)
    return str(path)


def _open(path, **options):
    """Through the registry, so the options are checked to arrive."""
    return open_dataset(
        path, format="xyz", key_spec=KeySpecification.from_defaults(), **options
    )


def test_the_xyz_backend_keeps_the_contract(tmp_path):
    path = _write(tmp_path, [_water(-14.0), _water(-14.5), _water(-15.0)])
    check_data_backend(_open(path, head="pbe"), expected_length=3)


def test_properties_are_keyed_by_convention_name_and_carry_the_head(tmp_path):
    backend = _open(_write(tmp_path, [_water(), _water()]), head="pbe")
    structure = backend[0]
    assert structure.properties["energy"] == -14.0
    assert structure.properties["forces"].shape == (3, 3)
    assert "REF_energy" not in structure.properties
    assert structure.head == "pbe"


def test_config_type_weights_scale_the_structure_weight(tmp_path):
    path = _write(tmp_path, [_water(config_type="MD"), _water(config_type="opt")])
    backend = _open(path, config_type_weights={"MD": 0.5})
    assert [s.weight for s in backend.iter_range()] == [0.5, 1.0]


def test_isolated_atoms_are_references_and_not_structures(tmp_path):
    path = _write(tmp_path, [_isolated(1, -0.5), _isolated(8, -2.0), _water()])
    backend = _open(path)
    assert backend.isolated_atom_energies() == {1: -0.5, 8: -2.0}
    assert len(backend) == 1


def test_keep_isolated_atoms_keeps_them_among_the_structures_too(tmp_path):
    path = _write(tmp_path, [_isolated(8, -2.0), _water()])
    backend = _open(path, keep_isolated_atoms=True)
    assert backend.isolated_atom_energies() == {8: -2.0}
    assert len(backend) == 2


def test_a_file_of_isolated_atoms_only_has_nothing_to_train_on(tmp_path):
    path = _write(tmp_path, [_isolated(1, -0.5), _isolated(8, -2.0)])
    with pytest.raises(DataBackendError, match="only isolated-atom references"):
        _open(path)
    assert _open(path, require_labels=False).isolated_atom_energies() == {
        1: -0.5,
        8: -2.0,
    }


def test_two_isolated_atoms_of_one_element_that_disagree_raise(tmp_path):
    path = _write(tmp_path, [_isolated(8, -2.0), _isolated(8, -2.5), _water()])
    with pytest.raises(DataBackendError, match="different reference energies"):
        _open(path)


def test_an_unlabelled_file_is_refused_unless_labels_are_not_required(tmp_path):
    path = _write(tmp_path, [_water(None), _water(None)])
    with pytest.raises(DataBackendError, match="REF_energy") as error:
        _open(path)
    assert error.value.source == path
    assert len(_open(path, require_labels=False)) == 2


@pytest.mark.parametrize(
    "content",
    ["3\nnot a comment line ase can parse = = =\nO 0 0\n", "", "1\n\nQq 0 0 0\n"],
    ids=["malformed", "empty", "unknown-element"],
)
def test_an_unreadable_file_raises_naming_it(tmp_path, content):
    path = tmp_path / "broken.xyz"
    path.write_text(content)
    with pytest.raises(DataBackendError, match=r"broken\.xyz"):
        _open(str(path))


def test_sniff_claims_existing_xyz_files_only(tmp_path):
    path = _write(tmp_path, [_water()])
    assert XYZBackend.sniff(path)
    assert XYZBackend.sniff(_write(tmp_path, [_water()], "train.extxyz"))
    assert not XYZBackend.sniff(str(tmp_path / "missing.xyz"))
    assert not XYZBackend.sniff(str(tmp_path / "train.h5"))


def test_a_non_xyz_ase_format_is_opened_by_naming_the_backend(tmp_path):
    path = _write(tmp_path, [_water(), _water()], "train.traj")
    key_spec = KeySpecification.from_defaults()
    backend = open_dataset(path, format="xyz", key_spec=key_spec, require_labels=False)
    assert len(backend) == 2
