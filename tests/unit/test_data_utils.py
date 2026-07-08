"""Unit tests for mace/data/utils.py (configuration parsing).

Covers KeySpecification / update_keyspec_from_kwargs, config_from_atoms
(custom info/arrays keys, missing keys, config_type / weights, pbc/cell),
load_from_xyz round-trip through a temporary extxyz file (including
IsolatedAtom E0 extraction and keep_isolated_atoms), compute_average_E0s
on a hand-solvable linear system, test_config_types grouping, and
random_train_valid_split. No network, no subprocesses.
"""

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms

from mace.data import utils as data_utils
from mace.data.utils import (
    KeySpecification,
    compute_average_E0s,
    config_from_atoms,
    config_from_atoms_list,
    load_from_xyz,
    random_train_valid_split,
    update_keyspec_from_kwargs,
)
from mace.tools import AtomicNumberTable
from tests.helpers import make_fitting_configs


def ref_keyspec():
    return KeySpecification(
        info_keys={"energy": "REF_energy", "stress": "REF_stress", "head": "head"},
        arrays_keys={"forces": "REF_forces"},
    )


def make_water(**info):
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        cell=[4.0] * 3,
        pbc=[True] * 3,
    )
    atoms.info.update(info)
    return atoms


# ---------------------------------------------------------------------------
# KeySpecification
# ---------------------------------------------------------------------------


def test_keyspec_from_defaults():
    keyspec = KeySpecification.from_defaults()
    assert keyspec.info_keys["energy"] == "REF_energy"
    assert keyspec.info_keys["stress"] == "REF_stress"
    assert keyspec.info_keys["virials"] == "REF_virials"
    assert keyspec.info_keys["dipole"] == "dipole"
    assert keyspec.info_keys["head"] == "head"
    assert keyspec.arrays_keys["forces"] == "REF_forces"
    assert keyspec.arrays_keys["charges"] == "REF_charges"


def test_keyspec_update_merges_and_returns_self():
    keyspec = KeySpecification(info_keys={"energy": "a"})
    out = keyspec.update(info_keys={"energy": "b"}, arrays_keys={"forces": "f"})
    assert out is keyspec
    assert keyspec.info_keys["energy"] == "b"
    assert keyspec.arrays_keys["forces"] == "f"


def test_update_keyspec_from_kwargs():
    keyspec = KeySpecification()
    update_keyspec_from_kwargs(
        keyspec, {"energy_key": "E", "forces_key": "F", "unrelated": "x"}
    )
    assert keyspec.info_keys == {"energy": "E"}
    assert keyspec.arrays_keys == {"forces": "F"}


def test_update_keyspec_from_kwargs_embedding_specs():
    keyspec = KeySpecification()
    update_keyspec_from_kwargs(
        keyspec,
        {
            "embedding_specs": {
                "spin": {"per": "atom", "key": "site_spin"},
                "charge": {"per": "graph"},
            }
        },
    )
    assert keyspec.arrays_keys["spin"] == "site_spin"
    assert keyspec.info_keys["charge"] == "charge"  # key defaults to the name
    with pytest.raises(ValueError, match="Unsupported embedding_specs"):
        update_keyspec_from_kwargs(
            KeySpecification(), {"embedding_specs": {"bad": {"per": "bond"}}}
        )


# ---------------------------------------------------------------------------
# config_from_atoms
# ---------------------------------------------------------------------------


def test_config_from_atoms_maps_custom_keys():
    atoms = make_water(REF_energy=-1.5)
    atoms.info["REF_stress"] = np.linspace(0.0, 0.5, 6)
    forces = np.arange(9.0).reshape(3, 3)
    atoms.new_array("REF_forces", forces)

    config = config_from_atoms(atoms, key_specification=ref_keyspec())

    assert np.array_equal(config.atomic_numbers, [8, 1, 1])
    assert np.allclose(config.positions, atoms.get_positions())
    assert config.properties["energy"] == -1.5
    assert np.allclose(config.properties["forces"], forces)
    assert np.allclose(config.properties["stress"], atoms.info["REF_stress"])
    assert config.property_weights["energy"] == 1.0
    assert config.property_weights["forces"] == 1.0
    assert config.property_weights["stress"] == 1.0
    assert config.pbc == (True, True, True)
    assert np.allclose(config.cell, np.eye(3) * 4.0)
    assert config.config_type == "Default"
    assert config.weight == 1.0
    assert config.head == "Default"


def test_config_from_atoms_missing_keys_get_zero_weight():
    atoms = make_water()  # no energy/forces/stress present
    config = config_from_atoms(atoms, key_specification=ref_keyspec())
    assert config.properties["energy"] is None
    assert config.properties["forces"] is None
    assert config.properties["stress"] is None
    assert config.property_weights["energy"] == 0.0
    assert config.property_weights["forces"] == 0.0
    assert config.property_weights["stress"] == 0.0


def test_config_from_atoms_config_type_and_weights():
    atoms = make_water(
        REF_energy=0.0, config_type="slab", config_weight=2.0, config_energy_weight=5.0
    )
    config = config_from_atoms(
        atoms,
        key_specification=ref_keyspec(),
        config_type_weights={"slab": 3.0},
        head_name="dft",
    )
    assert config.config_type == "slab"
    assert config.weight == pytest.approx(6.0)  # config_weight * type weight
    assert config.property_weights["energy"] == pytest.approx(5.0)
    assert config.head == "dft"


def test_config_from_atoms_isolated_atom_nonperiodic():
    atoms = Atoms(numbers=[8], positions=[[0.0, 0.0, 0.0]])
    atoms.info["config_type"] = "IsolatedAtom"
    atoms.info["REF_energy"] = -3.0
    config = config_from_atoms(atoms, key_specification=ref_keyspec())
    assert config.config_type == "IsolatedAtom"
    assert config.pbc == (False, False, False)
    assert np.allclose(config.cell, np.zeros((3, 3)))
    assert config.properties["energy"] == -3.0


def test_config_from_atoms_dipole_and_charges():
    keyspec = KeySpecification(
        info_keys={"energy": "REF_energy", "dipole": "REF_dipole", "head": "head"},
        arrays_keys={"forces": "REF_forces", "charges": "REF_charges"},
    )
    atoms = make_water(REF_energy=0.0)
    atoms.info["REF_dipole"] = np.array([0.1, 0.2, 0.3])
    atoms.new_array("REF_charges", np.array([-2.0, 1.0, 1.0]))
    config = config_from_atoms(atoms, key_specification=keyspec)
    assert np.allclose(config.properties["dipole"], [0.1, 0.2, 0.3])
    assert np.allclose(config.properties["charges"], [-2.0, 1.0, 1.0])
    assert config.property_weights["dipole"] == 1.0
    assert config.property_weights["charges"] == 1.0


def test_config_from_atoms_list_matches_single():
    atoms_list = [make_water(REF_energy=float(i)) for i in range(3)]
    configs = config_from_atoms_list(atoms_list, key_specification=ref_keyspec())
    assert len(configs) == 3
    assert [c.properties["energy"] for c in configs] == [0.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# load_from_xyz round-trip
# ---------------------------------------------------------------------------


@pytest.fixture(name="fit_xyz")
def fixture_fit_xyz(tmp_path):
    configs = make_fitting_configs()  # 2 IsolatedAtom + 20 water configs
    path = tmp_path / "fit.xyz"
    ase.io.write(path, configs)
    return path, configs


def test_load_from_xyz_roundtrip(fit_xyz):
    path, written = fit_xyz
    keyspec = KeySpecification.from_defaults()
    e0s, configs = load_from_xyz(
        str(path), key_specification=keyspec, extract_atomic_energies=True
    )
    # both isolated atoms extracted as E0s and dropped from the configs
    assert e0s == {1: 0.0, 8: 0.0}
    assert len(configs) == 20
    waters = written[2:]
    for config, atoms in zip(configs, waters):
        assert np.array_equal(config.atomic_numbers, atoms.numbers)
        assert config.properties["energy"] == pytest.approx(
            atoms.info["REF_energy"], rel=1e-6
        )
        assert np.allclose(config.properties["forces"], atoms.arrays["REF_forces"])
        assert np.allclose(
            config.properties["stress"], atoms.info["REF_stress"], rtol=1e-6
        )
        assert config.config_type == "Default"
        assert config.head == "Default"
        assert config.pbc == (True, True, True)


def test_load_from_xyz_keep_isolated_atoms(fit_xyz):
    path, _ = fit_xyz
    e0s, configs = load_from_xyz(
        str(path),
        key_specification=KeySpecification.from_defaults(),
        extract_atomic_energies=True,
        keep_isolated_atoms=True,
    )
    assert e0s == {1: 0.0, 8: 0.0}
    assert len(configs) == 22
    assert configs[0].config_type == "IsolatedAtom"
    assert configs[1].config_type == "IsolatedAtom"


def test_load_from_xyz_no_extraction(fit_xyz):
    path, _ = fit_xyz
    e0s, configs = load_from_xyz(
        str(path),
        key_specification=KeySpecification.from_defaults(),
        extract_atomic_energies=False,
    )
    assert e0s == {}
    assert len(configs) == 22


def test_load_from_xyz_head_name(fit_xyz):
    path, _ = fit_xyz
    _, configs = load_from_xyz(
        str(path),
        key_specification=KeySpecification.from_defaults(),
        head_name="dft_head",
    )
    assert all(c.head == "dft_head" for c in configs)


def test_load_from_xyz_missing_keys_raises(fit_xyz):
    path, _ = fit_xyz
    keyspec = KeySpecification.from_defaults()
    keyspec.update(
        info_keys={"energy": "MISSING_energy", "stress": "MISSING_stress"},
        arrays_keys={"forces": "MISSING_forces"},
    )
    with pytest.raises(ValueError, match="MISSING_energy"):
        load_from_xyz(str(path), key_specification=keyspec)
    # same situation is tolerated with no_data_ok=True
    _, configs = load_from_xyz(str(path), key_specification=keyspec, no_data_ok=True)
    assert len(configs) == 22
    assert configs[-1].properties["energy"] is None


def test_load_from_xyz_restores_keyspec(fit_xyz):
    """load_from_xyz temporarily rewrites unsafe 'energy'/'forces'/'stress'
    keys but must restore the caller's keyspec afterwards."""
    path, _ = fit_xyz
    keyspec = KeySpecification.from_defaults()
    keyspec.update(
        info_keys={"energy": "energy", "stress": "stress"},
        arrays_keys={"forces": "forces"},
    )
    load_from_xyz(str(path), key_specification=keyspec, no_data_ok=True)
    assert keyspec.info_keys["energy"] == "energy"
    assert keyspec.arrays_keys["forces"] == "forces"
    assert keyspec.info_keys["stress"] == "stress"


# ---------------------------------------------------------------------------
# compute_average_E0s
# ---------------------------------------------------------------------------


def make_config(numbers, energy):
    numbers = np.array(numbers)
    return data_utils.Configuration(
        atomic_numbers=numbers,
        positions=np.zeros((len(numbers), 3)),
        properties={"energy": energy},
        property_weights={"energy": 1.0},
    )


def test_compute_average_E0s_exact_system():
    # Exactly solvable: E0(H) = -1, E0(O) = -3
    z_table = AtomicNumberTable([1, 8])
    configs = [
        make_config([1, 1], -2.0),  # H2
        make_config([8, 8], -6.0),  # O2
        make_config([1, 1, 8], -5.0),  # H2O
    ]
    e0s = compute_average_E0s(configs, z_table)
    assert set(e0s) == {1, 8}
    assert e0s[1] == pytest.approx(-1.0)
    assert e0s[8] == pytest.approx(-3.0)


def test_compute_average_E0s_least_squares_average():
    # Inconsistent data: lstsq averages. Two H2 molecules at -2 and -4
    # => E0(H) = -1.5 minimizes the squared residual.
    z_table = AtomicNumberTable([1])
    configs = [make_config([1, 1], -2.0), make_config([1, 1], -4.0)]
    e0s = compute_average_E0s(configs, z_table)
    assert e0s[1] == pytest.approx(-1.5)


# ---------------------------------------------------------------------------
# test_config_types + random_train_valid_split
# ---------------------------------------------------------------------------


def test_config_types_grouping():
    def cfg(config_type, head):
        c = make_config([1], 0.0)
        c.config_type = config_type
        c.head = head
        return c

    configs = [
        cfg("bulk", "Default"),
        cfg("slab", "Default"),
        cfg("bulk", "Default"),
        cfg("bulk", "dft"),
    ]
    grouped = data_utils.test_config_types(configs)
    names = [name for name, _ in grouped]
    assert names == ["bulk_Default", "slab_Default", "bulk_dft"]
    sizes = {name: len(confs) for name, confs in grouped}
    assert sizes == {"bulk_Default": 2, "slab_Default": 1, "bulk_dft": 1}


def test_random_train_valid_split_small(tmp_path):
    items = list(range(20))
    train, valid = random_train_valid_split(
        items, valid_fraction=0.1, seed=1, work_dir=str(tmp_path)
    )
    assert len(train) == 18
    assert len(valid) == 2
    assert sorted(train + valid) == items
    # fewer than 10 validation items: indices logged, not written to file
    assert not (tmp_path / "valid_indices_1.txt").exists()


def test_random_train_valid_split_writes_indices_file(tmp_path):
    items = list(range(100))
    train, valid = random_train_valid_split(
        items, valid_fraction=0.2, seed=7, work_dir=str(tmp_path)
    )
    assert len(train) == 80
    assert len(valid) == 20
    index_file = tmp_path / "valid_indices_7.txt"
    assert index_file.exists()
    saved = [int(line) for line in index_file.read_text().split()]
    assert sorted(saved) == sorted(items.index(v) for v in valid)
