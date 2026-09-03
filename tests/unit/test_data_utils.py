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
from mace.tools import AtomicNumberTable, DefaultKeys
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


# ---------------------------------------------------------------------------
# The default property keys are a data contract
#
# Every labelled dataset on disk is written against these names, so a rename
# is not a refactor -- it silently stops reading somebody's forces. The names
# are therefore asserted one at a time, spelled out, rather than compared
# against DefaultKeys (which would pass whatever DefaultKeys said today).
# ---------------------------------------------------------------------------


DEFAULT_KEY_NAMES = {
    "ENERGY": "REF_energy",
    "FORCES": "REF_forces",
    "STRESS": "REF_stress",
    "VIRIALS": "REF_virials",
    "DIPOLE": "dipole",
    "POLARIZABILITY": "polarizability",
    "HEAD": "head",
    "CHARGES": "REF_charges",
    "TOTAL_CHARGE": "total_charge",
    "TOTAL_SPIN": "total_spin",
    "ELEC_TEMP": "elec_temp",
    "MAGMOM": "REF_magmom",
    "MAGFORCES": "REF_magforces",
}


def test_default_keys_are_exactly_these_thirteen():
    assert {member.name: member.value for member in DefaultKeys} == DEFAULT_KEY_NAMES
    # spelled out again, so a bulk edit of the dict above still fails
    assert DefaultKeys.ENERGY.value == "REF_energy"
    assert DefaultKeys.FORCES.value == "REF_forces"
    assert DefaultKeys.STRESS.value == "REF_stress"
    assert DefaultKeys.VIRIALS.value == "REF_virials"
    assert DefaultKeys.DIPOLE.value == "dipole"
    assert DefaultKeys.POLARIZABILITY.value == "polarizability"
    assert DefaultKeys.HEAD.value == "head"
    assert DefaultKeys.CHARGES.value == "REF_charges"
    assert DefaultKeys.TOTAL_CHARGE.value == "total_charge"
    assert DefaultKeys.TOTAL_SPIN.value == "total_spin"
    assert DefaultKeys.ELEC_TEMP.value == "elec_temp"
    assert DefaultKeys.MAGMOM.value == "REF_magmom"
    assert DefaultKeys.MAGFORCES.value == "REF_magforces"


def test_keydict_derives_one_cli_argument_per_member():
    keydict = DefaultKeys.keydict()
    assert keydict == {
        f"{name.lower()}_key": value for name, value in DEFAULT_KEY_NAMES.items()
    }
    # the derivation is `<member name lowercased>_key`, which is exactly the
    # --energy_key / --magforces_key spelling the CLI exposes
    assert keydict["energy_key"] == "REF_energy"
    assert keydict["magforces_key"] == "REF_magforces"


def test_every_default_key_reaches_the_keyspec():
    """`update_keyspec_from_kwargs` routes each `<name>_key` into info_keys or
    arrays_keys from two hardcoded lists. A member missing from both is
    silently dropped -- the property is then never parsed and no error says
    so -- so assert the partition covers all thirteen."""
    keyspec = KeySpecification.from_defaults()
    routed = set(keyspec.info_keys) | set(keyspec.arrays_keys)
    assert routed == {member.name.lower() for member in DefaultKeys}
    assert not set(keyspec.info_keys) & set(keyspec.arrays_keys)
    # per-atom quantities are arrays, per-graph ones are info
    assert set(keyspec.arrays_keys) == {"forces", "charges", "magmom", "magforces"}
    assert keyspec.info_keys["elec_temp"] == "elec_temp"
    assert keyspec.info_keys["total_spin"] == "total_spin"
    assert keyspec.info_keys["total_charge"] == "total_charge"
    assert keyspec.info_keys["polarizability"] == "polarizability"


# ---------------------------------------------------------------------------
# Round-tripping every default key through a real extxyz file
# ---------------------------------------------------------------------------


def labelled_water():
    """A water molecule carrying a value under all thirteen default keys."""
    atoms = make_water()
    atoms.info["REF_energy"] = -14.5
    atoms.info["REF_stress"] = np.linspace(0.1, 0.6, 6)
    atoms.info["REF_virials"] = np.linspace(-0.3, 0.3, 6)
    atoms.info["dipole"] = np.array([0.1, -0.2, 0.3])
    atoms.info["polarizability"] = np.arange(9.0).reshape(3, 3)
    atoms.info["total_charge"] = -1.0
    atoms.info["total_spin"] = 2.0
    atoms.info["elec_temp"] = 300.0
    atoms.new_array("REF_forces", np.arange(9.0).reshape(3, 3) / 10.0)
    atoms.new_array("REF_charges", np.array([-0.8, 0.4, 0.4]))
    atoms.new_array("REF_magmom", np.array([[0.0, 0.0, 2.2]] * 3))
    atoms.new_array("REF_magforces", np.array([[0.1, 0.2, 0.3]] * 3))
    return atoms


def test_all_default_keys_survive_a_write_read_roundtrip(tmp_path):
    written = labelled_water()
    path = tmp_path / "labelled.xyz"
    ase.io.write(path, [written])

    _, configs = load_from_xyz(
        str(path), key_specification=KeySpecification.from_defaults()
    )
    (config,) = configs
    properties = config.properties

    assert properties["energy"] == pytest.approx(-14.5)
    assert np.allclose(properties["forces"], written.arrays["REF_forces"])
    assert np.allclose(properties["stress"], written.info["REF_stress"])
    assert np.allclose(properties["virials"], written.info["REF_virials"])
    assert np.allclose(
        np.reshape(properties["polarizability"], (3, 3)), np.arange(9.0).reshape(3, 3)
    )
    assert np.allclose(properties["charges"], [-0.8, 0.4, 0.4])
    # the graph-level model inputs
    assert properties["total_charge"] == pytest.approx(-1.0)
    assert properties["total_spin"] == pytest.approx(2.0)
    assert properties["elec_temp"] == pytest.approx(300.0)
    # the two magnetic per-atom arrays
    assert np.allclose(properties["magmom"], [[0.0, 0.0, 2.2]] * 3)
    assert np.allclose(properties["magforces"], [[0.1, 0.2, 0.3]] * 3)
    # head is written by load_from_xyz itself, not read from the file
    assert config.head == "Default"
    # twelve of the thirteen came back; `dipole` is the exception, below
    assert set(config.property_weights) == set(properties)
    assert {
        name for name, weight in config.property_weights.items() if weight == 1.0
    } == {name.lower() for name in DEFAULT_KEY_NAMES} - {"dipole"}


def test_the_default_dipole_key_does_not_survive_extxyz(tmp_path):
    """The one default key that cannot be read back from a file.

    ase reserves `dipole` as a per-config *calculator* property
    (`ase.io.extxyz.per_config_properties`, ase 3.29), so a value written into
    `atoms.info["dipole"]` is read back into `atoms.calc.results` and never
    reaches `atoms.info` -- where `config_from_atoms` is the only place that
    looks. The property silently becomes None with weight 0, i.e. a dipole
    dataset labelled with the documented default trains on nothing.

    `load_from_xyz` already rewrites the other three unsafe spellings
    ('energy', 'forces', 'stress') onto their REF_ equivalents; `dipole` has
    no such branch, which is why every dipole workflow in this repository
    passes `--dipole_key REF_dipole`. Pinned as the current behaviour, not
    endorsed: if the port fixes it, this test is what says the fix changed
    something.
    """
    atoms = make_water(REF_energy=0.0)
    atoms.new_array("REF_forces", np.zeros((3, 3)))
    atoms.info["dipole"] = np.array([0.1, -0.2, 0.3])
    atoms.info["REF_dipole"] = np.array([0.1, -0.2, 0.3])
    path = tmp_path / "dipole.xyz"
    ase.io.write(path, [atoms])

    reread = ase.io.read(path, index=0)
    assert "dipole" not in reread.info
    assert np.allclose(reread.calc.results["dipole"], [0.1, -0.2, 0.3])

    _, configs = load_from_xyz(
        str(path), key_specification=KeySpecification.from_defaults()
    )
    assert configs[0].properties["dipole"] is None
    assert configs[0].property_weights["dipole"] == 0.0

    # the same value under a non-reserved name reads fine, which is what the
    # dipole/polarizability workflows actually do
    keyspec = KeySpecification.from_defaults()
    update_keyspec_from_kwargs(keyspec, {"dipole_key": "REF_dipole"})
    _, renamed = load_from_xyz(str(path), key_specification=keyspec)
    assert np.allclose(renamed[0].properties["dipole"], [0.1, -0.2, 0.3])


@pytest.mark.parametrize(
    "name,cli_argument,custom,store",
    [
        ("energy", "energy_key", "pbe_energy", "info"),
        ("forces", "forces_key", "pbe_forces", "arrays"),
        ("stress", "stress_key", "pbe_stress", "info"),
        ("total_charge", "total_charge_key", "qtot", "info"),
        ("total_spin", "total_spin_key", "multiplicity", "info"),
        ("elec_temp", "elec_temp_key", "smearing_T", "info"),
        ("magmom", "magmom_key", "spins", "arrays"),
        ("magforces", "magforces_key", "spin_forces", "arrays"),
    ],
)
def test_custom_property_keys_round_trip(tmp_path, name, cli_argument, custom, store):
    """The `--<name>_key custom_name` path, end to end: a file labelled with a
    non-default name is read only when the keyspec is told about it, and the
    default name then reads nothing."""
    atoms = make_water(REF_energy=0.0)
    atoms.new_array("REF_forces", np.zeros((3, 3)))
    value = (
        np.arange(9.0).reshape(3, 3) if store == "arrays" else np.float64(7.5)
    )
    if store == "info":
        atoms.info[custom] = value
    else:
        atoms.new_array(custom, value)
    path = tmp_path / "custom.xyz"
    ase.io.write(path, [atoms])

    keyspec = KeySpecification.from_defaults()
    update_keyspec_from_kwargs(keyspec, {cli_argument: custom})
    _, configs = load_from_xyz(str(path), key_specification=keyspec)
    got = configs[0].properties[name]
    assert np.allclose(got, value)
    assert configs[0].property_weights[name] == 1.0

    # the same file read with the default keys finds nothing under that name,
    # and says so through a zero weight rather than by raising
    _, defaults = load_from_xyz(
        str(path), key_specification=KeySpecification.from_defaults()
    )
    if name not in ("energy", "forces"):  # those two are present as defaults
        assert defaults[0].properties[name] is None
        assert defaults[0].property_weights[name] == 0.0


def test_config_type_and_per_property_weights_round_trip(tmp_path):
    atoms = labelled_water()
    atoms.info["config_type"] = "slab"
    atoms.info["config_weight"] = 2.0
    atoms.info["config_energy_weight"] = 5.0
    atoms.info["config_forces_weight"] = 0.25
    atoms.info["config_magmom_weight"] = 3.0
    path = tmp_path / "weighted.xyz"
    ase.io.write(path, [atoms])

    _, configs = load_from_xyz(
        str(path),
        key_specification=KeySpecification.from_defaults(),
        config_type_weights={"slab": 3.0},
    )
    (config,) = configs
    assert config.config_type == "slab"
    assert config.weight == pytest.approx(6.0)  # config_weight * type weight
    assert config.property_weights["energy"] == pytest.approx(5.0)
    assert config.property_weights["forces"] == pytest.approx(0.25)
    assert config.property_weights["magmom"] == pytest.approx(3.0)
    # an unweighted property keeps 1.0
    assert config.property_weights["stress"] == pytest.approx(1.0)
    # an unknown config_type falls back to weight 1.0, it is not an error
    _, unknown = load_from_xyz(
        str(path),
        key_specification=KeySpecification.from_defaults(),
        config_type_weights={"bulk": 9.0},
    )
    assert unknown[0].weight == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Isolated-atom detection (the behaviour behind --keep_isolated_atoms)
# ---------------------------------------------------------------------------


def isolated_atom(number, energy=None, config_type="IsolatedAtom"):
    atoms = Atoms(numbers=[number], positions=[[0.0, 0.0, 0.0]])
    atoms.info["config_type"] = config_type
    if energy is not None:
        atoms.info["REF_energy"] = energy
    return atoms


def test_isolated_atom_detection_needs_both_one_atom_and_the_config_type(tmp_path):
    """Both conditions, because the test is `len(atoms) == 1 and
    config_type == "IsolatedAtom"`: a two-atom config labelled IsolatedAtom is
    training data, and a lone atom without the label is too."""
    two_atoms = Atoms(numbers=[8, 8], positions=[[0.0] * 3, [1.2, 0.0, 0.0]])
    two_atoms.info.update({"config_type": "IsolatedAtom", "REF_energy": -7.0})
    lone_unlabelled = isolated_atom(1, energy=-0.5, config_type="Default")
    path = tmp_path / "iso.xyz"
    ase.io.write(path, [isolated_atom(8, -3.0), two_atoms, lone_unlabelled])

    e0s, configs = load_from_xyz(
        str(path),
        key_specification=KeySpecification.from_defaults(),
        extract_atomic_energies=True,
    )
    assert e0s == {8: -3.0}
    assert len(configs) == 2  # only the real isolated atom was removed


def test_isolated_atom_without_an_energy_contributes_zero(tmp_path, caplog):
    path = tmp_path / "iso_noenergy.xyz"
    ase.io.write(path, [isolated_atom(8, -3.0), isolated_atom(1, None)])
    with caplog.at_level("WARNING"):
        e0s, configs = load_from_xyz(
            str(path),
            key_specification=KeySpecification.from_defaults(),
            extract_atomic_energies=True,
            no_data_ok=True,
        )
    assert e0s == {8: -3.0, 1: 0.0}
    assert "Zero energy will be used" in caplog.text
    assert configs == []


def test_keep_isolated_atoms_keeps_them_and_still_extracts(tmp_path):
    water = make_water(REF_energy=-14.0)
    water.new_array("REF_forces", np.zeros((3, 3)))
    path = tmp_path / "iso_keep.xyz"
    ase.io.write(path, [isolated_atom(8, -3.0), water])

    keyspec = KeySpecification.from_defaults()
    e0s, kept = load_from_xyz(
        str(path),
        key_specification=keyspec,
        extract_atomic_energies=True,
        keep_isolated_atoms=True,
    )
    assert e0s == {8: -3.0}
    assert [c.config_type for c in kept] == ["IsolatedAtom", "Default"]
    _, dropped = load_from_xyz(
        str(path),
        key_specification=KeySpecification.from_defaults(),
        extract_atomic_energies=True,
    )
    assert [c.config_type for c in dropped] == ["Default"]


# ---------------------------------------------------------------------------
# The remaining parsing-layer surface: prefixed split files, a headless
# config, and the two HDF5 writers that mace.data exports and nothing in the
# suite had ever called.
# ---------------------------------------------------------------------------


def test_random_train_valid_split_prefixes_the_indices_file(tmp_path):
    train, valid = random_train_valid_split(
        list(range(100)),
        valid_fraction=0.2,
        seed=3,
        work_dir=str(tmp_path),
        prefix="run7",
    )
    assert len(train) == 80 and len(valid) == 20
    assert (tmp_path / "run7_valid_indices_3.txt").exists()
    assert not (tmp_path / "valid_indices_3.txt").exists()


def test_config_types_treats_a_missing_head_as_the_empty_string():
    config = make_config([1], 0.0)
    config.config_type = "bulk"
    config.head = None
    (name, group), = data_utils.test_config_types([config])
    assert name == "bulk_"
    assert group == [config]
    assert config.head == ""  # normalised in place


def test_save_dataset_as_hdf5_writes_one_group_per_graph(tmp_path):
    """`save_dataset_as_HDF5` / `save_AtomicData_to_HDF5` are exported from
    `mace.data` and were reached by nothing in the suite. They write built
    graphs (not configurations), one group per graph, so the on-disk layout
    they define is pinned here before DATA-2 has to reproduce or replace it.
    """
    import h5py  # noqa: PLC0415  (only this test needs it)

    from mace.data import AtomicData, save_dataset_as_HDF5

    z_table = AtomicNumberTable([1, 8])
    configs = [
        config_from_atoms(make_water(REF_energy=float(i)), key_specification=ref_keyspec())
        for i in range(2)
    ]
    graphs = [
        AtomicData.from_config(config, z_table=z_table, cutoff=3.0)
        for config in configs
    ]
    path = tmp_path / "graphs.h5"
    save_dataset_as_HDF5(graphs, str(path))

    with h5py.File(path, "r") as handle:
        assert sorted(handle) == ["config_0", "config_1"]
        group = handle["config_0"]
        assert group["num_nodes"][()] == 3
        assert group["positions"].shape == (3, 3)
        assert group["edge_index"].shape == graphs[0].edge_index.shape
        assert np.allclose(group["cell"][()], graphs[0].cell.numpy())
        assert np.allclose(group["node_attrs"][()], graphs[0].node_attrs.numpy())
        assert group["energy"][()] == pytest.approx(0.0)
        assert handle["config_1"]["energy"][()] == pytest.approx(1.0)
