"""Parsing labelled structure files into configurations.

These cases are ported from the characterization suite that pinned the legacy
parser against real files. What is asserted is the *behaviour* the rewrite owes
that suite: which key each property is read from, what a missing key does to
its weight, how the weights compose, and which single-atom structures are
consumed as E0 references.

Two places where this deliberately does not match legacy are asserted here as
well, each next to the case it changes, so neither is a surprise found later by
somebody diffing outputs.
"""

from __future__ import annotations

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from mace_core.data import (
    ARRAYS_CONVENTION_NAMES,
    AtomicNumberTable,
    Configuration,
    DefaultKeys,
    EmbeddingFeatureSpec,
    KeySpecification,
    atomic_number_table_from_zs,
    configuration_from_atoms,
    group_by_config_type,
    random_train_valid_split,
    read_configurations,
)

#: The default file keys, written out rather than read from the enum. The point
#: of the table is that these exact strings are what every labelled dataset on
#: disk uses, so a test that compared the enum against itself would pass
#: through a rename that broke every one of those files.
DEFAULT_KEY_TABLE = {
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


def water(**info) -> Atoms:
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        cell=[4.0] * 3,
        pbc=[True] * 3,
    )
    atoms.info.update(info)
    return atoms


def labelled_water() -> Atoms:
    """A structure carrying a value under all thirteen default keys."""
    atoms = water()
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


def isolated_atom(number: int, energy=None, config_type="IsolatedAtom") -> Atoms:
    atoms = Atoms(numbers=[number], positions=[[0.0, 0.0, 0.0]])
    atoms.info["config_type"] = config_type
    if energy is not None:
        atoms.info["REF_energy"] = energy
    return atoms


def write(tmp_path, atoms_list, name="structures.xyz") -> str:
    path = tmp_path / name
    ase.io.write(path, atoms_list)
    return str(path)


# ---------------------------------------------------------------------------
# The default key table is a data contract
# ---------------------------------------------------------------------------


def test_the_default_keys_are_exactly_these_thirteen():
    assert {member.name: member.value for member in DefaultKeys} == DEFAULT_KEY_TABLE
    assert len(DEFAULT_KEY_TABLE) == 13


def test_the_command_line_spelling_is_derived_from_the_member_name():
    assert DefaultKeys.keydict() == {
        f"{name.lower()}_key": value for name, value in DEFAULT_KEY_TABLE.items()
    }
    assert DefaultKeys.keydict()["energy_key"] == "REF_energy"
    assert DefaultKeys.keydict()["magforces_key"] == "REF_magforces"


def test_every_default_key_is_routed_to_exactly_one_half():
    """A convention name in neither half is never parsed, and nothing says so."""
    key_spec = KeySpecification.from_defaults()
    routed = set(key_spec.info_keys) | set(key_spec.arrays_keys)
    assert routed == {member.name.lower() for member in DefaultKeys}
    assert not set(key_spec.info_keys) & set(key_spec.arrays_keys)


def test_the_magnetic_arrays_resolve_with_no_magnetic_flag_set():
    """They are part of the default table, not a magnetic-model extra."""
    key_spec = KeySpecification.from_defaults()
    assert set(key_spec.arrays_keys) == ARRAYS_CONVENTION_NAMES
    assert key_spec.arrays_keys["magmom"] == "REF_magmom"
    assert key_spec.arrays_keys["magforces"] == "REF_magforces"


def test_the_graph_level_inputs_are_per_structure_keys():
    key_spec = KeySpecification.from_defaults()
    assert key_spec.info_keys["elec_temp"] == "elec_temp"
    assert key_spec.info_keys["total_spin"] == "total_spin"
    assert key_spec.info_keys["total_charge"] == "total_charge"
    assert key_spec.info_keys["polarizability"] == "polarizability"


# ---------------------------------------------------------------------------
# KeySpecification
# ---------------------------------------------------------------------------


def test_update_merges_and_returns_self():
    key_spec = KeySpecification(info_keys={"energy": "a"})
    returned = key_spec.update(info_keys={"energy": "b"}, arrays_keys={"forces": "f"})
    assert returned is key_spec
    assert key_spec.info_keys["energy"] == "b"
    assert key_spec.arrays_keys["forces"] == "f"


def test_overrides_route_by_name_and_ignore_everything_else():
    key_spec = KeySpecification()
    key_spec.apply_overrides(
        {"energy_key": "E", "forces_key": "F", "unrelated": "x", "seed": 3}
    )
    assert key_spec.info_keys == {"energy": "E"}
    assert key_spec.arrays_keys == {"forces": "F"}


def test_an_override_for_an_unknown_property_is_an_error():
    """Legacy dropped it, and the property then silently never parsed."""
    with pytest.raises(ValueError, match="enthalpy_key"):
        KeySpecification().apply_overrides({"enthalpy_key": "REF_enthalpy"})


def test_a_copy_does_not_share_its_dictionaries():
    original = KeySpecification.from_defaults()
    clone = original.copy()
    clone.info_keys["energy"] = "somewhere_else"
    assert original.info_keys["energy"] == "REF_energy"


def test_an_embedding_feature_lands_in_the_half_its_per_names():
    key_spec = KeySpecification()
    key_spec.add_embedding_features(
        {
            "site_spin": {"per": "atom", "key": "spin_array"},
            "applied_field": {"per": "graph"},
        }
    )
    assert key_spec.arrays_keys["site_spin"] == "spin_array"
    # no declared key: the feature is read from its own name
    assert key_spec.info_keys["applied_field"] == "applied_field"


def test_an_embedding_feature_can_be_given_as_a_typed_spec():
    key_spec = KeySpecification()
    key_spec.add_embedding_features(
        {"site_spin": EmbeddingFeatureSpec(per="atom", key="spin_array")}
    )
    assert key_spec.arrays_keys == {"site_spin": "spin_array"}


@pytest.mark.parametrize("per", ["bond", "cell", ""])
def test_an_embedding_feature_stored_nowhere_real_raises(per):
    with pytest.raises(ValueError, match="site_spin"):
        KeySpecification().add_embedding_features({"site_spin": {"per": per}})


def test_an_embedding_feature_that_says_nothing_about_storage_raises():
    with pytest.raises(ValueError, match="per"):
        KeySpecification().add_embedding_features({"site_spin": {"key": "s"}})


def test_an_embedding_feature_reads_through_a_real_file(tmp_path):
    """The declaration is what makes a user's own input nameable downstream."""
    atoms = water(REF_energy=0.0, applied_field=0.5)
    atoms.new_array("REF_forces", np.zeros((3, 3)))
    atoms.new_array("spin_array", np.array([0.2, -0.1, -0.1]))
    key_spec = KeySpecification.from_defaults().add_embedding_features(
        {
            "site_spin": {"per": "atom", "key": "spin_array"},
            "applied_field": {"per": "graph"},
        }
    )

    parsed = read_configurations(write(tmp_path, [atoms]), key_spec)
    (config,) = parsed.configurations
    assert np.allclose(config.properties["site_spin"], [0.2, -0.1, -0.1])
    assert config.properties["applied_field"] == pytest.approx(0.5)
    assert config.property_weights["site_spin"] == 1.0


# ---------------------------------------------------------------------------
# One structure at a time
# ---------------------------------------------------------------------------


def reference_key_spec() -> KeySpecification:
    return KeySpecification(
        info_keys={"energy": "REF_energy", "stress": "REF_stress", "head": "head"},
        arrays_keys={"forces": "REF_forces"},
    )


def test_a_configuration_carries_the_structure_and_its_labels():
    atoms = water(REF_energy=-1.5)
    atoms.info["REF_stress"] = np.linspace(0.0, 0.5, 6)
    forces = np.arange(9.0).reshape(3, 3)
    atoms.new_array("REF_forces", forces)

    config = configuration_from_atoms(atoms, reference_key_spec())

    assert np.array_equal(config.atomic_numbers, [8, 1, 1])
    assert np.allclose(config.positions, atoms.get_positions())
    assert config.properties["energy"] == -1.5
    assert np.allclose(config.properties["forces"], forces)
    assert np.allclose(config.properties["stress"], atoms.info["REF_stress"])
    assert config.pbc == (True, True, True)
    assert config.cell is not None
    assert np.allclose(config.cell, np.eye(3) * 4.0)
    assert config.config_type == "Default"
    assert config.weight == 1.0
    assert config.head == "Default"
    assert len(config) == 3


def test_a_property_the_file_does_not_carry_is_none_at_zero_weight():
    config = configuration_from_atoms(water(), reference_key_spec())
    for name in ("energy", "forces", "stress"):
        assert config.properties[name] is None
        assert config.property_weights[name] == 0.0


def test_the_structure_weight_is_the_product_of_both_weights():
    atoms = water(
        REF_energy=0.0, config_type="slab", config_weight=2.0, config_energy_weight=5.0
    )
    config = configuration_from_atoms(
        atoms,
        reference_key_spec(),
        config_type_weights={"slab": 3.0},
        head_name="dft",
    )
    assert config.config_type == "slab"
    assert config.weight == pytest.approx(6.0)
    assert config.property_weights["energy"] == pytest.approx(5.0)
    assert config.head == "dft"


def test_an_unknown_config_type_weighs_one_rather_than_raising():
    atoms = water(REF_energy=0.0, config_type="slab", config_weight=2.0)
    config = configuration_from_atoms(
        atoms, reference_key_spec(), config_type_weights={"bulk": 9.0}
    )
    assert config.weight == pytest.approx(2.0)


def test_an_aperiodic_structure_keeps_the_zero_cell_it_was_given():
    """The cell is what the file said. An artificial box for a neighbour
    search is built where the search happens, and never written back here."""
    atoms = Atoms(numbers=[8], positions=[[0.0, 0.0, 0.0]])
    config = configuration_from_atoms(atoms, reference_key_spec())
    assert config.pbc == (False, False, False)
    assert config.cell is not None
    assert np.allclose(config.cell, np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Reading a file
# ---------------------------------------------------------------------------


def test_all_thirteen_default_keys_survive_a_write_and_a_read(tmp_path):
    written = labelled_water()
    parsed = read_configurations(
        write(tmp_path, [written]), KeySpecification.from_defaults()
    )
    (config,) = parsed.configurations
    properties = config.properties

    assert properties["energy"] == pytest.approx(-14.5)
    assert np.allclose(properties["forces"], written.arrays["REF_forces"])
    assert np.allclose(properties["stress"], written.info["REF_stress"])
    assert np.allclose(properties["virials"], written.info["REF_virials"])
    assert np.allclose(
        np.reshape(properties["polarizability"], (3, 3)), np.arange(9.0).reshape(3, 3)
    )
    assert np.allclose(properties["charges"], [-0.8, 0.4, 0.4])
    assert properties["total_charge"] == pytest.approx(-1.0)
    assert properties["total_spin"] == pytest.approx(2.0)
    assert properties["elec_temp"] == pytest.approx(300.0)
    assert np.allclose(properties["magmom"], [[0.0, 0.0, 2.2]] * 3)
    assert np.allclose(properties["magforces"], [[0.1, 0.2, 0.3]] * 3)
    # the head is stamped by the read, not read from the file
    assert config.head == "Default"
    assert properties["head"] == "Default"

    # every declared property has a weight, and twelve of the thirteen read
    assert set(config.property_weights) == set(properties)
    assert {n for n, w in config.property_weights.items() if w == 1.0} == {
        name.lower() for name in DEFAULT_KEY_TABLE
    } - {"dipole"}


def test_the_default_dipole_key_cannot_be_read_back_from_a_file(tmp_path):
    """The one default key that does not survive a round trip.

    ase reserves ``dipole`` as a per-structure calculator property, so a value
    written into ``info`` is read back into ``calc.results`` and never reaches
    the place the parser looks. The property becomes None at weight zero, which
    is why every dipole workflow names a different key. Pinned as behaviour the
    rewrite inherited, not endorsed: this test is what would notice a fix.
    """
    atoms = water(REF_energy=0.0)
    atoms.new_array("REF_forces", np.zeros((3, 3)))
    atoms.info["dipole"] = np.array([0.1, -0.2, 0.3])
    atoms.info["REF_dipole"] = np.array([0.1, -0.2, 0.3])
    path = write(tmp_path, [atoms])

    parsed = read_configurations(path, KeySpecification.from_defaults())
    assert parsed.configurations[0].properties["dipole"] is None
    assert parsed.configurations[0].property_weights["dipole"] == 0.0

    renamed = read_configurations(
        path,
        KeySpecification.from_defaults().apply_overrides({"dipole_key": "REF_dipole"}),
    )
    assert np.allclose(renamed.configurations[0].properties["dipole"], [0.1, -0.2, 0.3])


@pytest.mark.parametrize(
    ("name", "setting", "custom", "store"),
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
def test_a_custom_file_key_reads_only_when_the_specification_names_it(
    tmp_path, name, setting, custom, store
):
    """Multi-level-of-theory labelling, end to end: a ``pbe_energy``-style key
    reads when declared, and the default name then finds nothing and says so
    through a zero weight rather than by raising."""
    atoms = water(REF_energy=0.0)
    atoms.new_array("REF_forces", np.zeros((3, 3)))
    value = np.arange(9.0).reshape(3, 3) if store == "arrays" else np.float64(7.5)
    if store == "info":
        atoms.info[custom] = value
    else:
        atoms.new_array(custom, value)
    path = write(tmp_path, [atoms])

    declared = read_configurations(
        path, KeySpecification.from_defaults().apply_overrides({setting: custom})
    )
    assert np.allclose(declared.configurations[0].properties[name], value)
    assert declared.configurations[0].property_weights[name] == 1.0

    defaults = read_configurations(path, KeySpecification.from_defaults())
    if name not in ("energy", "forces"):  # those two are present under defaults
        assert defaults.configurations[0].properties[name] is None
        assert defaults.configurations[0].property_weights[name] == 0.0


def test_weights_compose_through_a_real_file(tmp_path):
    atoms = labelled_water()
    atoms.info.update(
        {
            "config_type": "slab",
            "config_weight": 2.0,
            "config_energy_weight": 5.0,
            "config_forces_weight": 0.25,
            "config_magmom_weight": 3.0,
        }
    )
    parsed = read_configurations(
        write(tmp_path, [atoms]),
        KeySpecification.from_defaults(),
        config_type_weights={"slab": 3.0},
    )
    (config,) = parsed.configurations
    assert config.config_type == "slab"
    assert config.weight == pytest.approx(6.0)
    assert config.property_weights["energy"] == pytest.approx(5.0)
    assert config.property_weights["forces"] == pytest.approx(0.25)
    assert config.property_weights["magmom"] == pytest.approx(3.0)
    assert config.property_weights["stress"] == pytest.approx(1.0)


def test_the_head_name_is_stamped_on_every_structure(tmp_path):
    path = write(tmp_path, [labelled_water(), labelled_water()])
    parsed = read_configurations(
        path, KeySpecification.from_defaults(), head_name="dft_head"
    )
    assert [c.head for c in parsed.configurations] == ["dft_head"] * 2
    assert [c.properties["head"] for c in parsed.configurations] == ["dft_head"] * 2


# ---------------------------------------------------------------------------
# The keys ase reserved in 3.23
# ---------------------------------------------------------------------------


def calculator_labelled_water() -> Atoms:
    atoms = water()
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-14.5,
        forces=np.arange(9.0).reshape(3, 3) / 10.0,
        stress=np.linspace(0.1, 0.6, 6),
    )
    return atoms


def test_a_reserved_key_is_rewritten_and_recovered_from_the_calculator(tmp_path):
    path = write(tmp_path, [calculator_labelled_water()])
    key_spec = KeySpecification.from_defaults().update(
        info_keys={"energy": "energy", "stress": "stress"},
        arrays_keys={"forces": "forces"},
    )

    parsed = read_configurations(path, key_spec)
    (config,) = parsed.configurations

    assert config.properties["energy"] == pytest.approx(-14.5)
    assert np.allclose(config.properties["forces"], np.arange(9.0).reshape(3, 3) / 10.0)
    assert np.allclose(config.properties["stress"], np.linspace(0.1, 0.6, 6))


def test_the_callers_key_specification_is_never_touched(tmp_path):
    """Legacy rewrote the caller's object and restored it at the end. Here the
    rewrite happens on a copy, so a specification shared between two datasets
    cannot come back holding whatever the first parse rewrote it to."""
    path = write(tmp_path, [calculator_labelled_water()])
    key_spec = KeySpecification.from_defaults().update(
        info_keys={"energy": "energy", "stress": "stress"},
        arrays_keys={"forces": "forces"},
    )
    before = (dict(key_spec.info_keys), dict(key_spec.arrays_keys))

    read_configurations(path, key_spec)

    assert (key_spec.info_keys, key_spec.arrays_keys) == before
    assert key_spec.info_keys["energy"] == "energy"
    assert key_spec.arrays_keys["forces"] == "forces"
    assert key_spec.info_keys["stress"] == "stress"

    # and reading twice through the same object gives the same answer
    again = read_configurations(path, key_spec)
    assert again.configurations[0].properties["energy"] == pytest.approx(-14.5)


def test_a_reserved_key_with_nothing_to_recover_yields_none(tmp_path):
    """No calculator on the structure: the rewritten key is still written, so
    it counts as present and the label is None at full weight. Inherited from
    legacy, and the reason the presence check below does not fire on a file
    whose recovery found nothing."""
    atoms = water()
    atoms.new_array("REF_forces", np.zeros((3, 3)))
    key_spec = KeySpecification.from_defaults().update(info_keys={"energy": "energy"})

    parsed = read_configurations(write(tmp_path, [atoms]), key_spec)
    (config,) = parsed.configurations
    assert config.properties["energy"] is None
    assert config.property_weights["energy"] == 1.0


# ---------------------------------------------------------------------------
# The presence check
# ---------------------------------------------------------------------------


def test_a_file_with_none_of_the_three_labels_raises_naming_all_of_them(tmp_path):
    path = write(tmp_path, [labelled_water()])
    key_spec = KeySpecification.from_defaults().apply_overrides(
        {
            "energy_key": "MISSING_energy",
            "forces_key": "MISSING_forces",
            "dipole_key": "MISSING_dipole",
        }
    )
    with pytest.raises(ValueError) as caught:
        read_configurations(path, key_spec)

    message = str(caught.value)
    assert "MISSING_energy" in message
    assert "MISSING_forces" in message
    assert "MISSING_dipole" in message
    assert path in message


def test_the_same_file_reads_with_no_data_ok(tmp_path, caplog):
    path = write(tmp_path, [labelled_water()])
    key_spec = KeySpecification.from_defaults().apply_overrides(
        {
            "energy_key": "MISSING_energy",
            "forces_key": "MISSING_forces",
            "dipole_key": "MISSING_dipole",
        }
    )
    with caplog.at_level("WARNING"):
        parsed = read_configurations(path, key_spec, no_data_ok=True)

    assert len(parsed.configurations) == 1
    assert parsed.configurations[0].properties["energy"] is None
    assert "MISSING_energy" in caplog.text


def test_forces_alone_are_enough_to_pass_the_check(tmp_path):
    """Only all three missing is an error; one of them missing is a warning."""
    atoms = water()
    atoms.new_array("REF_forces", np.zeros((3, 3)))
    parsed = read_configurations(
        write(tmp_path, [atoms]), KeySpecification.from_defaults()
    )
    assert parsed.configurations[0].properties["energy"] is None


# ---------------------------------------------------------------------------
# Isolated atoms
# ---------------------------------------------------------------------------


def test_isolated_atoms_become_reference_energies_and_leave_the_dataset(tmp_path):
    water_atoms = water(REF_energy=-14.0)
    water_atoms.new_array("REF_forces", np.zeros((3, 3)))
    path = write(
        tmp_path, [isolated_atom(8, -3.0), isolated_atom(1, -0.5), water_atoms]
    )

    parsed = read_configurations(
        path, KeySpecification.from_defaults(), extract_isolated_atom_energies=True
    )
    assert parsed.isolated_atom_energies == {8: -3.0, 1: -0.5}
    assert [c.config_type for c in parsed.configurations] == ["Default"]


def test_keeping_them_extracts_the_energies_anyway(tmp_path):
    water_atoms = water(REF_energy=-14.0)
    water_atoms.new_array("REF_forces", np.zeros((3, 3)))
    path = write(tmp_path, [isolated_atom(8, -3.0), water_atoms])

    parsed = read_configurations(
        path,
        KeySpecification.from_defaults(),
        extract_isolated_atom_energies=True,
        keep_isolated_atoms=True,
    )
    assert parsed.isolated_atom_energies == {8: -3.0}
    assert [c.config_type for c in parsed.configurations] == ["IsolatedAtom", "Default"]


def test_without_extraction_nothing_is_removed_and_nothing_is_read(tmp_path):
    water_atoms = water(REF_energy=-14.0)
    water_atoms.new_array("REF_forces", np.zeros((3, 3)))
    path = write(tmp_path, [isolated_atom(8, -3.0), water_atoms])

    parsed = read_configurations(path, KeySpecification.from_defaults())
    assert parsed.isolated_atom_energies == {}
    assert len(parsed.configurations) == 2


def test_detection_needs_both_one_atom_and_the_label(tmp_path):
    """A two-atom structure labelled as an isolated atom is mislabelled
    training data, and a lone atom without the label is a one-atom structure to
    fit. Consuming either as a reference energy would be wrong."""
    pair = Atoms(numbers=[8, 8], positions=[[0.0] * 3, [1.2, 0.0, 0.0]])
    pair.info.update({"config_type": "IsolatedAtom", "REF_energy": -7.0})
    lone_unlabelled = isolated_atom(1, energy=-0.5, config_type="Default")
    path = write(tmp_path, [isolated_atom(8, -3.0), pair, lone_unlabelled])

    parsed = read_configurations(
        path, KeySpecification.from_defaults(), extract_isolated_atom_energies=True
    )
    assert parsed.isolated_atom_energies == {8: -3.0}
    assert len(parsed.configurations) == 2


def test_a_marked_isolated_atom_without_an_energy_records_zero(tmp_path, caplog):
    path = write(tmp_path, [isolated_atom(8, -3.0), isolated_atom(1, None)])
    with caplog.at_level("WARNING"):
        parsed = read_configurations(
            path,
            KeySpecification.from_defaults(),
            extract_isolated_atom_energies=True,
            no_data_ok=True,
        )
    assert parsed.isolated_atom_energies == {8: -3.0, 1: 0.0}
    assert "Recording zero" in caplog.text
    assert parsed.configurations == []


def test_a_reference_energy_is_read_through_the_rewritten_key(tmp_path):
    """A deliberate divergence from legacy, and the only one that changes a
    number. Legacy captured the energy key *before* the reserved-key rewrite
    and then looked up the pre-rewrite spelling, so asking for the reserved key
    recorded every reference energy as zero. Here the rewritten key is used, so
    the value the calculator carries is the one that is read."""
    reference = Atoms(numbers=[8], positions=[[0.0, 0.0, 0.0]])
    reference.info["config_type"] = "IsolatedAtom"
    reference.calc = SinglePointCalculator(reference, energy=-3.0)
    path = write(tmp_path, [reference, calculator_labelled_water()])

    parsed = read_configurations(
        path,
        KeySpecification.from_defaults().update(info_keys={"energy": "energy"}),
        extract_isolated_atom_energies=True,
    )
    assert parsed.isolated_atom_energies == {8: -3.0}


# ---------------------------------------------------------------------------
# Splitting and grouping
# ---------------------------------------------------------------------------


def test_a_split_is_reproducible_and_leaves_no_file_for_a_small_holdout(tmp_path):
    """The indices are pinned, not just their count: the split has to be the
    same one a run recorded before the rewrite."""
    items = list(range(20))
    train, valid = random_train_valid_split(items, 0.1, seed=1, work_dir=tmp_path)

    assert valid == [6, 19]
    assert train == [1, 10, 18, 16, 7, 11, 12, 17, 15, 2, 3, 4, 5, 8, 0, 9, 14, 13]
    assert sorted(train + valid) == items
    assert not (tmp_path / "valid_indices_1.txt").exists()


def test_ten_or_more_validation_items_are_written_to_a_named_file(tmp_path):
    items = list(range(100))
    train, valid = random_train_valid_split(items, 0.2, seed=7, work_dir=tmp_path)

    assert len(train) == 80
    assert valid == [
        18,
        31,
        34,
        5,
        72,
        71,
        33,
        15,
        38,
        87,
        52,
        25,
        69,
        43,
        66,
        95,
        60,
        21,
        41,
        11,
    ]
    index_file = tmp_path / "valid_indices_7.txt"
    assert index_file.exists()
    assert [int(line) for line in index_file.read_text().split()] == valid


def test_the_prefix_keeps_two_runs_in_one_directory_apart(tmp_path):
    random_train_valid_split(
        list(range(100)), 0.2, seed=3, work_dir=tmp_path, prefix="run7"
    )
    assert (tmp_path / "run7_valid_indices_3.txt").exists()
    assert not (tmp_path / "valid_indices_3.txt").exists()


def test_the_validation_set_is_never_empty(tmp_path):
    train, valid = random_train_valid_split(
        list(range(2)), 0.1, seed=0, work_dir=tmp_path
    )
    assert len(train) == 1
    assert len(valid) == 1


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 2.0])
def test_a_fraction_that_holds_out_nothing_or_everything_raises(tmp_path, fraction):
    with pytest.raises(ValueError, match="valid_fraction"):
        random_train_valid_split(list(range(10)), fraction, seed=0, work_dir=tmp_path)


def test_a_single_item_cannot_be_split(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        random_train_valid_split([1], 0.5, seed=0, work_dir=tmp_path)


def configuration(config_type: str, head: str) -> Configuration:
    return Configuration(
        atomic_numbers=np.array([1]),
        positions=np.zeros((1, 3)),
        config_type=config_type,
        head=head,
    )


def test_grouping_keys_on_the_config_type_and_the_head_together():
    configs = [
        configuration("bulk", "Default"),
        configuration("slab", "Default"),
        configuration("bulk", "Default"),
        configuration("bulk", "dft"),
    ]
    grouped = group_by_config_type(configs)

    assert [name for name, _ in grouped] == ["bulk_Default", "slab_Default", "bulk_dft"]
    assert {name: len(group) for name, group in grouped} == {
        "bulk_Default": 2,
        "slab_Default": 1,
        "bulk_dft": 1,
    }


def test_grouping_reads_an_absent_head_as_the_empty_string():
    """Legacy reached the same name from a head of ``None``, which it
    normalised to ``""`` by writing it back onto the configuration. Here the
    field is typed as a string, so the empty head is the only way in, and the
    grouping edits nothing it is reporting on."""
    config = configuration("bulk", "")
    ((name, group),) = group_by_config_type([config])
    assert name == "bulk_"
    assert group == [config]
    assert config.head == ""


# ---------------------------------------------------------------------------
# The element table
# ---------------------------------------------------------------------------


def test_the_factory_sorts_and_de_duplicates():
    table = atomic_number_table_from_zs([8, 1, 8, 6, 1])
    assert list(table.zs) == [1, 6, 8]
    assert len(table) == 3
    assert table.index_to_z(2) == 8
    assert table.z_to_index(6) == 1


def test_the_class_takes_the_order_it_is_given():
    """A trained model's element order comes out of its checkpoint and has to
    be adopted exactly: sorting it would repoint every embedding weight."""
    table = AtomicNumberTable([8, 1, 6])
    assert list(table.zs) == [8, 1, 6]
    assert table.z_to_index(8) == 0


def test_an_element_the_table_does_not_have_raises():
    with pytest.raises(ValueError):
        atomic_number_table_from_zs([1, 8]).z_to_index(79)


def test_two_tables_with_the_same_order_are_equal():
    assert atomic_number_table_from_zs([8, 1]) == AtomicNumberTable([1, 8])
    assert AtomicNumberTable([1, 8]) != AtomicNumberTable([8, 1])
