"""Parsing labelled structure files into configurations.

These cases are ported from the characterization suite that pinned the legacy
parser against real files. What is asserted is the *behaviour* the rewrite owes
that suite: which key each property is read from, what a missing key does to
its weight, how the weights compose, and which single-atom structures are
consumed as E0 references.

Where this deliberately does not match legacy, that is asserted here as well,
next to the case it changes, so none of it is a surprise found later by
somebody diffing outputs.
"""

from __future__ import annotations

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import voigt_6_to_full_3x3_stress
from mace_core.data import (
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

#: The default file keys, where each is stored and the shape of one value (of
#: one atom's entry, for a per-atom property), written out rather than read
#: from the enum. The point of the table is that these exact strings, places
#: and shapes are what every labelled dataset on disk uses, so a test that
#: compared the enum against itself would pass through a rename, or a property
#: moving from one half to the other, that broke every one of those files.
DEFAULT_KEY_TABLE = {
    "ENERGY": ("REF_energy", "graph", ()),
    "FORCES": ("REF_forces", "atom", (3,)),
    "STRESS": ("REF_stress", "graph", (3, 3)),
    "VIRIALS": ("REF_virials", "graph", (3, 3)),
    "DIPOLE": ("dipole", "graph", (3,)),
    "POLARIZABILITY": ("polarizability", "graph", (3, 3)),
    "CHARGES": ("REF_charges", "atom", ()),
    "TOTAL_CHARGE": ("total_charge", "graph", ()),
    "TOTAL_SPIN": ("total_spin", "graph", ()),
    "ELEC_TEMP": ("elec_temp", "graph", ()),
    "MAGMOM": ("REF_magmom", "atom", (3,)),
    "MAGFORCES": ("REF_magforces", "atom", (3,)),
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
    """A structure carrying a value under all twelve default keys."""
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


def test_the_default_keys_are_exactly_these_twelve():
    assert {
        member.name: (member.value, member.storage, member.value_shape)
        for member in DefaultKeys
    } == DEFAULT_KEY_TABLE
    assert len(DEFAULT_KEY_TABLE) == 12


@pytest.mark.parametrize("storage", ["atoms", "arrays", "info", ""])
def test_a_storage_that_does_not_exist_raises_rather_than_matching_nothing(storage):
    with pytest.raises(ValueError, match=repr(storage)):
        DefaultKeys.names_stored_per(storage)


def test_the_command_line_spelling_is_derived_from_the_member_name():
    assert DefaultKeys.keydict() == {
        f"{name.lower()}_key": key for name, (key, _, _) in DEFAULT_KEY_TABLE.items()
    }
    assert DefaultKeys.keydict()["energy_key"] == "REF_energy"
    assert DefaultKeys.keydict()["magforces_key"] == "REF_magforces"


def test_from_defaults_routes_every_key_to_the_half_the_table_names():
    """Checked against the written-out table, not against the enum it is
    built from."""
    key_spec = KeySpecification.from_defaults()
    assert key_spec.graph_keys == {
        name.lower(): key
        for name, (key, per, _) in DEFAULT_KEY_TABLE.items()
        if per == "graph"
    }
    assert key_spec.atom_keys == {
        name.lower(): key
        for name, (key, per, _) in DEFAULT_KEY_TABLE.items()
        if per == "atom"
    }


# ---------------------------------------------------------------------------
# KeySpecification
# ---------------------------------------------------------------------------


def test_update_merges_and_returns_self():
    key_spec = KeySpecification(graph_keys={"energy": "a"})
    returned = key_spec.update(graph_keys={"energy": "b"}, atom_keys={"forces": "f"})
    assert returned is key_spec
    assert key_spec.graph_keys["energy"] == "b"
    assert key_spec.atom_keys["forces"] == "f"


def test_overrides_route_by_name_and_ignore_everything_else():
    key_spec = KeySpecification()
    key_spec.apply_overrides(
        {"energy_key": "E", "forces_key": "F", "unrelated": "x", "seed": 3}
    )
    assert key_spec.graph_keys == {"energy": "E"}
    assert key_spec.atom_keys == {"forces": "F"}


def test_an_override_for_an_unknown_property_is_an_error():
    """Legacy dropped it, and the property then silently never parsed."""
    with pytest.raises(ValueError, match="enthalpy_key"):
        KeySpecification().apply_overrides({"enthalpy_key": "REF_enthalpy"})


def test_a_copy_does_not_share_its_dictionaries():
    original = KeySpecification.from_defaults()
    clone = original.copy()
    clone.graph_keys["energy"] = "somewhere_else"
    assert original.graph_keys["energy"] == "REF_energy"


def test_a_name_in_both_halves_is_refused_by_the_constructor():
    with pytest.raises(ValueError, match="'site_spin' is in both halves"):
        KeySpecification(graph_keys={"site_spin": "a"}, atom_keys={"site_spin": "b"})


def test_a_name_in_both_halves_is_refused_by_update():
    """The per-structure value would otherwise be silently lost: the parser
    keeps whichever half it reads last."""
    key_spec = KeySpecification.from_defaults()
    with pytest.raises(ValueError, match="'charges' is in both halves"):
        key_spec.update(graph_keys={"charges": "qq"})
    # and the failed update left the specification as it was
    assert "charges" not in key_spec.graph_keys
    assert key_spec.atom_keys["charges"] == "REF_charges"


@pytest.mark.parametrize(
    ("graph_keys", "atom_keys", "name"),
    [
        ({"energy": "REF_energy", "forces": "REF_forces"}, {}, "forces"),
        ({}, {"energy": "REF_energy"}, "energy"),
        ({}, {"stress": "REF_stress"}, "stress"),
        ({"magmom": "REF_magmom"}, {}, "magmom"),
    ],
)
def test_a_default_name_in_the_wrong_half_is_refused(graph_keys, atom_keys, name):
    """Forces read from the per-structure values find nothing, and would be
    reported as an absent label at weight zero."""
    with pytest.raises(ValueError, match=rf"'{name}' is in the \w+ half"):
        KeySpecification(graph_keys=graph_keys, atom_keys=atom_keys)


@pytest.mark.parametrize(
    "name",
    [
        "atomic_numbers",
        "positions",
        "properties",
        "property_weights",
        "cell",
        "pbc",
        "weight",
        "config_type",
        "head",
    ],
)
def test_a_configuration_field_name_is_refused_on_every_path(name):
    """The head, for one, would otherwise be stored twice: on the
    configuration, from the caller, and among the properties, from the file,
    holding two different values."""
    with pytest.raises(ValueError, match=f"'{name}' is a field of Configuration"):
        KeySpecification.from_defaults().add_embedding_features(
            {name: {"per": "graph"}}
        )
    with pytest.raises(ValueError, match=f"'{name}' is a field of Configuration"):
        KeySpecification.from_defaults().update(graph_keys={name: name})
    with pytest.raises(ValueError, match=f"'{name}' is a field of Configuration"):
        KeySpecification(atom_keys={name: name})


@pytest.mark.parametrize("per", ["graph", "atom"])
def test_a_feature_named_like_a_default_property_raises_in_either_order(per):
    """Before or after the overrides, and whether or not the specification
    already resolves the name: the answer cannot depend on the order the two
    were applied in."""
    overrides = {"charges_key": "qq"}
    with pytest.raises(ValueError, match="charges_key"):
        KeySpecification().add_embedding_features({"charges": {"per": per}})
    with pytest.raises(ValueError, match="charges_key"):
        KeySpecification().apply_overrides(overrides).add_embedding_features(
            {"charges": {"per": per}}
        )


def test_a_directly_edited_specification_is_checked_again_when_copied():
    """`read_configurations` copies the specification before anything else,
    so this is where a parse refuses it."""
    key_spec = KeySpecification.from_defaults()
    key_spec.graph_keys["forces"] = "REF_forces"
    with pytest.raises(ValueError, match="'forces' is in both halves"):
        key_spec.copy()


def test_an_embedding_feature_lands_in_the_half_its_per_names():
    key_spec = KeySpecification()
    key_spec.add_embedding_features(
        {
            "site_spin": {"per": "atom", "key": "spin_array"},
            "applied_field": {"per": "graph"},
        }
    )
    assert key_spec.atom_keys["site_spin"] == "spin_array"
    # no declared key: the feature is read from its own name
    assert key_spec.graph_keys["applied_field"] == "applied_field"


def test_an_embedding_feature_can_be_given_as_a_typed_spec():
    key_spec = KeySpecification()
    key_spec.add_embedding_features(
        {"site_spin": EmbeddingFeatureSpec(per="atom", key="spin_array")}
    )
    assert key_spec.atom_keys == {"site_spin": "spin_array"}


@pytest.mark.parametrize("per", ["bond", "cell", ""])
def test_an_embedding_feature_stored_nowhere_real_raises(per):
    with pytest.raises(ValueError, match="site_spin"):
        KeySpecification().add_embedding_features({"site_spin": {"per": per}})


@pytest.mark.parametrize("per", ["bond", "cell", ""])
def test_a_typed_spec_is_held_to_the_same_rule(per):
    """Built directly, not through a mapping. It used to be accepted and
    routed to the per-structure half, since only "atom" was tested for."""
    with pytest.raises(ValueError, match="per"):
        EmbeddingFeatureSpec(per=per)


@pytest.mark.parametrize(("name", "per"), [("charges", "graph"), ("charges", "atom")])
def test_a_feature_named_like_a_default_property_raises(name, per):
    """In the other half it would sit in both, and the parser would keep
    whichever half it read last. In the same half it would replace the file
    key the property is read from, which is the override's job."""
    with pytest.raises(ValueError, match=f"{name}_key"):
        KeySpecification.from_defaults().add_embedding_features({name: {"per": per}})


def test_a_feature_declared_twice_raises():
    key_spec = KeySpecification().add_embedding_features({"spin": {"per": "atom"}})
    with pytest.raises(ValueError, match="declared twice"):
        key_spec.add_embedding_features({"spin": {"per": "graph"}})


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
        graph_keys={"energy": "REF_energy", "stress": "REF_stress"},
        atom_keys={"forces": "REF_forces"},
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
    assert np.allclose(
        config.properties["stress"],
        voigt_6_to_full_3x3_stress(atoms.info["REF_stress"]),
    )
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


def test_the_head_is_the_callers_and_is_stored_once():
    """Legacy also kept it among the properties, where it held whatever the
    file said when the structure was converted directly, and the caller's head
    when it was read through a file. Now it lives on the configuration only."""
    atoms = water(REF_energy=0.0, head="from_the_file")
    config = configuration_from_atoms(
        atoms, KeySpecification.from_defaults(), head_name="dft"
    )
    assert config.head == "dft"
    assert "head" not in KeySpecification.from_defaults().property_names()
    assert "head" not in config.properties
    assert "head" not in config.property_weights


@pytest.mark.parametrize("name", ["stress", "virials"])
@pytest.mark.parametrize(
    "value",
    [np.arange(9.0).reshape(3, 3), np.array([0.0, 4.0, 8.0, 5.0, 2.0, 1.0])],
    ids=["matrix", "voigt"],
)
def test_a_stress_or_virial_is_always_a_full_matrix(name, value):
    """The six Voigt components are ase's order: xx, yy, zz, yz, xz, xy."""
    atoms = water(REF_energy=0.0)
    atoms.info[DefaultKeys[name.upper()].value] = value
    config = configuration_from_atoms(atoms, KeySpecification.from_defaults())
    expected = value if value.shape == (3, 3) else voigt_6_to_full_3x3_stress(value)
    assert config.properties[name].shape == (3, 3)
    assert np.array_equal(config.properties[name], expected)
    if value.shape == (6,):
        assert config.properties[name][1, 2] == config.properties[name][2, 1] == 5.0
        assert config.properties[name][0, 1] == config.properties[name][1, 0] == 1.0


@pytest.mark.parametrize("shape", [(9,), (3,), (1, 9), ()])
@pytest.mark.parametrize("name", ["stress", "virials"])
def test_a_stress_or_virial_of_any_other_shape_raises(name, shape):
    """A flat nine included: reshaping it would guess the order it was
    written in."""
    atoms = water(REF_energy=0.0)
    atoms.info[DefaultKeys[name.upper()].value] = np.zeros(shape)
    with pytest.raises(ValueError, match=rf"{name} has shape"):
        configuration_from_atoms(atoms, KeySpecification.from_defaults())


def test_a_directly_built_configuration_holds_the_same_shapes():
    """The shape rule belongs to the configuration, not to one parser, so a
    configuration built by hand is held to it too."""
    with pytest.raises(ValueError, match="stress has shape"):
        Configuration(
            atomic_numbers=np.array([1]),
            positions=np.zeros((1, 3)),
            properties={"stress": np.zeros(9)},
        )
    voigt = np.array([0.0, 4.0, 8.0, 5.0, 2.0, 1.0])
    config = Configuration(
        atomic_numbers=np.array([1]),
        positions=np.zeros((1, 3)),
        properties={"virials": voigt},
    )
    assert np.array_equal(
        config.properties["virials"], voigt_6_to_full_3x3_stress(voigt)
    )


#: Shapes a label must be refused in, on a three-atom structure. Each is a
#: layout somebody could plausibly write: a flat matrix, a per-atom array with
#: a component missing or an extra axis, a scalar wrapped in a list.
WRONG_SHAPES = [
    ("energy", (1,)),
    ("forces", (3, 2)),
    ("forces", (9,)),
    ("forces", (2, 3)),
    ("dipole", (5,)),
    ("dipole", (1, 3)),
    ("polarizability", (9,)),
    ("polarizability", (6,)),
    ("charges", (3, 1)),
    ("charges", (2,)),
    ("total_charge", (2,)),
    ("total_spin", (1,)),
    ("elec_temp", (3,)),
    ("magmom", (3,)),
    ("magforces", (3, 1)),
]


@pytest.mark.parametrize(("name", "shape"), WRONG_SHAPES)
def test_every_default_label_of_the_wrong_shape_raises(name, shape):
    with pytest.raises(ValueError, match=rf"{name} has shape") as caught:
        Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=np.zeros((3, 3)),
            properties={name: np.zeros(shape)},
        )
    _, per, value_shape = DEFAULT_KEY_TABLE[name.upper()]
    full_shape = (3, *value_shape) if per == "atom" else value_shape
    assert f"Expected {full_shape}" in str(caught.value)


def test_a_name_outside_the_default_table_keeps_its_shape():
    """A multi-level-of-theory name or an embedding feature has no row to
    check against, so it is carried as given."""
    config = Configuration(
        atomic_numbers=np.array([1]),
        positions=np.zeros((1, 3)),
        properties={"pbe_stress": np.zeros(9), "applied_field": np.zeros((2, 2))},
    )
    assert config.properties["pbe_stress"].shape == (9,)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("positions", np.zeros((2, 3)), "positions has shape"),
        ("positions", np.zeros(3), "positions has shape"),
        ("cell", np.zeros(9), "cell has shape"),
        ("pbc", (True, True), "pbc is"),
    ],
)
def test_the_structure_itself_is_shape_checked(field, value, message):
    arguments = {"atomic_numbers": np.array([1]), "positions": np.zeros((1, 3))}
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        Configuration(**arguments)


def test_the_head_is_refused_as_a_property_of_a_configuration():
    with pytest.raises(ValueError, match="'head'"):
        Configuration(
            atomic_numbers=np.array([1]),
            positions=np.zeros((1, 3)),
            properties={"head": "dft"},
        )


def test_the_shape_error_names_the_structure_and_the_file(tmp_path):
    good = water(REF_energy=0.0)
    bad = water(REF_energy=0.0)
    bad.info["REF_stress"] = np.zeros(9)
    path = write(tmp_path, [good, bad])
    with pytest.raises(ValueError, match="structure 1 of") as caught:
        read_configurations(path, KeySpecification.from_defaults())
    assert path in str(caught.value)


def test_the_shape_error_counts_the_isolated_atoms_that_were_dropped(tmp_path):
    """The index is the structure's position in the file, the same one the
    isolated-atom errors use, not its position after the E0 extraction."""
    bad = water(REF_energy=0.0)
    bad.info["REF_stress"] = np.zeros(9)
    path = write(tmp_path, [isolated_atom(8, -3.0), water(REF_energy=0.0), bad])
    with pytest.raises(ValueError, match="structure 2 of"):
        read_configurations(
            path, KeySpecification.from_defaults(), extract_isolated_atom_energies=True
        )


def test_two_configurations_compare_by_identity():
    """A field-by-field comparison would put numpy arrays through `==`, which
    raises for any array longer than one."""
    first = configuration_from_atoms(water(), KeySpecification.from_defaults())
    second = configuration_from_atoms(water(), KeySpecification.from_defaults())
    assert first == first
    assert first != second
    assert len({first, second}) == 2


# ---------------------------------------------------------------------------
# Whether a label is present
# ---------------------------------------------------------------------------


def test_a_zero_weight_and_an_absent_label_are_told_apart():
    """The weight cannot say which is which, so it must not be asked.

    A file is free to write `config_forces_weight=0.0` for a structure whose
    forces are there, and an absent label is zeroed too. Both give `0.0`, so a
    consumer reading the weight to find out whether a label exists is reading
    two different facts through one number.
    """
    key_spec = KeySpecification.from_defaults()

    weighted_to_zero = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    weighted_to_zero.arrays["REF_forces"] = np.zeros((2, 3))
    weighted_to_zero.info["config_forces_weight"] = 0.0
    absent = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    present = configuration_from_atoms(weighted_to_zero, key_spec)
    missing = configuration_from_atoms(absent, key_spec)

    assert (
        present.property_weights["forces"] == missing.property_weights["forces"] == 0.0
    )
    assert present.is_labelled("forces")
    assert not missing.is_labelled("forces")


def test_asking_about_an_undeclared_property_raises():
    """Every declared property is present, as None when absent, so a name that
    is missing is a misspelling, and False would read as "unlabelled"."""
    configuration = configuration_from_atoms(
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        KeySpecification.from_defaults(),
    )
    with pytest.raises(ValueError, match="'force'") as caught:
        configuration.is_labelled("force")
    assert "'forces'" in str(caught.value)


# ---------------------------------------------------------------------------
# Reading a file
# ---------------------------------------------------------------------------


def test_all_twelve_default_keys_survive_a_write_and_a_read(tmp_path):
    written = labelled_water()
    parsed = read_configurations(
        write(tmp_path, [written]), KeySpecification.from_defaults()
    )
    (config,) = parsed.configurations
    properties = config.properties

    assert properties["energy"] == pytest.approx(-14.5)
    assert np.allclose(properties["forces"], written.arrays["REF_forces"])
    assert np.allclose(
        properties["stress"], voigt_6_to_full_3x3_stress(written.info["REF_stress"])
    )
    assert np.allclose(
        properties["virials"], voigt_6_to_full_3x3_stress(written.info["REF_virials"])
    )
    assert np.array_equal(properties["polarizability"], np.arange(9.0).reshape(3, 3))
    assert np.allclose(properties["charges"], [-0.8, 0.4, 0.4])
    assert properties["total_charge"] == pytest.approx(-1.0)
    assert properties["total_spin"] == pytest.approx(2.0)
    assert properties["elec_temp"] == pytest.approx(300.0)
    assert np.allclose(properties["magmom"], [[0.0, 0.0, 2.2]] * 3)
    assert np.allclose(properties["magforces"], [[0.1, 0.2, 0.3]] * 3)
    # the head is the caller's, and it is not a property
    assert config.head == "Default"
    assert "head" not in properties

    # every property has a weight, and all twelve read
    assert set(config.property_weights) == set(properties)
    assert {n for n, w in config.property_weights.items() if w == 1.0} == {
        name.lower() for name in DEFAULT_KEY_TABLE
    }


def test_the_default_dipole_key_is_read_back_from_the_calculator(tmp_path):
    """ase reserves ``dipole`` as a per-structure calculator property, so a
    value written into ``info`` is read back into ``calc.results``. The parser
    reads it from there; a renamed key reads from ``info`` as before."""
    atoms = water(REF_energy=0.0)
    atoms.new_array("REF_forces", np.zeros((3, 3)))
    atoms.info["dipole"] = np.array([0.1, -0.2, 0.3])
    atoms.info["REF_dipole"] = np.array([0.4, 0.5, 0.6])
    path = write(tmp_path, [atoms])
    read_back = ase.io.read(path)
    assert isinstance(read_back, Atoms)
    assert "dipole" not in read_back.info

    parsed = read_configurations(path, KeySpecification.from_defaults())
    assert np.allclose(parsed.configurations[0].properties["dipole"], [0.1, -0.2, 0.3])
    assert parsed.configurations[0].property_weights["dipole"] == 1.0

    renamed = read_configurations(
        path,
        KeySpecification.from_defaults().apply_overrides({"dipole_key": "REF_dipole"}),
    )
    assert np.allclose(renamed.configurations[0].properties["dipole"], [0.4, 0.5, 0.6])


@pytest.mark.parametrize(
    ("setting", "file_key", "name", "store", "value"),
    [
        ("energy_key", "free_energy", "energy", "info", -3.25),
        ("charges_key", "charges", "charges", "arrays", np.array([-0.8, 0.4, 0.4])),
        ("magmom_key", "magmoms", "magmom", "arrays", np.full((3, 3), 0.5)),
    ],
)
def test_any_key_ase_moves_into_the_calculator_is_read_from_there(
    tmp_path, setting, file_key, name, store, value
):
    """Not only the three keys the parse rewrites: every name ase treats as a
    calculator property disappears from ``info`` and ``arrays`` on reading,
    and would otherwise come back as None at weight zero."""
    atoms = water(REF_energy=0.0)
    atoms.new_array("REF_forces", np.zeros((3, 3)))
    if store == "info":
        atoms.info[file_key] = value
    else:
        atoms.new_array(file_key, value)
    path = write(tmp_path, [atoms])
    read_back = ase.io.read(path)
    assert isinstance(read_back, Atoms)
    assert file_key not in read_back.info
    assert file_key not in read_back.arrays

    parsed = read_configurations(
        path, KeySpecification.from_defaults().apply_overrides({setting: file_key})
    )
    (config,) = parsed.configurations
    assert np.allclose(config.properties[name], value)
    assert config.property_weights[name] == 1.0


def test_a_dipole_in_the_calculator_passes_the_presence_check(tmp_path):
    """A dipole-only file has to count as labelled wherever ase put the
    dipole."""
    atoms = water()
    atoms.info["dipole"] = np.array([0.1, -0.2, 0.3])
    parsed = read_configurations(
        write(tmp_path, [atoms]), KeySpecification.from_defaults()
    )
    assert np.allclose(parsed.configurations[0].properties["dipole"], [0.1, -0.2, 0.3])


def test_an_undeclared_dipole_does_not_pass_the_presence_check(tmp_path):
    """A dipole the specification does not name is never read, so a file whose
    energy and forces keys both miss must still raise, even though ase holds a
    dipole in the calculator. The message names only the keys searched."""
    atoms = water()
    atoms.info["dipole"] = np.array([0.1, -0.2, 0.3])
    path = write(tmp_path, [atoms])
    key_spec = KeySpecification(
        graph_keys={"energy": "REF_energy"}, atom_keys={"forces": "REF_forces"}
    )
    with pytest.raises(ValueError, match="none of 'REF_energy', 'REF_forces' is"):
        read_configurations(path, key_spec)


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
    value = (
        np.arange(9.0).reshape(3, 3)
        if store == "arrays" or name == "stress"
        else np.float64(7.5)
    )
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
    assert all("head" not in c.properties for c in parsed.configurations)


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
        graph_keys={"energy": "energy", "stress": "stress"},
        atom_keys={"forces": "forces"},
    )

    parsed = read_configurations(path, key_spec)
    (config,) = parsed.configurations

    assert config.properties["energy"] == pytest.approx(-14.5)
    assert np.allclose(config.properties["forces"], np.arange(9.0).reshape(3, 3) / 10.0)
    # the calculator holds Voigt components; the configuration holds the matrix
    assert np.allclose(
        config.properties["stress"],
        voigt_6_to_full_3x3_stress(np.linspace(0.1, 0.6, 6)),
    )


def test_the_callers_key_specification_is_never_touched(tmp_path):
    """Legacy rewrote the caller's object and restored it at the end. Here the
    rewrite happens on a copy, so a specification shared between two datasets
    cannot come back holding whatever the first parse rewrote it to."""
    path = write(tmp_path, [calculator_labelled_water()])
    key_spec = KeySpecification.from_defaults().update(
        graph_keys={"energy": "energy", "stress": "stress"},
        atom_keys={"forces": "forces"},
    )
    before = (dict(key_spec.graph_keys), dict(key_spec.atom_keys))

    read_configurations(path, key_spec)

    assert (key_spec.graph_keys, key_spec.atom_keys) == before
    assert key_spec.graph_keys["energy"] == "energy"
    assert key_spec.atom_keys["forces"] == "forces"
    assert key_spec.graph_keys["stress"] == "stress"

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
    key_spec = KeySpecification.from_defaults().update(graph_keys={"energy": "energy"})

    parsed = read_configurations(write(tmp_path, [atoms]), key_spec)
    (config,) = parsed.configurations
    assert config.properties["energy"] is None
    assert config.property_weights["energy"] == 1.0


def test_a_reserved_key_is_rewritten_onto_its_default_spelling():
    """The rewrite has to land where the parser then looks.

    Spelled once, on the key table. A second copy here would keep saying
    `REF_energy` while the table moved.
    """
    from mace_core.data.xyz import _RESERVED_KEYS

    for name, reserved in _RESERVED_KEYS.items():
        assert reserved.rewritten == DefaultKeys[name.upper()].value


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


def test_repeated_isolated_atoms_that_agree_are_accepted(tmp_path):
    path = write(tmp_path, [isolated_atom(8, -3.0), isolated_atom(8, -3.0)])
    parsed = read_configurations(
        path,
        KeySpecification.from_defaults(),
        extract_isolated_atom_energies=True,
        no_data_ok=True,
    )
    assert parsed.isolated_atom_energies == {8: -3.0}


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_isolated_atoms_that_disagree_raise_in_either_order(tmp_path, order):
    """A deliberate divergence from legacy, which kept the last one. Two spin
    states of an oxygen atom are both legitimate references, and keeping the
    last made the E0 depend on the order of the file."""
    atoms = [isolated_atom(8, -2.0), isolated_atom(8, -1.5)]
    path = write(tmp_path, [atoms[i] for i in order] + [isolated_atom(1, -0.5)])
    with pytest.raises(ValueError, match="element 8") as caught:
        read_configurations(
            path,
            KeySpecification.from_defaults(),
            extract_isolated_atom_energies=True,
            no_data_ok=True,
        )
    assert "-2.0" in str(caught.value)
    assert "-1.5" in str(caught.value)
    assert "element 1" not in str(caught.value)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_an_energy_wins_over_a_marked_atom_without_one(tmp_path, order):
    atoms = [isolated_atom(8, -3.0), isolated_atom(8, None)]
    path = write(tmp_path, [atoms[i] for i in order])
    parsed = read_configurations(
        path,
        KeySpecification.from_defaults(),
        extract_isolated_atom_energies=True,
        no_data_ok=True,
    )
    assert parsed.isolated_atom_energies == {8: -3.0}


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
        KeySpecification.from_defaults().update(graph_keys={"energy": "energy"}),
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


def test_the_class_refuses_an_element_listed_twice():
    """Two indices would name one element, and dropping the repeat would shift
    every index after it."""
    with pytest.raises(ValueError, match=r"atomic numbers \[1\] appear more"):
        AtomicNumberTable([1, 8, 1])


def test_an_element_the_table_does_not_have_raises():
    with pytest.raises(ValueError):
        atomic_number_table_from_zs([1, 8]).z_to_index(79)


def test_two_tables_with_the_same_order_are_equal():
    assert atomic_number_table_from_zs([8, 1]) == AtomicNumberTable([1, 8])
    assert AtomicNumberTable([1, 8]) != AtomicNumberTable([8, 1])
