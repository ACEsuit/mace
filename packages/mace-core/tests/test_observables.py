"""The declarative observable specification.

The acceptance bar is "zero new code", so most of these tests declare something
as plain data, the rows a configuration would hold, and assert what comes out.
If any of them needed a new branch in ``mace_core`` to pass, the abstraction
would not be doing its job.
"""

import pytest
from mace_core.observables import (
    DEFAULT_CATALOGUE,
    DerivativeRequest,
    InputSpec,
    IrrepsGrammarError,
    IrrepTerm,
    ObservableCatalogue,
    ObservableSpec,
    default_derivative_name,
    irreps_dimension,
    parse_irreps,
)
from pydantic import ValidationError

# The two inputs every model has, written out so that the tests below can add
# one row at a time to them.
BASE_INPUTS = [
    {"name": "pos", "irreps": "1o", "per_atom": True, "units": "Å"},
    {"name": "strain", "irreps": "0e+2e", "per_atom": False, "units": "1"},
]

ENERGY_ROW = {"name": "energy", "irreps": "0e", "per_atom": False, "units": "eV"}


def catalogue_from(observables, inputs=()):
    return ObservableCatalogue.model_validate(
        {"inputs": [*BASE_INPUTS, *inputs], "observables": list(observables)}
    )


# ---------------------------------------------------------------------------
# The irreps grammar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "dimension"),
    [
        ("0e", 1),
        ("1o", 3),
        ("1e", 3),
        ("0e+2e", 6),
        ("128x0e+128x1o+128x2e", 128 * (1 + 3 + 5)),
        (" 0e + 1o ", 4),
    ],
)
def test_the_grammar_accepts_a_sum_of_multiplied_irreps(text, dimension):
    assert irreps_dimension(text) == dimension


def test_a_term_keeps_its_written_order_and_its_parity():
    terms = parse_irreps("2x1o+0e")
    assert [(t.multiplicity, t.degree, t.parity) for t in terms] == [
        (2, 1, "o"),
        (1, 0, "e"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "1",
        "1x",
        "x0e",
        "0e+",
        "1u",
        "-1o",
        # A multiplicity of zero spans no components.
        "0x1o",
        "2x0e+0x1o",
        # A leading zero would make `007e` read as l = 7.
        "007e",
        "01x0e",
        # Non-ASCII digits, which `int` would otherwise accept.
        "\u0661o",
    ],
)
def test_a_malformed_declaration_is_rejected(text):
    with pytest.raises(IrrepsGrammarError):
        parse_irreps(text)


@pytest.mark.parametrize(
    ("multiplicity", "degree", "parity"),
    [
        (0, 1, "o"),
        (-2, 1, "o"),
        (1, -1, "e"),
        (1, 0, "x"),
        (True, 0, "e"),
        (1, 1.0, "o"),
    ],
)
def test_a_term_built_directly_is_held_to_the_grammar(multiplicity, degree, parity):
    with pytest.raises(IrrepsGrammarError):
        IrrepTerm(multiplicity=multiplicity, degree=degree, parity=parity)


def test_a_grammar_error_names_the_observable_and_the_grammar():
    with pytest.raises(IrrepsGrammarError) as caught:
        parse_irreps("1u", observable="quadrupole")
    message = str(caught.value)
    assert "quadrupole" in message
    assert "'1u'" in message
    assert "multiplicity" in message and "parity" in message


def test_a_malformed_spec_names_the_observable_and_the_grammar():
    """The same contract through pydantic, which is how a user meets it."""
    with pytest.raises(ValidationError) as caught:
        ObservableSpec(
            name="quadrupole",
            irreps="rank2",
            per_atom=True,
            units="e*Å^2",
        )
    message = str(caught.value)
    assert "quadrupole" in message
    assert "'rank2'" in message
    assert "Examples: '0e'" in message


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_an_unknown_field_is_an_error_rather_than_ignored():
    with pytest.raises(ValidationError):
        ObservableSpec(
            name="q",
            irreps="0e",
            per_atom=False,
            units="eV",
            weight=3.0,  # ty: ignore[unknown-argument]
        )


def test_a_name_that_is_not_an_identifier_is_rejected():
    with pytest.raises(ValidationError) as caught:
        ObservableSpec(
            name="latent charges",
            irreps="0e",
            per_atom=True,
            units="e",
        )
    assert "identifier" in str(caught.value)


def test_units_may_not_be_empty():
    with pytest.raises(ValidationError):
        ObservableSpec(name="q", irreps="0e", per_atom=False, units="")


# ---------------------------------------------------------------------------
# Derivative naming and signs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quantity", "wrt", "name"),
    [
        ("dipole", "pos", "d_dipole_d_pos"),
        ("dipole", "strain", "d_dipole_d_strain"),
        ("polarizability", "pos", "d_polarizability_d_pos"),
        ("energy", "elec_temp", "d_energy_d_elec_temp"),
        ("quadrupole", "magmom", "d_quadrupole_d_magmom"),
    ],
)
def test_an_undeclared_name_follows_the_rule(quantity, wrt, name):
    assert default_derivative_name(quantity, wrt) == name
    spec = ObservableSpec(name=quantity, irreps="0e", per_atom=False, units="eV")
    assert spec.derivative_name(wrt) == name
    assert spec.derivative_sign(wrt) == +1


def test_a_declared_name_and_sign_win_over_the_rule():
    """The whole point: a quantity with a name of its own says so in the file."""
    spec = ObservableSpec(
        name="energy",
        irreps="0e",
        per_atom=False,
        units="eV",
        derivatives=[{"wrt": "magmom", "name": "magforces", "sign": -1}],
    )
    assert spec.derivative_name("magmom") == "magforces"
    assert spec.derivative_sign("magmom") == -1


def test_an_input_that_was_not_requested_still_has_a_name():
    """Naming works for any declared input, requested or not."""
    spec = ObservableSpec(
        name="energy",
        irreps="0e",
        per_atom=False,
        units="eV",
        derivatives=[{"wrt": "pos", "name": "forces", "sign": -1}],
    )
    assert spec.derivative_name("strain") == "d_energy_d_strain"
    assert spec.derivative_sign("strain") == +1


def test_a_name_without_a_sign_is_refused():
    """A renamed derivative inheriting +1 in silence trains inverted forces."""
    with pytest.raises(ValidationError, match="declares no sign"):
        DerivativeRequest(wrt="pos", name="forces")


@pytest.mark.parametrize("sign", [0, 2, -3, True, False, 1.0, -1.0, "-1"])
def test_a_sign_that_is_not_the_integer_plus_or_minus_one_is_refused(sign):
    """`True` and `1.0` compare equal to 1, and pydantic would coerce `"-1"`."""
    with pytest.raises(ValidationError, match="A sign is the integer \\+1 or -1"):
        DerivativeRequest(wrt="pos", name="forces", sign=sign)


def test_a_custom_name_may_not_imitate_the_generated_spelling():
    """`d_<q>_d_<x>` states which quantity was differentiated. A custom name
    wearing that spelling would be stating something that is not true."""
    with pytest.raises(ValidationError, match="spelled like the grammar"):
        DerivativeRequest(wrt="pos", name="d_dipole_d_pos", sign=+1)


def test_a_sign_alone_needs_no_name():
    request = DerivativeRequest(wrt="pos", sign=-1)
    assert request.name is None
    assert request.sign == -1


# ---------------------------------------------------------------------------
# The default catalogue
# ---------------------------------------------------------------------------


def test_the_defaults_declare_energy_and_its_two_derivatives():
    assert DEFAULT_CATALOGUE.names() == ("energy", "forces", "stress")


def test_the_default_names_and_signs_come_from_the_declaration_and_not_from_code():
    """The reason the table was removed: this is now a property of the data.

    If these were still special-cased in code, the assertion would pass with
    the declaration saying nothing at all.
    """
    resolved = {
        spec.name: spec.sign for spec in DEFAULT_CATALOGUE.requested_derivatives()
    }
    assert resolved == {"forces": -1, "stress": +1}

    stripped = DEFAULT_CATALOGUE.model_dump()
    for observable in stripped["observables"]:
        for request in observable["derivatives"]:
            request.pop("name")
            request.pop("sign")
    bare = ObservableCatalogue.model_validate(stripped)
    assert {spec.name for spec in bare.requested_derivatives()} == {
        "d_energy_d_pos",
        "d_energy_d_strain",
    }
    assert [spec.name for spec in DEFAULT_CATALOGUE.inputs] == ["pos", "strain"]


def test_the_default_forces_row_is_the_negative_position_gradient():
    forces = DEFAULT_CATALOGUE.derivative("energy", "pos")
    assert forces.name == "forces"
    assert forces.sign == -1
    assert forces.per_atom is True
    assert forces.irreps == "1o"
    assert forces.units == "eV/Å"


def test_the_default_stress_row_is_the_positive_strain_gradient():
    stress = DEFAULT_CATALOGUE.derivative("energy", "strain")
    assert stress.name == "stress"
    assert stress.sign == +1
    assert stress.per_atom is False
    assert stress.irreps == "0e+2e"


# ---------------------------------------------------------------------------
# The two "zero new code" acceptance cases
# ---------------------------------------------------------------------------


QUADRUPOLE_ROW = {
    "name": "quadrupole",
    "irreps": "0e+2e",
    "per_atom": True,
    "units": "e*Å^2",
}


def test_a_new_rank_two_per_atom_observable_is_one_declaration():
    catalogue = catalogue_from([{**QUADRUPOLE_ROW, "derivatives": ["strain"]}])
    quadrupole = catalogue.observable("quadrupole")
    assert quadrupole.per_atom is True
    assert quadrupole.dimension == 6
    assert catalogue.names() == ("quadrupole", "d_quadrupole_d_strain")
    derivative = catalogue.derivative("quadrupole", "strain")
    # One value per atom: a per-atom quantity against a per-graph input.
    assert derivative.per_atom is True
    # Not a scalar, so the irreps are a tensor product this package does not
    # compute.
    assert derivative.irreps is None


def test_a_per_atom_scalar_against_the_strain_is_per_atom():
    """The case the frozen tree's per-atom stresses belong to: one strain
    derivative per atom, carrying the strain's irreps."""
    catalogue = catalogue_from(
        [{"name": "site_energy", "irreps": "0e", "per_atom": True, "units": "eV"}]
    )
    derivative = catalogue.derivative("site_energy", "strain")
    assert derivative.per_atom is True
    assert derivative.irreps == "0e+2e"


def test_a_per_atom_observable_against_a_per_atom_input_is_refused():
    """d(per-atom q)/d(pos) is an (n_atoms, n_atoms, ...) block, which neither
    classification describes. Requested, it is refused at declaration."""
    with pytest.raises(ValidationError) as caught:
        catalogue_from([{**QUADRUPOLE_ROW, "derivatives": ["pos"]}])
    message = str(caught.value)
    assert "'quadrupole'" in message
    assert "'pos'" in message
    assert "n_atoms, n_atoms" in message


def test_an_unrepresentable_derivative_cannot_be_resolved_either():
    """Naming works for a pair nobody requested, so the refusal has to hold
    there too, or a consumer could still be handed a wrong classification."""
    catalogue = catalogue_from([QUADRUPOLE_ROW])
    with pytest.raises(ValueError, match="n_atoms, n_atoms"):
        catalogue.derivative("quadrupole", "pos")


def test_a_new_input_feature_makes_its_derivative_declarable():
    """`magmom` is the case that pays for the grammar being written over
    declared inputs rather than over positions and the strain.

    It is also the case that pays for the name and the sign living in the
    declaration. `magforces` used to be a row in a table inside this package,
    and this catalogue reaches it with no code at all.
    """
    catalogue = catalogue_from(
        [
            {
                **ENERGY_ROW,
                "derivatives": [
                    {
                        "wrt": "magmom",
                        "name": "magforces",
                        "sign": -1,
                        "units": "eV/muB",
                    }
                ],
            }
        ],
        inputs=[{"name": "magmom", "irreps": "1o", "per_atom": True, "units": "muB"}],
    )
    magforces = catalogue.derivative("energy", "magmom")
    assert magforces.name == "magforces"
    assert magforces.sign == -1
    assert magforces.per_atom is True
    # The magnetic model expands the moment in spherical harmonics, so it is a
    # polar `1o` vector there, and the gradient of a scalar against it is too.
    assert magforces.irreps == "1o"
    assert catalogue.names() == ("energy", "magforces")


def test_the_bare_string_form_and_the_mapping_form_agree():
    shorthand = catalogue_from([{**ENERGY_ROW, "derivatives": ["pos"]}])
    assert shorthand.observable("energy").derivatives == (DerivativeRequest(wrt="pos"),)


# ---------------------------------------------------------------------------
# Catalogue-level validation: the errors that live between rows
# ---------------------------------------------------------------------------


def test_a_derivative_against_an_undeclared_input_is_an_error():
    with pytest.raises(ValidationError) as caught:
        catalogue_from([{**ENERGY_ROW, "derivatives": ["elec_temp"]}])
    message = str(caught.value)
    assert "energy" in message
    assert "elec_temp" in message
    assert "['pos', 'strain']" in message


@pytest.mark.parametrize("requested", [["pos"], []])
def test_an_observable_may_not_wear_a_generated_derivative_name(requested):
    """Whether or not the derivative it names was requested: the spelling
    claims the observable is d(energy)/d(pos), and a declared row is not."""
    with pytest.raises(ValidationError) as caught:
        catalogue_from(
            [
                {
                    "name": "d_energy_d_pos",
                    "irreps": "1o",
                    "per_atom": True,
                    "units": "eV/Å",
                },
                {**ENERGY_ROW, "derivatives": requested},
            ]
        )
    assert "'d_energy_d_pos'" in str(caught.value)
    assert "spelled like a generated derivative name" in str(caught.value)


def test_an_observable_may_not_share_a_name_with_an_input():
    with pytest.raises(ValidationError) as caught:
        catalogue_from(
            [{"name": "pos", "irreps": "1o", "per_atom": True, "units": "Å"}]
        )
    assert "observable 'pos' has the name of a declared input" in str(caught.value)


def test_a_declared_derivative_name_may_not_share_a_name_with_an_input():
    with pytest.raises(ValidationError) as caught:
        catalogue_from(
            [
                {
                    **ENERGY_ROW,
                    "derivatives": [{"wrt": "pos", "name": "strain", "sign": -1}],
                }
            ]
        )
    assert "'strain'" in str(caught.value)
    assert "already the name of a declared input" in str(caught.value)


@pytest.mark.parametrize(
    ("name", "instead"), [("total_energy", "energy"), ("node_energy", "node_energies")]
)
def test_a_name_the_output_type_reserves_is_refused(name, instead):
    """`total_energy` is where `energy` is stored and `node_energy` is a retired
    spelling. Either would pass here and fail only when a model filled it."""
    with pytest.raises(ValidationError) as caught:
        ObservableSpec(name=name, irreps="0e", per_atom=False, units="eV")
    assert f"Use {instead!r} instead" in str(caught.value)
    with pytest.raises(ValidationError) as caught:
        DerivativeRequest(wrt="pos", name=name, sign=-1)
    assert f"Use {instead!r} instead" in str(caught.value)


def test_a_declared_name_may_not_collide_with_a_declared_observable():
    """The same guard, now reachable through a name the declaration chose."""
    with pytest.raises(ValidationError) as caught:
        catalogue_from(
            [
                {"name": "forces", "irreps": "1o", "per_atom": True, "units": "eV/Å"},
                {
                    **ENERGY_ROW,
                    "derivatives": [{"wrt": "pos", "name": "forces", "sign": -1}],
                },
            ]
        )
    assert "'forces'" in str(caught.value)


def test_a_name_declared_twice_is_an_error():
    with pytest.raises(ValidationError) as caught:
        catalogue_from([ENERGY_ROW, ENERGY_ROW])
    assert "declared twice" in str(caught.value)


def test_asking_for_the_same_derivative_twice_is_an_error():
    with pytest.raises(ValidationError):
        catalogue_from([{**ENERGY_ROW, "derivatives": ["pos", "pos"]}])


def test_an_unknown_observable_or_input_says_what_is_declared():
    catalogue = DEFAULT_CATALOGUE
    with pytest.raises(KeyError) as caught:
        catalogue.observable("dipole")
    assert "['energy']" in str(caught.value)
    with pytest.raises(KeyError) as caught:
        catalogue.input("magmom")
    assert "['pos', 'strain']" in str(caught.value)


def test_a_derivative_can_be_named_without_having_been_requested():
    """Naming is a property of the pair; requesting is what says "compute it"."""
    catalogue = ObservableCatalogue(
        inputs=[InputSpec(name="pos", irreps="1o", per_atom=True, units="Å")],
        observables=[
            ObservableSpec(
                name="dipole",
                irreps="1o",
                per_atom=False,
                units="Debye",
            )
        ],
    )
    assert catalogue.names() == ("dipole",)
    derived = catalogue.derivative("dipole", "pos")
    assert derived.name == "d_dipole_d_pos"
    # The differentiated quantity is not a scalar, so the gradient's irreps are
    # a tensor product this package deliberately does not compute.
    assert derived.irreps is None
