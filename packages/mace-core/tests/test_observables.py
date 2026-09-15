"""The declarative observable specification.

The acceptance bar is "zero new code", so most of these tests declare something
in YAML text and assert what comes out. If any of them needed a new branch in
``mace_core`` to pass, the abstraction would not be doing its job.
"""

import pytest
from mace_core.observables import (
    DerivativeRequest,
    InputSpec,
    IrrepsGrammarError,
    ObservableCatalogue,
    ObservableSpec,
    derivative_name,
    derivative_sign,
    irreps_dimension,
    load_catalogue,
    load_default_catalogue,
    parse_irreps,
)
from pydantic import ValidationError

# A catalogue with the two inputs every model has, written out so that the
# tests below can add one row at a time to it.
BASE_INPUTS = """
inputs:
  - name: pos
    irreps: "1o"
    per_atom: true
    units: "Å"
  - name: cell
    irreps: "0e+2e"
    per_atom: false
    units: "1"
"""


def catalogue_from(text, tmp_path):
    path = tmp_path / "observables.yaml"
    path.write_text(BASE_INPUTS + text, encoding="utf-8")
    return load_catalogue(path)


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


@pytest.mark.parametrize("text", ["", "   ", "1", "1x", "x0e", "0e+", "1u", "-1o"])
def test_a_malformed_declaration_is_rejected(text):
    with pytest.raises(IrrepsGrammarError):
        parse_irreps(text)


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
            normalization="none",
        )
    message = str(caught.value)
    assert "quadrupole" in message
    assert "'rank2'" in message
    assert "Examples: '0e'" in message


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_normalization_is_required_and_closed():
    # Both calls are rejected by the type checker as well, which is the point:
    # the field is a closed Literal, so a wrong value is caught statically and
    # at runtime. The ignores are what let the runtime half be tested.
    with pytest.raises(ValidationError):
        ObservableSpec(name="q", irreps="0e", per_atom=False, units="eV")  # ty: ignore[missing-argument]
    with pytest.raises(ValidationError):
        ObservableSpec(
            name="q",
            irreps="0e",
            per_atom=False,
            units="eV",
            normalization="minmax",  # ty: ignore[invalid-argument-type]
        )


def test_an_unknown_field_is_an_error_rather_than_ignored():
    with pytest.raises(ValidationError):
        ObservableSpec(
            name="q",
            irreps="0e",
            per_atom=False,
            units="eV",
            normalization="none",
            weight=3.0,  # ty: ignore[unknown-argument]
        )


def test_a_name_that_is_not_an_identifier_is_rejected():
    with pytest.raises(ValidationError) as caught:
        ObservableSpec(
            name="latent charges",
            irreps="0e",
            per_atom=True,
            units="e",
            normalization="none",
        )
    assert "identifier" in str(caught.value)


def test_units_may_not_be_empty():
    with pytest.raises(ValidationError):
        ObservableSpec(
            name="q", irreps="0e", per_atom=False, units="", normalization="none"
        )


# ---------------------------------------------------------------------------
# Derivative naming and signs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quantity", "wrt", "name", "sign"),
    [
        ("energy", "pos", "forces", -1),
        ("energy", "cell", "stress", +1),
        ("energy", "magmom", "magforces", -1),
    ],
)
def test_the_three_special_cases_keep_their_names_and_signs(quantity, wrt, name, sign):
    assert derivative_name(quantity, wrt) == name
    assert derivative_sign(quantity, wrt) == sign


@pytest.mark.parametrize(
    ("quantity", "wrt", "name"),
    [
        ("dipole", "pos", "d_dipole_d_pos"),
        ("dipole", "cell", "d_dipole_d_cell"),
        ("polarizability", "pos", "d_polarizability_d_pos"),
        ("energy", "elec_temp", "d_energy_d_elec_temp"),
        ("quadrupole", "magmom", "d_quadrupole_d_magmom"),
    ],
)
def test_everything_else_follows_the_rule(quantity, wrt, name):
    assert derivative_name(quantity, wrt) == name
    assert derivative_sign(quantity, wrt) == +1


# ---------------------------------------------------------------------------
# The shipped defaults
# ---------------------------------------------------------------------------


def test_the_defaults_declare_energy_and_its_two_derivatives():
    catalogue = load_default_catalogue()
    assert catalogue.names() == ("energy", "forces", "stress")
    assert [spec.name for spec in catalogue.inputs] == ["pos", "cell"]


def test_the_default_forces_row_is_the_negative_position_gradient():
    forces = load_default_catalogue().derivative("energy", "pos")
    assert forces.name == "forces"
    assert forces.sign == -1
    assert forces.per_atom is True
    assert forces.irreps == "1o"
    assert forces.units == "eV/Å"
    assert forces.default_loss_weight == 100.0


def test_the_default_stress_row_is_the_positive_strain_gradient():
    stress = load_default_catalogue().derivative("energy", "cell")
    assert stress.name == "stress"
    assert stress.sign == +1
    assert stress.per_atom is False
    assert stress.irreps == "0e+2e"


# ---------------------------------------------------------------------------
# The two "zero new code" acceptance cases
# ---------------------------------------------------------------------------


def test_a_new_rank_two_per_atom_observable_is_a_row_in_yaml(tmp_path):
    catalogue = catalogue_from(
        """
observables:
  - name: quadrupole
    irreps: "0e+2e"
    per_atom: true
    units: "e*Å^2"
    normalization: "rms"
    default_loss_weight: 2.5
    derivatives: [pos, cell]
""",
        tmp_path,
    )
    quadrupole = catalogue.observable("quadrupole")
    assert quadrupole.per_atom is True
    assert quadrupole.dimension == 6
    assert quadrupole.default_loss_weight == 2.5
    assert catalogue.names() == (
        "quadrupole",
        "d_quadrupole_d_pos",
        "d_quadrupole_d_cell",
    )


def test_a_new_input_feature_makes_its_derivative_declarable(tmp_path):
    """`magmom` is the case that pays for the grammar being written over
    declared inputs rather than over positions and the cell."""
    catalogue = catalogue_from(
        """
  - name: magmom
    irreps: "1e"
    per_atom: true
    units: "muB"

observables:
  - name: energy
    irreps: "0e"
    per_atom: false
    units: "eV"
    normalization: "std"
    derivatives:
      - wrt: magmom
        units: "eV/muB"
        default_loss_weight: 10.0
""",
        tmp_path,
    )
    magforces = catalogue.derivative("energy", "magmom")
    assert magforces.name == "magforces"
    assert magforces.sign == -1
    assert magforces.per_atom is True
    # A magnetic moment is an axial vector, so its conjugate force is too.
    assert magforces.irreps == "1e"
    assert magforces.default_loss_weight == 10.0
    assert catalogue.names() == ("energy", "magforces")


def test_the_bare_string_form_and_the_mapping_form_agree(tmp_path):
    shorthand = catalogue_from(
        """
observables:
  - name: energy
    irreps: "0e"
    per_atom: false
    units: "eV"
    normalization: "std"
    derivatives: [pos]
""",
        tmp_path,
    )
    assert shorthand.observable("energy").derivatives == (DerivativeRequest(wrt="pos"),)


# ---------------------------------------------------------------------------
# Catalogue-level validation: the errors that live between rows
# ---------------------------------------------------------------------------


def test_a_derivative_against_an_undeclared_input_is_an_error(tmp_path):
    with pytest.raises(ValidationError) as caught:
        catalogue_from(
            """
observables:
  - name: energy
    irreps: "0e"
    per_atom: false
    units: "eV"
    normalization: "std"
    derivatives: [elec_temp]
""",
            tmp_path,
        )
    message = str(caught.value)
    assert "energy" in message
    assert "elec_temp" in message
    assert "['cell', 'pos']" in message


def test_a_derived_name_may_not_collide_with_a_declared_observable(tmp_path):
    with pytest.raises(ValidationError) as caught:
        catalogue_from(
            """
observables:
  - name: forces
    irreps: "1o"
    per_atom: true
    units: "eV/Å"
    normalization: "rms"
  - name: energy
    irreps: "0e"
    per_atom: false
    units: "eV"
    normalization: "std"
    derivatives: [pos]
""",
            tmp_path,
        )
    assert "'forces'" in str(caught.value)


def test_a_name_declared_twice_is_an_error(tmp_path):
    with pytest.raises(ValidationError) as caught:
        catalogue_from(
            """
observables:
  - name: energy
    irreps: "0e"
    per_atom: false
    units: "eV"
    normalization: "std"
  - name: energy
    irreps: "0e"
    per_atom: false
    units: "eV"
    normalization: "none"
""",
            tmp_path,
        )
    assert "declared twice" in str(caught.value)


def test_asking_for_the_same_derivative_twice_is_an_error(tmp_path):
    with pytest.raises(ValidationError):
        catalogue_from(
            """
observables:
  - name: energy
    irreps: "0e"
    per_atom: false
    units: "eV"
    normalization: "std"
    derivatives: [pos, pos]
""",
            tmp_path,
        )


def test_an_unknown_observable_or_input_says_what_is_declared():
    catalogue = load_default_catalogue()
    with pytest.raises(KeyError) as caught:
        catalogue.observable("dipole")
    assert "['energy']" in str(caught.value)
    with pytest.raises(KeyError) as caught:
        catalogue.input("magmom")
    assert "['cell', 'pos']" in str(caught.value)


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
                normalization="rms",
            )
        ],
    )
    assert catalogue.names() == ("dipole",)
    derived = catalogue.derivative("dipole", "pos")
    assert derived.name == "d_dipole_d_pos"
    # The differentiated quantity is not a scalar, so the gradient's irreps are
    # a tensor product this package deliberately does not compute.
    assert derived.irreps is None
