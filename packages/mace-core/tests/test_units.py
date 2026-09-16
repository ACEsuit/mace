"""Unit constants and the sign-convention statements.

The constants are read from ase, so what is worth asserting is not that the
arithmetic is right but that the values are the ones the rest of the stack was
built against. A CODATA revision in ase is allowed to move them; it is not
allowed to move them quietly.

The verbatim match between the convention statements and the characterization
suite that pinned them is checked in ``tests/architecture``, which can see both
trees. This file asserts what the statements say.
"""

from __future__ import annotations

import ase.units
import pytest
from mace_core import units

#: The values in force today. A mismatch means ase changed a physical constant,
#: which is a fact about every number the stack produces and belongs in a
#: reviewed commit rather than in whatever ase release pip happened to resolve.
PINNED = {
    "BOHR": 0.5291772105638411,
    "HARTREE": 27.211386024367243,
    "RYDBERG": 13.605693012183622,
    "KCAL_PER_MOL": 0.04336410390059322,
    "KJ_PER_MOL": 0.010364269574711572,
    "GIGAPASCAL": 0.006241509125883258,
    "DEBYE": 0.20819433442462576,
    "BOLTZMANN": 8.617330337217213e-05,
}


@pytest.mark.parametrize(("name", "value"), sorted(PINNED.items()))
def test_the_constants_are_the_values_the_stack_was_built_against(name, value):
    assert getattr(units, name) == pytest.approx(value, rel=1e-12)


def test_the_constants_are_ase_s_and_not_a_second_copy():
    """Compared as a whole mapping, so a constant added above without a line
    here fails rather than going unchecked."""
    assert {name: getattr(units, name) for name in PINNED} == {
        "BOHR": ase.units.Bohr,
        "HARTREE": ase.units.Hartree,
        "RYDBERG": ase.units.Rydberg,
        "KCAL_PER_MOL": ase.units.kcal / ase.units.mol,
        "KJ_PER_MOL": ase.units.kJ / ase.units.mol,
        "GIGAPASCAL": ase.units.GPa,
        "DEBYE": ase.units.Debye,
        "BOLTZMANN": ase.units.kB,
    }


def test_the_base_units_are_ase_s_base_units():
    """eV and Angstrom are exactly 1.0 there, which is why no factor for them
    is defined: a quantity already in the base needs no conversion."""
    assert ase.units.eV == 1.0
    assert ase.units.Ang == 1.0
    assert (units.ENERGY_UNIT, units.LENGTH_UNIT) == ("eV", "Angstrom")


# ---------------------------------------------------------------------------
# The sign conventions
# ---------------------------------------------------------------------------


def test_the_three_derivative_conventions_say_what_they_must():
    assert units.FORCE_SIGN_CONVENTION == "forces = -dE/d(positions), in eV/Ang."
    assert units.STRESS_SIGN_CONVENTION == (
        "stress = (1/V) dE/d(strain), in eV/Ang^3, with V = |det(cell)|."
    )
    assert units.VIRIAL_SIGN_CONVENTION == (
        "virials = -stress * V = -dE/d(strain), in eV."
    )


def test_the_magnetic_force_carries_the_same_sign_as_a_force():
    assert units.MAGFORCE_SIGN_CONVENTION == "magforces = -dE/d(magmom)."


def test_the_stress_is_the_one_quantity_that_is_not_negated():
    """The asymmetry is the thing a port normalises away: forces and virials
    are negated gradients, the stress is not, and the virial is the negative of
    the very quantity the stress is built from."""
    assert "-dE/d(positions)" in units.FORCE_SIGN_CONVENTION
    assert "-dE/d(strain)" in units.VIRIAL_SIGN_CONVENTION
    assert "-dE" not in units.STRESS_SIGN_CONVENTION
    assert "(1/V) dE/d(strain)" in units.STRESS_SIGN_CONVENTION


def test_every_convention_is_reachable_by_the_name_of_its_quantity():
    assert units.SIGN_CONVENTIONS == {
        "forces": units.FORCE_SIGN_CONVENTION,
        "stress": units.STRESS_SIGN_CONVENTION,
        "virials": units.VIRIAL_SIGN_CONVENTION,
        "magforces": units.MAGFORCE_SIGN_CONVENTION,
    }
