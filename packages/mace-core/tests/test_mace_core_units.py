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


def test_every_factor_the_module_defines_is_pinned():
    """A factor added to `units` without a line in PINNED fails here, whether
    or not it is also added to `__all__`, rather than going unchecked by the
    two tests that iterate over PINNED."""
    defined = {
        name
        for name, value in vars(units).items()
        if not name.startswith("_") and isinstance(value, float)
    }
    assert defined == set(PINNED)
    assert defined <= set(units.__all__)


def test_the_constants_are_ase_s_and_not_a_second_copy():
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
#
# The wording of the three derivative conventions is checked character for
# character against the characterization suite by
# tests/architecture/test_convention_statements.py, and is not restated here.
# ---------------------------------------------------------------------------


def test_the_magnetic_force_carries_the_same_sign_as_a_force():
    assert units.MAGFORCE_SIGN_CONVENTION == "magforces = -dE/d(magmom)."


def test_every_convention_is_filed_under_the_quantity_it_states():
    """Each statement opens with the name of the quantity it defines, so the
    key it is filed under is checked against the statement rather than
    against a second copy of the mapping. A convention constant left out of
    the mapping fails too."""
    for quantity, statement in units.SIGN_CONVENTIONS.items():
        assert statement.startswith(f"{quantity} = "), (quantity, statement)
    constants = {
        getattr(units, name)
        for name in vars(units)
        if name.endswith("_SIGN_CONVENTION")
    }
    assert set(units.SIGN_CONVENTIONS.values()) == constants
    assert len(units.SIGN_CONVENTIONS) == len(constants)
