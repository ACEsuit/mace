"""Units and the physics sign conventions, for the whole v1 stack.

**Base units are ase's**: energies in eV, lengths in Angstrom. That is why
``ase.units.eV`` and ``ase.units.Ang`` are both exactly ``1.0`` and are not
re-exported here: a quantity already in the base needs no factor. Everything
below is the factor that takes a quantity *into* that base, so
``energy_in_eV = energy_in_hartree * HARTREE``.

The factors are read from :mod:`ase.units` rather than written out, so a CODATA
revision reaches the two stacks together instead of leaving one of them on the
old numbers. ``test_units.py`` pins the values that revision would move, which
is what keeps "follows ase" from meaning "changes without anyone noticing".

Sign conventions
----------------

**This module is the single source for the prose, and only for the prose.** A
convention restated next to the code that applies it is a second copy that can
disagree with the first, so consumers import the constants below and cite them;
they do not paraphrase them. Every statement is reproduced verbatim from the
characterization suite that pinned it against finite differences, and a test
asserts the two texts are still character-for-character identical:

* forces = -dE/d(positions), in eV/Ang.
* stress = (1/V) dE/d(strain), in eV/Ang^3, with V = |det(cell)|.
* virials = -stress * V = -dE/d(strain), in eV.

The asymmetry is deliberate and is the thing most likely to be normalised away
by a port: of the three, the stress is the only one that is *not* negated, and
the virial is the negative of the very quantity the stress is built from.

**The machine-readable sign is not here.** A ``+1`` or ``-1`` that code can act
on belongs to the declaration of the quantity it governs, beside its name and
its units, and the observables work brings that. Prose cannot be derived from a
number, which is why this half stays, but the two are copies of one fact and a
test has to hold them together the moment both exist in one tree. Until then,
what this module owns is the wording.

The fourth statement is the magnetic family's, taken from the autograd pass
that produces it beside the forces. Legacy fixes no unit for a magnetic moment,
so none is claimed here.
"""

from __future__ import annotations

from ase import units as _ase_units

__all__ = [
    "BOHR",
    "BOLTZMANN",
    "DEBYE",
    "ENERGY_UNIT",
    "FORCE_SIGN_CONVENTION",
    "GIGAPASCAL",
    "HARTREE",
    "KCAL_PER_MOL",
    "KJ_PER_MOL",
    "LENGTH_UNIT",
    "MAGFORCE_SIGN_CONVENTION",
    "RYDBERG",
    "SIGN_CONVENTIONS",
    "STRESS_SIGN_CONVENTION",
    "VIRIAL_SIGN_CONVENTION",
]

#: Name of the base energy unit. Present so an error message or a written
#: artifact can state the unit without hardcoding the string in ten places.
ENERGY_UNIT = "eV"

#: Name of the base length unit.
LENGTH_UNIT = "Angstrom"

#: One Bohr radius, in Angstrom.
BOHR: float = _ase_units.Bohr

#: One Hartree, in eV.
HARTREE: float = _ase_units.Hartree

#: One Rydberg, in eV.
RYDBERG: float = _ase_units.Rydberg

#: One kcal/mol, in eV.
KCAL_PER_MOL: float = _ase_units.kcal / _ase_units.mol

#: One kJ/mol, in eV.
KJ_PER_MOL: float = _ase_units.kJ / _ase_units.mol

#: One GPa, in eV/Angstrom^3. The conversion a stress is reported through.
GIGAPASCAL: float = _ase_units.GPa

#: One Debye, in the base dipole unit (elementary charge times Angstrom).
DEBYE: float = _ase_units.Debye

#: Boltzmann's constant, in eV/K. Reaches the model as the scale of the
#: electronic temperature input.
BOLTZMANN: float = _ase_units.kB

FORCE_SIGN_CONVENTION = "forces = -dE/d(positions), in eV/Ang."

STRESS_SIGN_CONVENTION = (
    "stress = (1/V) dE/d(strain), in eV/Ang^3, with V = |det(cell)|."
)

VIRIAL_SIGN_CONVENTION = "virials = -stress * V = -dE/d(strain), in eV."

#: Not part of the finite-difference trio: a magnetic moment has no finite
#: displacement to differentiate against in the same harness. The sign is the
#: one the autograd pass that produces it beside the forces applies, and it is
#: the same sign for the same reason -- a reported force is the negative
#: gradient of the energy with respect to its conjugate input.
MAGFORCE_SIGN_CONVENTION = "magforces = -dE/d(magmom)."

#: Every convention, keyed by the name of the quantity it governs. A consumer
#: that has a quantity name and wants its convention looks here rather than
#: mapping names to constants itself.
SIGN_CONVENTIONS: dict[str, str] = {
    "forces": FORCE_SIGN_CONVENTION,
    "stress": STRESS_SIGN_CONVENTION,
    "virials": VIRIAL_SIGN_CONVENTION,
    "magforces": MAGFORCE_SIGN_CONVENTION,
}
