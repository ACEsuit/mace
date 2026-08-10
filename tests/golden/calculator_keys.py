"""The spellings this repository's ASE calculators use for their results.

The golden harness is deliberately framework-agnostic: it knows what a
channel *is* -- its shape and its unit -- but not what any particular
implementation calls it. That knowledge lives here, and it is not a detail.
The model forwards and the calculator that wraps them do not agree on four
names, and the registry could only ever be keyed on one of the two. Measured
against ``mace/calculators/mace.py`` on this tree:

    calculator writes          model forward / registry calls it
    -------------------        ---------------------------------
    LES_alphas                 latent_alphas
    LES_kappas                 latent_kappas
    bec                        BEC
    MACE_magmoms               equilibrated_magmom

Before these aliases existed, a LES or magnetic golden taken through the
calculator recorded energy, forces and stress, dropped everything else
without a word, and committed a reference that claimed to pin the family.
The reference passed and pinned nothing. That is why the harness now refuses
an unrecognised key outright, and why every spelling either resolves to a
channel here or appears on the allowlist below with its reason.

This module is imported by ``tests/golden/__init__.py`` for its side effect,
so anything that can import the harness already has these registrations.
"""

from __future__ import annotations

from tests.golden import harness

# ---------------------------------------------------------------------------
# Alternative spellings
#
# Each of these is written by MACECalculator or MagneticMACECalculator under
# a name the model's own forward does not use.
# ---------------------------------------------------------------------------

#: mace/calculators/mace.py: results["LES_alphas"] <- ret_tensors["latent_alphas"]
harness.register_alias("LES_alphas", "latent_alphas")

#: mace/calculators/mace.py: results["LES_kappas"] <- ret_tensors["latent_kappas"]
harness.register_alias("LES_kappas", "latent_kappas")

#: mace/calculators/mace.py: results["bec"] <- ret_tensors["BEC"]
harness.register_alias("bec", "BEC")

#: mace/calculators/mace.py: MagneticMACECalculator writes the relaxed moments
#: as results["MACE_magmoms"]; the model returns them as "equilibrated_magmom".
harness.register_alias("MACE_magmoms", "equilibrated_magmom")


# ---------------------------------------------------------------------------
# The allowlist
#
# Committee statistics, and only committee statistics. Two properties put
# them outside the schema rather than merely unpinned:
#
#   * they are not predictions of a model, they are the spread of an
#     ensemble of models, so their value depends on how many checkpoints the
#     caller happened to pass and which ones;
#   * the "_comm" members carry a leading axis of committee size, which no
#     channel kind expresses, and "stress_var" is not even self-consistent
#     across the two calculators -- MACECalculator leaves it 3x3 (the Voigt
#     conversion below it touches only results["stress"]) while
#     MagneticMACECalculator converts it to Voigt-6. One kind cannot describe
#     both, and picking one would silently mis-shape the other.
#
# A committee golden that wants these pinned should add a kind keyed on the
# committee size and reconcile the stress_var layout first; the allowlist is
# the record that nobody has, not an oversight.
# ---------------------------------------------------------------------------

_COMMITTEE_SPREAD = {
    "energy_comm": "per-member energies of a committee; leading axis is the "
    "committee size, not a property of any one model",
    "energy_var": "variance of a committee's energies; an ensemble statistic, "
    "not a prediction, and it changes with the committee's membership",
    "forces_comm": "per-member forces of a committee; leading axis is the "
    "committee size",
    "forces_var": "variance of a committee's forces; an ensemble statistic",
    "stress_comm": "per-member stresses of a committee; leading axis is the "
    "committee size",
    "stress_var": "variance of a committee's stresses; an ensemble statistic, "
    "and 3x3 from MACECalculator against Voigt-6 from "
    "MagneticMACECalculator, so no single kind fits it",
    "dipole_comm": "per-member dipoles of a committee; leading axis is the "
    "committee size",
    "dipole_var": "variance of a committee's dipoles; an ensemble statistic",
}

for _key, _reason in _COMMITTEE_SPREAD.items():
    harness.ignore_key(_key, _reason)


# ---------------------------------------------------------------------------
# A divergence worth stating rather than encoding
#
# MACECalculator writes results["energies"] as the per-atom energies before
# the E0 subtraction, which is the ase meaning of the property and what the
# harness declares. MagneticMACECalculator reuses the same name for a
# committee's *total* energies, shape (n_models,). The two cannot both be
# right and the harness pins the ase meaning, so a magnetic committee
# snapshot fails on the shape and says so. That is the intended outcome: the
# collision is in the calculator, and a harness that quietly accepted either
# shape would be hiding it.
# ---------------------------------------------------------------------------
