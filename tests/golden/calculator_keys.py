"""The spellings this repository's ASE calculators use for their results.

The golden harness is deliberately framework-agnostic: it knows what a
channel *is* -- its shape and its unit -- but not what any particular
implementation calls it. That knowledge lives here (for the calculator
surface) and in ``model_keys`` (for the forward surface), and it is not a
detail. Measured against ``mace/calculators/mace.py`` on this tree, the two
surfaces disagree in three different ways, and each needs a different repair:

    calculator writes    model forward calls it    what differs
    -----------------    ----------------------    -----------------------
    LES_alphas           latent_alphas             the name only
    LES_kappas           latent_kappas             the name only
    bec                  BEC                       the name only
    MACE_magmoms         equilibrated_magmom       the name only
    stresses             atomic_stresses           name *and* layout
    virials              atomic_virials            name, and `virials`
                                                   already means something
                                                   else on the other surface

Before these registrations existed, a LES or magnetic golden taken through
the calculator recorded energy, forces and stress, dropped everything else
without a word, and committed a reference that claimed to pin the family.
The reference passed and pinned nothing. That is why the harness now refuses
an unrecognised key outright, and why every spelling either resolves to a
channel here or appears on the allowlist below with its reason.

This module is imported by ``tests/golden/__init__.py`` for its side effect,
so anything that can import the harness already has these registrations.
"""

from __future__ import annotations

from tests.golden import harness

CALC = harness.SURFACE_CALCULATOR

# ---------------------------------------------------------------------------
# Plain renames
#
# Each of these is written by MACECalculator or MagneticMACECalculator under
# a name the model's own forward does not use. No layout changes, so a
# spelling-only alias is the whole repair, and it is registered for both
# surfaces because nothing else uses these words.
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
# One quantity, two representations
#
# `results_map` (mace/calculators/mace.py:719-738) renames the two per-atom
# decompositions, and for the stress it also *reshapes*: results["stresses"]
# is put through `full_3x3_to_voigt_6_stress` per atom (:791-797) while the
# model's `atomic_stresses` stays (n_atoms, 3, 3). results["virials"] is
# renamed but not reshaped.
#
# Neither of the two obvious repairs is usable:
#
#   * a plain alias onto one channel is a shape failure -- (n_atoms, 6)
#     against (n_atoms, 3, 3);
#   * a channel each is worse than the bug it replaces. Both channels would
#     hold the same physics, and a golden taken through the model route and
#     one taken through the calculator route would then never compare. That
#     is the silent split this registry exists to prevent, arrived at
#     deliberately instead of by accident.
#
# So the representation is canonicalised on ingest: one channel, holding the
# full 3x3, and the calculator's Voigt-6 is expanded as it arrives. The
# direction matters, and it was chosen by measurement rather than by taste.
#
# Voigt-6 can only carry a symmetric tensor: `full_3x3_to_voigt_6_stress`
# averages each off-diagonal pair, so it discards the antisymmetric part.
# Choosing the 3x3 as canonical is therefore only safe if MACE's per-atom
# tensors are symmetric to begin with -- and they are, by construction:
# `get_atomic_virials_stresses` symmetrises explicitly at
# mace/modules/utils.py:382 (`atom_virial + atom_virial.transpose(-1, -2)) / 2`)
# before dividing by the cell volume, so both the virial and the stress
# inherit it.
#
# Measured on the committed `tiny_scaleshift` anchor in float64 over all six
# fixtures (dimer_short, isolated_atom, slab_vacuum, slab_zero_vacuum,
# triclinic_bulk, water_cluster), with the process default dtype set to
# float64 so the graph is not built through float32:
#
#     max |A - A^T| over every per-atom stress and virial   0.0
#     max |voigt_6_to_full_3x3(to_voigt(A)) - A|            0.0
#     max |calculator route - model route|, atomic_stresses 0.0
#     max |calculator route - model route|, atomic_virials  0.0
#
# Bit-exact, not "within tolerance". The conversion is lossless here and the
# two surfaces land on identical numbers, which is the property the single
# channel depends on. If a future per-atom decomposition drops the
# symmetrisation at utils.py:382, this stops being true silently -- so
# tests/golden/test_harness.py re-measures the asymmetry rather than trusting
# this comment.
# ---------------------------------------------------------------------------

harness.register_alias(
    "stresses",
    "atomic_stresses",
    surface=CALC,
    convert=harness.voigt_6_to_full_3x3,
    note=(
        "ase stores a per-atom stress in Voigt-6; the channel is the model's "
        "full 3x3. Lossless because the per-atom virial is symmetrised at "
        "mace/modules/utils.py:382 -- measured 0.0 on every fixture."
    ),
)

#: The graph-level stress is the same story one level up: ase's `stress`
#: property is Voigt-6 by convention (and MACECalculator converts to it at
#: mace/calculators/mace.py:790), while the channel is the model's 3x3. Same
#: conversion, same reason, and registering it here rather than special-casing
#: it inside the harness keeps one representation change in one place.
harness.register_alias(
    "stress",
    "stress",
    surface=CALC,
    convert=harness.voigt_6_to_full_3x3,
    note=(
        "ase's stress property is Voigt-6; the channel is the 3x3 the model "
        "returns. Same channel, one layout."
    ),
)

#: The collision, not a rename. `virials` is the *graph* virial in every
#: model forward (mace/modules/models.py:433) and the *per-atom* virial in the
#: calculator's results (mace/calculators/mace.py:729-733), and the calculator
#: never exposes the graph one at all. A single spelling->channel map has to
#: pick one and mis-shape the other; scoping the alias to this surface is what
#: lets both be true.
harness.register_alias(
    "virials",
    "atomic_virials",
    surface=CALC,
    note=(
        "MACECalculator's results['virials'] is the per-atom virial "
        "(mace/calculators/mace.py:729-733). The model's forward uses the same "
        "word for the graph-level virial, which is what the `virials` channel "
        "is; the calculator has no key for that one."
    ),
)


# ---------------------------------------------------------------------------
# Where the inputs are read from
#
# Both of these are constructor arguments, so the static default in the
# harness is a guess about a specific instance. A structure whose moments live
# under a non-default key matched nothing, was recorded as no magmom at all,
# and then compared clean -- an input the model reads and the reference does
# not record is the same silence as an output nobody pins.
# ---------------------------------------------------------------------------

#: MagneticMACECalculator(magmom_key=...) -> self.magmom_key
#: (mace/calculators/mace.py:1153), fed to the keyspec at :1304.
harness.register_input_probe("magmom", attribute="magmom_key", store="arrays")

#: MACECalculator(charges_key="Qs") / MagneticMACECalculator(charges_key=...)
#: -> self.charges_key (mace/calculators/mace.py:298, :1152).
harness.register_input_probe("input_charges", attribute="charges_key", store="arrays")


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
# Two divergences worth stating rather than encoding
#
# 1. MACECalculator writes results["energies"] as the per-atom energies before
#    the E0 subtraction, which is the ase meaning of the property and what the
#    harness declares. MagneticMACECalculator reuses the same name for a
#    committee's *total* energies, shape (n_models,). The two cannot both be
#    right and the harness pins the ase meaning, so a magnetic committee
#    snapshot fails on the shape and says so. That is the intended outcome:
#    the collision is in the calculator, and a harness that quietly accepted
#    either shape would be hiding it.
#
# 2. results["virials"] is scaled by `energy_units_to_eV / length_units_to_A**3`
#    (mace/calculators/mace.py:729-733) -- a *stress* conversion applied to a
#    virial, which is an energy. It is the identity under the default units, so
#    nothing in this repository sees it, and the harness declares the channel
#    "eV" because that is what the quantity is. A calculator constructed with
#    non-default units would hand this schema a mislabelled number; that is a
#    defect in the calculator, not something to encode here.
# ---------------------------------------------------------------------------
