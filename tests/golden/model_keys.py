"""The spellings the model ``forward`` methods use for their outputs.

The sibling of ``calculator_keys``, for the other door. It exists because the
calculator is not the whole surface and never was: ``edge_forces`` and
``hessian`` are returned by every energy model and appear in no calculator's
``results`` at all, so a golden that wants either has to go through
``golden_outputs`` -- and then it meets the forward's full key set, not the
calculator's subset. A schema derived from ``mace/calculators/mace.py`` alone
resolves every calculator key and leaves thirteen forward keys resolving to
nothing, which is a golden that fails on the first model-route snapshot
rather than a golden that pins less than it claims. The registry now covers
every surface, and ``tests/golden/surface_scan.py`` derives the expectation
from the package so the next divergence fails in the guard instead.

The key sets, read out of the ``forward`` return dicts on this tree
(``mace/modules/models.py`` and ``mace/modules/extensions.py``):

    MACE                     energy, node_energy, contributions, forces,
                             edge_forces, virials, stress, atomic_virials,
                             atomic_stresses, displacement, hessian,
                             node_feats
    ScaleShiftMACE           the same, with interaction_energy replacing
                             contributions
    AtomicDipolesMACE        dipole, atomic_dipoles
    AtomicDielectricMACE     + charges, polarizability, polarizability_sh,
                             dmu_dr, dalpha_dr
    EnergyDipolesMACE        the energy set plus the dipole set
    MACELES                  + les_energy, latent_{charges,dipoles,alphas,
                             kappas,quads}, BEC
    PolarMACE                + density_coefficients, spin_density,
                             charges_history, fermi_level, external_field,
                             spins, total_charge, electrostatic_energy,
                             electron_energy, electrostatic_potentials,
                             spin_charge_density, fukui_functions
    MagneticScaleShiftMACE   + magforces
    MagneticSCFMACE          + equilibrated_magmom, scf_energy_history,
                             scf_steps

...and two more classes in two more files, which is the second half of this
file's history. The guard read ``models.py`` and ``extensions.py`` and called
that the model surface, but four files in ``mace/`` define a ``forward`` that
returns an output dict:

    mace/calculators/lammps_mace.py::LAMMPS_MACE
                             total_energy_local, node_energy, forces, virials
    mace/calculators/mace_torchsim.py::MaceTorchSimModel
                             energy, forces, stress (plus whatever the wrapped
                             model returned, forwarded verbatim)

Both are deployment wrappers rather than models, which is presumably why they
were not looked at -- and both are exactly what a deployment golden evaluates.
The guard now discovers the files instead of listing them, so a fifth writer
is found rather than waited for, and ``total_energy_local`` is a declared
channel because of it: it was the one key in the whole set that resolved to
nothing.

Almost everything else is a declared channel under exactly the model's name,
because the registry was keyed on the model's vocabulary in the first place.
What is left is this file: one allowlist entry, and the notes below.
"""

from __future__ import annotations

from tests.golden import harness

# ---------------------------------------------------------------------------
# The one forward key that is deliberately not pinned
#
# `displacement` is not an observable. It is the symmetric strain handle the
# stress is differentiated against: `get_symmetric_displacement` creates it as
# `torch.zeros((num_graphs, 3, 3)) + positions.sum() * 0.0`
# (mace/modules/utils.py:100-106) purely to attach it to the autograd graph,
# and nothing ever writes to it, so the value a forward returns is identically
# zero on every structure. Measured on the committed `tiny_scaleshift` anchor
# over all six fixtures: max |displacement| = 0.0, and PolarMACE's is the same
# (1, 3, 3) of zeros.
#
# Pinning a constant records nine zeros per fixture and catches nothing that
# the stress channel does not already catch, since a change to the strain
# convention shows up in the stress it produces. Allowlisting it says that on
# the record, with a date, instead of leaving it to be rediscovered.
# ---------------------------------------------------------------------------

harness.ignore_key(
    "displacement",
    "the symmetric strain handle the stress is differentiated against, not an "
    "output: mace/modules/utils.py:100-106 creates it as zeros and nothing "
    "writes to it, so its value is identically zero on every structure "
    "(measured 0.0 on all six fixtures with the tiny_scaleshift anchor). A "
    "change to the strain convention is caught by the stress channel it "
    "produces, not by nine pinned zeros.",
)


# ---------------------------------------------------------------------------
# Four things that are true of this surface and are not registrations
#
# 1. `virials` on this surface is the graph-level virial, shape (n_graphs, 3,
#    3), which is exactly what the `virials` channel declares. No alias is
#    needed here; the alias is on the calculator surface, which uses the same
#    word for the per-atom virial. See calculator_keys.py.
#
# 2. A forward returns its whole key set on every call and fills in `None` for
#    anything the call did not request -- `hessian` unless
#    `compute_hessian=True`, `atomic_stresses` unless
#    `compute_atomic_stresses=True`, and `electrostatic_potentials` always, on
#    this tree, since `esps` is only ever assigned `None`
#    (mace/modules/extensions.py:1154 is its single assignment). The harness
#    resolves the key and then skips the missing value, so an unknown key
#    still raises while a known-but-absent one leaves the channel out; a
#    reference that pins it then fails with "channel vanished".
#
# 3. `total_energy_local` is a channel and not an alias for `energy`. The
#    LAMMPS wrapper masks the site energies by `local_or_ghost` before summing
#    (mace/calculators/lammps_mace.py:71-74), so it is the part of the energy
#    this domain owns; the full energy is only recovered by adding the domains
#    up. Aliasing the two would have made a single-domain golden pass and a
#    decomposed one compare a part against a whole.
#
# 4. Graph-level channels are declared per graph -- `energy` is a `()` scalar,
#    not `(n_graphs,)`. A forward returns the batched form, so a
#    `golden_outputs` hook over single-structure fixtures has to index the one
#    graph out. That is the hook's job and not the harness's: squeezing a
#    leading axis of one inside the schema would silently accept a two-graph
#    batch as a one-graph result.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The measured extents of the position_gradient kind
#
# `harness.POSITION_GRADIENT` pins the atom axis and leaves the leading one
# free, because that extent is a property of the differentiated quantity
# rather than of the structure, and says the measurement belongs here. Taken
# on the published MACE-MDP model (AtomicDielectricMACE) over the molecular
# fixtures, with `compute_dielectric_derivatives=True`:
#
#     dmu_dr      (3, n_atoms, 3)   d(dipole)/dr, 3 dipole components
#     dalpha_dr   (9, n_atoms, 3)   d(polarizability)/dr, the 3x3 flattened
#                                   by `total_polarizability.flatten(-2)`
#                                   (mace/modules/models.py:1160)
#
# Neither appears in any calculator's `results`: `results_map`
# (mace/calculators/mace.py:719-738) has no entry for them, so the model
# surface is the only door. That is why tests/golden/test_mdp_foundation.py
# takes its reference through the forward and asserts the calculator's four
# shared channels against the same file, rather than the other way round.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# One channel, two families, and they do not agree on its unit
#
# The `dipole` channel is declared "Debye", which is what MACECalculator's own
# docstring promises (mace/calculators/mace.py:100). Both dipole families add
# a fixed-charge baseline to their readout sum, and the two baselines are not
# in the same unit:
#
#     AtomicDipolesMACE      compute_fixed_charge_dipole       divides by
#                            (mace/modules/utils.py:622)       1e-11 / c / e
#                                                              -> Debye
#     AtomicDielectricMACE   compute_fixed_charge_dipole_polar the same
#                            (mace/modules/utils.py:634-636)   division, but
#                                                              commented out
#                                                              -> e*Ang
#
# The ratio is exactly the e*Ang -> Debye factor, 4.8032. PolarMACE is a third
# case and sits with the second: its `dipole` is `compute_total_charge_dipole_
# permuted` (mace/modules/utils.py:640-648), a bare sum of charge times
# position with no conversion at all.
#
# Which of these is right is a physics question for the electrostatics work,
# not something a schema can decide, and declaring a channel per family would
# only mean the three could never be compared. So one channel holds them all,
# the discrepancy between the two baselines is pinned as a number by
# tests/golden/test_tiny_dipoles.py::test_the_two_fixed_charge_baselines_do_not_share_a_unit,
# and a rewrite that unifies the two functions fails there instead of silently
# rescaling one of the committed references by 4.8.
# ---------------------------------------------------------------------------
