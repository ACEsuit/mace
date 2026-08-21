"""The committed training regression set is what it claims to be.

``datasets/regression_train.xyz`` is the dataset the legacy-versus-rewrite
training comparison runs on, and it is committed, so nothing regenerates it
between runs -- which means nothing would notice if it drifted, went stale
against its own recipe, or lost the families it exists to cover. These tests
are that notice. They are cheap, need no model, and run in the required unit
suite rather than in the slow workflow directory.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.io import read as ase_read

from tests.golden import harness
from tests.golden.make_regression_set import OUTPUT, label, self_check


@pytest.fixture(name="configs", scope="module")
def fixture_configs():
    assert OUTPUT.exists(), (
        f"{OUTPUT} is missing; regenerate it with "
        f"`python tests/golden/regenerate.py --target regression-set "
        f"--i-know-what-i-am-doing`"
    )
    return ase_read(OUTPUT, index=":")


def test_the_set_covers_the_four_families_the_comparison_needs(configs):
    """Surfaces, molecules, bulk and isolated atoms, each for its own reason.

    A training regression on molecules alone would never exercise the stress,
    and one on bulk alone would never exercise the aperiodic neighbour
    regime; the isolated atoms are what fixes the E0 table, without which
    every energy is offset by a constant nobody can see.
    """
    families = {atoms.info.get("config_type") for atoms in configs}
    assert {"IsolatedAtom", "molecule", "surface", "bulk", "bulk_triclinic"} <= (
        families
    ), sorted(families)

    periodicity = {tuple(np.asarray(atoms.pbc).tolist()) for atoms in configs}
    assert (False, False, False) in periodicity
    assert (True, True, False) in periodicity, "no mixed per-axis pbc"
    assert (True, True, True) in periodicity

    elements = {int(z) for atoms in configs for z in atoms.numbers}
    assert elements == {1, 6, 8}, (
        f"the set uses elements {sorted(elements)}; the committed anchors "
        f"carry the z-table [1, 6, 8], so anything else cannot be fine-tuned "
        f"from them"
    )


def test_every_configuration_carries_both_electrostatic_labels(configs):
    """Partial charges and the dipole, which is what makes it usable for
    electrostatics -- and the charges are neutral per configuration, so the
    dipole is origin-independent and a model has something well defined to
    learn.

    The dipole is under ``REF_dipole`` rather than the package's default
    ``dipole``, and that is not a style choice: ``dipole`` is one of ase's
    recognised calculator properties, so a value written into ``info`` comes
    back from an extxyz round trip in ``atoms.calc.results`` instead, where
    the parser never looks. Asserted here, on the committed file, so the
    workaround stays justified rather than becoming folklore.
    """
    for index, atoms in enumerate(configs):
        assert "REF_charges" in atoms.arrays, index
        assert "REF_dipole" in atoms.info, index
        assert "dipole" not in atoms.info, (
            f"configuration {index} carries a 'dipole' key in info after an "
            f"extxyz round trip; if ase stopped intercepting the name, the "
            f"set can go back to the default key"
        )
        charges = np.asarray(atoms.arrays["REF_charges"], dtype=float)
        assert charges.shape == (len(atoms),), index
        # Neutral to the fp64 row rather than exactly: extxyz writes eight
        # decimals, so a set the recipe made neutral to machine precision
        # comes back with a residual of order 1e-8 -- two orders inside the
        # bound, and a real net charge would be orders outside it.
        assert abs(charges.sum()) < harness.FP64_CPU_REFERENCE.atol, (
            f"configuration {index} carries a net charge of {charges.sum():.3g}"
        )
        assert np.asarray(atoms.info["REF_dipole"], dtype=float) == pytest.approx(
            charges @ atoms.positions, abs=harness.FP64_CPU_REFERENCE.atol
        ), f"configuration {index}: the dipole is not the moment of its charges"


def test_the_labels_still_match_the_recipe_that_produced_them(configs):
    """The committed file against its own generator, configuration by
    configuration.

    This is what catches a dataset that was hand-edited, half-regenerated, or
    written by an older version of the recipe. Comparing at the fp64 row
    rather than bit-for-bit because the labels have been through an extxyz
    round trip.
    """
    tol = harness.FP64_CPU_REFERENCE
    for index, atoms in enumerate(configs):
        relabelled = label(atoms)
        assert relabelled.info["REF_energy"] == pytest.approx(
            atoms.info["REF_energy"], abs=tol.atol
        ), f"configuration {index}: the committed energy is not the recipe's"
        assert np.abs(
            np.asarray(relabelled.arrays["REF_forces"])
            - np.asarray(atoms.arrays["REF_forces"])
        ).max() <= tol.atol, index
        assert ("REF_stress" in relabelled.info) == ("REF_stress" in atoms.info), (
            f"configuration {index} disagrees with the recipe about whether "
            f"it carries a stress"
        )
        if "REF_stress" in atoms.info:
            assert np.abs(
                np.asarray(relabelled.info["REF_stress"])
                - np.asarray(atoms.info["REF_stress"]).ravel()
            ).max() <= tol.atol, index


def test_the_forces_and_stress_are_the_derivatives_of_the_energy(configs):
    """The property that makes the set learnable at all.

    A training set whose forces are not the gradient of its energies has no
    consistent minimum, so a model fitted to both cannot do better than the
    inconsistency -- and "the rewrite trains to a worse error" would then be
    a property of the data. Checked by central differences against the
    recipe, on one configuration of each family (the check is O(N) energy
    evaluations per configuration, so all thirty would be wasteful).
    """
    seen = set()
    checked = 0
    for atoms in configs:
        family = atoms.info.get("config_type")
        if family in seen or len(atoms) == 1:
            continue
        seen.add(family)
        force_error, stress_error = self_check(atoms)
        assert force_error < 1e-7, (
            f"{family}: the labelled forces differ from -dE/dx by "
            f"{force_error:.3e} eV/Ang"
        )
        assert stress_error < 1e-7, (
            f"{family}: the labelled stress differs from (1/V) dE/dstrain by "
            f"{stress_error:.3e} eV/Ang^3"
        )
        checked += 1
    assert checked >= 4, f"only {checked} families were checked"


def test_the_degenerate_slab_carries_no_stress_and_says_so_by_omission(configs):
    """A zero-volume cell has no stress, and the recipe writes none.

    Kept in the set on purpose -- a slab built with no vacuum is a real user
    input and the neighbour list has a patch for exactly it -- so the
    training path has to cope with one configuration missing a property the
    others have. That is a masking case worth having in the data rather than
    only in a unit test.
    """
    degenerate = [
        atoms
        for atoms in configs
        if atoms.info.get("config_type") == "surface_zero_vacuum"
    ]
    assert len(degenerate) == 1, "the zero-vacuum slab left the set"
    atoms = degenerate[0]
    assert abs(np.linalg.det(np.asarray(atoms.cell))) == 0.0
    assert "REF_stress" not in atoms.info
    assert "REF_energy" in atoms.info and "REF_forces" in atoms.arrays


def test_the_set_stays_small_enough_to_train_in_a_test(configs):
    """A committed dataset is a permanent cost in every clone, and this one is
    also run inside the contract suite, so its size is a contract too."""
    assert 20 <= len(configs) <= 60, len(configs)
    assert max(len(atoms) for atoms in configs) <= 12
    size_kb = OUTPUT.stat().st_size / 1000
    assert size_kb < 100, f"{OUTPUT.name} is {size_kb:.0f} kB"
