"""The v1 calculator against the frozen tree's, on the committed anchor.

The anchor is converted into a v1 checkpoint with its record, the v1
calculator reads it back, and both calculators evaluate the same structures in
one process at fp64: the anchor's own training structures, which are periodic,
and one of them as a molecule. The results have to agree under the names the
frozen tree gives them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from tests.parity.test_anchor_as_foundation import (
    ANCHOR,
    TRAIN_SET,
    write_anchor_checkpoint,
)
from tests.parity.test_fm00_training_step import load_anchor

#: fp64 agreement between two implementations of the same arithmetic, summed in
#: different orders.
RTOL, ATOL = 1e-10, 1e-10


def structures() -> list[Atoms]:
    frames = [
        atoms
        for atoms in read(TRAIN_SET, ":")
        if atoms.info.get("config_type") != "IsolatedAtom"
    ][:3]
    molecule = frames[0].copy()
    molecule.set_pbc(False)
    molecule.set_cell(np.zeros((3, 3)))
    return [*frames, molecule]


@pytest.fixture(name="calculators")
def fixture_calculators(fp64, tmp_path):
    from mace_torch.calculators import MACECalculator

    from mace.calculators.mace import MACECalculator as LegacyCalculator

    legacy = load_anchor(ANCHOR)
    checkpoint = write_anchor_checkpoint(legacy, tmp_path)
    return (
        LegacyCalculator(models=legacy, device="cpu", default_dtype="float64"),
        MACECalculator(model_paths=Path(checkpoint)),
    )


@pytest.mark.parametrize("index", range(4))
def test_the_two_calculators_report_the_same_results(calculators, index):
    legacy, v1 = calculators
    atoms = structures()[index]
    results = {}
    for name, calculator in (("legacy", legacy), ("v1", v1)):
        copy = atoms.copy()
        copy.calc = calculator
        copy.get_potential_energy()
        results[name] = dict(calculator.results)
    shared = sorted(set(results["legacy"]) & set(results["v1"]))
    assert {
        "energy",
        "free_energy",
        "forces",
        "stress",
        "energies",
        "node_energy",
    } <= set(shared)
    for key in shared:
        np.testing.assert_allclose(
            results["v1"][key],
            results["legacy"][key],
            rtol=RTOL,
            atol=ATOL,
            err_msg=key,
        )


def test_the_descriptors_agree(calculators):
    legacy, v1 = calculators
    for atoms in structures():
        for invariants_only in (True, False):
            np.testing.assert_allclose(
                v1.get_descriptors(atoms, invariants_only=invariants_only),
                legacy.get_descriptors(atoms, invariants_only=invariants_only),
                rtol=RTOL,
                atol=ATOL,
            )


def test_the_first_layer_s_descriptors_agree(calculators):
    legacy, v1 = calculators
    atoms = structures()[0]
    np.testing.assert_allclose(
        v1.get_descriptors(atoms, num_layers=1),
        legacy.get_descriptors(atoms, num_layers=1),
        rtol=RTOL,
        atol=ATOL,
    )


def test_the_hessians_agree(calculators):
    legacy, v1 = calculators
    for atoms in structures()[:2]:
        np.testing.assert_allclose(
            v1.get_hessian(atoms), legacy.get_hessian(atoms), rtol=1e-8, atol=1e-10
        )
