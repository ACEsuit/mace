"""The retired potential request never produced a value in the frozen model."""

import pytest
import torch
from mace_torch.physics import DerivativeEngine

from tests.golden.harness import FP64_CPU_REFERENCE
from tests.parity.test_polar_parity import (
    CASES,
    ENERGY,
    convert,
    legacy_batch,
    legacy_polar,
    v1_graph,
    with_inputs,
)

pytestmark = [pytest.mark.polar, pytest.mark.usefixtures("fp64", "isolated")]


@pytest.mark.parametrize("name", sorted(CASES))
def test_retired_potential_output_remains_absent_with_matching_predictions(name):
    atoms, (profile, normal), inputs = CASES[name]
    atoms = with_inputs(atoms, *inputs)
    legacy = legacy_polar()
    model, config = convert(legacy, profile, normal)
    actual = DerivativeEngine(model, ENERGY)(
        v1_graph(atoms, config["atomic_numbers"], config["cutoff"]),
        compute=("forces",),
    )
    assert actual.get("electrostatic_potentials") is None
    assert "electrostatic_potentials" not in actual.names()
    row = FP64_CPU_REFERENCE
    for enabled in (False, True):
        legacy.return_electrostatic_potentials = enabled
        expected = legacy(legacy_batch(legacy, atoms), compute_force=True)
        assert expected["electrostatic_potentials"] is None
        for key in ("energy", "forces", "dipole"):
            torch.testing.assert_close(
                actual.get(key), expected[key], atol=row.atol, rtol=row.rtol
            )
