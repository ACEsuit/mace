"""Recover the historical final-density snapshot from the canonical output."""

import pytest
import torch
from mace_torch.physics import DerivativeEngine

from tests.golden.harness import FP64_ACCELERATED_BACKEND, FP64_CPU_REFERENCE
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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize("name", sorted(CASES))
def test_final_density_recovers_the_complete_historical_snapshot(device, steps, name):
    atoms, (profile, normal), inputs = CASES[name]
    atoms = with_inputs(atoms, *inputs)
    legacy = legacy_polar(num_recursion_steps=steps)
    model, config = convert(legacy, profile, normal)
    engine = DerivativeEngine(model, ENERGY).to(device)
    legacy.to(device)
    old = legacy_batch(legacy, atoms)
    old = {key: value.to(device) for key, value in old.items()}
    new = v1_graph(atoms, config["atomic_numbers"], config["cutoff"])
    new = {key: value.to(device) for key, value in new.items()}
    expected = legacy(old, training=True)
    actual = engine(new, compute=("forces",), training=True)
    history = expected["charges_history"]
    density = actual.extras["spin_charge_density"]
    recovered = density.detach().unsqueeze(-1)
    assert history.shape == (len(atoms), 2, 4, 1)
    assert not history.requires_grad and not recovered.requires_grad
    assert density.requires_grad
    assert torch.equal(history[..., 0], expected["spin_charge_density"].detach())
    assert actual.get("charges_history") is None
    row = FP64_ACCELERATED_BACKEND if device == "cuda" else FP64_CPU_REFERENCE
    for value, reference in (
        (recovered, history),
        (actual.total_energy, expected["energy"]),
        (actual.forces, expected["forces"]),
    ):
        torch.testing.assert_close(value, reference, atol=row.atol, rtol=row.rtol)
