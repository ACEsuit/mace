"""Export edge gradients use the frozen coupling's positive derivative sign."""

import pytest
import torch

from tests.parity.test_energy_outputs_parity import (
    DEVICES,
    close,
    copy_graph,
    models_and_graph,
)
from tests.parity.test_energy_outputs_parity import (
    checkpoints as checkpoints,
)

pytestmark = pytest.mark.usefixtures("isolated", "fp64")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["water_cluster", "triclinic_bulk"])
def test_export_edge_force_matches_positive_finite_difference(
    checkpoints, device, name
):
    _, engine, graph = models_and_graph(checkpoints, (name,), torch.float64, device)
    result = engine(graph, compute=("forces", "edge_forces"))
    edge_forces = result.extras["edge_forces"]
    index = int(edge_forces.abs().argmax())
    edge, axis = divmod(index, 3)
    step = 1e-4
    energies = []
    for sign in (-1, 1):
        shifted = copy_graph(graph)
        shifted["shifts"][edge, axis] += sign * step
        energies.append(engine(shifted, compute=()).total_energy.detach().sum())
    derivative = (energies[1] - energies[0]) / (2 * step)
    assert abs(float(derivative)) > 1e-4
    close(edge_forces[edge, axis], derivative, torch.float64, device)
    sender, receiver = graph["edge_index"]
    assembled = torch.zeros_like(result.forces)
    assembled.index_add_(0, sender, edge_forces)
    assembled.index_add_(0, receiver, -edge_forces)
    close(assembled, result.forces, torch.float64, device)


@pytest.mark.polar
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["neutral molecule", "crystal", "slab"])
def test_polar_export_edges_preserve_values_and_training_derivatives(device, name):
    from mace_torch.physics import DerivativeEngine

    from tests.parity.test_polar_parity import (
        CASES,
        ENERGY,
        convert,
        legacy_batch,
        legacy_polar,
        v1_graph,
        with_inputs,
    )

    legacy = legacy_polar()
    atoms, (profile, normal), inputs = CASES[name]
    atoms = with_inputs(atoms, *inputs)
    model, config = convert(legacy, profile, normal)
    engine = DerivativeEngine(model, ENERGY).to(device)
    legacy.to(device)
    old = {key: value.to(device) for key, value in legacy_batch(legacy, atoms).items()}
    new = v1_graph(atoms, config["atomic_numbers"], config["cutoff"])
    new = {key: value.to(device) for key, value in new.items()}
    # Edge rows are a topology ordering, unlike the per-atom predictions.
    for key in ("edge_index", "shifts", "unit_shifts"):
        new[key] = old[key].clone()
    old["positions"].requires_grad_(True)
    new["positions"].requires_grad_(True)
    expected = legacy(old, training=True, compute_edge_forces=True)
    actual = engine(new, compute=("forces", "edge_forces"), training=True)
    close(actual.total_energy, expected["energy"], torch.float64, device)
    close(actual.forces, expected["forces"], torch.float64, device)
    edges = actual.extras["edge_forces"]
    close(edges, expected["edge_forces"], torch.float64, device)
    direction = torch.linspace(0.1, 0.9, edges.numel(), device=device).reshape_as(edges)
    actual_gradient = torch.autograd.grad((edges * direction).sum(), new["positions"])[
        0
    ]
    expected_gradient = torch.autograd.grad(
        (expected["edge_forces"] * direction).sum(), old["positions"]
    )[0]
    close(actual_gradient, expected_gradient, torch.float64, device)
