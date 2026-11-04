"""Energy decomposition and atomic strain outputs against both frozen anchors."""

import pytest
import torch
from ase.io import read
from mace_core.config.precision import PrecisionConfig
from mace_torch.deploy.loader import load_deployed
from mace_torch.physics import prepare_inputs

from tests.golden.anchors import anchor_batch
from tests.golden.harness import FP32, FP64_ACCELERATED_BACKEND, FP64_CPU_REFERENCE
from tests.parity.test_device_workflows_parity import FIXTURES
from tests.parity.test_device_workflows_parity import checkpoints as checkpoints

pytestmark = pytest.mark.usefixtures("isolated")
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def close(actual, expected, dtype, device):
    row = (
        FP32
        if dtype == torch.float32
        else (FP64_ACCELERATED_BACKEND if device == "cuda" else FP64_CPU_REFERENCE)
    )
    torch.testing.assert_close(actual, expected, atol=row.atol, rtol=row.rtol)


def models_and_graph(checkpoints, names, dtype, device):
    source, artifact = checkpoints
    torch.set_default_dtype(dtype)
    # The pickle's nested TorchScript buffers need Module.to as well as the
    # ordinary tensor storage restored by map_location.
    legacy = torch.load(source, weights_only=False, map_location="cpu").to(
        device=device, dtype=dtype
    )
    native = load_deployed(artifact, device=device).with_precision(
        PrecisionConfig(model=str(dtype).removeprefix("torch.")), device=device
    )
    atoms = [read(FIXTURES / f"{name}.xyz") for name in names]
    graph = anchor_batch(legacy, atoms, dtype).to(device).to_dict()
    graph["cell"] = graph["cell"].reshape(-1, 3, 3)
    graph["node_atomic_numbers"] = legacy.atomic_numbers[graph["node_attrs"].argmax(-1)]
    return legacy, native.engine, graph


def copy_graph(graph):
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in graph.items()
    }


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "names",
    [
        ("water_cluster",),
        ("triclinic_bulk",),
        ("water_cluster", "triclinic_bulk", "isolated_atom"),
    ],
)
def test_atomic_and_total_strain_outputs_match_legacy(
    checkpoints, names, dtype, device
):
    legacy, native, graph = models_and_graph(checkpoints, names, dtype, device)
    wanted = ("forces", "stress", "virials", "atomic_virials", "atomic_stresses")
    original = copy_graph(graph)
    values = []
    for training in (False, True):
        old_graph, new_graph = copy_graph(graph), copy_graph(graph)
        old_positions = old_graph["positions"].requires_grad_(True)
        new_positions = new_graph["positions"].requires_grad_(True)
        expected = legacy(
            old_graph,
            training=training,
            compute_force=True,
            compute_stress=True,
            compute_virials=True,
            compute_edge_forces=True,
            compute_atomic_stresses=True,
        )
        actual = native(new_graph, compute=wanted, training=training)
        pairs = [
            (actual.total_energy, expected["energy"]),
            (actual.forces, expected["forces"]),
            (actual.virials, expected["virials"]),
            (actual.stress, expected["stress"]),
        ]
        for name in ("atomic_virials", "atomic_stresses"):
            pairs.append((actual.extras[name], expected[name]))
            assert actual.extras[name].shape == (graph["positions"].shape[0], 3, 3)
        for left, right in pairs:
            close(left, right, dtype, device)
        for name, total in (
            ("atomic_virials", actual.virials),
            ("atomic_stresses", actual.stress),
        ):
            summed = torch.zeros_like(total).index_add(
                0, graph["batch"], actual.extras[name]
            )
            close(summed, total, dtype, device)
        nonperiodic = ~graph["pbc"].reshape(-1, 3).any(-1)
        assert torch.count_nonzero(actual.stress[nonperiodic]) == 0
        assert (
            torch.count_nonzero(
                actual.extras["atomic_stresses"][nonperiodic[graph["batch"]]]
            )
            == 0
        )
        if "contributions" in expected:
            contributions = actual.extras["contributions"]
            close(contributions, expected["contributions"], dtype, device)
            close(contributions.sum(-1), actual.total_energy, dtype, device)
            assert contributions.shape[1] == 2 + int(legacy.num_interactions)
            pairs.append((contributions, expected["contributions"]))
        else:
            assert "contributions" not in actual.extras
        if training:
            # A shared nonuniform cotangent checks derivative training, not just
            # agreement of a sum that could cancel opposite component errors.
            objectives = [0, 0]
            for left, right in pairs:
                direction = torch.linspace(
                    0.1, 0.9, left.numel(), dtype=dtype, device=device
                ).reshape(left.shape)
                objectives[0] += (left * direction).sum()
                objectives[1] += (right * direction).sum()
            gradients = [
                torch.autograd.grad(objective, positions)[0]
                for objective, positions in zip(
                    objectives, (new_positions, old_positions), strict=True
                )
            ]
            close(*gradients, dtype, device)
        values.append([value.detach() for value, _ in pairs])
        for key, before in original.items():
            if isinstance(before, torch.Tensor):
                torch.testing.assert_close(new_graph[key], before, atol=0, rtol=0)
        assert new_graph.keys() == original.keys()
    for inference, training in zip(*values, strict=True):
        close(inference, training, dtype, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("strain", ["zero", "symmetric", "antisymmetric"])
def test_external_displacement_matches_legacy_preparation_and_virial(
    checkpoints, dtype, device, strain
):
    from mace.modules.utils import get_symmetric_displacement

    legacy, native, graph = models_and_graph(
        checkpoints, ("triclinic_bulk",), dtype, device
    )
    displacement = torch.tensor(
        [[[0.01, 0.004, -0.002], [0.004, -0.003, 0.001], [-0.002, 0.001, 0.005]]],
        dtype=dtype,
        device=device,
    )
    if strain == "zero":
        displacement.zero_()
    elif strain == "antisymmetric":
        displacement[0, 1, 0] = -0.004
        displacement[0, 2, 0] = 0.002
        displacement[0, 2, 1] = -0.001
        displacement.diagonal(dim1=-1, dim2=-2).zero_()
    displacement.requires_grad_(True)
    graph["displacement"] = displacement
    graph["positions"].requires_grad_(True)
    before = copy_graph(graph)
    prepared, position_leaf, leaf = prepare_inputs(
        graph, need_forces=True, need_stress=True
    )
    assert position_leaf is graph["positions"]
    old_positions, old_shifts, old_leaf = get_symmetric_displacement(
        positions=graph["positions"],
        unit_shifts=graph["unit_shifts"],
        cell=graph["cell"],
        edge_index=graph["edge_index"],
        num_graphs=1,
        batch=graph["batch"],
        displacement=displacement,
    )
    assert leaf is old_leaf is displacement
    close(prepared["positions"], old_positions, dtype, device)
    close(prepared["shifts"], old_shifts, dtype, device)
    expected = legacy(
        copy_graph(graph),
        training=True,
        compute_force=True,
        compute_virials=True,
        compute_stress=True,
        compute_edge_forces=True,
        compute_atomic_stresses=True,
    )
    actual = native(
        graph,
        compute=("forces", "virials", "stress", "atomic_virials", "atomic_stresses"),
        training=True,
    )
    close(actual.total_energy, expected["energy"], dtype, device)
    close(actual.forces, expected["forces"], dtype, device)
    close(actual.virials, expected["virials"], dtype, device)
    close(actual.stress, expected["stress"], dtype, device)
    for name in ("atomic_virials", "atomic_stresses"):
        close(actual.extras[name], expected[name], dtype, device)
    strain_gradient, position_gradient = torch.autograd.grad(
        actual.total_energy.sum(), (displacement, graph["positions"])
    )
    close(-strain_gradient, actual.virials, dtype, device)
    close(-position_gradient, actual.forces, dtype, device)
    if dtype == torch.float64 and device == "cpu" and strain == "symmetric":
        for atom, axis in ((0, 0), (4, 1)):
            energies = []
            step = 1e-4
            for sign in (-1, 1):
                shifted = copy_graph(graph)
                shifted["positions"] = graph["positions"].detach().clone()
                shifted["positions"][atom, axis] += sign * step
                energies.append(
                    native(shifted, compute=("virials",)).total_energy.detach()
                )
            difference = -(energies[1] - energies[0]).sum() / (2 * step)
            close(actual.forces[atom, axis], difference, dtype, device)
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(graph[key], value, atol=0, rtol=0)
    inert, _, missing = prepare_inputs(graph, need_forces=False, need_stress=False)
    assert missing is None
    torch.testing.assert_close(inert["positions"], graph["positions"], atol=0, rtol=0)
