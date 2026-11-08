"""The declared joint model against legacy, including its fixed-charge physics.

Shared legacy readouts are split by output copy using the existing dipole
conversion. Gradients of the shared weights are compared through that linear
map, not by assuming that independently parameterized heads have equal updates.
"""

import copy

import numpy as np
import pytest
import torch
from ase import Atoms
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.models.build import build_model

from tests.golden.harness import FP64_ACCELERATED_BACKEND, FP64_CPU_REFERENCE
from tests.parity.dipole_convert import dipole_config, transfer_dipole_weights
from tests.parity.fm00_convert import model_arguments
from tests.parity.test_dipole_parity import legacy_batch, v1_graph

pytestmark = pytest.mark.usefixtures("isolated")


def legacy_model():
    from e3nn import o3

    from mace.modules import interaction_classes
    from mace.modules.models import EnergyDipolesMACE

    torch.manual_seed(19)
    return EnergyDipolesMACE(
        r_max=3.5,
        num_bessel=6,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e+4x1o+4x2e"),
        MLP_irreps=o3.Irreps("4x0e+4x1o"),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
        correlation=2,
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([-0.5, 1.0]),
        radial_MLP=[64, 64, 64],
    )


def converted(legacy, backend="reference"):
    architecture = model_arguments(dipole_config(legacy))["config"].model_dump(
        mode="json"
    )
    architecture.update(
        model="plain", backend=backend, observables=["energy", "forces", "dipole"]
    )
    config = ResolvedConfig.from_dict(
        {"model": architecture, "data": {"heads": {"default": {}}}}
    )
    engine, _ = build_model(
        config,
        DEFAULT_CATALOGUE,
        z_table=AtomicNumberTable([1, 8]),
        heads=("default",),
        e0s=ResolvedE0s({"default": {1: -0.5, 8: 1.0}}),
        statistics=DatasetStatistics(avg_num_neighbors=3.0, mean=0.0, std=1.0),
        initialize=False,
    )
    transfer_dipole_weights(legacy, engine.backbone, "energy_dipole")
    return engine, config


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    return request.param


def close(actual, expected, device="cpu"):
    row = FP64_CPU_REFERENCE if device == "cpu" else FP64_ACCELERATED_BACKEND
    torch.testing.assert_close(actual, expected, atol=row.atol, rtol=row.rtol)


def inputs(legacy, charged=False):
    atoms = Atoms(
        "OH2", positions=[[0.13, -0.1, 0.2], [1.02, 0.07, -0.11], [-0.2, 0.94, 0.13]]
    )
    charges = [-0.8, 0.4, 0.4] if not charged else [-0.8, 0.4, -0.6]
    return (
        atoms,
        legacy_batch(legacy, atoms, charges, sum(charges)),
        v1_graph(atoms, [1, 8], 3.5, charges, sum(charges)),
    )


@pytest.mark.parametrize("charged", [False, True])
def test_joint_values_and_derivatives_match_legacy(fp64, device, charged):
    from mace.modules.utils import compute_dielectric_gradients

    legacy = legacy_model()
    native, _ = converted(legacy)
    assert native.backbone.backbone.layer_irreps[-1] == "4x0e+4x1o"
    assert native.backbone.outputs.heads["dipole"].reachable == [0, 1]
    _, old, new = inputs(legacy, charged)
    old = {key: value.to(device) for key, value in old.items()}
    new = {key: value.to(device) for key, value in new.items()}
    expected = legacy.to(device)(old, training=True)
    derivative = compute_dielectric_gradients(expected["dipole"], old["positions"])
    actual = native.to(device)(new, compute=("forces", "dmu_dr"), training=True)
    close(actual.total_energy, expected["energy"], device)
    close(actual.forces, expected["forces"], device)
    close(actual.dipole, expected["dipole"], device)
    close(actual.extras["dmu_dr"], derivative, device)
    loss = (
        actual.total_energy.square().sum()
        + actual.forces.square().sum()
        + actual.dipole.square().sum()
    )
    oracle_loss = (
        expected["energy"].square().sum()
        + expected["forces"].square().sum()
        + expected["dipole"].square().sum()
    )
    loss.backward()
    oracle_loss.backward()
    # Radial weights share coordinates; force training exercises double backward.
    for old_layer, new_layer in zip(
        legacy.interactions, native.backbone.backbone.interactions, strict=True
    ):
        for index, weight in enumerate(new_layer.body.radial_mlp.weights):
            close(
                weight.grad,
                getattr(old_layer.conv_tp_weights, f"layer{index}").weight.grad,
                device,
            )


def test_joint_shared_parameter_gradients_obey_the_conversion(fp64):
    legacy = legacy_model()
    native, _ = converted(legacy)
    _, old, new = inputs(legacy, True)
    expected = legacy(old, training=True)
    actual = native(new, compute=("forces",), training=True)
    (
        expected["energy"].square().sum()
        + expected["forces"].square().sum()
        + expected["dipole"].square().sum()
    ).backward()
    (
        actual.total_energy.square().sum()
        + actual.forces.square().sum()
        + actual.dipole.square().sum()
    ).backward()
    # The conversion is linear in all trainable weights, so converting a tangent
    # gives J v exactly, including full-to-reduced CG projection and shared heads.
    for seed in (12, 43):
        tangent = copy.deepcopy(legacy)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter in tangent.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
        mapped, _ = converted(tangent)
        oracle = sum(
            (parameter.grad * direction).sum()
            for parameter, direction in zip(
                legacy.parameters(), tangent.parameters(), strict=True
            )
        )
        actual_gradient = sum(
            (parameter.grad * direction).sum()
            for parameter, direction in zip(
                native.parameters(), mapped.parameters(), strict=True
            )
        )
        close(actual_gradient, oracle)


@pytest.mark.parametrize(
    "backend",
    ["reference", pytest.param("cueq", marks=[pytest.mark.gpu, pytest.mark.cueq])],
)
def test_joint_checkpoint_and_calculator_match_legacy(fp64, tmp_path, backend):
    from mace_core.config.provenance import e0_details
    from mace_core.metadata import ConfigRecord, HeadSummary, ModelMetadata, Provenance
    from mace_torch.calculators import MACECalculator
    from mace_torch.train.checkpoint import write_model

    from mace.calculators import MACECalculator as LegacyCalculator

    legacy = legacy_model()
    native, config = converted(legacy)
    # Select the backend through recorded configuration; weights stay canonical.
    config = config.model_copy(
        update={"model": config.model.model_copy(update={"backend": backend})}
    )
    metadata = ModelMetadata(
        elements=[1, 8],
        config=ConfigRecord.from_config(config),
        provenance=Provenance(versions={"mace-torch-v1": "parity"}),
        heads={
            "default": HeadSummary(
                e0=e0_details(config.data.heads["default"].e0s, {"H": -0.5, "O": 1.0})
            )
        },
    )
    path = write_model(tmp_path / "joint", native, metadata)
    device = "cuda" if backend == "cueq" else "cpu"
    frozen = LegacyCalculator(
        models=legacy,
        model_type="EnergyDipoleMACE",
        device="cpu",
        default_dtype="float64",
    )
    calc = MACECalculator(
        model_paths=path, device=device, pad_num_atoms=8, pad_num_edges=128
    )
    if backend == "cueq":
        from mace_torch.backends.cueq import compiled_operations_available

        assert compiled_operations_available()
        assert any(
            type(module).__name__.startswith("CuEq")
            for module in calc.models[0].engine.modules()
        )
    atoms, _, _ = inputs(legacy, True)
    atoms.arrays["Qs"] = np.array([-0.8, 0.4, -0.6])
    for periodic in (False, True):
        atoms.set_cell([4.1, 4.3, 4.7])
        atoms.set_pbc(periodic)
        for calculator in (frozen, calc):
            calculator.calculate(atoms)
        for name in ("energy", "forces", "stress", "dipole"):
            close(
                torch.as_tensor(calc.results[name]),
                torch.as_tensor(frozen.results[name]),
                device,
            )


@pytest.mark.parametrize("operation", ["rotation", "translation", "permutation"])
def test_joint_geometry_transformations(fp64, operation):
    legacy = legacy_model()
    native, _ = converted(legacy)
    atoms, _, graph = inputs(legacy, True)
    original = native(graph, compute=("forces",))
    charges = np.array([-0.8, 0.4, -0.6])
    rotation = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    translation = torch.tensor([0.5, -0.2, 0.3])
    order = [2, 0, 1]
    if operation == "rotation":
        atoms.positions = atoms.positions @ rotation.numpy().T
    elif operation == "translation":
        atoms.positions += translation.numpy()
    else:
        atoms = atoms[order]
        charges = charges[order]
    transformed = native(
        v1_graph(atoms, [1, 8], 3.5, charges, charges.sum()), compute=("forces",)
    )
    close(transformed.total_energy, original.total_energy)
    if operation == "rotation":
        close(transformed.dipole, original.dipole @ rotation.T)
        close(transformed.forces, original.forces @ rotation.T)
    elif operation == "translation":
        from mace_torch.models.dipoles import E_ANGSTROM_PER_DEBYE

        close(
            transformed.dipole,
            original.dipole + charges.sum() * translation / E_ANGSTROM_PER_DEBYE,
        )
        close(transformed.forces, original.forces)
    else:
        close(transformed.dipole, original.dipole)
        close(transformed.forces, original.forces[order])
