"""Retained readout activations against the frozen gate and readout classes."""

from types import SimpleNamespace

import pytest
import torch
from e3nn import o3
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models.heads import _Gate, _GatedReadout

from mace.modules import gate_dict
from mace.modules.blocks import NonLinearBiasReadoutBlock, NonLinearReadoutBlock
from mace.modules.gate import GatedEquivariantBlock
from tests.golden.harness import (
    CLOSED_FORM_FP64,
    FP32,
    FP64_ACCELERATED_BACKEND,
    FP64_CPU_REFERENCE,
)
from tests.parity.fm00_convert import linear_weights_to_canonical


def close(actual, expected, device):
    row = (
        FP32
        if actual.dtype == torch.float32
        else (CLOSED_FORM_FP64 if device == "cpu" else FP64_ACCELERATED_BACKEND)
    )
    torch.testing.assert_close(actual, expected, atol=row.atol, rtol=row.rtol)


def canonical_weights(linear, weights):
    # Bias instructions have no entry in the independently stored weight vector.
    view = SimpleNamespace(
        weight=weights,
        instructions=[
            instruction for instruction in linear.instructions if instruction.i_in >= 0
        ],
    )
    return torch.tensor(
        linear_weights_to_canonical(view, str(linear.irreps_in), str(linear.irreps_out))
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("gate", ["silu", "tanh", "abs", "None"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_vector_tensor_gate_values_gradients_and_curvature(
    fp64, gate, device, dtype
):
    native = _Gate("3x0e", "2x1o+1x2e", gate=gate).to(device)
    legacy = GatedEquivariantBlock(
        "3x0e", [gate_dict[gate]], "3x0e", [gate_dict[gate]], "2x1o+1x2e"
    ).to(device)
    torch.manual_seed(14)
    features = torch.randn(3, 17, device=device, dtype=dtype, requires_grad=True)
    # Pin abs's derivative convention at zero as well as both signs.
    with torch.no_grad():
        features[0, :6] = 0
    compiled = torch.compile(native, backend="eager", fullgraph=True)
    actual, expected = compiled(features), legacy(features)
    close(actual, expected, device)
    rotation = o3.rand_matrix(device=device, dtype=dtype)
    for matrix in (rotation, -rotation):
        inputs = legacy.irreps_in.D_from_matrix(matrix)
        outputs = legacy.irreps_out.D_from_matrix(matrix)
        close(native(features @ inputs.T), actual @ outputs.T, device)
    close(native(features.flip(0)), actual.flip(0), device)
    direction = torch.randn_like(actual)
    gradients = [
        torch.autograd.grad((value * direction).sum(), features, create_graph=True)[0]
        for value in (actual, expected)
    ]
    close(*gradients, device)
    direction = torch.randn_like(features)
    curvature = [
        torch.autograd.grad((value * direction).sum(), features)[0]
        for value in gradients
    ]
    close(*curvature, device)
    # Finite differences stay away from abs's nondifferentiable origin.
    if dtype == torch.float64:
        probe = (
            torch.linspace(0.3, 1.7, 17, device=device).reshape(1, -1).requires_grad_()
        )
        assert torch.autograd.gradgradcheck(native, (probe,))


@pytest.mark.parametrize("gate", ["silu", "tanh", "abs", "None"])
@pytest.mark.parametrize("biased", [False, True])
def test_multihead_readout_values_and_training_gradients(fp64, gate, biased):
    torch.manual_seed(42)
    kind = NonLinearBiasReadoutBlock if biased else NonLinearReadoutBlock
    legacy = kind(
        o3.Irreps("4x0e+2x1o"),
        o3.Irreps("6x0e"),
        gate_dict[gate],
        o3.Irreps("2x0e"),
        num_heads=2,
    )
    native = _GatedReadout(
        ReferenceBackend(),
        "4x0e+2x1o",
        "0e",
        3,
        "float64",
        num_heads=2,
        biased=biased,
        gate=gate,
    )
    pairs = [(legacy.linear_1, native.first), (legacy.linear_2, native.second)]
    if biased:
        pairs.append((legacy.linear_mid, native.middle))
    for old, new in pairs:
        with torch.no_grad():
            if old.bias.numel():
                old.bias.copy_(torch.randn_like(old.bias))
            new.weight.copy_(canonical_weights(old, old.weight))
            new.bias.copy_(old.bias)
    features = torch.randn(4, 10, requires_grad=True)
    heads = torch.tensor([0, 1, 1, 0])
    actual, expected = native(features, heads), legacy(features, heads)
    close(actual, expected, "cpu")
    gradients = [
        torch.autograd.grad(value.square().sum(), features, create_graph=True)[0]
        for value in (actual, expected)
    ]
    close(*gradients, "cpu")
    # A force-style objective backpropagates through the first derivative.
    sum(value.square().sum() for value in gradients).backward()
    for old, new in pairs:
        mapped = canonical_weights(old, old.weight.grad)
        close(new.weight.grad, mapped, "cpu")
        if old.bias.numel():
            close(new.bias.grad, old.bias.grad, "cpu")


@pytest.mark.parametrize("gate", ["silu", "tanh", "abs", "None"])
@pytest.mark.parametrize("gate_kind", ["native", "e3nn"])
def test_converted_dielectric_preserves_readout_activation(
    fp64, isolated, tmp_path, gate, gate_kind
):
    from tests.parity.test_dipole_parity import (
        CASES,
        legacy_batch,
        legacy_dielectric,
        v1_graph,
    )
    from tests.parity.test_output_variants import converted

    legacy = legacy_dielectric("4x0e+4x1o+4x2e", gate=gate)
    if gate_kind == "e3nn":
        from e3nn.nn import Gate

        old = legacy.readouts[-1].equivariant_nonlin
        legacy.readouts[-1].equivariant_nonlin = Gate(
            old.irreps_scalars,
            [gate_dict[gate]] * len(old.irreps_scalars),
            old.irreps_gates,
            [gate_dict[gate]] * len(old.irreps_gates),
            old.irreps_gated,
        )
    native = converted(legacy, tmp_path)
    assert native.config.model.readout.gate == gate
    atoms, charges, total = CASES["charged pair"]
    batch = legacy_batch(legacy, atoms, charges, total)
    expected = legacy(batch, training=True, compute_dielectric_derivatives=True)
    graph = v1_graph(atoms, [1, 6, 8], 3.5, charges, total)
    actual = native.engine(graph, compute=("dmu_dr", "dalpha_dr"), training=True)
    close(actual.dipole, expected["dipole"], "cpu")
    for name in ("polarizability", "dmu_dr", "dalpha_dr"):
        close(actual.extras[name], expected[name], "cpu")
    # Legacy's vmap helper gives correct Jacobian values but different higher
    # gradients for the non-SiLU gates on the tested torch version. Compare
    # training to direct differentiation and check it with finite differences.
    for name, prediction in (
        ("dmu_dr", expected["dipole"]),
        ("dalpha_dr", expected["polarizability"]),
    ):
        jacobian = torch.stack(
            [
                torch.autograd.grad(
                    component, batch["positions"], create_graph=True, retain_graph=True
                )[0]
                for component in prediction.flatten()
            ]
        )
        close(jacobian, expected[name], "cpu")
        expected[name] = jacobian
    sum(
        actual.extras[name].square().sum()
        for name in ("polarizability", "dmu_dr", "dalpha_dr")
    ).backward()
    sum(
        expected[name].square().sum()
        for name in ("polarizability", "dmu_dr", "dalpha_dr")
    ).backward()
    for old, new in zip(
        legacy.interactions, native.engine.backbone.backbone.interactions, strict=True
    ):
        for index, weight in enumerate(new.body.radial_mlp.weights):
            close(
                weight.grad,
                getattr(old.conv_tp_weights, f"layer{index}").weight.grad,
                "cpu",
            )
    weight = legacy.interactions[0].conv_tp_weights.layer0.weight
    index = int(weight.grad.abs().argmax())
    original = weight.flatten()[index].item()
    step = 1e-5
    evaluations = []
    for displacement in (-step, step):
        with torch.no_grad():
            weight.flatten()[index] = original + displacement
        output = legacy(
            legacy_batch(legacy, atoms, charges, total),
            training=True,
            compute_dielectric_derivatives=True,
        )
        evaluations.append(
            sum(
                output[name].square().sum().item()
                for name in ("polarizability", "dmu_dr", "dalpha_dr")
            )
        )
    with torch.no_grad():
        weight.flatten()[index] = original
    torch.testing.assert_close(
        weight.grad.flatten()[index].item(),
        (evaluations[1] - evaluations[0]) / (2 * step),
        atol=FP64_CPU_REFERENCE.atol,
        rtol=FP64_CPU_REFERENCE.rtol,
    )


@pytest.mark.polar
@pytest.mark.parametrize("gate", ["silu", "tanh", "abs", "None"])
def test_converted_polar_preserves_energy_and_fukui_activation(
    fp64, isolated, tmp_path, gate
):
    from tests.parity.test_output_variants import converted
    from tests.parity.test_polar_parity import (
        CASES,
        assert_matches,
        legacy_batch,
        legacy_polar,
        v1_graph,
        with_inputs,
    )

    # Keep the converter's existing unit-field-normalization precondition.
    legacy = legacy_polar(gate=gate, field_norm_factor=1.0)
    native = converted(legacy, tmp_path)
    assert native.config.model.readout.gate == gate
    atoms, _, inputs = CASES["charged doublet in a field"]
    atoms = with_inputs(atoms, *inputs)
    expected = legacy(legacy_batch(legacy, atoms), training=False, compute_force=True)
    actual = native.engine(v1_graph(atoms, [1, 8], 4.0), compute=("forces",))
    assert_matches(expected, actual)
