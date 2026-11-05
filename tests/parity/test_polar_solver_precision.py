"""Check the corrected solver against legacy evaluated in high precision.

Both receive identical, already rounded multipoles and positions. The native
solver uses the configured reduction floor and returns the interface dtype.
Whole-model parity remains a separate obligation: this test isolates the
cancellation-prone electrostatic calculation from predicted density errors.
"""

import numpy as np
import pytest
import torch
from mace_core.electrostatics import ElectrostaticsSolverDescriptor, FeatureProjection
from mace_torch.backends.precision import cast_at_boundary
from mace_torch.electrostatics import (
    build_long_range,
    build_long_range_features,
    long_range_geometry,
)

from tests.golden import harness

pytestmark = [pytest.mark.polar, pytest.mark.usefixtures("isolated")]


def differentiated(function, multipoles, positions):
    inputs = tuple(
        value.detach().clone().requires_grad_() for value in (multipoles, positions)
    )
    output = function(*inputs)
    generator = torch.Generator().manual_seed(731)

    def direction(value):
        # Identical cotangents even when one output remains in float64.
        return torch.randn(value.shape, dtype=multipoles.dtype, generator=generator).to(
            device=value.device, dtype=value.dtype
        )

    gradients = torch.autograd.grad(
        (output * direction(output)).sum(), inputs, create_graph=True
    )
    hessian_product = torch.autograd.grad(
        sum((gradient * direction(gradient)).sum() for gradient in gradients), inputs
    )
    return (output, *gradients, *hessian_product)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("self_interaction", [False, True])
@pytest.mark.parametrize("kind", ["projection", "energy"])
def test_polar_solver_reductions_match_high_precision_legacy(
    device, dtype, self_interaction, kind
):
    from graph_longrange import realspace_electrostatics as legacy

    torch.set_default_dtype(torch.float64)
    fixtures = list(
        harness.load_fixtures(names=("water_cluster", "dimer_short")).values()
    )
    positions = torch.tensor(
        np.concatenate([atoms.positions for atoms in fixtures]),
        dtype=dtype,
        device=device,
    )
    batch = torch.repeat_interleave(
        torch.arange(len(fixtures), device=device),
        torch.tensor([len(atoms) for atoms in fixtures], device=device),
    )
    multipoles = 0.1 * torch.randn(
        (positions.shape[0], 4),
        dtype=dtype,
        generator=torch.Generator().manual_seed(149),
    ).to(device)
    descriptor = ElectrostaticsSolverDescriptor(
        "molecular",
        1,
        3.0,
        1.5,
        precision="float64",
        features=FeatureProjection(
            max_l=1, widths=(1.5, 3.0), include_self_interaction=self_interaction
        ),
        external_field_flags=frozenset({"include_self_interaction"})
        if self_interaction
        else frozenset(),
    )

    if kind == "projection":
        old = legacy.RealSpaceFiniteDifferenceElectrostaticFeatures(
            1, 1.5, 1, [1.5, 3.0], include_self_interaction=self_interaction
        ).to(device)
        new = build_long_range_features(descriptor).to(device)

        def reference(source, coords):
            return old(source.double().unsqueeze(-2), coords.double(), batch)[0]
    else:
        old = legacy.RealSpaceFiniteDiffereneEnergy(
            1, 1.5, include_self_interaction=self_interaction
        ).to(device)
        new = build_long_range(descriptor).to(device)

        def reference(source, coords):
            return old(source.double(), coords.double(), batch)

    cast_at_boundary(new, "float64", "float32" if dtype == torch.float32 else "float64")

    def candidate(source, coords):
        graph = {
            "positions": coords,
            "batch": batch,
            "cell": 10
            * torch.eye(3, dtype=dtype, device=device).expand(len(fixtures), 3, 3),
            "pbc": torch.zeros((len(fixtures), 3), dtype=torch.bool, device=device),
        }
        geometry = long_range_geometry(graph, descriptor)
        return (
            new(new.prepare(geometry), source)
            if kind == "projection"
            else new(graph, source, geometry)
        )

    expected = differentiated(reference, multipoles, positions)
    actual = differentiated(candidate, multipoles, positions)
    row = (
        harness.FP32
        if dtype == torch.float32
        else (
            harness.FP64_CPU_REFERENCE
            if device == "cpu"
            else harness.FP64_ACCELERATED_BACKEND
        )
    )
    assert actual[0].dtype == dtype
    for observed, wanted in zip(actual, expected, strict=True):
        assert torch.isfinite(observed).all() and torch.isfinite(wanted).all()
        torch.testing.assert_close(
            observed.double(), wanted.double(), atol=row.atol, rtol=row.rtol
        )
