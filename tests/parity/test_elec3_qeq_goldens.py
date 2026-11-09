"""QEq models against the frozen, external MACE 0.3.14 oracle.

No external source checkout is needed in CI. The canonical tiny weights and
outputs were captured together by the explicit ELEC-3 golden target.
"""

import json

import pytest
import torch

from tests.golden.harness import FP32, FP64_CPU_REFERENCE
from tests.parity.elec3_qeq import (
    PROFILES,
    VARIANTS,
    graph_for,
    load_model,
    outputs,
    reference_path,
)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32], ids=["float64", "float32"]
)
def test_qeq_frozen_external_reference(variant, profile, dtype):
    model = load_model(variant, profile, dtype)
    actual = outputs(model, graph_for(profile, dtype))
    expected = json.loads(reference_path().read_text())["cases"][f"{variant}/{profile}"]
    row = FP64_CPU_REFERENCE if dtype == torch.float64 else FP32
    assert set(actual) == set(expected)
    for name, value in actual.items():
        torch.testing.assert_close(
            value,
            torch.tensor(expected[name], dtype=value.dtype),
            atol=row.atol,
            rtol=row.rtol,
            msg=f"{variant} {profile} {name}",
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "backend", ["reference", pytest.param("cueq", marks=pytest.mark.cueq)]
)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32], ids=["float64", "float32"]
)
def test_qeq_cuda_reference_and_cueq(variant, profile, dtype, backend):
    from tests.golden.harness import FP64_ACCELERATED_BACKEND

    model = load_model(variant, profile, dtype, backend).to("cuda")
    graph = {
        key: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for key, value in graph_for(profile, dtype).items()
    }
    actual = outputs(model, graph)
    expected = json.loads(reference_path().read_text())["cases"][f"{variant}/{profile}"]
    row = FP64_ACCELERATED_BACKEND if dtype == torch.float64 else FP32
    for name, value in actual.items():
        torch.testing.assert_close(
            value.cpu(),
            torch.tensor(expected[name], dtype=value.dtype),
            atol=row.atol,
            rtol=row.rtol,
            msg=f"{variant} {profile} {backend} {name}",
        )
    (actual["forces"].square().sum() + actual["stress"].square().sum()).backward()
    gradients = [
        parameter.grad
        for parameter in model.source_maps.parameters()
        if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    "profile", ["molecular", "full_periodic", "z_slab", "partial", "per_structure"]
)
def test_qeq_projected_matrix_is_energy_hessian_on_fixed_charge_subspace(profile):
    from mace_torch.electrostatics import long_range_geometry
    from mace_torch.physics.charge_equilibration import constrained_charges

    model = load_model("fixed", profile)
    graph = graph_for(profile)
    geometry = long_range_geometry(graph, model.solver_spec)
    count = graph["positions"].shape[0]
    matrix = model.charge_solve.potential_matrix(graph, geometry)
    hessian = torch.autograd.functional.hessian(
        lambda q: model.coulomb(graph, q[:, None], geometry).sum(),
        torch.zeros(count, dtype=torch.float64),
    )
    start = 0
    for index, size in enumerate(graph["structure_sizes"]):
        end = start + size
        block, reference = matrix[start:end, start:end], hessian[start:end, start:end]
        # In a molecule-in-box the two potentials have a uniform gauge shift.
        difference = block - reference
        torch.testing.assert_close(
            difference, difference[0, 0].expand_as(difference), atol=1e-12, rtol=0
        )
        chi = graph["input_chi"][start:end].clone().requires_grad_()
        hardness = torch.diag(graph["input_hardness"][start:end])
        charges = constrained_charges(
            block + hardness, chi, graph["total_charge"][index]
        )
        expected = constrained_charges(
            reference + hardness, chi, graph["total_charge"][index]
        )
        torch.testing.assert_close(charges, expected, atol=1e-12, rtol=0)
        before = torch.autograd.grad(charges.square().sum(), chi, retain_graph=True)[0]
        after = torch.autograd.grad(expected.square().sum(), chi)[0]
        torch.testing.assert_close(before, after, atol=1e-12, rtol=0)
        start = end
