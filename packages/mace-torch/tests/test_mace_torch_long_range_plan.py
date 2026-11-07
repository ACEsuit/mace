"""Discrete geometry planning stays outside the differentiable model."""

from dataclasses import replace

import pytest
import torch
from conftest import assert_close, fp64_only
from mace_core.electrostatics import ElectrostaticsSolverDescriptor, FeatureProjection
from mace_torch.electrostatics import (
    LONG_RANGE_PLAN,
    LongRangeEnergy,
    LongRangeFeatures,
    long_range_geometry,
    long_range_plan,
)
from mace_torch.electrostatics.reference.realspace_electrostatics import (
    batch_complete_graph_excluding_self_duplicates_vector,
)
from mace_torch.physics import DerivativeEngine
from mace_torch_engine_fixtures import molecule
from test_mace_torch_polar_model import WITH_STRESS, build_model, polar_graph


@fp64_only
@pytest.mark.parametrize(
    "profile", ["molecular", "full_periodic", "partial", "per_structure"]
)
def test_whole_polar_forward_compiles_with_prepared_geometry(profile):
    from torch._dynamo.testing import CompileCounter

    periodic = profile != "molecular"
    graph = polar_graph(
        *molecule(), cell=(torch.eye(3) * 7).numpy(), pbc=(periodic,) * 3
    )
    model = build_model(profile=profile)
    engine = DerivativeEngine(model, WITH_STRESS, None).eval()
    expected = engine(graph, compute=("forces", "stress"))
    counter = CompileCounter()
    engine.compile_model(backend=counter)
    for _ in range(2):
        actual = engine(graph, compute=("forces", "stress"))
        for name in ("total_energy", "forces", "stress", "dipole"):
            assert_close(getattr(actual, name), getattr(expected, name).detach(), name)
    assert counter.frame_count == 1
    assert LONG_RANGE_PLAN not in graph


@pytest.mark.parametrize("profile", ["molecular", "full_periodic", "per_structure"])
def test_plan_preserves_energy_projection_and_cell_gradients(profile, dtype):
    graph = polar_graph(*molecule(), cell=(torch.eye(3) * 7).numpy(), pbc=(True,) * 3)
    graph = {
        key: value.to(dtype)
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        else value
        for key, value in graph.items()
    }
    descriptor = ElectrostaticsSolverDescriptor(
        profile,
        1,
        2.5,
        1.0,
        precision="float64" if dtype == torch.float64 else "float32",
        features=FeatureProjection(max_l=1, widths=(1.0, 1.5)),
    )
    energy = LongRangeEnergy(descriptor, "reference").to(dtype)
    features = LongRangeFeatures(descriptor, "reference").to(dtype)
    source = torch.randn(4, 4, dtype=dtype)
    plan = long_range_plan(graph, descriptor)
    values, gradients = [], []
    for supplied in ({}, {LONG_RANGE_PLAN: plan}):
        positions = graph["positions"].detach().clone().requires_grad_(True)
        cell = graph["cell"].detach().clone().requires_grad_(True)
        batch = {**graph, **supplied, "positions": positions, "cell": cell}
        geometry = long_range_geometry(batch, descriptor)
        output = torch.cat(
            (
                energy(batch, source, geometry).reshape(-1),
                features(features.prepare(geometry), source).reshape(-1),
            )
        )
        grad = torch.autograd.grad(output.sum(), (positions, cell), allow_unused=True)
        values.append(output.detach())
        gradients.append(grad)
    assert_close(values[0], values[1])
    for actual, expected in zip(*gradients, strict=True):
        assert (actual is None) == (expected is None)
        if actual is not None:
            assert expected is not None
            assert_close(actual, expected.detach())
    with pytest.raises(ValueError, match="cutoff"):
        long_range_geometry(
            {**graph, LONG_RANGE_PLAN: plan}, replace(descriptor, kspace_cutoff=3.0)
        )


def test_solver_specific_plans_share_one_geometry_without_held_model_state():
    graph = polar_graph(*molecule())
    descriptor = ElectrostaticsSolverDescriptor("molecular", 1, 2.5, 1.0)

    class Planner:
        def plan(self, geometry):
            assert geometry.sizes == (4,)
            return {"test_solver": geometry.positions.new_tensor([7.0])}

    plan = long_range_plan(graph, descriptor, (Planner(),))
    geometry = long_range_geometry({**graph, LONG_RANGE_PLAN: plan}, descriptor)
    assert geometry.plans["test_solver"].item() == 7
    assert geometry.prepared == {}
    geometry.prepared["test_solver"] = "one forward only"
    assert not long_range_geometry(
        {**graph, LONG_RANGE_PLAN: plan}, descriptor
    ).prepared


def test_realspace_pair_planning_handles_sparse_graph_ids():
    batch = torch.tensor([5, 5, 8])
    edges = batch_complete_graph_excluding_self_duplicates_vector(batch, 1)
    assert torch.equal(edges, torch.tensor([[0, 1], [1, 0]]))


@pytest.mark.gpu
@pytest.mark.parametrize("profile", ["molecular", "full_periodic"])
def test_planned_solver_compiles_and_replays_a_cuda_graph(profile, dtype):
    from mace_core.config.precision import PrecisionConfig
    from mace_torch.backends.conformance import _check_cuda_graph, _close
    from mace_torch_neighbor_fixtures import require_cuda

    require_cuda()
    graph = polar_graph(*molecule(), cell=(torch.eye(3) * 7).numpy(), pbc=(True,) * 3)
    graph = {
        name: value.to(
            "cuda", dtype=dtype if value.is_floating_point() else value.dtype
        )
        if isinstance(value, torch.Tensor)
        else value
        for name, value in graph.items()
    }
    precision = "float64" if dtype == torch.float64 else "float32"
    # Exercise the production precision boundary, including the wider solve
    # required by cancellation in the displaced-charge approximation.
    model = build_model(
        profile=profile,
        head_precision=PrecisionConfig(model=precision),
        precision=precision,
    ).to("cuda")
    descriptor = model.descriptor
    energy, features = model.coulomb, model.projection
    plan = long_range_plan(graph, descriptor)

    def run(positions, cell, source):
        batch = {**graph, "positions": positions, "cell": cell, LONG_RANGE_PLAN: plan}
        geometry = long_range_geometry(batch, descriptor)
        return torch.cat(
            (
                energy(batch, source, geometry),
                features(features.prepare(geometry), source).reshape(-1),
            )
        )

    arguments = [
        graph["positions"],
        graph["cell"],
        torch.randn(
            4, 4, dtype=dtype, generator=torch.Generator().manual_seed(149)
        ).cuda(),
    ]
    compiled = torch.compile(run, fullgraph=True, dynamic=False)
    with torch.no_grad():
        expected = run(*arguments)
        # The shared backend row compares different GPU arithmetic kernels.
        _close(compiled(*arguments), expected, precision, "compiled long-range plan")
        _check_cuda_graph(compiled, arguments, expected, precision, "long-range plan")
