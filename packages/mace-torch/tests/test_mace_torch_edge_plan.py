"""Ordering changes only an edge permutation; padding preserves stable runs."""

import itertools

import numpy as np
import pytest
import torch
from conftest import TOLERANCES, fp64_only
from mace_core.data import AtomicNumberTable, Configuration
from mace_core.edges import EDGE_FEATURES, plan_edges
from mace_core.graph import FixedPadding, graph_sample, validate_graph
from mace_core.neighbors import AllPairsBackend
from mace_torch.graph import EdgeRuns, apply_edge_plan, build_batch, collate, pad_batch
from mace_torch_compile_fixtures import deployed
from mace_torch_neighbor_fixtures import require_cuda

TABLE = AtomicNumberTable([1])
FEATURE_SETS = [
    frozenset(combination)
    for size in range(4)
    for combination in itertools.combinations(sorted(EDGE_FEATURES), size)
]


def graph(device="cpu", empty=False):
    config = Configuration(
        atomic_numbers=np.ones(4, dtype=np.int64),
        positions=np.array(
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.8, 0.0], [10.0, 0.0, 0.0]]
        ),
    )
    result = build_batch(
        [config],
        atomic_number_table=TABLE,
        cutoff=0.01 if empty else 1.3,
        neighbor_backend=AllPairsBackend(),
        device=device,
    )
    # Start from an arbitrary order, with one disconnected atom.
    order = torch.arange(result["edge_index"].shape[1] - 1, -1, -1, device=device)
    result["edge_index"] = result["edge_index"][:, order]
    for key in ("shifts", "unit_shifts"):
        result[key] = result[key][order]
    return result


def assert_features(data, features):
    validate_graph(data)
    if "sorted_by_sender" in features:
        assert torch.all(data["edge_index"][0, 1:] >= data["edge_index"][0, :-1])
    for side, row in (("sender", 0), ("receiver", 1)):
        if f"{side}_runs" in features:
            runs = EdgeRuns.from_graph(data, side)
            assert torch.equal(
                runs.order, torch.argsort(data["edge_index"][row], stable=True)
            )
            expected = torch.bincount(
                data["edge_index"][row], minlength=len(data["positions"])
            )
            assert torch.equal(runs.offsets[1:] - runs.offsets[:-1], expected)


@pytest.mark.parametrize("required", FEATURE_SETS)
@pytest.mark.parametrize("guaranteed", FEATURE_SETS)
def test_every_plan_preserves_edges_and_existing_features(required, guaranteed):
    original = graph()
    prepared = apply_edge_plan(original, plan_edges(guaranteed, set()))
    snapshot = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in prepared.items()
    }
    actual = apply_edge_plan(prepared, plan_edges(required, guaranteed))
    assert_features(actual, required | guaranteed)
    before = sorted(
        zip(
            original["edge_index"].T.tolist(),
            original["shifts"].tolist(),
            original["unit_shifts"].tolist(),
            strict=True,
        )
    )
    after = sorted(
        zip(
            actual["edge_index"].T.tolist(),
            actual["shifts"].tolist(),
            actual["unit_shifts"].tolist(),
            strict=True,
        )
    )
    assert before == after
    for key, value in snapshot.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(prepared[key], value)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("features", FEATURE_SETS)
def test_padding_preserves_order_and_derived_fields(empty, features):
    prepared = apply_edge_plan(graph(empty=empty), plan_edges(features, set()))
    padded, _ = pad_batch(
        prepared,
        FixedPadding(nodes_budget=7, edges_budget=32, graphs_budget=3),
        1.3,
        atomic_number_table=TABLE,
    )
    assert_features(padded, features)


@pytest.mark.parametrize(
    "key", ["sender_order", "receiver_order", "sender_counts", "receiver_counts"]
)
def test_a_sample_cannot_carry_collated_runs(key):
    config = Configuration(atomic_numbers=np.array([1]), positions=np.zeros((1, 3)))
    sample = graph_sample(
        config,
        atomic_number_table=TABLE,
        cutoff=1.0,
        neighbor_backend=AllPairsBackend(),
    )
    sample[key] = np.zeros(0, dtype=np.int64)
    with pytest.raises(ValueError, match=key):
        collate([sample])


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "counts", "unstable", "negative"]
)
def test_validation_refuses_inconsistent_runs(corruption):
    prepared = apply_edge_plan(graph(), plan_edges(EDGE_FEATURES, set()))
    if corruption == "missing":
        del prepared["receiver_counts"]
    elif corruption == "duplicate":
        prepared["receiver_order"][0] = prepared["receiver_order"][1]
    elif corruption == "counts":
        prepared["receiver_counts"][0] += 1
    elif corruption == "negative":
        prepared["receiver_order"][0] = -1
    else:
        prepared["receiver_order"] = prepared["receiver_order"].flip(0)
    with pytest.raises(ValueError, match="receiver_"):
        validate_graph(prepared)


@pytest.mark.gpu
def test_edge_adaptation_and_padding_remain_on_cuda(monkeypatch):
    require_cuda()
    prepared = graph(device="cuda")
    with monkeypatch.context() as patch:
        patch.setattr(
            torch.Tensor, "cpu", lambda *a, **k: pytest.fail("copied edges to CPU")
        )
        padded, _ = pad_batch(
            apply_edge_plan(prepared, plan_edges(EDGE_FEATURES, set())),
            FixedPadding(nodes_budget=7, edges_budget=32, graphs_budget=3),
            1.3,
            atomic_number_table=TABLE,
        )
    assert_features(padded, EDGE_FEATURES)
    assert padded["receiver_order"].device.type == "cuda"


@fp64_only
def test_ordering_and_padding_preserve_model_outputs_and_force_training_gradients():
    deployment = deployed()
    config = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.2, 0.9, 0.0]]),
        cell=np.eye(3) * 6.0,
        pbc=(True, True, True),
    )
    original = build_batch(
        [config],
        atomic_number_table=deployment.z_table,
        cutoff=4.0,
        neighbor_backend=AllPairsBackend(),
    )
    for key in ("edge_index", "unit_shifts", "shifts"):
        original[key] = original[key].flip(1 if key == "edge_index" else 0)
    ordered = apply_edge_plan(original, plan_edges(EDGE_FEATURES, set()))
    padded, info = pad_batch(
        ordered,
        FixedPadding(nodes_budget=6, edges_budget=32, graphs_budget=2),
        4.0,
        atomic_number_table=deployment.z_table,
    )
    parameters = [p for p in deployment.engine.parameters() if p.requires_grad]
    outputs, gradients = [], []
    for batch in (original, padded):
        output = deployment.engine(batch, compute=("forces", "stress"), training=True)
        values = {
            "total_energy": output.total_energy[: info.n_real_graphs],
            "forces": output.forces[: info.n_real_nodes],
            "stress": output.stress[: info.n_real_graphs],
        }
        loss = values["forces"].square().sum() + values["stress"].square().sum()
        gradients.append(torch.autograd.grad(loss, parameters, allow_unused=True))
        outputs.append(values)
    atol, rtol = TOLERANCES[torch.float64]
    for key in ("total_energy", "forces", "stress"):
        torch.testing.assert_close(
            outputs[0][key], outputs[1][key], atol=atol, rtol=rtol
        )
    for expected, actual in zip(*gradients, strict=True):
        assert (expected is None) == (actual is None)
        if expected is not None:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
