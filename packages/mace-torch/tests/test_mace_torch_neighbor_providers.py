"""Both entry-point groups obey the same edge set, cells, order and cache contract."""

import dataclasses
import itertools
import logging
from typing import Any, cast

import numpy as np
import pytest
import torch
from mace_core.data import AtomicNumberTable, Configuration
from mace_core.edges import EDGE_FEATURES, EdgeRequirementError
from mace_core.graph import AutoPadding, graph_sample, validate_graph
from mace_core.neighbors import (
    AllPairsBackend,
    NeighborBackendNotAvailableError,
    NeighborListConfig,
    available_neighbor_backends,
    get_neighbor_backend,
    get_neighborhood,
    neighbor_capabilities,
    search_geometry,
)
from mace_torch.graph import apply_edge_plan, build_batch, collate, pad_batch
from mace_torch.neighbors import resolve_neighbor_provider
from mace_torch_neighbor_fixtures import DeviceProvider, register, require_cuda

TABLE = AtomicNumberTable([1])
POINTS = np.array([[0.0, 0.0, 0.0], [0.7, 0.2, 0.1], [0.3, 0.9, 0.4]])
TRICLINIC = np.array([[3.1, 0.2, 0.1], [0.9, 2.8, -0.2], [0.4, -0.3, 3.3]])
CASES = [
    (POINTS, 3.2, (True,) * 3, TRICLINIC),
    (np.zeros((1, 3)), 3.0, (True,) * 3, np.eye(3) * 1.5),
    (POINTS, 2.0, (False,) * 3, None),
    (POINTS, 3.5, (True, True, False), np.diag([3.0, 3.0, 12.0])),
    (POINTS, 3.5, (True, True, False), np.diag([3.0, 3.0, 0.0])),
    (POINTS, 0.01, (False,) * 3, None),
    (np.zeros((1, 3)), np.sqrt(2), (True,) * 3, np.eye(3)),
    (np.empty((0, 3)), 2.0, (False,) * 3, None),
]


def edge_set(neighbors):
    def as_tensor(value):
        return torch.tensor(value) if isinstance(value, np.ndarray) else value

    rows = torch.cat(
        (
            as_tensor(neighbors.edge_index).T,
            as_tensor(neighbors.unit_shifts),
        ),
        dim=1,
    )
    return sorted(map(tuple, rows.cpu().tolist()))


@pytest.fixture
def providers(tmp_path, monkeypatch):
    for name, cls in [
        ("device", "DeviceProvider"),
        ("batch", "BatchProvider"),
        ("stateful", "StatefulProvider"),
    ]:
        register(tmp_path, monkeypatch, name, f"mace_torch_neighbor_fixtures:{cls}")


@pytest.mark.parametrize("case", CASES)
def test_default_cpu_providers_match_the_oracle(providers, case):
    positions, cutoff, pbc, cell = case
    tested = set()
    for entry in available_neighbor_backends("torch"):
        if not entry.loaded:
            continue
        section = getattr(entry.factory, "settings_section", None)
        if section is not None and any(
            field.is_required() for field in section.model_fields.values()
        ):
            # Configured native CUDA libraries have dedicated device parity tests.
            continue
        backend = get_neighbor_backend(entry.name, framework="torch")
        capabilities = neighbor_capabilities(backend, entry.name)
        if entry.framework is not None and "cpu" not in capabilities.devices:
            continue
        tested.add(entry.name)
        provider = resolve_neighbor_provider(NeighborListConfig(kind=entry.name))
        for motion in [0.0, 0.03]:
            moved = positions.copy()
            if len(moved):
                moved[0, 0] += motion
            geometry = search_geometry(moved, cutoff, pbc, cell)
            expected = get_neighborhood(
                moved, cutoff, pbc, cell, backend=AllPairsBackend()
            )
            result = provider.search_batch(
                [torch.tensor(moved, dtype=torch.float64)], cutoff, [geometry]
            )[0]
            assert edge_set(result) == edge_set(expected), entry.name
            assert torch.equal(result.cell, torch.tensor(expected.cell)), entry.name
            if "sorted_by_sender" in provider.capabilities.features:
                assert torch.all(result.edge_index[0, 1:] >= result.edge_index[0, :-1])
    assert {"all_pairs", "vesin", "device", "batch", "stateful"} <= tested


@pytest.mark.parametrize("enabled", tuple(itertools.product((False, True), repeat=3)))
def test_provider_guarantees_are_materialized_after_collation(
    providers, monkeypatch, enabled
):
    features = frozenset(
        name
        for name, present in zip(sorted(EDGE_FEATURES), enabled, strict=True)
        if present
    )
    declared = dataclasses.replace(DeviceProvider().capabilities(), features=features)
    monkeypatch.setattr(DeviceProvider, "capabilities", lambda self: declared)
    provider = resolve_neighbor_provider(
        NeighborListConfig(kind="device"), required_features=features
    )
    assert not provider.edge_plan.adapters
    configs = [
        Configuration(atomic_numbers=np.ones(3, dtype=np.int64), positions=POINTS),
        Configuration(atomic_numbers=np.ones(1, dtype=np.int64), positions=POINTS[:1]),
    ]
    graph = build_batch(
        configs, atomic_number_table=TABLE, cutoff=2.0, neighbor_backend=provider
    )
    graph, _ = pad_batch(graph, AutoPadding(), 2.0, atomic_number_table=TABLE)
    validate_graph(graph)
    for row, side in enumerate(("sender", "receiver")):
        if f"{side}_runs" in features:
            assert torch.equal(
                graph[f"{side}_order"],
                torch.argsort(graph["edge_index"][row], stable=True),
            )
    if "sorted_by_sender" in features:
        assert torch.all(graph["edge_index"][0, 1:] >= graph["edge_index"][0, :-1])


def test_host_data_plus_device_edges_matches_host_edges(providers, monkeypatch):
    configs = [
        Configuration(
            atomic_numbers=np.ones(len(p), dtype=np.int64),
            positions=p,
            cell=c,
            pbc=(b[0], b[1], b[2]),
        )
        for p, _, b, c in CASES[:6]
    ]
    provider = resolve_neighbor_provider(NeighborListConfig(kind="batch"))
    device_pointers = set()
    search_batch = provider.backend.search_batch
    cpu = torch.Tensor.cpu

    def record(*args):
        results = search_batch(*args)
        device_pointers.update(
            value.data_ptr()
            for n in results
            for value in (n.edge_index, n.unit_shifts, n.shifts)
            if value.numel()
        )
        return results

    def guarded_cpu(value, *args, **kwargs):
        assert not value.numel() or value.data_ptr() not in device_pointers
        return cpu(value, *args, **kwargs)

    monkeypatch.setattr(provider.backend, "search_batch", record)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", guarded_cpu)
        actual = build_batch(
            configs, atomic_number_table=TABLE, cutoff=2.0, neighbor_backend=provider
        )
    assert provider.backend.batch_calls == 1
    # The same provider's output, explicitly copied to host, is the reference route.
    samples = []
    native = DeviceProvider()
    for config in configs:
        geometry = search_geometry(config.positions, 2.0, config.pbc, config.cell)
        result = native.search(torch.tensor(config.positions), 2.0, geometry)
        host = dataclasses.replace(
            result,
            **{
                field.name: getattr(result, field.name).numpy()
                for field in dataclasses.fields(result)
            },
        )
        samples.append(
            graph_sample(
                config, atomic_number_table=TABLE, cutoff=2.0, neighborhood=host
            )
        )
    expected = collate(samples)
    validate_graph(actual)
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key]


def test_batch_search_matches_single_search_including_order(providers):
    provider = resolve_neighbor_provider(NeighborListConfig(kind="batch"))
    geometries = [search_geometry(p, 2.0, b, c) for p, _, b, c in CASES[:5]]
    positions = [torch.tensor(p) for p, _, _, _ in CASES[:5]]
    batch = provider.search_batch(positions, 2.0, geometries)
    for points, geometry, actual in zip(positions, geometries, batch, strict=True):
        expected = provider.backend.search(points, 2.0, geometry)
        assert torch.equal(actual.edge_index, expected.edge_index)
        assert torch.equal(actual.unit_shifts, expected.unit_shifts)


def test_stateful_provider_matches_fresh_search_across_motion(providers):
    provider = resolve_neighbor_provider(NeighborListConfig(kind="stateful"))
    assert not hasattr(provider.backend, "search_batch")
    for motion in [0.0, 0.03, 0.05, 0.4, -0.2]:
        points = torch.tensor(POINTS, dtype=torch.float64)
        points[1, 0] += motion
        geometry = search_geometry(points.numpy(), 1.0, (True,) * 3, TRICLINIC)
        actual = provider.search_batch([points], 1.0, [geometry])[0]
        expected = DeviceProvider().search(points, 1.0, geometry)
        assert edge_set(actual) == edge_set(expected)
    assert provider.backend.rebuilds == 3
    with pytest.raises(ValueError, match="one instance per system"):
        provider.search_batch([points, points], 1.0, [geometry, geometry])


def test_named_provider_warns_once_at_resolution_default_does_not(providers, caplog):
    with caplog.at_level(logging.WARNING):
        provider = resolve_neighbor_provider(
            NeighborListConfig(kind="device"), required_features=EDGE_FEATURES
        )
        assert len(caplog.records) == 1
        assert "receiver_runs" in caplog.text and "sender_runs" in caplog.text
        for _ in range(2):
            provider.search_batch(
                [torch.tensor(POINTS)], 2.0, [search_geometry(POINTS, 2.0)]
            )
        assert len(caplog.records) == 1
        caplog.clear()
        resolve_neighbor_provider(required_features=EDGE_FEATURES)
        assert not caplog.records
    with pytest.raises(EdgeRequirementError, match=r"device.*invented"):
        resolve_neighbor_provider(
            NeighborListConfig(kind="device"),
            required_features=cast(Any, frozenset({"invented"})),
        )


def test_import_failure_is_recorded_and_named_resolution_raises(tmp_path, monkeypatch):
    register(tmp_path, monkeypatch, "missing_device", "no_such_neighbor_module:Backend")
    assert resolve_neighbor_provider().name == "vesin"
    entries = {item.name: item for item in available_neighbor_backends("torch")}
    assert not entries["missing_device"].loaded
    with pytest.raises(NeighborBackendNotAvailableError, match="missing_device"):
        resolve_neighbor_provider(NeighborListConfig(kind="missing_device"))


def test_default_provider_is_selected_per_device_with_explicit_names_preserved(
    tmp_path, monkeypatch
):
    register(
        tmp_path,
        monkeypatch,
        "nvalchemiops",
        "mace_torch_neighbor_fixtures:DeviceProvider",
    )
    monkeypatch.setattr(torch.version, "hip", None)
    assert resolve_neighbor_provider(device="cpu").name == "vesin"
    provider = resolve_neighbor_provider(device="cuda:0")
    assert provider.name == "nvalchemiops"
    assert "sorted_by_sender" in provider.capabilities.features
    assert (
        resolve_neighbor_provider(
            NeighborListConfig(kind="vesin"), device="cuda:0"
        ).name
        == "vesin"
    )
    assert resolve_neighbor_provider(device="cuda:0", default="vesin").name == "vesin"
    monkeypatch.setattr(torch.version, "hip", "7.0")
    assert resolve_neighbor_provider(device="cuda:0").name == "vesin"


def test_unavailable_cuda_default_is_reported_and_explicit_request_fails(
    tmp_path, monkeypatch, caplog
):
    register(tmp_path, monkeypatch, "nvalchemiops", "no_such_neighbor_module:Backend")
    monkeypatch.setattr(torch.version, "hip", None)
    with caplog.at_level(logging.INFO):
        assert resolve_neighbor_provider(device="cuda:0").name == "vesin"
    assert "CUDA neighbour default unavailable" in caplog.text
    with pytest.raises(NeighborBackendNotAvailableError, match="nvalchemiops"):
        resolve_neighbor_provider(
            NeighborListConfig(kind="nvalchemiops"), device="cuda:0"
        )


@pytest.mark.parametrize("field", ["edge_index", "unit_shifts", "cell", "shifts"])
def test_bad_provider_results_are_refused(providers, monkeypatch, field):
    provider = resolve_neighbor_provider(NeighborListConfig(kind="device"))
    geometry = search_geometry(POINTS, 2.0)
    result = DeviceProvider().search(torch.tensor(POINTS), 2.0, geometry)
    value = getattr(result, field).clone()
    if field == "edge_index":
        value[0, 0] = len(POINTS)
    elif field == "unit_shifts":
        value = value.to(torch.float64)
    else:
        value[0, 0] += 1.0
    result = dataclasses.replace(result, **{field: value})
    monkeypatch.setattr(provider.backend, "search", lambda *args: result)
    with pytest.raises(ValueError, match=field):
        provider.search_batch([torch.tensor(POINTS)], 2.0, [geometry])


@pytest.mark.gpu
def test_native_provider_and_padding_keep_edges_on_cuda(providers, monkeypatch):
    require_cuda()
    provider = resolve_neighbor_provider(
        NeighborListConfig(kind="device"),
        device="cuda",
        required_features=EDGE_FEATURES,
    )
    config = Configuration(atomic_numbers=np.ones(3, dtype=np.int64), positions=POINTS)
    cpu = torch.Tensor.cpu

    def guarded_cpu(value, *args, **kwargs):
        assert value.device.type == "cpu", "copied device edges to host"
        return cpu(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", guarded_cpu)
        graph = build_batch(
            [config], atomic_number_table=TABLE, cutoff=2.0, neighbor_backend=provider
        )
        padded, _ = pad_batch(graph, AutoPadding(), 2.0, atomic_number_table=TABLE)
        padded = apply_edge_plan(padded, provider.edge_plan)
    validate_graph(padded)
    assert padded["edge_index"].device.type == "cuda"
