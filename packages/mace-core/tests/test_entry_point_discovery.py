"""Registration conflicts must not depend on installation order or import code."""

from importlib.metadata import EntryPoint

import pytest
from mace_core.cli import registry as commands
from mace_core.data import registry as data
from mace_core.electrostatics import registry as solvers
from mace_core.kernels import registry as kernels
from mace_core.neighbors import backends as neighbors


@pytest.fixture(
    params=[
        (kernels, kernels.available_backends, kernels.DuplicateBackendError),
        (
            neighbors,
            neighbors.available_neighbor_backends,
            neighbors.DuplicateNeighborBackendError,
        ),
        (data, data.available_backends, data.DuplicateDataBackendError),
        (solvers, solvers.available_solvers, solvers.DuplicateSolverError),
    ],
    ids=["kernels", "neighbors", "data", "solvers"],
)
def registry(request):
    return request.param


def test_conflicting_registrations_fail_before_import_in_either_order(
    registry, monkeypatch
):
    module, discover, error = registry
    entries = [
        EntryPoint(name="collision", value=value, group="test")
        for value in ("first:Backend", "second:Backend")
    ]
    loaded = []
    monkeypatch.setattr(EntryPoint, "load", lambda entry: loaded.append(entry.value))
    messages = []
    for order in (entries, entries[::-1]):
        monkeypatch.setattr(module, "entry_points", lambda group, order=order: order)
        with pytest.raises(error) as failure:
            discover()
        messages.append(str(failure.value))
    assert messages[0] == messages[1]
    assert all(
        value in messages[0]
        for value in ("collision", "first:Backend", "second:Backend")
    )
    assert not loaded


def test_identical_registrations_load_once(registry, monkeypatch):
    module, discover, _ = registry
    entry = EntryPoint(name="same", value="provider:Backend", group="test")
    loaded = []
    factory = object()

    def load(entry):
        loaded.append(entry.value)
        return factory

    monkeypatch.setattr(module, "entry_points", lambda group: [entry, entry])
    monkeypatch.setattr(EntryPoint, "load", load)
    result = discover()
    records = list(result.values()) if isinstance(result, dict) else result
    assert len(records) == 1
    assert records[0].loaded and records[0].factory is factory
    assert loaded == ["provider:Backend"]


def test_commands_keep_their_stricter_duplicate_path_rule(monkeypatch):
    entry = EntryPoint(name="train", value="provider:Command", group="mace.commands")
    monkeypatch.setattr(commands, "entry_points", lambda group: [entry, entry])
    monkeypatch.setattr(EntryPoint, "load", lambda entry: None)
    with pytest.raises(commands.CommandConflictError, match="train"):
        commands.discover()
