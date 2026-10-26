"""Collation against the frozen legacy collater, on the committed fixtures.

``ptr``, ``batch`` and ``shifts`` must equal the legacy vendored
``Batch.from_data_list`` bit for bit, which is the data exit gate, and
``cell`` value for value once the legacy ``[3 * n_graphs, 3]`` concatenation
is viewed as ``[n_graphs, 3, 3]``.
"""

import json

import pytest
import torch
from mace_core.data import AtomicNumberTable, KeySpecification, configuration_from_atoms
from mace_core.neighbors import get_neighbor_backend
from mace_torch.graph import build_batch

from mace import data as legacy_data
from mace.tools import AtomicNumberTable as LegacyAtomicNumberTable
from mace.tools import torch_geometric
from tests.golden import harness


def _fixtures():
    with open(harness.MANIFEST_PATH, encoding="utf-8") as handle:
        r_max = json.load(handle)["r_max_hint"]
    return harness.load_fixtures(), r_max


FIXTURES, R_MAX = _fixtures()


@pytest.fixture(name="default_dtype", params=[torch.float64, torch.float32], ids=str)
def fixture_default_dtype(request):
    """The legacy builder reads the default dtype; v1 is handed it."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(previous)


@pytest.fixture(name="batches")
def fixture_batches(default_dtype):
    names = sorted(FIXTURES)
    atoms = [FIXTURES[name] for name in names]
    zs = sorted({int(z) for a in atoms for z in a.numbers})

    legacy_keys = legacy_data.KeySpecification(info_keys={}, arrays_keys={})
    legacy_graphs = [
        legacy_data.AtomicData.from_config(
            legacy_data.config_from_atoms(a, key_specification=legacy_keys),
            z_table=LegacyAtomicNumberTable(zs),
            cutoff=R_MAX,
        )
        for a in atoms
    ]
    legacy = torch_geometric.Batch.from_data_list(legacy_graphs)

    structures = [configuration_from_atoms(a, KeySpecification()) for a in atoms]
    batch = build_batch(
        structures,
        atomic_number_table=AtomicNumberTable(zs),
        cutoff=R_MAX,
        neighbor_backend=get_neighbor_backend("matscipy"),
        dtype=default_dtype,
    )
    return legacy, batch


@pytest.mark.parametrize("key", ["ptr", "batch", "shifts"])
def test_collate_is_bit_exact_against_the_legacy_collater(batches, key):
    legacy, batch = batches
    assert batch[key].dtype == legacy[key].dtype
    assert torch.equal(batch[key], legacy[key])


@pytest.mark.parametrize(
    "key", ["edge_index", "unit_shifts", "positions", "node_attrs", "pbc"]
)
def test_collate_reproduces_the_legacy_structure_keys(batches, key):
    legacy, batch = batches
    assert torch.equal(batch[key], legacy[key])


def test_collate_cell_is_the_legacy_one_viewed_per_graph(batches):
    legacy, batch = batches
    assert torch.equal(batch["cell"], legacy["cell"].view(-1, 3, 3))


def test_collate_rcell_and_volume_follow_legacy(batches):
    legacy, batch = batches
    torch.testing.assert_close(batch["volume"], legacy["volume"], rtol=0, atol=0)
    torch.testing.assert_close(
        batch["rcell"], legacy["rcell"].view(-1, 3, 3), rtol=0, atol=0
    )


def test_collate_atomic_numbers_are_the_legacy_ones(batches):
    legacy, batch = batches
    assert torch.equal(batch["node_atomic_numbers"], legacy["atomic_numbers"])
