"""Collation against the frozen legacy collater, on the committed fixtures.

``ptr``, ``batch`` and ``shifts`` must equal the legacy vendored
``Batch.from_data_list`` bit for bit, which is the data exit gate. ``cell`` is
the legacy one value for value, once the legacy ``[3 * n_graphs, 3]``
concatenation is viewed as ``[n_graphs, 3, 3]``, for every structure with a
periodic axis and no all-zero row; elsewhere legacy carries a box built from
the atom extent and v1 the parsed cell, a recorded difference. The element
one-hot is no longer a key: the model forms it from ``node_atomic_numbers``,
and doing so reproduces the legacy ``node_attrs``.
"""

import json

import pytest
import torch
from mace_core.data import KeySpecification, configuration_from_atoms
from mace_core.graph import graph_layout
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
    layout = graph_layout(
        zs, cutoff=R_MAX, float_dtype=str(default_dtype).removeprefix("torch.")
    )
    batch, _, _ = build_batch(
        structures, layout, neighbor_backend=get_neighbor_backend("matscipy")
    )
    return legacy, batch, zs


@pytest.mark.parametrize("key", ["ptr", "batch", "shifts"])
def test_collate_is_bit_exact_against_the_legacy_collater(batches, key):
    legacy, batch, _ = batches
    assert batch[key].dtype == legacy[key].dtype
    assert torch.equal(batch[key], legacy[key])


@pytest.mark.parametrize("key", ["edge_index", "unit_shifts", "positions", "pbc"])
def test_collate_reproduces_the_legacy_structure_keys(batches, key):
    legacy, batch, _ = batches
    assert torch.equal(batch[key], legacy[key])


def test_collate_cell_is_the_legacy_one_wherever_legacy_keeps_the_parsed_cell(
    batches,
):
    legacy, batch, _ = batches
    legacy_cells = legacy["cell"].view(-1, 3, 3)
    kept = batch["pbc"].any(dim=1) & batch["cell"].ne(0).any(dim=2).all(dim=1)
    assert kept.any()
    assert torch.equal(batch["cell"][kept], legacy_cells[kept])
    for index in torch.nonzero(~kept).flatten().tolist():
        assert not torch.equal(batch["cell"][index], legacy_cells[index]), index


def test_the_one_hot_formed_from_atomic_numbers_is_the_legacy_node_attrs(batches):
    legacy, batch, zs = batches
    index = torch.tensor([zs.index(int(z)) for z in batch["node_atomic_numbers"]])
    one_hot = torch.nn.functional.one_hot(index, len(zs)).to(legacy["node_attrs"].dtype)
    assert torch.equal(one_hot, legacy["node_attrs"])


def test_the_batch_carries_none_of_the_legacy_derived_cell_keys(batches):
    _, batch, _ = batches
    assert not {"rcell", "volume", "node_attrs"} & set(batch)


def test_collate_atomic_numbers_are_the_legacy_ones(batches):
    legacy, batch, _ = batches
    assert torch.equal(batch["node_atomic_numbers"], legacy["atomic_numbers"])
