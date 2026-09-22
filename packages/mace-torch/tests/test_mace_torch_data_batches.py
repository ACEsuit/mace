"""From structures to batches, and what a batch has to satisfy.

The properties worth pinning are the ones whose failure is silent: a graph and
its labels drifting out of alignment, a per-graph field joined as if it were
per-atom, and a declared observable the data does not carry.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.observables import load_default_catalogue, resolve_requested
from mace_torch.data import (
    GraphDataset,
    MissingTargetError,
    collate_training,
    graph_from_configuration,
    make_loader,
    target_specs,
    targets_from_configuration,
)


def field(batch, name) -> torch.Tensor:
    """One graph field as a tensor. `num_graphs` is the one that is not."""
    value = batch.graph[name]
    assert isinstance(value, torch.Tensor)
    return value


CATALOGUE = load_default_catalogue()
REQUESTED = resolve_requested(["energy", "forces"], CATALOGUE)
SPECS = target_specs(REQUESTED)
Z_TABLE = AtomicNumberTable([1, 8])
CUTOFF = 5.0

WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def water(seed: int = 0, head: str = "default") -> Configuration:
    generator = np.random.default_rng(seed)
    return Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=WATER + generator.normal(scale=0.01, size=(3, 3)),
        properties={
            "energy": -2067.0 + 0.1 * seed,
            "forces": generator.normal(size=(3, 3)),
        },
        head=head,
    )


def test_a_structure_becomes_the_fields_the_schema_declares():
    graph = graph_from_configuration(water(), cutoff=CUTOFF, z_table=Z_TABLE)
    assert set(graph) == {
        "positions",
        "atomic_numbers",
        "edge_index",
        "shifts",
        "unit_shifts",
        "cell",
        "pbc",
        "weight",
        "head",
    }


def test_the_per_graph_fields_carry_no_leading_axis_before_collation():
    """The collation stacks them, so a leading one would make a batch of
    shape [n, 1] that indexes as if every graph had one of everything."""
    graph = graph_from_configuration(water(), cutoff=CUTOFF, z_table=Z_TABLE)
    assert graph["cell"].shape == (3, 3)
    assert graph["pbc"].shape == (3,)
    assert graph["head"].shape == ()


def test_an_element_the_model_was_not_built_for_is_refused_at_the_graph():
    """Otherwise it indexes out of the one-hot at the first forward, where
    nothing names the structure it came from."""
    carbon = Configuration(
        atomic_numbers=np.array([6]),
        positions=np.zeros((1, 3)),
        properties={"energy": 0.0, "forces": np.zeros((1, 3))},
    )
    with pytest.raises(ValueError, match=r"\[6\]"):
        graph_from_configuration(carbon, cutoff=CUTOFF, z_table=Z_TABLE)


def test_a_declared_observable_the_data_lacks_is_refused():
    unlabelled = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=WATER,
        properties={"energy": -2067.0},
    )
    with pytest.raises(MissingTargetError, match="forces"):
        targets_from_configuration(unlabelled, SPECS)


def test_a_batch_joins_the_labels_the_way_it_joins_the_nodes():
    """One energy per structure, one force per atom, in the same order."""
    dataset = GraphDataset(
        [water(0), water(1)], cutoff=CUTOFF, z_table=Z_TABLE, targets=SPECS
    )
    batch = collate_training([dataset[0], dataset[1]], z_table=Z_TABLE)
    assert batch.targets["energy"].shape == (2,)
    assert batch.targets["forces"].shape == (6, 3)
    assert batch.graph["num_graphs"] == 2


def test_the_labels_stay_with_their_own_structure():
    """The failure this guards is a run that trains on shuffled labels and
    reports a loss that simply stops going down."""
    first, second = water(0), water(1)
    dataset = GraphDataset(
        [first, second], cutoff=CUTOFF, z_table=Z_TABLE, targets=SPECS
    )
    batch = collate_training([dataset[0], dataset[1]], z_table=Z_TABLE)
    assert batch.targets["energy"][0].item() == pytest.approx(
        first.properties["energy"]
    )
    assert torch.allclose(
        batch.targets["forces"][3:],
        torch.as_tensor(second.properties["forces"], dtype=torch.float64),
    )


def test_the_element_index_is_the_position_in_the_model_s_table():
    dataset = GraphDataset([water()], cutoff=CUTOFF, z_table=Z_TABLE, targets=SPECS)
    batch = collate_training([dataset[0]], z_table=Z_TABLE)
    assert field(batch, "element_index").tolist() == [1, 0, 0]


def test_a_head_name_becomes_its_position_in_the_model_s_head_list():
    dataset = GraphDataset(
        [water(head="dft"), water(head="ccsd")],
        cutoff=CUTOFF,
        z_table=Z_TABLE,
        targets=SPECS,
        heads=("dft", "ccsd"),
    )
    batch = collate_training([dataset[0], dataset[1]], z_table=Z_TABLE)
    assert field(batch, "head").tolist() == [0, 1]


def test_the_graph_count_comes_from_ptr_and_is_a_plain_int():
    """A tensor here is a host read in every reduction that uses it."""
    dataset = GraphDataset(
        [water(0), water(1), water(2)], cutoff=CUTOFF, z_table=Z_TABLE, targets=SPECS
    )
    batch = collate_training([dataset[i] for i in range(3)], z_table=Z_TABLE)
    assert isinstance(batch.graph["num_graphs"], int)
    assert batch.graph["num_graphs"] == 3


def test_a_loader_yields_batches_of_the_asked_for_size():
    dataset = GraphDataset(
        [water(i) for i in range(5)], cutoff=CUTOFF, z_table=Z_TABLE, targets=SPECS
    )
    loader = make_loader(dataset, batch_size=2, shuffle=False)
    sizes = [batch.graph["num_graphs"] for batch in loader]
    assert sizes == [2, 2, 1]
