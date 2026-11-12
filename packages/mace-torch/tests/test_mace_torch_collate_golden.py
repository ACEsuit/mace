"""Collation into the flat graph dict, pinned against the frozen legacy collater.

The golden below is what the legacy vendored ``Batch.from_data_list`` produced
for three structures in one batch at a 2.9 Å cutoff: a water molecule (fully
aperiodic), a two-atom triclinic crystal, and a slab whose vacuum row is all
zero. The legacy package cannot be imported from here; the live comparison over
the committed fixtures is ``tests/parity/test_collate_parity.py``.

``ptr``, ``batch`` and ``shifts`` are compared bit for bit. Most shifts are a
single lattice vector, copied exactly; ``1.6400000000000001`` and
``2.7800000000000002`` are the rounded sums of two, so those rows also pin that
the shift is computed in float64. The shifts do not change with the cell the
graph carries: unit shifts are zero along every aperiodic axis, so the rows
legacy extends or patches never enter one.

``cell`` is the parsed cell. Legacy carried the extended search box for the
molecule and patched the slab's zero row; both are recorded differences.
"""

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.data import Configuration
from mace_core.graph import (
    GraphInfo,
    GraphLayout,
    check_batch_invariants,
    check_layout,
    graph_layout,
    graph_sample,
)
from mace_core.graph import collate as reference_collate
from mace_core.neighbors import get_neighbor_backend, get_neighborhood
from mace_core.observables import InputSpec, ObservableSpec
from mace_torch.graph import build_batch, collate, collate_targets

CUTOFF = 2.9
ELEMENTS = [1, 6, 8, 14]
INPUTS = [InputSpec(name="magmom", irreps="1e", per_atom=True, units="μB")]
TARGETS = [
    ObservableSpec(name="energy", irreps="0e", per_atom=False, units="eV"),
    ObservableSpec(name="forces", irreps="1o", per_atom=True, units="eV/Å"),
    ObservableSpec(name="stress", irreps="0e+2e", per_atom=False, units="eV/Å^3"),
    ObservableSpec(name="dipole", irreps="1o", per_atom=False, units="e·Å"),
]
LAYOUT = graph_layout(ELEMENTS, cutoff=CUTOFF, inputs=INPUTS, targets=TARGETS)
MATSCIPY = get_neighbor_backend("matscipy")


def _unlabelled(**given):
    """Every name the layout declares, absent unless given."""
    return {**{name: None for name in (*LAYOUT.inputs, *LAYOUT.targets)}, **given}


def _layout(dtype, layout=LAYOUT):
    """``layout`` at the floating dtype of the test, validated anew."""
    record = {**layout.model_dump(), "float_dtype": str(dtype).removeprefix("torch.")}
    return GraphLayout.model_validate(record)


STRUCTURES = [
    Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        properties=_unlabelled(),
    ),
    Configuration(
        atomic_numbers=np.array([14, 14]),
        positions=np.array([[0.0, 0.0, 0.0], [0.91, 0.73, 0.62]]),
        cell=np.array([[2.47, 0.0, 0.0], [0.83, 2.39, 0.0], [0.31, 0.57, 2.61]]),
        pbc=(True, True, True),
        properties=_unlabelled(),
    ),
    Configuration(
        atomic_numbers=np.array([6, 8]),
        positions=np.array([[0.0, 0.0, 0.0], [1.43, 0.81, 1.17]]),
        cell=np.array([[2.83, 0.0, 0.0], [1.41, 2.44, 0.0], [0.0, 0.0, 0.0]]),
        pbc=(True, True, False),
        properties=_unlabelled(),
    ),
]

PTR = [0, 3, 5, 7]
BATCH = [0, 0, 0, 1, 1, 2, 2]
EDGE_INDEX = [
    [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
     4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    [1, 2, 0, 2, 0, 1, 4, 3, 4, 3, 4, 3, 3, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3,
     4, 4, 3, 4, 3, 4, 3, 5, 6, 5, 5, 6, 6, 5, 5, 5, 6, 6, 6, 5, 5, 6, 6, 5, 6],
]  # fmt: skip
SHIFTS = [
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (-2.7800000000000002, -0.57, -2.61),
    (-0.31, -0.57, -2.61),
    (-0.31, -0.57, -2.61),
    (-0.83, -2.39, 0.0),
    (-0.83, -2.39, 0.0),
    (1.6400000000000001, -2.39, 0.0),
    (-2.47, 0.0, 0.0),
    (-2.47, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (2.47, 0.0, 0.0),
    (-1.6400000000000001, 2.39, 0.0),
    (0.83, 2.39, 0.0),
    (0.31, 0.57, 2.61),
    (-0.31, -0.57, -2.61),
    (-0.83, -2.39, 0.0),
    (1.6400000000000001, -2.39, 0.0),
    (-2.47, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (2.47, 0.0, 0.0),
    (2.47, 0.0, 0.0),
    (-1.6400000000000001, 2.39, 0.0),
    (0.83, 2.39, 0.0),
    (0.83, 2.39, 0.0),
    (0.31, 0.57, 2.61),
    (0.31, 0.57, 2.61),
    (2.7800000000000002, 0.57, 2.61),
    (-1.41, -2.44, 0.0),
    (-1.41, -2.44, 0.0),
    (1.4200000000000002, -2.44, 0.0),
    (-2.83, 0.0, 0.0),
    (-2.83, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (2.83, 0.0, 0.0),
    (-1.4200000000000002, 2.44, 0.0),
    (1.41, 2.44, 0.0),
    (-1.41, -2.44, 0.0),
    (1.4200000000000002, -2.44, 0.0),
    (-2.83, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (2.83, 0.0, 0.0),
    (2.83, 0.0, 0.0),
    (-1.4200000000000002, 2.44, 0.0),
    (1.41, 2.44, 0.0),
    (1.41, 2.44, 0.0),
]
#: The legacy collated cell, ``[3 * n_graphs, 3]``: the extended search cell of
#: the molecule, the physical triclinic cell, and the slab with its zero row
#: patched from the search cell.
LEGACY_CELL = [
    [8.0, 0.0, 0.0],
    [0.0, 7.7299999999999995, 0.0],
    [0.0, 0.0, 6.8],
    [2.47, 0.0, 0.0],
    [0.83, 2.39, 0.0],
    [0.31, 0.57, 2.61],
    [2.83, 0.0, 0.0],
    [1.41, 2.44, 0.0],
    [0.0, 0.0, 7.97],
]


#: The parsed cells the graph carries.
PARSED_CELL = [
    np.zeros((3, 3)),
    np.array(STRUCTURES[1].cell),
    np.array(STRUCTURES[2].cell),
]


def _search(structure):
    return get_neighborhood(
        structure.positions,
        CUTOFF,
        pbc=structure.pbc,
        cell=structure.cell,
        backend=MATSCIPY,
    )


def _samples(structures=STRUCTURES, layout=LAYOUT):
    return [
        graph_sample(structure, _search(structure), layout) for structure in structures
    ]


@pytest.fixture(name="batch")
def fixture_batch(dtype):
    graph, _, _ = build_batch(STRUCTURES, _layout(dtype), neighbor_backend=MATSCIPY)
    return graph


def test_ptr_and_batch_are_the_legacy_collaters_bit_for_bit(batch):
    assert torch.equal(batch["ptr"], torch.tensor(PTR, dtype=torch.int64))
    assert torch.equal(batch["batch"], torch.tensor(BATCH, dtype=torch.int64))


def test_shifts_are_the_legacy_collaters_bit_for_bit(batch, dtype):
    expected = torch.tensor(SHIFTS, dtype=torch.float64).to(dtype)
    assert batch["shifts"].dtype == dtype
    assert torch.equal(batch["shifts"], expected)


def test_edge_index_is_offset_by_the_nodes_before_each_structure(batch):
    assert torch.equal(batch["edge_index"], torch.tensor(EDGE_INDEX))


def test_the_cell_is_the_parsed_one_stacked_per_structure(batch, dtype):
    expected = torch.tensor(np.stack(PARSED_CELL), dtype=torch.float64).to(dtype)
    assert batch["cell"].shape == (3, 3, 3)
    assert torch.equal(batch["cell"], expected)


def test_the_crystal_cell_is_the_legacy_one_and_the_other_two_are_recorded_drops(
    batch, dtype
):
    legacy = torch.tensor(LEGACY_CELL, dtype=torch.float64).to(dtype).view(-1, 3, 3)
    assert torch.equal(batch["cell"][1], legacy[1])
    assert not torch.equal(batch["cell"][0], legacy[0])
    assert torch.equal(batch["cell"][2][:2], legacy[2][:2])
    assert not batch["cell"][2][2].any()


def test_every_value_of_the_dict_is_a_tensor(batch):
    assert all(isinstance(value, torch.Tensor) for value in batch.values())


def test_the_info_holds_the_sizes_beside_the_dict(dtype):
    _, _, info = build_batch(STRUCTURES, _layout(dtype), neighbor_backend=MATSCIPY)
    assert info == GraphInfo(
        num_graphs=3, num_nodes=7, num_edges=50, structure_sizes=(3, 2, 2)
    )


@fp64_only
def test_the_torch_collation_is_the_numpy_reference(dtype):
    graphs = [graph for graph, _ in _samples()]
    batch, info = collate(graphs, _layout(dtype))
    reference, reference_info = reference_collate(graphs, LAYOUT)
    assert info == reference_info
    assert set(batch) == set(reference)
    for key, value in reference.items():
        assert np.array_equal(batch[key].numpy(), value), key


def test_the_output_passes_the_layout_check_at_the_bound_dtype(dtype):
    batch, targets, info = build_batch(
        STRUCTURES, _layout(dtype), neighbor_backend=MATSCIPY
    )
    check_layout(batch, _layout(dtype), info)
    check_batch_invariants(batch, info)
    for key, spec in LAYOUT.graph_fields().items():
        if spec.dtype == "float" and key in batch:
            assert batch[key].dtype == dtype, key
    assert targets["energy"].dtype == dtype


def test_the_targets_come_beside_the_graph_and_never_in_it(dtype):
    labelled = [
        Configuration(
            atomic_numbers=s.atomic_numbers,
            positions=s.positions,
            cell=s.cell,
            pbc=s.pbc,
            properties=_unlabelled(energy=-float(index), forces=np.ones((len(s), 3))),
        )
        for index, s in enumerate(STRUCTURES)
    ]
    graph, targets, _ = build_batch(labelled, _layout(dtype), neighbor_backend=MATSCIPY)
    assert not set(LAYOUT.target_fields()) & set(graph)
    assert targets["energy"].tolist() == [0.0, -1.0, -2.0]
    assert targets["forces"].shape == (7, 3)
    assert targets["energy_present"].tolist() == [True, True, True]
    assert targets["stress_present"].tolist() == [False, False, False]
    assert targets["stress_weight"].tolist() == [0.0, 0.0, 0.0]


def test_collate_targets_follows_the_graph_order(dtype):
    pairs = _samples()
    targets = collate_targets([t for _, t in pairs], _layout(dtype))
    assert targets["forces_present"].shape == (7,)
    assert targets["dipole_weight"].shape == (3, 3)


def test_collation_does_not_modify_its_samples():
    graphs = [graph for graph, _ in _samples()]
    before = [{k: np.array(v, copy=True) for k, v in s.items()} for s in graphs]
    collate(graphs, LAYOUT)
    for sample, copy in zip(graphs, before, strict=True):
        for key, value in copy.items():
            assert np.array_equal(sample[key], value), key


def test_collation_refuses_samples_with_different_keys():
    first, second = (graph for graph, _ in _samples(STRUCTURES[:2]))
    del second["magmom"]
    with pytest.raises(ValueError, match="carry different keys"):
        collate([first, second], LAYOUT)


def test_collation_refuses_a_key_the_layout_does_not_declare():
    graphs = [graph for graph, _ in _samples(STRUCTURES[:1])]
    graphs[0]["tags"] = np.arange(3.0)
    with pytest.raises(ValueError, match="does not declare"):
        collate(graphs, LAYOUT)


def test_an_undeclared_property_never_reaches_the_batch():
    structures = [
        Configuration(
            atomic_numbers=s.atomic_numbers,
            positions=s.positions,
            cell=s.cell,
            pbc=s.pbc,
            properties=_unlabelled(tags=np.arange(len(s), dtype=float)),
        )
        for s in STRUCTURES
    ]
    graph, targets, _ = build_batch(structures, LAYOUT, neighbor_backend=MATSCIPY)
    assert "tags" not in graph
    assert "tags" not in targets


def test_a_declared_atom_feature_is_concatenated_as_given(dtype):
    structures = [
        Configuration(
            atomic_numbers=s.atomic_numbers,
            positions=s.positions,
            cell=s.cell,
            pbc=s.pbc,
            properties=_unlabelled(tags=np.arange(len(s), dtype=float)),
        )
        for s in STRUCTURES
    ]
    layout = graph_layout(
        ELEMENTS,
        cutoff=CUTOFF,
        features={"tags": "atom"},
        inputs=INPUTS,
        targets=TARGETS,
    )
    graph, _, _ = build_batch(
        structures, _layout(dtype, layout), neighbor_backend=MATSCIPY
    )
    assert graph["tags"].shape == (7,)
    assert graph["tags"].dtype == dtype
    assert graph["tags"].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0]


def test_samples_without_edges_collate_into_a_batch_a_device_search_completes(dtype):
    graphs = [graph_sample(s, None, LAYOUT)[0] for s in STRUCTURES]
    batch, info = collate(graphs, _layout(dtype))
    assert not {"edge_index", "shifts", "unit_shifts"} & set(batch)
    assert info.num_edges == 0
    assert torch.equal(batch["ptr"], torch.tensor(PTR))


def test_the_all_pairs_backend_builds_the_same_edges():
    def edge_set(batch):
        rows = torch.cat((batch["edge_index"].T, batch["unit_shifts"].long()), dim=1)
        return sorted(map(tuple, rows.tolist()))

    matscipy, _, _ = build_batch(STRUCTURES, LAYOUT, neighbor_backend=MATSCIPY)
    all_pairs, _, _ = build_batch(
        STRUCTURES, LAYOUT, neighbor_backend=get_neighbor_backend("all_pairs")
    )
    assert edge_set(matscipy) == edge_set(all_pairs)
    assert torch.equal(matscipy["cell"], all_pairs["cell"])


def test_the_float_dtype_is_the_layouts_and_not_the_process_default(dtype):
    """The suite runs under both default dtypes; the batch follows the layout."""
    batch, _, info = build_batch(STRUCTURES, LAYOUT, neighbor_backend=MATSCIPY)
    assert batch["positions"].dtype == torch.float64
    check_layout(batch, LAYOUT, info)


def test_samples_given_as_tensors_collate_like_numpy_ones(dtype):
    graphs = [graph for graph, _ in _samples()]
    as_tensors = [
        {key: torch.as_tensor(value) for key, value in sample.items()}
        for sample in graphs
    ]
    from_numpy, _ = collate(graphs, _layout(dtype))
    from_tensors, _ = collate(as_tensors, _layout(dtype))
    for key, value in from_numpy.items():
        assert torch.equal(from_tensors[key], value), key


def test_a_process_default_device_does_not_move_the_batch(dtype):
    """The batch is assembled on the host whatever ``torch.set_default_device``
    says, and goes to a device only when ``device`` names one."""
    previous = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        batch, targets, _ = build_batch(
            STRUCTURES, _layout(dtype), neighbor_backend=MATSCIPY
        )
    finally:
        torch.set_default_device(previous)
    devices = {value.device.type for value in [*batch.values(), *targets.values()]}
    assert devices == {"cpu"}
    assert torch.equal(batch["ptr"], torch.tensor(PTR))
