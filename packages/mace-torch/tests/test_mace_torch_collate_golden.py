"""Collation into the flat graph dict, pinned against the frozen legacy collater.

The golden below is what the legacy vendored ``Batch.from_data_list`` produced
for three structures in one batch at a 2.9 Å cutoff: a water molecule (fully
aperiodic, so it reaches the extended search cell), a two-atom triclinic
crystal, and a slab whose vacuum row is all zero (so that row is patched). The
legacy package cannot be imported from here; the live comparison over the
committed fixtures is ``tests/parity/test_graph_parity.py``.

``ptr``, ``batch`` and ``shifts`` are compared bit for bit. Most shifts are a
single lattice vector, copied exactly; ``1.6400000000000001`` and
``2.7800000000000002`` are the rounded sums of two, so those rows also pin
that the shift is computed against the returned cell in float64.
"""

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.data import AtomicNumberTable, Configuration
from mace_core.graph import (
    GRAPH_SCHEMA,
    graph_sample,
    validate_graph,
)
from mace_core.graph import (
    collate as reference_collate,
)
from mace_core.neighbors import get_neighbor_backend
from mace_torch.graph import build_batch, collate, dtype_name

CUTOFF = 2.9
TABLE = AtomicNumberTable([1, 6, 8, 14])
MATSCIPY = get_neighbor_backend("matscipy")

STRUCTURES = [
    Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
    ),
    Configuration(
        atomic_numbers=np.array([14, 14]),
        positions=np.array([[0.0, 0.0, 0.0], [0.91, 0.73, 0.62]]),
        cell=np.array([[2.47, 0.0, 0.0], [0.83, 2.39, 0.0], [0.31, 0.57, 2.61]]),
        pbc=(True, True, True),
    ),
    Configuration(
        atomic_numbers=np.array([6, 8]),
        positions=np.array([[0.0, 0.0, 0.0], [1.43, 0.81, 1.17]]),
        cell=np.array([[2.83, 0.0, 0.0], [1.41, 2.44, 0.0], [0.0, 0.0, 0.0]]),
        pbc=(True, True, False),
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


@pytest.fixture(name="batch")
def fixture_batch(dtype):
    return build_batch(
        STRUCTURES,
        atomic_number_table=TABLE,
        cutoff=CUTOFF,
        dtype=dtype,
        neighbor_backend=MATSCIPY,
    )


def test_ptr_and_batch_are_the_legacy_collaters_bit_for_bit(batch):
    assert torch.equal(batch["ptr"], torch.tensor(PTR, dtype=torch.int64))
    assert torch.equal(batch["batch"], torch.tensor(BATCH, dtype=torch.int64))


def test_shifts_are_the_legacy_collaters_bit_for_bit(batch, dtype):
    expected = torch.tensor(SHIFTS, dtype=torch.float64).to(dtype)
    assert batch["shifts"].dtype == dtype
    assert torch.equal(batch["shifts"], expected)


def test_edge_index_is_offset_by_the_nodes_before_each_structure(batch):
    assert torch.equal(batch["edge_index"], torch.tensor(EDGE_INDEX))


def test_the_cell_is_the_legacy_one_stacked_rather_than_concatenated(batch, dtype):
    legacy = torch.tensor(LEGACY_CELL, dtype=torch.float64).to(dtype)
    assert batch["cell"].shape == (3, 3, 3)
    assert torch.equal(batch["cell"], legacy.view(-1, 3, 3))


def test_rcell_and_volume_follow_from_the_returned_cell(batch, dtype):
    cell = torch.tensor(LEGACY_CELL, dtype=torch.float64).to(dtype).view(-1, 3, 3)
    for graph in range(3):
        volume = torch.linalg.det(cell[graph])
        assert torch.equal(batch["volume"][graph], volume)
        expected = 2 * torch.pi * torch.linalg.inv(cell[graph].mT)
        assert torch.equal(batch["rcell"][graph], expected)


def test_rcell_is_exactly_zero_for_a_cell_of_zero_volume(dtype):
    sample = graph_sample(
        STRUCTURES[1],
        atomic_number_table=TABLE,
        cutoff=CUTOFF,
        neighbor_backend=MATSCIPY,
    )
    sample["cell"] = np.zeros((3, 3))
    batch = collate([sample], dtype=dtype)
    assert torch.equal(batch["volume"], torch.zeros(1, dtype=dtype))
    assert torch.equal(batch["rcell"], torch.zeros(1, 3, 3, dtype=dtype))


@fp64_only
def test_the_torch_collation_is_the_numpy_reference(dtype):
    samples = [
        graph_sample(
            structure,
            atomic_number_table=TABLE,
            cutoff=CUTOFF,
            neighbor_backend=MATSCIPY,
        )
        for structure in STRUCTURES
    ]
    batch = collate(samples, dtype=dtype)
    reference = reference_collate(samples)
    assert set(batch) == set(reference)
    for key, value in reference.items():
        if isinstance(value, tuple):
            assert batch[key] == value, key
        elif key in ("rcell", "volume"):
            # One LAPACK call in numpy, another in torch: equal to round-off.
            np.testing.assert_allclose(batch[key].numpy(), value, rtol=0, atol=1e-12)
        else:
            assert np.array_equal(batch[key].numpy(), value), key


def test_the_output_validates_at_the_bound_dtype(batch, dtype):
    info = validate_graph(batch, float_dtype=dtype_name(dtype))
    assert (info.num_graphs, info.num_nodes, info.num_edges) == (3, 7, 50)
    for key, spec in GRAPH_SCHEMA.items():
        if spec.dtype == "float" and key in batch:
            assert batch[key].dtype == dtype, key


def test_the_host_tuples_agree_with_ptr_and_pbc(batch):
    assert batch["structure_sizes"] == (3, 2, 2)
    assert batch["structure_periodic"] == (False, True, True)
    assert batch["structure_padding"] == (False, False, False)


def test_collation_does_not_modify_its_samples():
    samples = [
        graph_sample(
            structure,
            atomic_number_table=TABLE,
            cutoff=CUTOFF,
            neighbor_backend=MATSCIPY,
        )
        for structure in STRUCTURES
    ]
    before = [{k: np.array(v, copy=True) for k, v in s.items()} for s in samples]
    collate(samples)
    for sample, copy in zip(samples, before, strict=True):
        for key, value in copy.items():
            assert np.array_equal(sample[key], value), key


def test_collation_refuses_samples_with_different_keys():
    first = graph_sample(
        STRUCTURES[0],
        atomic_number_table=TABLE,
        cutoff=CUTOFF,
        neighbor_backend=MATSCIPY,
    )
    second = graph_sample(
        STRUCTURES[1],
        atomic_number_table=TABLE,
        cutoff=CUTOFF,
        neighbor_backend=MATSCIPY,
    )
    second["density_coefficients"] = np.zeros((2, 4))
    with pytest.raises(ValueError, match="carry different keys"):
        collate([first, second])


def test_an_undeclared_property_is_carried_and_never_promoted():
    structures = [
        Configuration(
            atomic_numbers=s.atomic_numbers,
            positions=s.positions,
            cell=s.cell,
            pbc=s.pbc,
            properties={"tags": np.arange(len(s), dtype=float)},
        )
        for s in STRUCTURES
    ]
    batch = build_batch(
        structures, atomic_number_table=TABLE, cutoff=CUTOFF, neighbor_backend=MATSCIPY
    )
    assert isinstance(batch["tags"], tuple)
    assert len(batch["tags"]) == 3
    for value, structure in zip(batch["tags"], STRUCTURES, strict=True):
        assert np.shape(value) == (len(structure),)


def test_a_declared_atom_feature_is_concatenated_as_given(dtype):
    structures = [
        Configuration(
            atomic_numbers=s.atomic_numbers,
            positions=s.positions,
            cell=s.cell,
            pbc=s.pbc,
            properties={"tags": np.arange(len(s), dtype=float)},
        )
        for s in STRUCTURES
    ]
    batch = build_batch(
        structures,
        atomic_number_table=TABLE,
        cutoff=CUTOFF,
        features={"tags": "atom"},
        dtype=dtype,
        neighbor_backend=MATSCIPY,
    )
    assert batch["tags"].shape == (7,)
    assert batch["tags"].dtype == dtype
    assert batch["tags"].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0]


def test_the_all_pairs_backend_builds_the_same_edges():
    def edge_set(batch):
        rows = torch.cat((batch["edge_index"].T, batch["unit_shifts"].long()), dim=1)
        return sorted(map(tuple, rows.tolist()))

    matscipy = build_batch(
        STRUCTURES, atomic_number_table=TABLE, cutoff=CUTOFF, neighbor_backend=MATSCIPY
    )
    all_pairs = build_batch(
        STRUCTURES,
        atomic_number_table=TABLE,
        cutoff=CUTOFF,
        neighbor_backend=get_neighbor_backend("all_pairs"),
    )
    assert edge_set(matscipy) == edge_set(all_pairs)
    assert torch.equal(matscipy["cell"], all_pairs["cell"])


def test_the_float_dtype_is_the_callers_and_not_the_process_default(dtype):
    """The suite runs under both default dtypes; the batch follows the argument."""
    batch = build_batch(
        STRUCTURES, atomic_number_table=TABLE, cutoff=CUTOFF, neighbor_backend=MATSCIPY
    )
    assert batch["positions"].dtype == torch.float64
    validate_graph(batch)


def test_samples_given_as_tensors_collate_like_numpy_ones(dtype):
    samples = [
        graph_sample(
            structure,
            atomic_number_table=TABLE,
            cutoff=CUTOFF,
            neighbor_backend=MATSCIPY,
        )
        for structure in STRUCTURES
    ]
    as_tensors = [
        {key: torch.as_tensor(value) for key, value in sample.items()}
        for sample in samples
    ]
    from_numpy = collate(samples, dtype=dtype)
    from_tensors = collate(as_tensors, dtype=dtype)
    for key, value in from_numpy.items():
        if isinstance(value, tuple):
            assert from_tensors[key] == value, key
        else:
            assert torch.equal(from_tensors[key], value), key


def test_a_process_default_device_does_not_move_the_batch(dtype):
    """The batch is assembled on the host whatever ``torch.set_default_device``
    says, and goes to a device only when ``device`` names one."""
    previous = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        batch = build_batch(
            STRUCTURES,
            atomic_number_table=TABLE,
            cutoff=CUTOFF,
            dtype=dtype,
            neighbor_backend=MATSCIPY,
        )
    finally:
        torch.set_default_device(previous)
    devices = {
        value.device.type for value in batch.values() if isinstance(value, torch.Tensor)
    }
    assert devices == {"cpu"}
    assert torch.equal(batch["ptr"], torch.tensor(PTR))
