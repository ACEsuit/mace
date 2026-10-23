"""The static-shape padding contract: a padded batch is a valid batch.

The cases of the legacy ``tests/unit/test_padding.py`` are ported here as
properties of :func:`pad_batch` and :func:`unpad_outputs`: the padding atoms
take the first element, the padding cell is ``eye(3) * max(2 * r_max, 1)`` and
not periodic, padding edges stay in bounds, zero padding edges is legal, the
padded batch keeps the real structures first, and every per-atom output is
stripped of its padding rows.
"""

import numpy as np
import pytest
import torch
from mace_core.data import AtomicNumberTable, Configuration
from mace_core.graph import (
    AutoPadding,
    FixedPadding,
    NoPadding,
    PaddingInfo,
    PaddingOverflowError,
    validate_graph,
)
from mace_core.observables import (
    DEFAULT_CATALOGUE,
    DerivativeRequest,
    InputSpec,
    ObservableCatalogue,
    ObservableSpec,
)
from mace_core.outputs import MACEOutput
from mace_torch.backends.radial import polynomial_envelope
from mace_torch.graph import build_batch, dtype_name, pad_batch, unpad_outputs
from mace_torch.graph.collate import derived_cell_quantities

R_MAX = 3.5
TABLE = AtomicNumberTable([1, 8])


def _water(offset=0.0, periodic=False):
    return Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        + offset,
        cell=np.eye(3) * 6.0 if periodic else None,
        pbc=(True, True, True) if periodic else None,
        properties={"energy": -1.0, "forces": np.ones((3, 3)), "total_spin": None},
        property_weights={"energy": 1.0, "forces": 1.0},
    )


@pytest.fixture(name="batch")
def fixture_batch(dtype):
    return build_batch(
        [_water(periodic=True), _water(0.1)],
        atomic_number_table=TABLE,
        cutoff=R_MAX,
        dtype=dtype,
    )


@pytest.fixture(name="padded")
def fixture_padded(batch):
    policy = FixedPadding(nodes_budget=10, edges_budget=64, graphs_budget=4)
    return pad_batch(batch, policy, R_MAX, atomic_number_table=TABLE)


def test_the_padded_batch_passes_validation_unchanged(padded, dtype):
    data, _ = padded
    info = validate_graph(data, float_dtype=dtype_name(dtype))
    assert (info.num_nodes, info.num_edges, info.num_graphs) == (10, 64, 4)


def test_the_padding_info_holds_the_real_sizes(batch, padded):
    _, info = padded
    assert info == PaddingInfo(
        n_real_nodes=6,
        n_real_edges=int(batch["edge_index"].shape[1]),
        n_real_graphs=2,
    )


def test_padding_only_appends(batch, padded):
    data, info = padded
    for key, value in batch.items():
        if isinstance(value, tuple):
            assert data[key][: info.n_real_graphs] == value, key
        elif key == "edge_index":
            assert torch.equal(data[key][:, : info.n_real_edges], value)
        elif key == "ptr":
            assert torch.equal(data[key][: info.n_real_graphs + 1], value)
        else:
            assert torch.equal(data[key][: value.shape[0]], value), key


def test_padding_atoms_belong_to_the_last_structure(padded):
    data, info = padded
    assert data["batch"][info.n_real_nodes :].tolist() == [3] * 4
    assert data["ptr"].tolist() == [0, 3, 6, 6, 10]
    assert data["structure_sizes"] == (3, 3, 0, 4)
    assert data["structure_padding"] == (False, False, True, True)
    assert data["structure_periodic"] == (True, False, False, False)


def test_padding_atoms_take_the_first_element(padded):
    data, info = padded
    one_hot = data["node_attrs"][info.n_real_nodes :]
    assert torch.equal(one_hot[:, 0], torch.ones(4, dtype=one_hot.dtype))
    assert not one_hot[:, 1:].any()
    assert data["node_atomic_numbers"][info.n_real_nodes :].tolist() == [1] * 4
    assert not data["positions"][info.n_real_nodes :].any()


def test_the_padding_cell_is_scaled_identity_and_not_periodic(padded, dtype):
    data, info = padded
    expected = torch.eye(3, dtype=dtype) * max(2 * R_MAX, 1.0)
    for graph in range(info.n_real_graphs, 4):
        assert torch.equal(data["cell"][graph], expected)
        assert not data["pbc"][graph].any()
        assert data["volume"][graph] == (2 * R_MAX) ** 3


def test_the_padding_rcell_and_volume_are_derived_from_the_padding_cell(padded):
    """As for any structure, so a padded batch is a valid batch here too."""
    data, _ = padded
    rcell, volume = derived_cell_quantities(data["cell"])
    assert torch.equal(data["rcell"], rcell)
    assert torch.equal(data["volume"], volume)


def test_every_structure_has_a_finite_reciprocal_space_extent(padded):
    """The reciprocal-space grid is sized per lattice direction as
    ``cutoff / (rcell . a / |a|)`` and then maximised over the batch, so one
    structure with a zero ``rcell`` makes the grid of every structure
    unbounded."""
    data, _ = padded
    cell = data["cell"]
    directions = cell / torch.linalg.norm(cell, dim=-1, keepdim=True)
    spacing = torch.einsum("gij,gij->gi", data["rcell"], directions)
    assert torch.all(spacing > 0)
    assert torch.all(torch.isfinite(6.0 / spacing))


def test_padding_edges_are_self_loops_on_the_last_padding_atom(padded):
    data, info = padded
    padding_edges = data["edge_index"][:, info.n_real_edges :]
    assert padding_edges.unique().tolist() == [9]
    assert data["edge_index"].max() < 10
    assert data["edge_index"].min() >= 0


def test_the_cutoff_is_exactly_zero_on_a_padding_edge(padded, dtype):
    """An exact float fact, not a tolerance: the edge is 2 * r_max long and the
    envelope is masked to zero at r >= r_max."""
    data, info = padded
    sender, receiver = data["edge_index"][:, info.n_real_edges :]
    vectors = (
        data["positions"][receiver]
        - data["positions"][sender]
        + data["shifts"][info.n_real_edges :]
    )
    lengths = torch.linalg.norm(vectors, dim=-1)
    assert torch.equal(lengths, torch.full_like(lengths, 2 * R_MAX))
    envelope = polynomial_envelope(lengths, torch.tensor(R_MAX, dtype=dtype), 6)
    assert torch.equal(envelope, torch.zeros_like(envelope))
    unit_shifts = data["unit_shifts"][info.n_real_edges :]
    assert torch.equal(
        unit_shifts, torch.tensor([[1.0, 0.0, 0.0]] * len(unit_shifts), dtype=dtype)
    )


def test_padding_shifts_are_unit_shifts_times_the_padding_cell(padded):
    data, info = padded
    padding_shifts = data["shifts"][info.n_real_edges :]
    padding_cell = data["cell"][-1]
    assert torch.equal(
        padding_shifts, data["unit_shifts"][info.n_real_edges :] @ padding_cell
    )


def test_total_spin_pads_to_one_and_labels_to_absent(padded):
    data, info = padded
    assert data["total_spin"][info.n_real_graphs :].tolist() == [1.0, 1.0]
    assert not data["energy_present"][info.n_real_graphs :].any()
    assert not data["forces_present"][info.n_real_nodes :].any()
    assert not data["energy_weight"][info.n_real_graphs :].any()
    assert not data["weight"][info.n_real_graphs :].any()


def test_no_mask_tensor_appears_in_the_padded_dict(batch, padded):
    data, _ = padded
    assert set(data) == set(batch)


def test_zero_padding_edges_is_legal(batch, dtype):
    n_edges = int(batch["edge_index"].shape[1])
    policy = FixedPadding(nodes_budget=9, edges_budget=n_edges, graphs_budget=3)
    data, info = pad_batch(batch, policy, R_MAX, atomic_number_table=TABLE)
    assert data["edge_index"].shape[1] == info.n_real_edges
    assert data["node_attrs"].shape[0] == 9
    validate_graph(data, float_dtype=dtype_name(dtype))


def test_an_edge_budget_with_no_room_for_a_padding_atom_grows_the_nodes(batch, dtype):
    policy = FixedPadding(nodes_budget=6, edges_budget=64, graphs_budget=2)
    data, info = pad_batch(batch, policy, R_MAX, atomic_number_table=TABLE)
    assert data["positions"].shape[0] == info.n_real_nodes + 1
    assert data["ptr"].shape[0] == info.n_real_graphs + 2
    validate_graph(data, float_dtype=dtype_name(dtype))


def test_an_overflowing_fixed_budget_can_refuse_instead(batch):
    policy = FixedPadding(
        nodes_budget=4, edges_budget=64, graphs_budget=4, on_overflow="error"
    )
    with pytest.raises(PaddingOverflowError, match="does not fit"):
        pad_batch(batch, policy, R_MAX, atomic_number_table=TABLE)


def test_auto_padding_rounds_the_edges_up_with_headroom(batch, dtype):
    n_edges = int(batch["edge_index"].shape[1])
    data, _ = pad_batch(batch, AutoPadding(), R_MAX, atomic_number_table=TABLE)
    expected = -(-int(n_edges * 1.25) // 64) * 64
    assert data["edge_index"].shape[1] == expected
    assert data["positions"].shape[0] == 7
    validate_graph(data, float_dtype=dtype_name(dtype))


def test_no_padding_returns_the_batch_as_it_is(batch):
    data, info = pad_batch(batch, NoPadding(), R_MAX, atomic_number_table=TABLE)
    assert data.keys() == batch.keys()
    assert all(data[k] is batch[k] for k in batch)
    assert info.n_real_graphs == 2


def test_padding_does_not_modify_the_batch(batch):
    before = {k: v.clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    pad_batch(
        batch,
        FixedPadding(nodes_budget=10, edges_budget=64, graphs_budget=4),
        R_MAX,
        atomic_number_table=TABLE,
    )
    for key, value in before.items():
        assert torch.equal(batch[key], value), key


def test_a_padded_cell_scale_has_a_floor_of_one(dtype):
    batch = build_batch([_water()], atomic_number_table=TABLE, cutoff=0.3, dtype=dtype)
    data, _ = pad_batch(
        batch,
        FixedPadding(nodes_budget=4, edges_budget=2, graphs_budget=2),
        0.3,
        atomic_number_table=TABLE,
    )
    assert torch.equal(data["cell"][-1], torch.eye(3, dtype=data["cell"].dtype))


def test_unpad_after_pad_is_the_identity_on_real_slots(batch, padded):
    data, info = padded
    outputs = {
        "energy": data["energy"] * 2,
        "forces": data["forces"] * 3,
        "stress": torch.zeros(4, 3, 3, dtype=data["energy"].dtype),
    }
    real = unpad_outputs(outputs, info)
    assert torch.equal(real["energy"], batch["energy"] * 2)
    assert torch.equal(real["forces"], batch["forces"] * 3)
    assert real["stress"].shape == (2, 3, 3)


def test_every_per_atom_output_is_stripped_of_its_padding_rows():
    info = PaddingInfo(n_real_nodes=4, n_real_edges=0, n_real_graphs=1)
    output = MACEOutput(
        total_energy=torch.zeros(2),
        node_energies=torch.zeros(9),
        forces=torch.zeros(9, 3),
        stress=torch.zeros(2, 3, 3),
    )
    real = unpad_outputs(output, info)
    shapes = {
        name: None if value is None else tuple(value.shape)
        for name, value in (
            ("total_energy", real.total_energy),
            ("node_energies", real.node_energies),
            ("forces", real.forces),
            ("stress", real.stress),
            ("virials", real.virials),
        )
    }
    assert shapes == {
        "total_energy": (1,),
        "node_energies": (4,),
        "forces": (4, 3),
        "stress": (1, 3, 3),
        "virials": None,
    }


def test_an_output_the_catalogue_does_not_declare_is_refused():
    info = PaddingInfo(n_real_nodes=4, n_real_edges=0, n_real_graphs=1)
    with pytest.raises(KeyError, match="latent_charges"):
        unpad_outputs(
            {"latent_charges": torch.zeros(9)}, info, catalogue=DEFAULT_CATALOGUE
        )


def test_an_undeclared_tensor_cannot_be_padded(batch):
    batch = dict(batch)
    batch["mystery"] = torch.zeros(6)
    with pytest.raises(ValueError, match="mystery"):
        pad_batch(batch, AutoPadding(), R_MAX, atomic_number_table=TABLE)


def test_a_misspelled_feature_storage_is_refused_when_padding(batch):
    batch = dict(batch)
    batch["spin_state"] = torch.zeros(6)
    misspelled: dict = {"spin_state": "atoms"}
    with pytest.raises(ValueError, match="use 'atom' or 'graph'"):
        pad_batch(
            batch,
            AutoPadding(),
            R_MAX,
            atomic_number_table=TABLE,
            features=misspelled,
        )


def _electrostatic_catalogue():
    """A catalogue declaring per-atom extras and a per-atom derivative of a
    per-graph observable, the shape of the Born effective charges."""
    positions = InputSpec(name="pos", irreps="1o", per_atom=True, units="Å")
    return ObservableCatalogue(
        inputs=(positions,),
        observables=(
            ObservableSpec(name="energy", irreps="0e", per_atom=False, units="eV"),
            ObservableSpec(
                name="dipole",
                irreps="1o",
                per_atom=False,
                units="e Å",
                derivatives=(DerivativeRequest(wrt="pos"),),
            ),
            ObservableSpec(
                name="latent_charges", irreps="0e", per_atom=True, units="e"
            ),
        ),
    )


def test_declared_extras_and_derivatives_are_stripped_by_their_declaration():
    catalogue = _electrostatic_catalogue()
    born_charges = catalogue.derivative("dipole", "pos").name
    info = PaddingInfo(n_real_nodes=4, n_real_edges=0, n_real_graphs=1)
    output = MACEOutput(total_energy=torch.zeros(2), forces=torch.zeros(9, 3))
    output.extras = {
        "latent_charges": torch.zeros(9),
        born_charges: torch.zeros(9, 3, 3),
    }
    real = unpad_outputs(output, info, catalogue=catalogue)
    assert real.extras["latent_charges"].shape == (4,)
    assert real.extras[born_charges].shape == (4, 3, 3)
    mapping = unpad_outputs(
        {"dipole": torch.zeros(2, 3), born_charges: torch.zeros(9, 3, 3)},
        info,
        catalogue=catalogue,
    )
    assert mapping["dipole"].shape == (1, 3)
    assert mapping[born_charges].shape == (4, 3, 3)


def test_born_charge_forces_add_to_the_forces_after_unpadding():
    """forces += einsum(born charges, field) broadcasts only if both were
    stripped of the same padding rows."""
    catalogue = _electrostatic_catalogue()
    born_charges = catalogue.derivative("dipole", "pos").name
    info = PaddingInfo(n_real_nodes=4, n_real_edges=0, n_real_graphs=1)
    real = unpad_outputs(
        {"forces": torch.zeros(9, 3), born_charges: torch.ones(9, 3, 3)},
        info,
        catalogue=catalogue,
    )
    field = torch.tensor([0.1, 0.0, 0.0], dtype=real["forces"].dtype)
    total = real["forces"] + torch.einsum("nij,i->nj", real[born_charges], field)
    assert total.shape == (4, 3)
