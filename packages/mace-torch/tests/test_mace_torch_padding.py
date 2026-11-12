"""The static-shape padding contract: a padded batch is a valid batch, because
padding is a structure.

The cases of the legacy ``tests/unit/test_padding.py`` are ported here as
properties of :func:`pad_batch`, :func:`pad_targets` and
:func:`unpad_outputs`: the padding atoms carry Z = 0 and sit at the origin,
the padding cell is ``eye(3) * max(2 * r_max, 1)`` and not periodic, padding
edges stay in bounds and are inert under the cutoff, zero padding edges is
legal, the padded batch keeps the real structures first, and every per-atom
output is stripped of its padding rows. A padded batch is also compared key by
key with the numpy reference ``collate([*real, *padding])``.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pytest
import torch
from mace_core.data import Configuration
from mace_core.graph import (
    EDGE_KEYS,
    AutoPadding,
    FieldSpec,
    FixedPadding,
    GraphInfo,
    GraphLayout,
    GraphValidationError,
    NoPadding,
    PaddingBudget,
    PaddingInfo,
    PaddingOverflowError,
    check_batch_invariants,
    check_layout,
    graph_layout,
    graph_sample,
    padding_samples,
    unpad_outputs,
)
from mace_core.graph import collate as reference_collate
from mace_core.neighbors import get_neighbor_backend, get_neighborhood
from mace_core.observables import (
    DEFAULT_CATALOGUE,
    DerivativeRequest,
    InputSpec,
    ObservableCatalogue,
    ObservableSpec,
)
from mace_core.outputs import MACEOutput
from mace_torch.backends.radial import polynomial_envelope
from mace_torch.graph import (
    build_batch,
    collate,
    collate_targets,
    pad_batch,
    pad_targets,
)

R_MAX = 3.5
ELEMENTS = (1, 8)
INPUTS = [
    InputSpec(name="total_charge", irreps="0e", per_atom=False, units="e"),
    InputSpec(name="total_spin", irreps="0e", per_atom=False, units="1"),
]
TARGETS = [
    ObservableSpec(name="energy", irreps="0e", per_atom=False, units="eV"),
    ObservableSpec(name="forces", irreps="1o", per_atom=True, units="eV/Å"),
    ObservableSpec(name="dipole", irreps="1o", per_atom=False, units="e·Å"),
    ObservableSpec(
        name="polarizability", irreps="0e+1e+2e", per_atom=False, units="e·Å²/V"
    ),
]


def _layout(cutoff=R_MAX, *, targets=(), **declared):
    """The layout every test batches, at the floating dtype of the test."""
    layout = graph_layout(
        ELEMENTS,
        cutoff=cutoff,
        inputs=INPUTS,
        targets=[*TARGETS, *targets],
        **declared,
    )
    dtype = str(torch.get_default_dtype()).removeprefix("torch.")
    return GraphLayout.model_validate({**layout.model_dump(), "float_dtype": dtype})


LAYOUT = _layout()

#: Every name a test structure declares, labelled or not.
DECLARED_NAMES = (*LAYOUT.inputs, *LAYOUT.targets, "born_charges")


@pytest.fixture(autouse=True)
def _layout_at_the_test_dtype(dtype, monkeypatch):
    """The conftest runs every test once per dtype and sets it as the process
    default; ``LAYOUT`` follows it, so a batch and its check agree."""
    del dtype
    monkeypatch.setattr(sys.modules[__name__], "LAYOUT", _layout())


MATSCIPY = get_neighbor_backend("matscipy")
BUDGET = FixedPadding(nodes_budget=10, edges_budget=64, graphs_budget=4)


def _water(offset=0.0, periodic=False, **properties):
    return Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        + offset,
        cell=np.eye(3) * 6.0 if periodic else None,
        pbc=(True, True, True) if periodic else None,
        properties={
            **dict.fromkeys(DECLARED_NAMES),
            "energy": -1.0,
            "forces": np.ones((3, 3)),
            **properties,
        },
        property_weights={"energy": 1.0, "forces": 1.0},
    )


def _structures():
    return [_water(periodic=True), _water(0.1)]


def _search(structure, cutoff):
    return get_neighborhood(
        structure.positions,
        cutoff,
        pbc=structure.pbc,
        cell=structure.cell,
        backend=MATSCIPY,
    )


def _samples(structures, layout=None, *, edges=True):
    layout = LAYOUT if layout is None else layout
    return [
        graph_sample(
            structure, _search(structure, layout.cutoff) if edges else None, layout
        )
        for structure in structures
    ]


def _collated(structures, layout, *, edges=True):
    pairs = _samples(structures, layout, edges=edges)
    graph, info = collate([graph for graph, _ in pairs], layout)
    targets = collate_targets([targets for _, targets in pairs], layout)
    return graph, targets, info


@pytest.fixture(name="batch")
def fixture_batch():
    """``(graph, targets, info)`` of one periodic and one open water."""
    return build_batch(_structures(), LAYOUT, neighbor_backend=MATSCIPY)


@pytest.fixture(name="padded")
def fixture_padded(batch):
    graph, _, info = batch
    return pad_batch(graph, info, BUDGET, LAYOUT)


# --- a padded batch is a valid batch --------------------------------------------------


def test_the_padded_batch_passes_the_layout_check_unchanged(padded, dtype):
    graph, info = padded
    check_layout(graph, LAYOUT, info)
    check_batch_invariants(graph, info)
    assert (info.num_nodes, info.num_edges, info.num_graphs) == (10, 64, 4)


def test_the_padded_batch_is_the_reference_collation_of_real_and_padding_samples(
    batch, padded
):
    """Padding has no per-key rule: it is the numpy reference collate over the
    real samples followed by the padding structures."""
    _, _, info = batch
    graph, padded_info = padded
    real = [sample for sample, _ in _samples(_structures())]
    padding = padding_samples(
        PaddingBudget(nodes=10, edges=64, graphs=4),
        padded_info.padding,
        LAYOUT,
    )
    reference, reference_info = reference_collate([*real, *padding], LAYOUT)
    assert set(graph) == set(reference)
    for key, value in graph.items():
        expected = torch.as_tensor(reference[key])
        if expected.is_floating_point():
            expected = expected.to(value.dtype)
        assert value.dtype == expected.dtype, key
        assert torch.equal(value, expected), key
    assert reference_info.structure_sizes == padded_info.structure_sizes
    assert info.structure_sizes == padded_info.structure_sizes[: info.num_graphs]


def test_the_graph_info_holds_the_real_sizes_and_marks_the_padding(batch, padded):
    _, _, info = batch
    _, padded_info = padded
    assert padded_info.padding == PaddingInfo(
        n_real_nodes=6,
        n_real_edges=info.num_edges,
        n_real_graphs=2,
    )
    assert padded_info.structure_sizes == (3, 3, 0, 4)


def test_padding_only_appends(batch, padded):
    graph, _, info = batch
    data, _ = padded
    for key, value in graph.items():
        if key == "edge_index":
            assert torch.equal(data[key][:, : info.num_edges], value)
        else:
            assert torch.equal(data[key][: value.shape[0]], value), key


def test_padding_atoms_belong_to_the_last_structure(padded):
    data, info = padded
    assert data["batch"][info.padding.n_real_nodes :].tolist() == [3] * 4
    assert data["ptr"].tolist() == [0, 3, 6, 6, 10]


def test_the_padding_structure_holds_every_padding_atom_at_the_origin_with_z_zero(
    padded,
):
    data, info = padded
    real_nodes = info.padding.n_real_nodes
    assert data["node_atomic_numbers"][real_nodes:].tolist() == [0] * 4
    assert not data["positions"][real_nodes:].any()
    assert info.structure_sizes[-1] == info.num_nodes - real_nodes


def test_the_padding_cell_is_scaled_identity_and_not_periodic(padded, dtype):
    data, info = padded
    expected = torch.eye(3, dtype=dtype) * max(2 * R_MAX, 1.0)
    for graph in range(info.padding.n_real_graphs, info.num_graphs):
        assert torch.equal(data["cell"][graph], expected)
        assert not data["pbc"][graph].any()
        assert data["head"][graph] == 0


def test_padding_edges_are_self_loops_on_the_last_padding_atom(padded):
    data, info = padded
    padding_edges = data["edge_index"][:, info.padding.n_real_edges :]
    assert padding_edges.unique().tolist() == [9]
    assert data["edge_index"].max() < 10
    assert data["edge_index"].min() >= 0


def test_the_cutoff_is_exactly_zero_on_a_padding_edge(padded, dtype):
    """An exact float fact, not a tolerance: the edge is 2 * r_max long and the
    envelope is masked to zero at r >= r_max."""
    data, info = padded
    real_edges = info.padding.n_real_edges
    sender, receiver = data["edge_index"][:, real_edges:]
    vectors = (
        data["positions"][receiver]
        - data["positions"][sender]
        + data["shifts"][real_edges:]
    )
    lengths = torch.linalg.norm(vectors, dim=-1)
    assert torch.equal(lengths, torch.full_like(lengths, 2 * R_MAX))
    envelope = polynomial_envelope(lengths, torch.tensor(R_MAX, dtype=dtype), 6)
    assert torch.equal(envelope, torch.zeros_like(envelope))
    unit_shifts = data["unit_shifts"][real_edges:]
    assert torch.equal(
        unit_shifts, torch.tensor([[1.0, 0.0, 0.0]] * len(unit_shifts), dtype=dtype)
    )


def test_padding_shifts_are_unit_shifts_times_the_padding_cell(padded):
    data, info = padded
    real_edges = info.padding.n_real_edges
    assert torch.equal(
        data["shifts"][real_edges:],
        data["unit_shifts"][real_edges:] @ data["cell"][-1],
    )


def test_total_spin_pads_to_one_and_the_other_inputs_to_zero(padded):
    data, info = padded
    real_graphs = info.padding.n_real_graphs
    assert data["total_spin"][real_graphs:].tolist() == [1.0, 1.0]
    assert not data["total_charge"][real_graphs:].any()


def test_no_mask_tensor_appears_in_the_padded_dict(batch, padded):
    graph, _, _ = batch
    data, _ = padded
    assert set(data) == set(graph)


def test_the_padded_batch_keeps_the_device_and_dtype_of_the_batch(batch, padded):
    graph, _, _ = batch
    data, _ = padded
    for key, value in graph.items():
        assert data[key].dtype == value.dtype, key
        assert data[key].device == value.device, key


def test_padding_does_not_modify_the_batch(batch):
    graph, _, info = batch
    before = {key: value.clone() for key, value in graph.items()}
    pad_batch(graph, info, BUDGET, LAYOUT)
    assert set(graph) == set(before)
    for key, value in before.items():
        assert torch.equal(graph[key], value), key


def test_a_padded_cell_scale_has_a_floor_of_one(dtype):
    layout = _layout(0.3)
    graph, _, info = build_batch([_water()], layout, neighbor_backend=MATSCIPY)
    data, _ = pad_batch(
        graph,
        info,
        FixedPadding(nodes_budget=4, edges_budget=2, graphs_budget=2),
        layout,
    )
    assert torch.equal(data["cell"][-1], torch.eye(3, dtype=dtype))


# --- budgets ------------------------------------------------------------------------


def test_zero_padding_edges_is_legal(batch, dtype):
    graph, _, info = batch
    policy = FixedPadding(nodes_budget=9, edges_budget=info.num_edges, graphs_budget=3)
    data, padded_info = pad_batch(graph, info, policy, LAYOUT)
    assert data["edge_index"].shape[1] == info.num_edges
    assert data["positions"].shape[0] == 9
    check_layout(data, LAYOUT, padded_info)
    check_batch_invariants(data, padded_info)


def test_an_edge_budget_with_no_room_for_a_padding_atom_grows_the_nodes(
    batch, dtype, caplog
):
    graph, _, info = batch
    policy = FixedPadding(nodes_budget=6, edges_budget=64, graphs_budget=2)
    with caplog.at_level(logging.WARNING, logger="mace_torch.graph.padding"):
        data, padded_info = pad_batch(graph, info, policy, LAYOUT)
    assert data["positions"].shape[0] == info.num_nodes + 1
    assert data["ptr"].shape[0] == info.num_graphs + 2
    assert "exceeds the padding budget" in caplog.text
    check_layout(data, LAYOUT, padded_info)
    check_batch_invariants(data, padded_info)


def test_a_batch_over_every_fixed_budget_warns_though_nothing_is_padded(
    batch, dtype, caplog
):
    graph, _, info = batch
    policy = FixedPadding(
        nodes_budget=info.num_nodes - 1,
        edges_budget=info.num_edges - 1,
        graphs_budget=info.num_graphs - 1,
    )
    with caplog.at_level(logging.WARNING, logger="mace_torch.graph.padding"):
        data, padded_info = pad_batch(graph, info, policy, LAYOUT)
    assert data["positions"].shape[0] == info.num_nodes
    assert data["edge_index"].shape[1] == info.num_edges
    assert padded_info.num_graphs == info.num_graphs
    assert "exceeds the padding budget" in caplog.text
    check_layout(data, LAYOUT, padded_info)


@pytest.mark.parametrize(
    ("budget", "named"),
    [
        ({"nodes_budget": 4}, r"in nodes \(\d+ needed, budget 4\)"),
        ({"edges_budget": 1}, r"in edges \(\d+ needed, budget 1\)"),
        ({"graphs_budget": 1}, r"in structures \(\d+ needed, budget 1\)"),
    ],
)
def test_a_batch_over_a_fixed_budget_can_refuse_naming_the_dimension(
    batch, budget, named
):
    graph, _, info = batch
    sizes = {"nodes_budget": 10, "edges_budget": 64, "graphs_budget": 4, **budget}
    policy = FixedPadding(
        nodes_budget=sizes["nodes_budget"],
        edges_budget=sizes["edges_budget"],
        graphs_budget=sizes["graphs_budget"],
        on_overflow="error",
    )
    with pytest.raises(PaddingOverflowError, match=named):
        pad_batch(graph, info, policy, LAYOUT)


def test_auto_padding_rounds_the_edges_up_with_headroom(batch, dtype):
    graph, _, info = batch
    data, padded_info = pad_batch(graph, info, AutoPadding(), LAYOUT)
    expected = -(-int(info.num_edges * 1.25) // 64) * 64
    assert data["edge_index"].shape[1] == expected
    assert data["positions"].shape[0] == info.num_nodes + 1
    assert padded_info.structure_sizes == (*info.structure_sizes, 1)
    check_layout(data, LAYOUT, padded_info)
    check_batch_invariants(data, padded_info)


def test_auto_padding_follows_its_own_edge_multiple(batch):
    graph, _, info = batch
    data, _ = pad_batch(graph, info, AutoPadding(edge_multiple=7), LAYOUT)
    assert data["edge_index"].shape[1] % 7 == 0
    assert data["edge_index"].shape[1] >= info.num_edges


def test_no_padding_returns_the_batch_as_it_is(batch):
    graph, _, info = batch
    data, padded_info = pad_batch(graph, info, NoPadding(), LAYOUT)
    assert data.keys() == graph.keys()
    assert all(data[key] is graph[key] for key in graph)
    assert padded_info.padding == PaddingInfo(
        n_real_nodes=info.num_nodes,
        n_real_edges=info.num_edges,
        n_real_graphs=info.num_graphs,
    )


@pytest.mark.parametrize(
    "policy",
    [
        BUDGET,
        NoPadding(),
        FixedPadding(nodes_budget=6, edges_budget=1, graphs_budget=2),
    ],
    ids=["padded", "no-padding", "at-the-budget"],
)
def test_a_batch_bound_for_padding_is_checked_once(monkeypatch, policy):
    """Collated with ``check=False``, a batch gets its one layout check from
    pad_batch, whether or not anything is padded."""
    calls = []

    def counting(graph, *args, **kwargs):
        calls.append(graph)
        return check_layout(graph, *args, **kwargs)

    for module in ("mace_torch.graph.collate", "mace_torch.graph.padding"):
        monkeypatch.setattr(sys.modules[module], "check_layout", counting)
    graph, _, info = build_batch(
        _structures(), LAYOUT, neighbor_backend=MATSCIPY, check=False
    )
    if isinstance(policy, FixedPadding):
        policy = FixedPadding(
            nodes_budget=policy.nodes_budget,
            edges_budget=info.num_edges,
            graphs_budget=policy.graphs_budget,
        )
    result, _ = pad_batch(graph, info, policy, LAYOUT)
    assert len(calls) == 1
    assert calls[0] is result


def test_padding_a_batch_twice_is_refused(padded):
    data, info = padded
    with pytest.raises(ValueError, match="padded already"):
        pad_batch(data, info, BUDGET, LAYOUT)


#: A batch requirement's field: the edges of each node.
NODE_DEGREE = {"node_degree": FieldSpec(shape=("n_nodes",), dtype="int64")}


def test_a_batch_carrying_a_requirement_field_is_refused():
    layout = _layout(requirement_fields=NODE_DEGREE)
    graph, _, info = _collated(_structures(), layout)
    graph["node_degree"] = torch.bincount(
        graph["edge_index"][1], minlength=info.num_nodes
    )
    with pytest.raises(ValueError, match="node_degree"):
        pad_batch(graph, info, BUDGET, layout)


# --- what padding extends besides the full graph --------------------------------------


def test_a_batch_without_edges_pads_nodes_and_graphs_only(dtype):
    graph, _, info = _collated(_structures(), LAYOUT, edges=False)
    data, padded_info = pad_batch(graph, info, BUDGET, LAYOUT)
    assert not set(data) & EDGE_KEYS
    assert (padded_info.num_nodes, padded_info.num_edges, padded_info.num_graphs) == (
        10,
        0,
        4,
    )
    assert padded_info.padding == PaddingInfo(6, 0, 2)
    check_layout(data, LAYOUT, padded_info)
    check_batch_invariants(data, padded_info)


def test_a_batch_missing_a_declared_feature_is_refused_by_name():
    layout = _layout(features={"spin_state": "atom"})
    graph, _, info = _collated(_structures(), LAYOUT)
    with pytest.raises(ValueError, match="missing the embedding feature 'spin_state'"):
        pad_batch(graph, info, BUDGET, layout)


def test_embedding_features_pad_with_zeros_at_their_own_trailing_sizes(dtype):
    layout = _layout(features={"spin_state": "atom", "phase": "graph"})
    structures = [
        _water(periodic=True, spin_state=np.ones((3, 2)), phase=np.ones(3)),
        _water(0.1, spin_state=np.full((3, 2), 2.0), phase=np.full(3, 2.0)),
    ]
    graph, _, info = _collated(structures, layout)
    data, padded_info = pad_batch(graph, info, BUDGET, layout)
    assert data["spin_state"].shape == (10, 2)
    assert data["phase"].shape == (4, 3)
    assert torch.equal(data["spin_state"][:6], graph["spin_state"])
    assert not data["spin_state"][6:].any()
    assert not data["phase"][2:].any()
    check_layout(data, layout, padded_info)


def test_pad_targets_gives_zero_labels_false_flags_and_zero_weights_on_padding(
    batch, padded
):
    _, targets, _ = batch
    _, info = padded
    padded_targets = pad_targets(targets, info, LAYOUT)
    assert set(padded_targets) == set(targets)
    fields = LAYOUT.target_fields()
    for key, value in padded_targets.items():
        if fields[key].shape[0] == "n_nodes":
            rows, real = info.num_nodes, info.padding.n_real_nodes
        else:
            rows, real = info.num_graphs, info.padding.n_real_graphs
        assert value.shape[0] == rows, key
        assert value.dtype == targets[key].dtype, key
        assert torch.equal(value[:real], targets[key]), key
        assert not value[real:].any(), key
    assert padded_targets["forces"].shape == (10, 3)
    assert padded_targets["forces_present"].shape == (10,)
    assert padded_targets["energy_present"].tolist() == [True, True, False, False]
    assert padded_targets["dipole_weight"].shape == (4, 3)
    assert padded_targets["polarizability_weight"].shape == (4, 3, 3)
    assert padded_targets["weight"].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_pad_targets_refuses_a_key_the_layout_does_not_declare(batch, padded):
    _, targets, _ = batch
    _, info = padded
    targets = {**targets, "mystery": torch.zeros(2)}
    with pytest.raises(ValueError, match="'mystery' is not a target"):
        pad_targets(targets, info, LAYOUT)


def test_pad_targets_refuses_targets_that_are_padded_already(batch, padded):
    _, targets, _ = batch
    _, info = padded
    once = pad_targets(targets, info, LAYOUT)
    with pytest.raises(ValueError, match="rows; the real batch has"):
        pad_targets(once, info, LAYOUT)


def test_pad_targets_needs_the_info_of_a_padded_batch(batch):
    _, targets, info = batch
    with pytest.raises(ValueError, match="pad_batch"):
        pad_targets(targets, info, LAYOUT)


def test_a_declared_per_atom_observable_pads_as_a_target_with_no_new_code(dtype):
    born_charges = ObservableSpec(
        name="born_charges", irreps="0e+1e+2e", per_atom=True, units="e"
    )
    layout = _layout(targets=[born_charges])
    structures = [_water(periodic=True, born_charges=np.ones((3, 9))), _water(0.1)]
    graph, targets, info = _collated(structures, layout)
    _, padded_info = pad_batch(graph, info, BUDGET, layout)
    padded_targets = pad_targets(targets, padded_info, layout)
    assert padded_targets["born_charges"].shape == (10, 9)
    assert padded_targets["born_charges_present"].tolist() == [True] * 3 + [False] * 7
    assert padded_targets["born_charges_weight"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert not padded_targets["born_charges"][3:].any()


# --- unpadding outputs ----------------------------------------------------------------


def test_unpad_after_pad_is_the_identity_on_real_slots(batch, padded):
    _, targets, _ = batch
    _, info = padded
    padded_targets = pad_targets(targets, info, LAYOUT)
    outputs = {
        "energy": padded_targets["energy"] * 2,
        "forces": padded_targets["forces"] * 3,
        "stress": torch.zeros(4, 3, 3, dtype=padded_targets["energy"].dtype),
    }
    real = unpad_outputs(outputs, info)
    assert torch.equal(real["energy"], targets["energy"] * 2)
    assert torch.equal(real["forces"], targets["forces"] * 3)
    assert real["stress"].shape == (2, 3, 3)


def test_unpadding_an_unpadded_batch_keeps_every_slot(batch):
    _, targets, info = batch
    real = unpad_outputs({"energy": targets["energy"]}, info)
    assert torch.equal(real["energy"], targets["energy"])


#: A batch of one real structure of four atoms and one padding structure of five.
PADDED_INFO = GraphInfo(
    num_graphs=2,
    num_nodes=9,
    num_edges=0,
    structure_sizes=(4, 5),
    padding=PaddingInfo(n_real_nodes=4, n_real_edges=0, n_real_graphs=1),
)


def test_every_per_atom_output_is_stripped_of_its_padding_rows():
    output = MACEOutput(
        total_energy=torch.zeros(2),
        node_energies=torch.zeros(9),
        forces=torch.zeros(9, 3),
        stress=torch.zeros(2, 3, 3),
    )
    real = unpad_outputs(output, PADDED_INFO)
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
    with pytest.raises(KeyError, match="latent_charges"):
        unpad_outputs(
            {"latent_charges": torch.zeros(9)},
            PADDED_INFO,
            catalogue=DEFAULT_CATALOGUE,
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
    output = MACEOutput(total_energy=torch.zeros(2), forces=torch.zeros(9, 3))
    output.extras = {
        "latent_charges": torch.zeros(9),
        born_charges: torch.zeros(9, 3, 3),
    }
    real = unpad_outputs(output, PADDED_INFO, catalogue=catalogue)
    assert real.extras["latent_charges"].shape == (4,)
    assert real.extras[born_charges].shape == (4, 3, 3)
    mapping = unpad_outputs(
        {"dipole": torch.zeros(2, 3), born_charges: torch.zeros(9, 3, 3)},
        PADDED_INFO,
        catalogue=catalogue,
    )
    assert mapping["dipole"].shape == (1, 3)
    assert mapping[born_charges].shape == (4, 3, 3)


def test_born_charge_forces_add_to_the_forces_after_unpadding():
    """forces += einsum(born charges, field) broadcasts only if both were
    stripped of the same padding rows."""
    catalogue = _electrostatic_catalogue()
    born_charges = catalogue.derivative("dipole", "pos").name
    real = unpad_outputs(
        {"forces": torch.zeros(9, 3), born_charges: torch.ones(9, 3, 3)},
        PADDED_INFO,
        catalogue=catalogue,
    )
    field = torch.tensor([0.1, 0.0, 0.0], dtype=real["forces"].dtype)
    total = real["forces"] + torch.einsum("nij,i->nj", real[born_charges], field)
    assert total.shape == (4, 3)


def test_a_caller_supplied_displacement_pads_with_zero_strain(dtype):
    graph, _, info = _collated(_structures(), LAYOUT)
    graph["displacement"] = torch.full((info.num_graphs, 3, 3), 0.5, dtype=dtype)
    padded, padded_info = pad_batch(graph, info, BUDGET, LAYOUT)
    assert padded["displacement"].shape == (padded_info.num_graphs, 3, 3)
    assert torch.all(padded["displacement"][: info.num_graphs] == 0.5)
    assert not padded["displacement"][info.num_graphs :].any()


def test_a_key_the_layout_does_not_declare_is_refused_by_name(dtype):
    graph, _, info = _collated(_structures(), LAYOUT)
    graph["mystery"] = torch.zeros(info.num_nodes, dtype=dtype)
    with pytest.raises(ValueError, match="'mystery' is not declared"):
        pad_batch(graph, info, BUDGET, LAYOUT)


def test_requirement_fields_are_built_after_padding_and_checked_once_built(dtype):
    """Collation and padding run before the requirements exist; once they are
    built, the layout check holds every field to its declaration."""
    layout = _layout(requirement_fields=NODE_DEGREE)
    graph, _, info = _collated(_structures(), layout)
    padded, padded_info = pad_batch(graph, info, BUDGET, layout)
    built = dict(padded)
    built["node_degree"] = torch.bincount(
        padded["edge_index"][1], minlength=padded_info.num_nodes
    )
    check_layout(built, layout, padded_info)
    built["node_degree"] = built["node_degree"][:-1]
    with pytest.raises(GraphValidationError, match="node_degree"):
        check_layout(built, layout, padded_info)


@pytest.mark.parametrize("seed", range(8))
def test_padding_random_batches_keeps_every_invariant_and_unpads_exactly(seed):
    """Random sizes, periodicities and budgets: the padded batch passes the
    batch invariants, and stripping it gives the real batch back."""
    rng = np.random.default_rng(seed)
    structures = []
    for _ in range(int(rng.integers(1, 5))):
        n_atoms = int(rng.integers(1, 6))
        periodic = bool(rng.integers(0, 2))
        structures.append(
            Configuration(
                atomic_numbers=rng.choice(ELEMENTS, size=n_atoms),
                positions=rng.uniform(0.0, 4.0, size=(n_atoms, 3)),
                cell=np.eye(3) * 5.0 if periodic else None,
                pbc=(True, True, True) if periodic else None,
                properties=dict.fromkeys(DECLARED_NAMES),
            )
        )
    graph, _, info = build_batch(
        structures, LAYOUT, neighbor_backend=MATSCIPY, check=False
    )
    policy = FixedPadding(
        nodes_budget=info.num_nodes + int(rng.integers(0, 6)),
        edges_budget=info.num_edges + int(rng.integers(0, 40)),
        graphs_budget=info.num_graphs + int(rng.integers(0, 3)),
    )
    padded, padded_info = pad_batch(graph, info, policy, LAYOUT)
    check_batch_invariants(padded, padded_info)
    real = unpad_outputs(
        {
            "node_energies": padded["positions"][:, 0],
            "total_energy": padded["cell"][:, 0, 0],
        },
        padded_info,
    )
    assert torch.equal(real["node_energies"], graph["positions"][:, 0])
    assert torch.equal(real["total_energy"], graph["cell"][:, 0, 0])
