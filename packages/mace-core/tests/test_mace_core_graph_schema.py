"""The flat graph dict: its structural core, the layout a run declares, the
step from a Configuration to a sample, the numpy reference collation, and the
checks split by cost.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from typing import cast

import numpy as np
import pytest
from mace_core.data import Configuration
from mace_core.graph import (
    ANY_TRAILING,
    DEFAULT_INPUTS,
    DEFAULT_TARGETS,
    EDGE_KEYS,
    FLOAT,
    GRAPH_CORE,
    NOT_KEYS,
    FieldSpec,
    GraphInfo,
    GraphLayout,
    GraphValidationError,
    GraphView,
    PaddingInfo,
    SampleError,
    check_batch_invariants,
    check_layout,
    collate,
    collate_targets,
    default_layout,
    feature_field,
    graph_sample,
    input_field,
    presence_key,
    structure_graph,
    target_field,
    weight_key,
)
from mace_core.neighbors import AllPairsBackend, NeighborList
from mace_core.observables import InputSpec, ObservableSpec

ELEMENTS = (1, 8, 14)
CUTOFF = 2.5
BACKEND = AllPairsBackend()
LAYOUT = default_layout(ELEMENTS, cutoff=CUTOFF)

WATER_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
CRYSTAL_CELL = np.array([[2.47, 0.0, 0.0], [0.83, 2.39, 0.0], [0.31, 0.57, 2.61]])


def _water(**properties):
    return Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=WATER_POSITIONS.copy(),
        properties=properties,
    )


def _crystal(**properties):
    return Configuration(
        atomic_numbers=np.array([14, 14]),
        positions=np.array([[0.0, 0.0, 0.0], [0.91, 0.73, 0.62]]),
        cell=CRYSTAL_CELL.copy(),
        pbc=(True, True, True),
        properties=properties,
    )


def _slab(cell, **properties):
    """Two silicon atoms periodic in x and y, open along z."""
    return Configuration(
        atomic_numbers=np.array([14, 14]),
        positions=np.array([[0.0, 0.0, 0.0], [1.2, 1.2, 1.0]]),
        cell=np.asarray(cell, dtype=float),
        pbc=(True, True, False),
        properties=properties,
    )


def _sample(structure, layout=LAYOUT, *, edges=True, **kwargs):
    neighbors = structure_graph(structure, layout.cutoff, BACKEND) if edges else None
    return graph_sample(structure, neighbors, layout, **kwargs)


def _batch(*structures, layout=LAYOUT, edges=True):
    pairs = [_sample(s, layout, edges=edges) for s in structures]
    graph, info = collate([graph for graph, _ in pairs], layout)
    targets = collate_targets([targets for _, targets in pairs], layout)
    return graph, targets, info


def _born_charges():
    """A per-atom tensor observable outside the defaults: nine components."""
    return ObservableSpec(
        name="born_charges", irreps="0e+1e+2e", per_atom=True, units="e"
    )


# --- the structural core ---------------------------------------------------------


def test_the_core_holds_exactly_the_structural_keys():
    assert set(GRAPH_CORE) == {
        "positions",
        "node_atomic_numbers",
        "edge_index",
        "shifts",
        "unit_shifts",
        "cell",
        "pbc",
        "batch",
        "ptr",
        "head",
        "displacement",
    }


def test_the_core_cannot_be_edited_in_place():
    with pytest.raises(TypeError):
        cast(dict, GRAPH_CORE)["positions"] = GRAPH_CORE["cell"]


@pytest.mark.parametrize(
    ("key", "shape", "dtype", "required", "producer"),
    [
        ("positions", ("n_nodes", 3), FLOAT, True, "data"),
        ("node_atomic_numbers", ("n_nodes",), "int64", True, "data"),
        ("edge_index", (2, "n_edges"), "int64", True, "nbr"),
        ("shifts", ("n_edges", 3), FLOAT, True, "nbr"),
        ("unit_shifts", ("n_edges", 3), FLOAT, True, "nbr"),
        ("cell", ("n_graphs", 3, 3), FLOAT, True, "data"),
        ("pbc", ("n_graphs", 3), "bool", True, "data"),
        ("batch", ("n_nodes",), "int64", True, "coll"),
        ("ptr", ("n_graphs_plus_one",), "int64", True, "coll"),
        ("head", ("n_graphs",), "int64", False, "derived"),
        ("displacement", ("n_graphs", 3, 3), FLOAT, False, "caller"),
    ],
)
def test_a_row_of_the_core(key, shape, dtype, required, producer):
    spec = GRAPH_CORE[key]
    assert (spec.shape, spec.dtype, spec.required, spec.producer) == (
        shape,
        dtype,
        required,
        producer,
    )


def test_the_edge_keys_are_the_three_a_neighbour_list_writes():
    assert {"edge_index", "shifts", "unit_shifts"} == EDGE_KEYS
    assert {GRAPH_CORE[key].producer for key in EDGE_KEYS} == {"nbr"}


def test_derived_and_host_quantities_are_never_keys():
    assert {
        "num_nodes",
        "num_edges",
        "num_graphs",
        "edge_vectors",
        "edge_lengths",
        "node_attrs",
        "rcell",
        "volume",
        "structure_sizes",
        "structure_periodic",
        "structure_padding",
    } == NOT_KEYS
    assert not NOT_KEYS & set(GRAPH_CORE)


def test_labels_are_not_core_keys():
    assert not set(DEFAULT_TARGETS) & set(GRAPH_CORE)
    assert not set(DEFAULT_INPUTS) & set(GRAPH_CORE)


# --- the layout a run declares ------------------------------------------------------


@pytest.mark.parametrize("heads", [(), ("pbe", "pbe"), ("a", "b", "a")])
def test_a_head_list_that_is_empty_or_repeats_a_name_is_refused(heads):
    with pytest.raises(ValueError, match="head list"):
        default_layout(ELEMENTS, cutoff=CUTOFF, heads=heads)


@pytest.mark.parametrize("table", [(), (1, 1), (0, 1), (1, 119), (-8,)])
def test_an_element_table_that_is_empty_repeats_or_leaves_the_periodic_table_is_refused(
    table,
):
    with pytest.raises(ValueError, match="element table"):
        default_layout(table, cutoff=CUTOFF)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("nan"), float("inf")])
def test_a_cutoff_that_is_not_a_positive_finite_distance_is_refused(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        default_layout(ELEMENTS, cutoff=cutoff)


SCALAR_PER_STRUCTURE = FieldSpec(("n_graphs",), FLOAT, True, "data", "A scalar.")


@pytest.mark.parametrize(
    "name", ["positions", "batch", "head", "rcell", "node_attrs", "num_nodes"]
)
@pytest.mark.parametrize("kind", ["inputs", "targets"])
def test_a_declared_name_colliding_with_the_core_or_a_name_never_carried_is_refused(
    name, kind
):
    with pytest.raises(ValueError, match="named like a key of the graph core"):
        GraphLayout(
            atomic_numbers=ELEMENTS,
            heads=("Default",),
            cutoff=CUTOFF,
            **{kind: {name: SCALAR_PER_STRUCTURE}},  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("name", ["energy_present", "spin_present", "spin_weight"])
@pytest.mark.parametrize("kind", ["inputs", "targets"])
def test_a_declared_name_ending_like_a_flag_or_a_weight_is_refused(name, kind):
    with pytest.raises(ValueError, match="presence flag or a loss weight"):
        GraphLayout(
            atomic_numbers=ELEMENTS,
            heads=("Default",),
            cutoff=CUTOFF,
            **{kind: {name: SCALAR_PER_STRUCTURE}},  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("name", ["", "  "])
def test_an_empty_declared_name_is_refused(name):
    with pytest.raises(ValueError, match="empty name"):
        GraphLayout(
            atomic_numbers=ELEMENTS,
            heads=("Default",),
            cutoff=CUTOFF,
            inputs={name: SCALAR_PER_STRUCTURE},
        )


def test_an_input_that_is_also_a_target_is_refused():
    with pytest.raises(ValueError, match="both as an input and as a target"):
        GraphLayout(
            atomic_numbers=ELEMENTS,
            heads=("Default",),
            cutoff=CUTOFF,
            inputs={"total_charge": SCALAR_PER_STRUCTURE},
            targets={"total_charge": SCALAR_PER_STRUCTURE},
        )


def test_a_feature_named_like_a_declared_input_is_refused():
    with pytest.raises(ValueError, match="name of a declared input"):
        default_layout(ELEMENTS, cutoff=CUTOFF, features={"total_charge": "graph"})


@pytest.mark.parametrize("storage", ["atoms", "structure", ""])
def test_a_misspelled_feature_storage_is_refused(storage):
    with pytest.raises(ValueError, match="use 'atom' or 'graph'"):
        default_layout(ELEMENTS, cutoff=CUTOFF, features={"spin_state": storage})


def test_equal_layouts_are_equal_and_hash_equal():
    first = default_layout(list(ELEMENTS), cutoff=CUTOFF)
    second = default_layout(np.array(ELEMENTS), cutoff=int(CUTOFF * 2) / 2)
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    assert first != default_layout(ELEMENTS, cutoff=CUTOFF + 1)
    assert first != default_layout(ELEMENTS, cutoff=CUTOFF, heads=("pbe",))
    assert first != default_layout(ELEMENTS, cutoff=CUTOFF, targets=[_born_charges()])


def test_a_layout_cannot_be_edited():
    with pytest.raises(dataclasses.FrozenInstanceError):
        LAYOUT.cutoff = 3.0  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        cast(dict, LAYOUT.inputs)["mystery"] = SCALAR_PER_STRUCTURE


def test_the_element_index_is_the_position_in_the_table():
    assert [LAYOUT.element_index(z) for z in ELEMENTS] == [0, 1, 2]
    with pytest.raises(ValueError, match="26"):
        LAYOUT.element_index(26)


# --- the default declarations ---------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "shape", "absent_value"),
    [
        ("magmom", ("n_nodes", 3), 0.0),
        ("total_charge", ("n_graphs",), 0.0),
        ("total_spin", ("n_graphs",), 1.0),
        ("elec_temp", ("n_graphs",), 0.0),
        ("fermi_level", ("n_graphs",), 0.0),
        ("external_field", ("n_graphs", 3), 0.0),
    ],
)
def test_the_default_inputs_are_the_ones_develop_batches(name, shape, absent_value):
    assert set(LAYOUT.inputs) == set(DEFAULT_INPUTS)
    spec = LAYOUT.inputs[name]
    assert (spec.shape, spec.dtype, spec.absent_value) == (shape, FLOAT, absent_value)


@pytest.mark.parametrize(
    ("name", "shape", "weight_shape"),
    [
        ("energy", ("n_graphs",), ("n_graphs",)),
        ("forces", ("n_nodes", 3), ("n_graphs",)),
        ("magforces", ("n_nodes", 3), ("n_graphs",)),
        ("stress", ("n_graphs", 3, 3), ("n_graphs",)),
        ("virials", ("n_graphs", 3, 3), ("n_graphs",)),
        ("dipole", ("n_graphs", 3), ("n_graphs", 3)),
        ("charges", ("n_nodes",), ("n_graphs",)),
        ("polarizability", ("n_graphs", 3, 3), ("n_graphs", 3, 3)),
    ],
)
def test_the_eight_default_targets_carry_a_flag_and_a_weight(name, shape, weight_shape):
    assert len(DEFAULT_TARGETS) == 8
    fields = LAYOUT.target_fields()
    assert fields[name].shape == shape
    assert fields[presence_key(name)].shape == (shape[0],)
    assert fields[presence_key(name)].dtype == "bool"
    assert fields[weight_key(name)].shape == weight_shape
    assert fields[weight_key(name)].dtype == FLOAT


def test_the_targets_carry_the_structure_weight_and_nothing_of_the_graph():
    fields = LAYOUT.target_fields()
    assert fields["weight"].shape == ("n_graphs",)
    assert len(fields) == 1 + 3 * len(DEFAULT_TARGETS)
    assert not set(fields) & set(LAYOUT.graph_fields())


def test_the_graph_fields_are_the_core_and_the_inputs():
    assert set(LAYOUT.graph_fields()) == set(GRAPH_CORE) | set(DEFAULT_INPUTS)


def test_an_input_field_takes_its_level_from_per_atom_and_its_shape_from_the_irreps():
    def shape(name, irreps, per_atom):
        spec = InputSpec(name=name, irreps=irreps, per_atom=per_atom, units="a.u.")
        return input_field(spec).shape

    assert shape("hubbard_u", "0e", False) == ("n_graphs",)
    assert shape("site_field", "1o", True) == ("n_nodes", 3)
    assert shape("site_field", "0e+1o", True) == ("n_nodes", 4)
    assert shape("charge_states", "2x0e", False) == ("n_graphs", 2)


def test_a_default_name_keeps_its_contract_shape_whatever_its_irreps_spell():
    stress = ObservableSpec(name="stress", irreps="0e+2e", per_atom=False, units="eV")
    assert target_field(stress).shape == ("n_graphs", 3, 3)
    magmom = InputSpec(name="magmom", irreps="1e", per_atom=True, units="μB")
    assert input_field(magmom).shape == ("n_nodes", 3)


@pytest.mark.parametrize("name", ["pos", "strain"])
def test_the_inputs_the_derivative_engine_creates_are_never_batched(name):
    spec = InputSpec(name=name, irreps="1o", per_atom=name == "pos", units="Å")
    with pytest.raises(ValueError, match="derivative engine"):
        input_field(spec)


def test_a_feature_field_closes_its_shape_with_the_datas_own_sizes():
    assert feature_field("spin_state", "atom").shape == ("n_nodes", ANY_TRAILING)
    assert feature_field("phase", "graph").shape == ("n_graphs", ANY_TRAILING)


# --- an observable outside the defaults -----------------------------------------------


def test_a_declared_observable_becomes_a_target_with_a_flag_and_a_weight():
    layout = default_layout(ELEMENTS, cutoff=CUTOFF, targets=[_born_charges()])
    fields = layout.target_fields()
    assert fields["born_charges"].shape == ("n_nodes", 9)
    assert fields["born_charges_present"].shape == ("n_nodes",)
    assert fields["born_charges_weight"].shape == ("n_graphs",)
    assert "born_charges" not in layout.graph_fields()


def test_a_declared_observable_is_batched_with_no_new_code():
    layout = default_layout(ELEMENTS, cutoff=CUTOFF, targets=[_born_charges()])
    charges = np.arange(27.0).reshape(3, 9)
    graph, targets, info = _batch(
        _water(born_charges=charges), _crystal(), layout=layout
    )
    assert "born_charges" not in graph
    assert targets["born_charges"].shape == (5, 9)
    np.testing.assert_array_equal(targets["born_charges"][:3], charges)
    assert not targets["born_charges"][3:].any()
    assert targets["born_charges_present"].tolist() == [True] * 3 + [False] * 2
    assert targets["born_charges_weight"].tolist() == [1.0, 0.0]
    check_layout(graph, layout, info)


def test_a_declared_input_outside_the_defaults_is_batched_with_no_new_code():
    site_field = InputSpec(name="site_field", irreps="1o", per_atom=True, units="V/Å")
    layout = default_layout(ELEMENTS, cutoff=CUTOFF, inputs=[site_field])
    values = np.full((3, 3), 0.5)
    real = [
        _sample(_water(site_field=values), layout)[0],
        _sample(_crystal(), layout)[0],
    ]
    graph, info = collate(real, layout)
    np.testing.assert_array_equal(graph["site_field"][:3], values)
    assert not graph["site_field"][3:].any()
    assert info.structure_sizes == (3, 2)


# --- from a Configuration to a sample -------------------------------------------------


def test_a_sample_is_a_graph_and_its_targets_and_the_graph_holds_no_label():
    graph, targets = _sample(_crystal(energy=-2.0, forces=np.ones((2, 3))))
    assert set(graph) == (set(GRAPH_CORE) - {"batch", "ptr", "displacement"}) | set(
        DEFAULT_INPUTS
    )
    assert set(targets) == set(LAYOUT.target_fields())
    assert targets["energy"] == -2.0
    assert targets["weight"] == 1.0


def test_sample_arrays_have_the_documented_dtypes():
    graph, targets = _sample(_crystal(energy=-2.0))
    assert graph["positions"].dtype == np.float64
    assert graph["unit_shifts"].dtype == np.float64
    assert graph["edge_index"].dtype == np.int64
    assert graph["node_atomic_numbers"].dtype == np.int64
    assert graph["pbc"].dtype == np.bool_
    assert targets["energy_present"].dtype == np.bool_


def test_the_head_is_its_index_in_the_models_head_list():
    layout = default_layout(ELEMENTS, cutoff=CUTOFF, heads=("Default", "pbe"))
    structure = Configuration(
        atomic_numbers=np.array([1]), positions=np.zeros((1, 3)), head="pbe"
    )
    graph, _ = _sample(structure, layout)
    assert graph["head"] == 1


def test_an_unknown_head_is_an_error_rather_than_a_fallback():
    layout = default_layout(ELEMENTS, cutoff=CUTOFF, heads=("Default", "pbe"))
    structure = Configuration(
        atomic_numbers=np.array([1]), positions=np.zeros((1, 3)), head="r2scan"
    )
    with pytest.raises(SampleError, match="r2scan"):
        _sample(structure, layout)


def test_an_element_outside_the_table_is_refused():
    structure = Configuration(atomic_numbers=np.array([26]), positions=np.zeros((1, 3)))
    with pytest.raises(SampleError, match="element 26"):
        _sample(structure, edges=False)


def test_a_refused_structure_is_named_by_its_source():
    structure = Configuration(atomic_numbers=np.array([26]), positions=np.zeros((1, 3)))
    with pytest.raises(SampleError, match=r"train\.xyz"):
        _sample(structure, edges=False, source="train.xyz")


def test_a_non_finite_position_is_refused():
    structure = _water()
    structure.positions[1, 0] = np.nan
    with pytest.raises(SampleError, match="non-finite position"):
        _sample(structure, edges=False)


def test_a_non_finite_cell_is_refused():
    structure = _crystal()
    structure.cell[0, 0] = np.inf
    with pytest.raises(SampleError, match="non-finite cell"):
        _sample(structure, edges=False)


@pytest.mark.parametrize(
    ("label", "value"),
    [("energy", np.nan), ("forces", np.full((3, 3), np.inf))],
)
def test_a_non_finite_label_is_refused(label, value):
    with pytest.raises(SampleError, match=f"non-finite value in label '{label}'"):
        _sample(_water(**{label: value}), edges=False)


def test_a_non_finite_input_is_refused():
    with pytest.raises(SampleError, match="non-finite value in input 'total_charge'"):
        _sample(_water(total_charge=np.inf), edges=False)


def test_a_periodic_axis_with_an_all_zero_cell_row_is_refused():
    structure = _crystal()
    structure.cell[2] = 0.0
    with pytest.raises(SampleError, match="axis 2"):
        _sample(structure, edges=False)


def test_an_all_zero_cell_with_a_periodic_axis_stands_for_the_identity():
    structure = Configuration(
        atomic_numbers=np.array([1]),
        positions=np.zeros((1, 3)),
        cell=np.zeros((3, 3)),
        pbc=(True, False, False),
    )
    graph, _ = _sample(structure)
    np.testing.assert_array_equal(graph["cell"], np.eye(3))


def test_a_stress_label_on_a_slab_of_zero_volume_is_refused():
    open_slab = _slab(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]], stress=np.eye(3)
    )
    with pytest.raises(SampleError, match="zero volume"):
        _sample(open_slab, edges=False)


def test_a_slab_of_zero_volume_without_a_stress_label_keeps_its_zero_row():
    cell = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]
    graph, targets = _sample(_slab(cell, energy=-1.0, virials=np.eye(3)))
    np.testing.assert_array_equal(graph["cell"], cell)
    assert not graph["unit_shifts"][:, 2].any()
    assert targets["virials_present"]


def test_a_stress_label_on_a_slab_with_its_vacuum_row_is_accepted():
    cell = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]]
    _, targets = _sample(_slab(cell, stress=np.eye(3)), edges=False)
    assert targets["stress_present"]


def test_a_stress_label_on_an_aperiodic_structure_is_accepted():
    """The stress of an open structure is masked downstream, so the label is
    carried rather than refused."""
    graph, targets = _sample(_water(stress=np.eye(3)), edges=False)
    assert targets["stress_present"]
    assert not graph["cell"].any()


def test_an_aperiodic_structure_carries_its_parsed_cell():
    parsed = np.diag([10.0, 11.0, 12.0])
    structure = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=WATER_POSITIONS.copy(),
        cell=parsed,
        pbc=(False, False, False),
    )
    graph, _ = _sample(structure)
    np.testing.assert_array_equal(graph["cell"], parsed)
    assert not graph["unit_shifts"].any()


def test_a_present_label_with_zero_weight_keeps_its_flag():
    structure = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=WATER_POSITIONS.copy(),
        properties={"energy": -1.0, "forces": np.ones((3, 3))},
        property_weights={"energy": 0.0, "forces": 0.0},
    )
    _, targets = _sample(structure, edges=False)
    assert targets["energy_present"]
    assert targets["forces_present"].tolist() == [True, True, True]
    assert targets["energy_weight"] == 0.0
    assert targets["forces_weight"] == 0.0


def test_an_absent_label_is_zero_flagged_absent_and_weighted_zero():
    _, targets = _sample(_water(energy=None, stress=None), edges=False)
    assert not targets["energy_present"]
    assert targets["energy"] == 0.0
    assert targets["energy_weight"] == 0.0
    assert targets["stress"].shape == (3, 3)
    assert not targets["stress"].any()
    assert not targets["dipole_present"]
    assert targets["dipole_weight"].tolist() == [0.0, 0.0, 0.0]
    assert targets["polarizability_weight"].shape == (3, 3)


def test_a_property_weight_is_carried_for_a_present_label():
    structure = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=WATER_POSITIONS.copy(),
        properties={"dipole": np.ones(3)},
        property_weights={"dipole": 2.5},
        weight=0.5,
    )
    _, targets = _sample(structure, edges=False)
    assert targets["dipole_weight"].tolist() == [2.5, 2.5, 2.5]
    assert targets["weight"] == 0.5


def test_an_absent_total_spin_is_one_and_other_inputs_are_zero():
    graph, _ = _sample(
        _water(total_spin=None, total_charge=None, magmom=None), edges=False
    )
    assert graph["total_spin"] == 1.0
    assert graph["total_charge"] == 0.0
    assert graph["external_field"].tolist() == [0.0, 0.0, 0.0]
    assert graph["magmom"].shape == (3, 3)
    assert not graph["magmom"].any()


def test_a_label_of_the_wrong_shape_is_refused():
    layout = default_layout(ELEMENTS, cutoff=CUTOFF, targets=[_born_charges()])
    with pytest.raises(SampleError, match="label 'born_charges' of shape"):
        _sample(_water(born_charges=np.zeros((3, 3, 3))), layout, edges=False)


def test_an_input_of_the_wrong_shape_is_refused():
    with pytest.raises(SampleError, match="input 'external_field' of shape"):
        _sample(_water(external_field=np.zeros(2)), edges=False)


def test_a_declared_atom_feature_with_the_wrong_rows_is_not_reshaped():
    layout = default_layout(ELEMENTS, cutoff=CUTOFF, features={"spin_state": "atom"})
    with pytest.raises(SampleError, match="not reshaped"):
        _sample(_water(spin_state=np.array([0, 1])), layout, edges=False)


def test_a_declared_feature_that_is_not_a_real_number_is_refused():
    layout = default_layout(ELEMENTS, cutoff=CUTOFF, features={"spin_state": "atom"})
    with pytest.raises(SampleError, match="real number"):
        _sample(_water(spin_state=np.array(["up", "down", "up"])), layout, edges=False)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_a_non_finite_feature_is_refused(value):
    layout = default_layout(ELEMENTS, cutoff=CUTOFF, features={"spin_state": "atom"})
    spin_state = np.array([0.0, 1.0, value])
    with pytest.raises(
        SampleError, match="non-finite value in embedding feature 'spin_state'"
    ):
        _sample(_water(spin_state=spin_state), layout, edges=False)


@pytest.mark.parametrize("properties", [{}, {"spin_state": None}])
def test_a_declared_feature_the_structure_lacks_is_refused(properties):
    layout = default_layout(ELEMENTS, cutoff=CUTOFF, features={"spin_state": "atom"})
    with pytest.raises(SampleError, match="cannot be made up"):
        _sample(_water(**properties), layout, edges=False)


def test_a_declared_feature_is_batched_by_its_storage():
    layout = default_layout(
        ELEMENTS, cutoff=CUTOFF, features={"spin_state": "atom", "phase": "graph"}
    )
    graph, _, _ = _batch(
        _water(spin_state=np.array([0, 1, 1]), phase=3.0),
        _crystal(spin_state=np.array([2, 2]), phase=4.0),
        layout=layout,
    )
    assert graph["spin_state"].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0]
    assert graph["phase"].tolist() == [3.0, 4.0]


def test_a_sample_without_a_neighbour_list_has_no_edge_keys():
    graph, _ = _sample(_crystal(), edges=False)
    assert not set(graph) & EDGE_KEYS


def test_a_neighbour_list_searched_with_another_cutoff_is_refused():
    structure = _crystal()
    neighbors = structure_graph(structure, CUTOFF + 0.5, BACKEND)
    with pytest.raises(SampleError, match="cutoff"):
        graph_sample(structure, neighbors, LAYOUT)


def test_a_neighbour_list_reaching_outside_the_structure_is_refused():
    neighbors = NeighborList(
        edge_index=np.array([[0], [5]]),
        unit_shifts=np.zeros((1, 3), dtype=np.int64),
        shifts=np.zeros((1, 3)),
        cutoff=CUTOFF,
    )
    with pytest.raises(SampleError, match="reach atom 5"):
        graph_sample(_water(), neighbors, LAYOUT)


def test_a_neighbour_list_with_images_along_an_open_axis_is_refused():
    neighbors = NeighborList(
        edge_index=np.array([[0], [1]]),
        unit_shifts=np.array([[0, 0, 1]]),
        shifts=np.zeros((1, 3)),
        cutoff=CUTOFF,
    )
    with pytest.raises(SampleError, match="aperiodic"):
        graph_sample(_water(), neighbors, LAYOUT)


def test_a_stored_neighbour_list_is_carried_as_given():
    structure = _crystal()
    neighbors = structure_graph(structure, CUTOFF, BACKEND)
    graph, _ = graph_sample(structure, neighbors, LAYOUT)
    np.testing.assert_array_equal(graph["edge_index"], neighbors.edge_index)
    np.testing.assert_array_equal(graph["unit_shifts"], neighbors.unit_shifts)
    np.testing.assert_array_equal(
        graph["shifts"], neighbors.unit_shifts @ graph["cell"]
    )


def test_an_undeclared_property_never_reaches_the_graph_or_the_targets():
    """Legacy promoted an unknown property into the batch with a guessed
    shape. Here a property nobody declares is left behind."""
    graph, targets = _sample(_water(tags=np.array([1.0, 2.0, 3.0])))
    assert "tags" not in graph
    assert "tags" not in targets


def test_pbc_given_as_a_numpy_array_is_read():
    structure = _crystal()
    structure.pbc = np.array([True, True, True])
    graph, _ = _sample(structure)
    assert graph["pbc"].tolist() == [True, True, True]


def test_building_a_sample_does_not_modify_the_configuration():
    structure = _crystal(energy=-2.0)
    positions, cell = structure.positions.copy(), structure.cell.copy()
    _sample(structure)
    np.testing.assert_array_equal(structure.positions, positions)
    np.testing.assert_array_equal(structure.cell, cell)


def test_a_structure_without_a_cell_carries_a_zero_cell_and_keeps_none():
    structure = _water()
    graph, _ = _sample(structure)
    assert structure.cell is None
    np.testing.assert_array_equal(graph["cell"], np.zeros((3, 3)))


# --- the numpy reference collation ----------------------------------------------------


@pytest.fixture(name="batch")
def fixture_batch():
    return _batch(_water(energy=-1.0), _crystal(energy=-2.0, forces=np.ones((2, 3))))


def test_collate_returns_the_graph_and_its_info(batch):
    graph, _, info = batch
    assert info == GraphInfo(
        num_graphs=2,
        num_nodes=5,
        num_edges=int(graph["edge_index"].shape[1]),
        structure_sizes=(3, 2),
    )
    assert info.padding is None
    assert not set(graph) & NOT_KEYS


def test_edge_index_is_offset_and_ptr_and_batch_cover_the_nodes(batch):
    graph, _, _ = batch
    water, _ = _sample(_water())
    n_water_edges = water["edge_index"].shape[1]
    assert graph["ptr"].tolist() == [0, 3, 5]
    assert graph["batch"].tolist() == [0, 0, 0, 1, 1]
    np.testing.assert_array_equal(
        graph["edge_index"][:, :n_water_edges], water["edge_index"]
    )
    assert graph["edge_index"][:, n_water_edges:].min() == 3


def test_every_key_is_joined_along_its_level(batch):
    graph, _, _ = batch
    assert graph["cell"].shape == (2, 3, 3)
    assert graph["pbc"].tolist() == [[False, False, False], [True, True, True]]
    assert graph["head"].tolist() == [0, 0]
    assert graph["total_spin"].tolist() == [1.0, 1.0]
    assert graph["magmom"].shape == (5, 3)
    assert graph["positions"].shape == (5, 3)
    assert graph["shifts"].shape == (graph["edge_index"].shape[1], 3)


def test_the_targets_are_joined_along_their_levels(batch):
    _, targets, _ = batch
    assert targets["energy"].tolist() == [-1.0, -2.0]
    assert targets["energy_present"].tolist() == [True, True]
    assert targets["forces"].shape == (5, 3)
    assert targets["forces_present"].tolist() == [False] * 3 + [True] * 2
    assert targets["forces_weight"].tolist() == [0.0, 1.0]
    assert targets["dipole_weight"].shape == (2, 3)
    assert targets["polarizability_weight"].shape == (2, 3, 3)
    assert targets["weight"].tolist() == [1.0, 1.0]


def test_collation_does_not_modify_the_samples():
    sample, _ = _sample(_crystal())
    before = {key: np.copy(value) for key, value in sample.items()}
    collate([sample, sample], LAYOUT)
    for key, value in before.items():
        np.testing.assert_array_equal(sample[key], value)


def test_collate_needs_a_sample():
    with pytest.raises(ValueError, match="at least one"):
        collate([], LAYOUT)


def test_samples_carrying_different_keys_are_refused():
    with_edges, _ = _sample(_water())
    without_edges, _ = _sample(_crystal(), edges=False)
    with pytest.raises(ValueError, match="different keys"):
        collate([with_edges, without_edges], LAYOUT)


@pytest.mark.parametrize("key", ["batch", "ptr"])
def test_a_sample_carrying_a_collated_key_is_refused(key):
    sample, _ = _sample(_water())
    sample[key] = np.zeros(3, dtype=np.int64)
    with pytest.raises(ValueError, match="collation computes"):
        collate([sample], LAYOUT)


@pytest.mark.parametrize("key", ["tags", "rcell", "energy"])
def test_a_sample_carrying_an_undeclared_key_is_refused(key):
    sample, _ = _sample(_water())
    sample[key] = np.zeros(3)
    with pytest.raises(ValueError, match="does not declare"):
        collate([sample], LAYOUT)


def test_a_target_sample_carrying_an_undeclared_key_is_refused():
    _, targets = _sample(_water())
    targets["tags"] = np.zeros(3)
    with pytest.raises(ValueError, match="does not declare"):
        collate_targets([targets], LAYOUT)


def test_a_batch_without_edges_collates_and_passes_the_layout_check():
    graph, _, info = _batch(_water(), _crystal(), edges=False)
    assert not set(graph) & EDGE_KEYS
    assert info.num_edges == 0
    check_layout(graph, LAYOUT, info)
    check_batch_invariants(graph, info)


def test_validate_runs_the_batch_invariants_on_a_hand_built_sample():
    water, _ = _sample(_water())
    crystal, _ = _sample(_crystal())
    water["edge_index"] = water["edge_index"].copy()
    water["edge_index"][1, 0] = 3
    collate([water, crystal], LAYOUT)
    with pytest.raises(GraphValidationError, match="between two structures"):
        collate([water, crystal], LAYOUT, validate=True)


# --- check_layout: every batch, metadata only -----------------------------------------


def test_a_collated_batch_passes_the_layout_check(batch):
    graph, _, info = batch
    check_layout(graph, LAYOUT, info)
    check_batch_invariants(graph, info)


def test_a_missing_required_key_is_named(batch):
    graph, _, info = batch
    del graph["total_spin"]
    with pytest.raises(GraphValidationError, match="total_spin"):
        check_layout(graph, LAYOUT, info)


@pytest.mark.parametrize("key", ["ptr", "positions"])
def test_a_missing_key_that_binds_the_sizes_is_named(batch, key):
    graph, _, info = batch
    del graph[key]
    with pytest.raises(GraphValidationError, match=key):
        check_layout(graph, LAYOUT, info)


def test_an_undeclared_key_is_named(batch):
    graph, _, info = batch
    graph["my_annotation"] = np.zeros(2)
    with pytest.raises(GraphValidationError, match="'my_annotation' is not declared"):
        check_layout(graph, LAYOUT, info)


@pytest.mark.parametrize("key", ["rcell", "volume", "node_attrs", "structure_sizes"])
def test_a_key_the_graph_never_carries_is_named(batch, key):
    graph, _, info = batch
    graph[key] = np.zeros(2)
    with pytest.raises(GraphValidationError, match=f"'{key}' is never a key"):
        check_layout(graph, LAYOUT, info)


def test_a_label_in_the_graph_is_refused(batch):
    graph, targets, info = batch
    graph["energy"] = targets["energy"]
    with pytest.raises(GraphValidationError, match="'energy' is not declared"):
        check_layout(graph, LAYOUT, info)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("magmom", np.zeros((5, 2))),
        ("magmom", np.zeros((4, 3))),
        ("cell", np.zeros((6, 3))),
        ("total_charge", np.zeros((2, 1))),
    ],
)
def test_a_wrong_shape_is_named(batch, key, value):
    graph, _, info = batch
    graph[key] = value
    with pytest.raises(GraphValidationError, match=f"'{key}' has shape"):
        check_layout(graph, LAYOUT, info)


@pytest.mark.parametrize(
    ("key", "dtype"),
    [("edge_index", np.int32), ("positions", np.float32), ("pbc", np.int64)],
)
def test_a_wrong_dtype_is_named(batch, key, dtype):
    graph, _, info = batch
    graph[key] = graph[key].astype(dtype)
    with pytest.raises(GraphValidationError, match=f"'{key}' has dtype"):
        check_layout(graph, LAYOUT, info)


@pytest.mark.parametrize(
    "change",
    [
        lambda info: {"num_edges": info.num_edges + 1},
        lambda info: {"num_graphs": 1, "structure_sizes": (5,)},
        lambda info: {"num_nodes": 6, "structure_sizes": (3, 3)},
    ],
    ids=["edges", "graphs", "nodes"],
)
def test_a_graph_info_that_disagrees_with_the_batch_is_refused(batch, change):
    graph, _, info = batch
    wrong = dataclasses.replace(info, **change(info))
    with pytest.raises(GraphValidationError, match="GraphInfo"):
        check_layout(graph, LAYOUT, wrong)


@pytest.mark.parametrize("dropped", sorted(EDGE_KEYS))
def test_one_edge_key_without_the_others_is_refused(batch, dropped):
    graph, _, info = batch
    del graph[dropped]
    with pytest.raises(GraphValidationError, match="edge keys come together"):
        check_layout(graph, LAYOUT, info)


def test_the_float_dtype_is_bound_by_the_caller(batch):
    graph, _, info = batch
    with pytest.raises(GraphValidationError, match="float32"):
        check_layout(graph, LAYOUT, info, float_dtype="float32")
    as_float32 = {
        key: value.astype(np.float32) if value.dtype == np.float64 else value
        for key, value in graph.items()
    }
    check_layout(as_float32, LAYOUT, info, float_dtype="float32")


@pytest.mark.parametrize("float_dtype", ["float6", "int64", "double"])
def test_a_float_dtype_that_is_not_a_floating_dtype_name_is_refused(batch, float_dtype):
    graph, _, info = batch
    with pytest.raises(ValueError, match="float_dtype"):
        check_layout(graph, LAYOUT, info, float_dtype=float_dtype)


def test_the_optional_core_keys_may_be_absent_or_present(batch):
    graph, _, info = batch
    del graph["head"]
    check_layout(graph, LAYOUT, info)
    graph["displacement"] = np.zeros((2, 3, 3))
    check_layout(graph, LAYOUT, info)


def test_an_embedding_feature_is_checked_on_its_leading_dimension_only():
    layout = default_layout(ELEMENTS, cutoff=CUTOFF, features={"spin_state": "atom"})
    graph, _, info = _batch(
        _water(spin_state=np.zeros((3, 4))),
        _crystal(spin_state=np.zeros((2, 4))),
        layout=layout,
    )
    assert graph["spin_state"].shape == (5, 4)
    check_layout(graph, layout, info)
    graph["spin_state"] = np.zeros((4, 4))
    with pytest.raises(GraphValidationError, match="'spin_state' has shape"):
        check_layout(graph, layout, info)


# --- check_batch_invariants: the index values, in tests and validate=True -------------


def test_an_edge_to_a_node_that_does_not_exist_is_refused(batch):
    graph, _, info = batch
    graph["edge_index"] = graph["edge_index"].copy()
    graph["edge_index"][0, 0] = 5
    with pytest.raises(GraphValidationError, match=r"\[0, 5\)"):
        check_batch_invariants(graph, info)


def test_an_edge_between_structures_is_refused(batch):
    graph, _, info = batch
    graph["edge_index"] = graph["edge_index"].copy()
    graph["edge_index"][1, 0] = 4
    with pytest.raises(GraphValidationError, match="between two structures"):
        check_batch_invariants(graph, info)


@pytest.mark.parametrize("ptr", [[1, 3, 5], [0, 6, 5], [0, 3, 6], [0, 3, 4]])
def test_a_ptr_that_does_not_partition_the_nodes_is_refused(batch, ptr):
    graph, _, info = batch
    graph["ptr"] = np.array(ptr)
    with pytest.raises(GraphValidationError, match="'ptr'"):
        check_batch_invariants(graph, info)


def test_a_batch_inconsistent_with_ptr_is_refused(batch):
    graph, _, info = batch
    graph["batch"] = np.array([0, 0, 1, 1, 1])
    with pytest.raises(GraphValidationError, match="'batch' does not match 'ptr'"):
        check_batch_invariants(graph, info)


def test_structure_sizes_that_disagree_with_ptr_are_refused(batch):
    graph, _, info = batch
    wrong = dataclasses.replace(info, structure_sizes=(2, 3))
    with pytest.raises(GraphValidationError, match="structure_sizes"):
        check_batch_invariants(graph, wrong)


def test_the_batch_invariants_need_no_info(batch):
    graph, _, _ = batch
    check_batch_invariants(graph)


# --- GraphInfo and GraphView ----------------------------------------------------------


def test_graph_info_refuses_sizes_that_disagree_with_its_counts():
    with pytest.raises(ValueError, match="entries"):
        GraphInfo(num_graphs=2, num_nodes=5, num_edges=0, structure_sizes=(5,))
    with pytest.raises(ValueError, match="sums to"):
        GraphInfo(num_graphs=2, num_nodes=5, num_edges=0, structure_sizes=(3, 3))


def test_graph_info_marks_padding_structures_from_its_padding_info():
    unpadded = GraphInfo(num_graphs=2, num_nodes=5, num_edges=0, structure_sizes=(3, 2))
    assert unpadded.is_padding == (False, False)
    padded = GraphInfo(
        num_graphs=4,
        num_nodes=8,
        num_edges=0,
        structure_sizes=(3, 2, 0, 3),
        padding=PaddingInfo(n_real_nodes=5, n_real_edges=0, n_real_graphs=2),
    )
    assert padded.is_padding == (False, False, True, True)


def test_graph_info_cannot_be_edited():
    info = GraphInfo(num_graphs=1, num_nodes=1, num_edges=0, structure_sizes=(1,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.num_nodes = 2  # ty: ignore[invalid-assignment]


def test_the_view_reads_the_dict_without_copying(batch):
    graph, _, info = batch
    view = GraphView(graph, LAYOUT, info)
    assert view.positions is graph["positions"]
    assert view["cell"] is graph["cell"]
    assert "edge_index" in view
    assert view.info.num_graphs == 2
    with pytest.raises(AttributeError, match="displacement"):
        _ = view.displacement


def test_the_view_checks_the_layout(batch):
    graph, _, info = batch
    graph["rcell"] = np.zeros((2, 3, 3))
    with pytest.raises(GraphValidationError, match="rcell"):
        GraphView(graph, LAYOUT, info)


# --- framework independence -----------------------------------------------------------


def test_the_graph_package_imports_no_framework():
    """In a fresh interpreter, since a pytest session may have imported a
    framework already."""
    probe = (
        "import json, sys, mace_core.graph\n"
        "print(json.dumps(sorted({m.split('.', 1)[0] for m in sys.modules})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    modules = set(json.loads(result.stdout.splitlines()[-1]))
    assert "mace_core" in modules
    assert not modules & {"torch", "jax", "jaxlib"}
