"""The flat graph dict: the schema as data, the validator, the numpy reference
collation, and the step from a Configuration to a sample.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from mace_core.data import AtomicNumberTable, Configuration
from mace_core.graph import (
    FLOAT,
    GRAPH_SCHEMA,
    HOST_KEYS,
    LABEL_KEYS,
    GraphInfo,
    GraphValidationError,
    GraphView,
    collate,
    graph_sample,
    presence_key,
    validate_graph,
)

TABLE = AtomicNumberTable([1, 8, 14])
CUTOFF = 2.5


def _water(**properties):
    return Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        properties=properties,
    )


def _crystal(**properties):
    return Configuration(
        atomic_numbers=np.array([14, 14]),
        positions=np.array([[0.0, 0.0, 0.0], [0.91, 0.73, 0.62]]),
        cell=np.array([[2.47, 0.0, 0.0], [0.83, 2.39, 0.0], [0.31, 0.57, 2.61]]),
        pbc=(True, True, True),
        properties=properties,
    )


def _graph(*structures, **kwargs):
    samples = [
        graph_sample(s, atomic_number_table=TABLE, cutoff=CUTOFF, **kwargs)
        for s in structures
    ]
    return collate(samples, features=kwargs.get("features"))


# --- the schema is data -------------------------------------------------------


EXPECTED_KEYS = {
    "positions", "node_attrs", "node_atomic_numbers", "edge_index", "shifts",
    "unit_shifts", "cell", "pbc", "rcell", "volume", "batch", "ptr", "head",
    "displacement", "magmom", "weight", "energy_weight", "forces_weight",
    "stress_weight", "virials_weight", "charges_weight", "magforces_weight",
    "dipole_weight", "polarizability_weight", "energy", "forces", "magforces",
    "stress", "virials", "dipole", "charges", "polarizability", "total_charge",
    "total_spin", "elec_temp", "fermi_level", "external_field",
    "density_coefficients", "structure_sizes", "structure_periodic",
    "structure_padding",
}  # fmt: skip


def test_the_schema_has_exactly_these_keys_and_one_presence_flag_per_label():
    flags = {presence_key(label) for label in LABEL_KEYS}
    assert set(GRAPH_SCHEMA) == EXPECTED_KEYS | flags
    assert {GRAPH_SCHEMA[f].flags for f in flags} == LABEL_KEYS


def test_the_schema_cannot_be_edited_in_place():
    with pytest.raises(TypeError):
        cast(dict, GRAPH_SCHEMA)["positions"] = GRAPH_SCHEMA["cell"]


@pytest.mark.parametrize(
    ("key", "shape", "dtype", "required", "pad_value"),
    [
        ("positions", ("n_nodes", 3), FLOAT, True, "zero"),
        ("node_attrs", ("n_nodes", "n_elements"), FLOAT, True, "element_zero_one_hot"),
        (
            "node_atomic_numbers",
            ("n_nodes",),
            "int64",
            True,
            "element_zero_atomic_number",
        ),
        ("edge_index", (2, "n_edges"), "int64", True, "self_loop"),
        ("shifts", ("n_edges", 3), FLOAT, True, "padding_shift"),
        ("unit_shifts", ("n_edges", 3), FLOAT, True, "unit_x"),
        ("cell", ("n_graphs", 3, 3), FLOAT, True, "padding_cell"),
        ("pbc", ("n_graphs", 3), "bool", True, "false"),
        ("rcell", ("n_graphs", 3, 3), FLOAT, True, "padding_reciprocal_cell"),
        ("volume", ("n_graphs",), FLOAT, True, "padding_volume"),
        ("batch", ("n_nodes",), "int64", True, "padding_graph_id"),
        ("ptr", ("n_graphs_plus_one",), "int64", True, "node_count"),
        ("head", ("n_graphs",), "int64", False, "zero"),
        ("total_spin", ("n_graphs",), FLOAT, False, "one"),
        ("total_charge", ("n_graphs",), FLOAT, False, "zero"),
        ("dipole_weight", ("n_graphs", 3), FLOAT, False, "zero"),
        ("polarizability_weight", ("n_graphs", 3, 3), FLOAT, False, "zero"),
        ("forces_present", ("n_nodes",), "bool", False, "false"),
        ("energy_present", ("n_graphs",), "bool", False, "false"),
        (
            "density_coefficients",
            ("n_nodes", "n_density_coefficients"),
            FLOAT,
            False,
            "zero",
        ),
        ("structure_sizes", ("n_graphs",), "int", True, "padding_structure_size"),
        ("structure_padding", ("n_graphs",), "bool", True, "true"),
    ],
)
def test_a_schema_row(key, shape, dtype, required, pad_value):
    spec = GRAPH_SCHEMA[key]
    assert (spec.shape, spec.dtype, spec.required, spec.pad_value) == (
        shape,
        dtype,
        required,
        pad_value,
    )


def test_the_host_keys_are_the_three_structure_tuples():
    assert {"structure_sizes", "structure_periodic", "structure_padding"} == HOST_KEYS


# --- validate_graph ---------------------------------------------------------------


@pytest.fixture(name="graph")
def fixture_graph():
    return _graph(_water(energy=-1.0), _crystal(energy=-2.0))


def test_a_collated_graph_validates(graph):
    assert validate_graph(graph) == GraphInfo(
        num_graphs=2, num_nodes=5, num_edges=graph["edge_index"].shape[1]
    )


def test_a_missing_required_key_is_named(graph):
    del graph["unit_shifts"]
    with pytest.raises(GraphValidationError, match="unit_shifts"):
        validate_graph(graph)


def test_inconsistent_symbolic_dimensions_are_refused(graph):
    graph["forces"] = np.zeros((4, 3))
    graph["forces_present"] = np.zeros(4, dtype=bool)
    with pytest.raises(GraphValidationError, match="n_nodes is 5"):
        validate_graph(graph)


def test_a_wrong_dtype_name_is_refused(graph):
    graph["edge_index"] = graph["edge_index"].astype(np.int32)
    with pytest.raises(GraphValidationError, match="int32"):
        validate_graph(graph)


def test_the_float_dtype_is_bound_by_the_caller(graph):
    with pytest.raises(GraphValidationError, match="float32"):
        validate_graph(graph, float_dtype="float32")


def test_an_edge_to_a_node_that_does_not_exist_is_refused(graph):
    graph["edge_index"] = graph["edge_index"].copy()
    graph["edge_index"][0, 0] = 5
    with pytest.raises(GraphValidationError, match=r"\[0, 5\)"):
        validate_graph(graph)


@pytest.mark.parametrize("ptr", [[1, 3, 5], [0, 4, 3], [0, 3, 6]])
def test_a_ptr_that_does_not_partition_the_nodes_is_refused(graph, ptr):
    graph["ptr"] = np.array(ptr)
    with pytest.raises(GraphValidationError, match="ptr"):
        validate_graph(graph)


def test_a_batch_inconsistent_with_ptr_is_refused(graph):
    graph["batch"] = np.array([0, 0, 1, 1, 1])
    with pytest.raises(GraphValidationError, match="batch"):
        validate_graph(graph)


def test_a_label_without_its_presence_flag_is_refused(graph):
    del graph["energy_present"]
    with pytest.raises(GraphValidationError, match="energy_present"):
        validate_graph(graph)


def test_labels_can_be_required(graph):
    validate_graph(graph, require_labels=True)
    unlabelled = _graph(_water())
    with pytest.raises(GraphValidationError, match="no structure"):
        validate_graph(unlabelled, require_labels=True)


def test_extra_keys_are_carried_but_can_be_refused(graph):
    graph["my_annotation"] = ("a", "b")
    validate_graph(graph)
    with pytest.raises(GraphValidationError, match="my_annotation"):
        validate_graph(graph, allow_extra=False)


@pytest.mark.parametrize("key", ["num_graphs", "edge_lengths", "spin_present"])
def test_keys_that_collide_with_the_schema_are_refused(graph, key):
    graph[key] = np.zeros(1)
    with pytest.raises(GraphValidationError, match=key):
        validate_graph(graph)


def test_host_keys_must_be_tuples_of_python_values(graph):
    graph["structure_sizes"] = np.array([3, 2])
    with pytest.raises(GraphValidationError, match="host value"):
        validate_graph(graph)
    graph["structure_sizes"] = (np.int64(3), np.int64(2))
    with pytest.raises(GraphValidationError, match="Python int"):
        validate_graph(graph)


def test_host_keys_must_agree_with_ptr_and_pbc(graph):
    graph["structure_sizes"] = (2, 3)
    with pytest.raises(GraphValidationError, match="structure_sizes"):
        validate_graph(graph)
    graph["structure_sizes"] = (3, 2)
    graph["structure_periodic"] = (True, True)
    with pytest.raises(GraphValidationError, match="structure_periodic"):
        validate_graph(graph)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("edge_index", np.zeros(3, dtype=np.int64), "has shape"),
        ("positions", [[0.0, 0.0, 0.0]] * 5, "is a list"),
        ("ptr", np.zeros((2, 2), dtype=np.int64), "has shape"),
    ],
)
def test_malformed_arrays_raise_the_validation_error(graph, key, value, message):
    graph[key] = value
    with pytest.raises(GraphValidationError, match=message):
        validate_graph(graph)


def test_a_misspelled_float_dtype_is_refused(graph):
    with pytest.raises(ValueError, match="float_dtype"):
        validate_graph(graph, float_dtype="float6")


def test_padding_structures_must_come_last(graph):
    graph["structure_padding"] = (True, False)
    with pytest.raises(GraphValidationError, match="appended"):
        validate_graph(graph)


def test_a_periodic_structure_cannot_be_padding(graph):
    graph["structure_padding"] = (False, True)
    with pytest.raises(GraphValidationError, match="aperiodic"):
        validate_graph(graph)


def test_a_labelled_structure_cannot_be_padding():
    graph = _graph(_crystal(energy=-2.0), _water(energy=-1.0))
    graph["structure_padding"] = (False, True)
    graph["positions"] = graph["positions"].copy()
    graph["positions"][2:] = 0.0
    with pytest.raises(GraphValidationError, match="energy_present"):
        validate_graph(graph)


def test_the_view_reads_the_dict_without_copying(graph):
    view = GraphView(graph)
    assert view.positions is graph["positions"]
    assert view.info.num_graphs == 2
    with pytest.raises(AttributeError, match="density_coefficients"):
        _ = view.density_coefficients


# --- the numpy reference collation --------------------------------------------------


def test_edge_index_is_offset_and_ptr_batch_cover_the_nodes(graph):
    water = graph_sample(_water(energy=-1.0), atomic_number_table=TABLE, cutoff=CUTOFF)
    n_water_edges = water["edge_index"].shape[1]
    assert graph["ptr"].tolist() == [0, 3, 5]
    assert graph["batch"].tolist() == [0, 0, 0, 1, 1]
    assert graph["edge_index"][:, n_water_edges:].min() == 3


def test_graph_level_values_are_stacked_on_a_new_leading_dimension(graph):
    assert graph["cell"].shape == (2, 3, 3)
    assert graph["energy"].tolist() == [-1.0, -2.0]
    assert graph["pbc"].tolist() == [[False, False, False], [True, True, True]]


def test_rcell_is_two_pi_times_the_inverse_transpose(graph):
    np.testing.assert_allclose(
        graph["rcell"] @ np.swapaxes(graph["cell"], -1, -2),
        np.broadcast_to(2 * np.pi * np.eye(3), (2, 3, 3)),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(graph["volume"], np.linalg.det(graph["cell"]))


def test_a_sample_carrying_a_collated_key_is_refused():
    sample = graph_sample(_water(), atomic_number_table=TABLE, cutoff=CUTOFF)
    sample["batch"] = np.zeros(3, dtype=np.int64)
    with pytest.raises(ValueError, match="collation computes"):
        collate([sample])


def test_collate_needs_a_sample():
    with pytest.raises(ValueError, match="at least one"):
        collate([])


# --- the step from a Configuration to a sample ----------------------------------------


def test_the_head_is_its_index_in_the_models_head_list():
    structure = Configuration(
        atomic_numbers=np.array([1]), positions=np.zeros((1, 3)), head="pbe"
    )
    sample = graph_sample(
        structure, atomic_number_table=TABLE, cutoff=CUTOFF, heads=("Default", "pbe")
    )
    assert sample["head"] == 1


def test_an_unknown_head_is_an_error_rather_than_a_fallback():
    structure = Configuration(
        atomic_numbers=np.array([1]), positions=np.zeros((1, 3)), head="r2scan"
    )
    with pytest.raises(ValueError, match="r2scan"):
        graph_sample(
            structure,
            atomic_number_table=TABLE,
            cutoff=CUTOFF,
            heads=("Default", "pbe"),
        )


def test_heads_that_repeat_a_name_are_refused():
    with pytest.raises(ValueError, match="repeat"):
        graph_sample(
            _water(), atomic_number_table=TABLE, cutoff=CUTOFF, heads=("a", "a")
        )


def test_pbc_given_as_a_numpy_array_is_read():
    structure = _crystal()
    structure.pbc = np.array([True, True, True])
    sample = graph_sample(structure, atomic_number_table=TABLE, cutoff=CUTOFF)
    assert sample["pbc"].tolist() == [True, True, True]


def test_an_undeclared_label_is_written_absent_with_zero_weight():
    sample = graph_sample(_water(energy=-1.0), atomic_number_table=TABLE, cutoff=CUTOFF)
    assert set(LABEL_KEYS) <= set(sample)
    assert not sample["dipole_present"]
    assert sample["dipole_weight"].tolist() == [0.0, 0.0, 0.0]
    assert sample["energy_present"]


def test_structures_parsed_with_different_key_specifications_share_a_batch():
    """One head reads only energies, another energies and forces: both
    structures carry every label, and the flags say which are present."""
    energy_only = _water(energy=-1.0)
    with_forces = _crystal(energy=-2.0, forces=np.ones((2, 3)), total_charge=1.0)
    graph = _graph(energy_only, with_forces)
    assert graph["energy_present"].tolist() == [True, True]
    assert graph["forces_present"].tolist() == [False] * 3 + [True] * 2
    assert graph["total_charge"].tolist() == [0.0, 1.0]
    validate_graph(graph, require_labels=True)


def test_a_present_label_with_zero_weight_is_still_present():
    structure = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=_water().positions,
        properties={"energy": -1.0, "forces": np.ones((3, 3))},
        property_weights={"energy": 0.0, "forces": 0.0},
    )
    sample = graph_sample(structure, atomic_number_table=TABLE, cutoff=CUTOFF)
    assert sample["energy_present"]
    assert sample["forces_present"].tolist() == [True, True, True]
    assert sample["energy_weight"] == 0.0


def test_an_absent_label_is_zero_and_flagged_absent():
    sample = graph_sample(
        _water(energy=None, stress=None), atomic_number_table=TABLE, cutoff=CUTOFF
    )
    assert not sample["energy_present"]
    assert sample["energy"] == 0.0
    assert sample["stress"].shape == (3, 3)
    assert not sample["stress"].any()


def test_an_absent_total_spin_is_one_and_other_inputs_are_zero():
    sample = graph_sample(
        _water(total_spin=None, total_charge=None, magmom=None),
        atomic_number_table=TABLE,
        cutoff=CUTOFF,
    )
    assert sample["total_spin"] == 1.0
    assert sample["total_charge"] == 0.0
    assert sample["magmom"].shape == (3, 3)


def test_the_one_hot_is_over_the_element_table():
    sample = graph_sample(_water(), atomic_number_table=TABLE, cutoff=CUTOFF)
    assert sample["node_attrs"].tolist() == [[0, 1, 0], [1, 0, 0], [1, 0, 0]]
    assert sample["node_atomic_numbers"].tolist() == [8, 1, 1]


def test_an_element_outside_the_table_is_refused():
    structure = Configuration(atomic_numbers=np.array([26]), positions=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="26"):
        graph_sample(structure, atomic_number_table=TABLE, cutoff=CUTOFF)


def test_an_undeclared_property_is_transported_untouched():
    """The legacy builder promoted an unknown one-dimensional property to
    ``(n, 1)``. Nothing here guesses: the value is carried as it came."""
    tags = np.array([1.0, 2.0, 3.0])
    sample = graph_sample(_water(tags=tags), atomic_number_table=TABLE, cutoff=CUTOFF)
    assert sample["tags"] is tags
    graph = collate([sample])
    assert isinstance(graph["tags"], tuple)
    assert graph["tags"][0] is tags


def test_a_declared_feature_is_batched_by_its_storage():
    structures = [
        _water(spin_state=np.array([0, 1, 1])),
        _crystal(spin_state=np.array([2, 2])),
    ]
    graph = _graph(*structures, features={"spin_state": "atom"})
    assert graph["spin_state"].tolist() == [0, 1, 1, 2, 2]
    per_graph = _graph(
        _water(phase=3.0), _crystal(phase=4.0), features={"phase": "graph"}
    )
    assert per_graph["phase"].tolist() == [3.0, 4.0]


def test_a_declared_atom_feature_with_the_wrong_rows_is_not_reshaped():
    with pytest.raises(ValueError, match="not reshaped"):
        graph_sample(
            _water(spin_state=np.array([0, 1])),
            atomic_number_table=TABLE,
            cutoff=CUTOFF,
            features={"spin_state": "atom"},
        )


@pytest.mark.parametrize("storage", ["atoms", "structure", ""])
def test_a_misspelled_feature_storage_is_refused(storage):
    with pytest.raises(ValueError, match="use 'atom' or 'graph'"):
        graph_sample(
            _water(spin_state=np.array([0, 1, 1])),
            atomic_number_table=TABLE,
            cutoff=CUTOFF,
            features={"spin_state": storage},
        )


@pytest.mark.parametrize(
    ("name", "storage"),
    [
        ("energy", "graph"),
        ("positions", "atom"),
        ("total_charge", "graph"),
        ("batch", "atom"),
        ("energy_present", "graph"),
        ("spin_present", "graph"),
        ("num_nodes", "graph"),
    ],
)
def test_a_feature_named_like_a_schema_key_is_refused(name, storage):
    with pytest.raises(ValueError, match="named like a key of the graph schema"):
        graph_sample(
            _water(), atomic_number_table=TABLE, cutoff=CUTOFF, features={name: storage}
        )
    with pytest.raises(ValueError, match="named like a key of the graph schema"):
        collate(
            [graph_sample(_water(), atomic_number_table=TABLE, cutoff=CUTOFF)],
            features={name: storage},
        )


def test_a_declared_feature_without_a_value_is_an_error():
    with pytest.raises(ValueError, match="cannot be made up"):
        graph_sample(
            _water(spin_state=None),
            atomic_number_table=TABLE,
            cutoff=CUTOFF,
            features={"spin_state": "atom"},
        )


def test_a_property_named_like_a_built_key_is_refused():
    with pytest.raises(ValueError, match="edge_index"):
        graph_sample(
            _water(edge_index=np.zeros(2)), atomic_number_table=TABLE, cutoff=CUTOFF
        )


def test_building_a_sample_does_not_modify_the_configuration():
    structure = _crystal(energy=-2.0)
    positions, cell = structure.positions.copy(), structure.cell.copy()
    graph_sample(structure, atomic_number_table=TABLE, cutoff=CUTOFF)
    assert np.array_equal(structure.positions, positions)
    assert np.array_equal(structure.cell, cell)


def test_the_configuration_keeps_its_physical_cell():
    """The effective cell lives on the graph only."""
    structure = _water()
    sample = graph_sample(structure, atomic_number_table=TABLE, cutoff=CUTOFF)
    assert structure.cell is None
    assert np.linalg.det(sample["cell"]) > 0
