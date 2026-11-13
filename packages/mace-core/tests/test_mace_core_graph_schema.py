"""The flat graph dict: its structural core, the layout a run declares, the
step from a Configuration to a sample, the numpy reference collation, and the
checks split by cost.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import numpy as np
import pytest
from mace_core.data import Configuration
from mace_core.graph import (
    EDGE_KEYS,
    GRAPH_CORE,
    FieldSpec,
    GraphInfo,
    GraphLayout,
    GraphValidationError,
    SampleError,
    check_batch_invariants,
    check_layout,
    collate,
    collate_targets,
    graph_layout,
    graph_sample,
)
from mace_core.graph.layout import input_field, target_field
from mace_core.neighbors import AllPairsBackend, NeighborList, get_neighborhood
from mace_core.observables import InputSpec, ObservableSpec

ELEMENTS = (1, 8, 14)
CUTOFF = 2.5
BACKEND = AllPairsBackend()

ENERGY = ObservableSpec(name="energy", irreps="0e", per_atom=False, units="eV")
FORCES = ObservableSpec(name="forces", irreps="1o", per_atom=True, units="eV/Å")
STRESS = ObservableSpec(name="stress", irreps="0e+2e", per_atom=False, units="eV/Å^3")
VIRIALS = ObservableSpec(name="virials", irreps="0e+2e", per_atom=False, units="eV")
DIPOLE = ObservableSpec(name="dipole", irreps="1o", per_atom=False, units="e·Å")
POLARIZABILITY = ObservableSpec(
    name="polarizability", irreps="0e+1e+2e", per_atom=False, units="e·Å²/V"
)
TOTAL_CHARGE = InputSpec(name="total_charge", irreps="0e", per_atom=False, units="e")
TOTAL_SPIN = InputSpec(name="total_spin", irreps="0e", per_atom=False, units="1")
MAGMOM = InputSpec(name="magmom", irreps="1e", per_atom=True, units="μB")

LAYOUT = graph_layout(
    ELEMENTS,
    cutoff=CUTOFF,
    inputs=[TOTAL_CHARGE, TOTAL_SPIN, MAGMOM],
    targets=[ENERGY, FORCES, STRESS, VIRIALS, DIPOLE, POLARIZABILITY],
)

WATER_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
CRYSTAL_CELL = np.array([[2.47, 0.0, 0.0], [0.83, 2.39, 0.0], [0.31, 0.57, 2.61]])


def _properties(layout, given):
    """Every name the layout declares, ``None`` unless given, as a reader
    resolving the run's key specification produces them."""
    declared = {
        name: None for name in (*layout.inputs, *layout.features, *layout.targets)
    }
    return {**declared, **given}


def _water(layout=LAYOUT, **properties):
    return Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=WATER_POSITIONS.copy(),
        properties=_properties(layout, properties),
    )


def _crystal(layout=LAYOUT, **properties):
    return Configuration(
        atomic_numbers=np.array([14, 14]),
        positions=np.array([[0.0, 0.0, 0.0], [0.91, 0.73, 0.62]]),
        cell=CRYSTAL_CELL.copy(),
        pbc=(True, True, True),
        properties=_properties(layout, properties),
    )


def _slab(cell, layout=LAYOUT, **properties):
    """Two silicon atoms periodic in x and y, open along z."""
    return Configuration(
        atomic_numbers=np.array([14, 14]),
        positions=np.array([[0.0, 0.0, 0.0], [1.2, 1.2, 1.0]]),
        cell=np.asarray(cell, dtype=float),
        pbc=(True, True, False),
        properties=_properties(layout, properties),
    )


def _atom(z=1, layout=LAYOUT, **fields):
    return Configuration(
        atomic_numbers=np.array([z]),
        positions=np.zeros((1, 3)),
        properties=_properties(layout, {}),
        **fields,
    )


def _search(structure, cutoff=CUTOFF):
    return get_neighborhood(
        structure.positions,
        cutoff,
        pbc=structure.pbc,
        cell=structure.cell,
        backend=BACKEND,
    )


def _sample(structure, layout=LAYOUT, *, edges=True, **kwargs):
    neighbors = _search(structure, layout.cutoff) if edges else None
    return graph_sample(structure, neighbors, layout, **kwargs)


def _batch(*structures, layout=LAYOUT, edges=True):
    pairs = [_sample(s, layout, edges=edges) for s in structures]
    graph, info = collate([graph for graph, _ in pairs], layout)
    targets = collate_targets([targets for _, targets in pairs], layout)
    return graph, targets, info


def _layout(**fields):
    defaults = {"atomic_numbers": ELEMENTS, "heads": ("Default",), "cutoff": CUTOFF}
    return GraphLayout.model_validate({**defaults, **fields})


def _with(layout, **update):
    """``layout`` with ``update`` applied, validated as a new layout."""
    return GraphLayout.model_validate({**layout.model_dump(), **update})


def _born_charges():
    """A per-atom tensor observable outside the default names: nine components."""
    return ObservableSpec(
        name="born_charges", irreps="0e+1e+2e", per_atom=True, units="e"
    )


SCALAR_PER_STRUCTURE = FieldSpec(shape=("n_graphs",), dtype="float")


# --- the structural core ---------------------------------------------------------


def test_the_core_is_the_written_out_contract():
    """One snapshot of the whole core: a change here is a schema change."""
    assert {
        key: (spec.shape, spec.dtype, spec.required) for key, spec in GRAPH_CORE.items()
    } == {
        "positions": (("n_nodes", 3), "float", True),
        "node_atomic_numbers": (("n_nodes",), "int64", True),
        "edge_index": ((2, "n_edges"), "int64", True),
        "shifts": (("n_edges", 3), "float", True),
        "unit_shifts": (("n_edges", 3), "float", True),
        "cell": (("n_graphs", 3, 3), "float", True),
        "pbc": (("n_graphs", 3), "bool", True),
        "batch": (("n_nodes",), "int64", True),
        "ptr": (("n_graphs_plus_one",), "int64", True),
        "head": (("n_graphs",), "int64", False),
        "displacement": (("n_graphs", 3, 3), "float", False),
    }


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        (("n_nodes", -1), "float"),
        (("n_nodes", ""), "float"),
        (("n_nodes",), "float32"),
    ],
)
def test_a_malformed_field_spec_is_refused(shape, dtype):
    with pytest.raises(ValueError):
        FieldSpec(shape=shape, dtype=dtype)


# --- the layout a run declares ------------------------------------------------------


@pytest.mark.parametrize("heads", [(), ("pbe", "pbe"), ("a", "b", "a")])
def test_a_head_list_that_is_empty_or_repeats_a_name_is_refused(heads):
    with pytest.raises(ValueError, match="head list"):
        graph_layout(ELEMENTS, cutoff=CUTOFF, heads=heads)


@pytest.mark.parametrize("heads", [(0,), ("",), ("Default", " ")])
def test_a_head_that_is_not_a_name_is_refused(heads):
    with pytest.raises(ValueError, match="head"):
        _layout(heads=heads)


@pytest.mark.parametrize("table", [(), (1, 1), (0, 1), (1, 119), (-8,), (1.7, 8.2)])
def test_an_element_table_that_is_empty_repeats_or_is_not_a_set_of_elements_is_refused(
    table,
):
    with pytest.raises(ValueError, match=r"element table|atomic numbers"):
        graph_layout(table, cutoff=CUTOFF)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("nan"), float("inf"), True])
def test_a_cutoff_that_is_not_a_positive_finite_distance_is_refused(cutoff):
    with pytest.raises(ValueError, match="cutoff"):
        graph_layout(ELEMENTS, cutoff=cutoff)


@pytest.mark.parametrize("name", sorted(GRAPH_CORE))
@pytest.mark.parametrize("kind", ["inputs", "targets", "requirement_fields"])
def test_a_declared_name_colliding_with_the_core_is_refused(name, kind):
    with pytest.raises(ValueError, match="named like a key of the graph core"):
        _layout(**{kind: {name: SCALAR_PER_STRUCTURE}})


@pytest.mark.parametrize("name", ["energy_present", "spin_present", "spin_weight"])
@pytest.mark.parametrize("kind", ["inputs", "targets", "requirement_fields"])
def test_a_declared_name_ending_like_a_flag_or_a_weight_is_refused(name, kind):
    with pytest.raises(ValueError, match="presence flag or a loss weight"):
        _layout(**{kind: {name: SCALAR_PER_STRUCTURE}})


@pytest.mark.parametrize("name", ["", "  "])
def test_an_empty_declared_name_is_refused(name):
    with pytest.raises(ValueError, match="empty name"):
        _layout(inputs={name: SCALAR_PER_STRUCTURE})


def test_a_target_named_like_the_structure_weight_is_refused():
    weight = ObservableSpec(name="weight", irreps="0e", per_atom=False, units="1")
    with pytest.raises(ValueError, match="loss weight of a whole structure"):
        graph_layout(ELEMENTS, cutoff=CUTOFF, targets=[weight])


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("inputs", "targets"),
        ("inputs", "requirement_fields"),
        ("targets", "requirement_fields"),
    ],
)
def test_a_name_declared_as_two_kinds_of_field_is_refused(first, second):
    with pytest.raises(ValueError, match="declared twice"):
        _layout(
            **{first: {"spin": SCALAR_PER_STRUCTURE}},
            **{second: {"spin": SCALAR_PER_STRUCTURE}},
        )


def test_a_name_declared_twice_is_refused_whatever_the_order():
    scalar = ObservableSpec(name="born", irreps="0e", per_atom=False, units="e")
    vector = ObservableSpec(name="born", irreps="1o", per_atom=True, units="e")
    for targets in ([scalar, vector], [vector, scalar]):
        with pytest.raises(ValueError, match="'born' is declared twice"):
            graph_layout(ELEMENTS, cutoff=CUTOFF, targets=targets)
    with pytest.raises(ValueError, match="'total_charge' is declared twice"):
        graph_layout(ELEMENTS, cutoff=CUTOFF, inputs=[TOTAL_CHARGE, TOTAL_CHARGE])


def test_a_feature_named_like_a_declared_input_is_refused():
    with pytest.raises(ValueError, match="declared twice"):
        graph_layout(
            ELEMENTS,
            cutoff=CUTOFF,
            inputs=[TOTAL_CHARGE],
            features={"total_charge": ("graph", ())},
        )


@pytest.mark.parametrize("storage", ["atoms", "structure", ""])
def test_a_misspelled_feature_storage_is_refused(storage):
    with pytest.raises(ValueError, match="use 'atom' or 'graph'"):
        graph_layout(ELEMENTS, cutoff=CUTOFF, features={"spin_state": (storage, ())})


@pytest.mark.parametrize(
    ("value_shape", "message"),
    [
        ((0,), "positive integer"),
        (("width",), "positive integer"),
        ((-2,), "negative size"),
    ],
)
def test_a_feature_width_that_is_not_a_positive_integer_is_refused(
    value_shape, message
):
    with pytest.raises(ValueError, match=message):
        graph_layout(
            ELEMENTS, cutoff=CUTOFF, features={"spin_state": ("atom", value_shape)}
        )


def test_a_layout_carries_only_what_is_declared():
    layout = graph_layout(ELEMENTS, cutoff=CUTOFF, targets=[ENERGY])
    assert set(layout.graph_fields()) == set(GRAPH_CORE)
    assert set(layout.target_fields()) == {
        "weight",
        "energy",
        "energy_present",
        "energy_weight",
    }


def test_a_layout_round_trips_through_json():
    layout = graph_layout(
        ELEMENTS,
        cutoff=CUTOFF,
        heads=("Default", "pbe"),
        float_dtype="float32",
        inputs=[TOTAL_SPIN],
        features={"spin_state": ("atom", ()), "phase": ("graph", (4,))},
        targets=[ENERGY, _born_charges()],
        requirement_fields={"pairs": FieldSpec(shape=(2, "n_pairs"), dtype="int64")},
    )
    assert GraphLayout.model_validate_json(layout.model_dump_json()) == layout


# --- declared fields ------------------------------------------------------------------


def test_an_input_field_takes_its_level_from_per_atom_and_its_shape_from_the_irreps():
    def shape(name, irreps, per_atom):
        spec = InputSpec(name=name, irreps=irreps, per_atom=per_atom, units="a.u.")
        return input_field(spec).shape

    assert shape("hubbard_u", "0e", False) == ("n_graphs",)
    assert shape("site_field", "1o", True) == ("n_nodes", 3)
    assert shape("site_field", "0e+1o", True) == ("n_nodes", 4)
    assert shape("charge_states", "2x0e", False) == ("n_graphs", 2)


@pytest.mark.parametrize("irreps", ["0e+2e", "0e+1e+2e", "9x0e"])
def test_a_rank_two_default_name_takes_nine_components_or_the_symmetric_pair(irreps):
    stress = ObservableSpec(name="stress", irreps=irreps, per_atom=False, units="eV")
    assert target_field(stress).shape == ("n_graphs", 3, 3)


@pytest.mark.parametrize(
    ("name", "irreps", "per_atom"),
    [
        ("stress", "0e", False),
        ("stress", "2e", False),
        ("forces", "0e", True),
        ("energy", "1o", False),
    ],
)
def test_irreps_that_do_not_fill_a_default_names_shape_are_refused(
    name, irreps, per_atom
):
    spec = ObservableSpec(name=name, irreps=irreps, per_atom=per_atom, units="eV")
    with pytest.raises(ValueError, match="Declare irreps that fill it"):
        target_field(spec)
    with pytest.raises(ValueError, match="Declare irreps that fill it"):
        input_field(InputSpec(name=name, irreps=irreps, per_atom=per_atom, units="eV"))


@pytest.mark.parametrize(
    ("name", "irreps", "per_atom"),
    [("forces", "1o", False), ("energy", "0e", True), ("magmom", "1e", False)],
)
def test_a_default_name_declared_at_the_wrong_level_is_refused(name, irreps, per_atom):
    """The data stores a default name per atom or per structure, and a
    declaration that disagrees would fail on every structure instead."""
    with pytest.raises(ValueError, match=f"per_atom={not per_atom}"):
        graph_layout(
            ELEMENTS,
            cutoff=CUTOFF,
            targets=[
                ObservableSpec(name=name, irreps=irreps, per_atom=per_atom, units="x")
            ],
        )
    with pytest.raises(ValueError, match=f"per_atom={not per_atom}"):
        graph_layout(
            ELEMENTS,
            cutoff=CUTOFF,
            inputs=[InputSpec(name=name, irreps=irreps, per_atom=per_atom, units="x")],
        )


@pytest.mark.parametrize("name", ["pos", "strain"])
def test_the_inputs_the_derivative_engine_creates_are_never_batched(name):
    spec = InputSpec(name=name, irreps="1o", per_atom=name == "pos", units="Å")
    with pytest.raises(ValueError, match="derivative engine"):
        input_field(spec)
    with pytest.raises(ValueError, match="derivative engine"):
        graph_layout(ELEMENTS, cutoff=CUTOFF, features={name: ("atom", (3,))})
    with pytest.raises(ValueError, match="derivative engine"):
        _layout(inputs={name: SCALAR_PER_STRUCTURE})


@pytest.mark.parametrize(
    ("kind", "name", "spec", "message"),
    [
        ("targets", "stress", FieldSpec(shape=("n_graphs",), dtype="float"), "stores"),
        ("inputs", "magmom", FieldSpec(shape=("n_nodes", 4), dtype="float"), "stores"),
        (
            "targets",
            "forces",
            FieldSpec(shape=("n_graphs", 3), dtype="float"),
            "stores",
        ),
        ("targets", "born", FieldSpec(shape=("n_edges",), dtype="float"), "per atom"),
        ("targets", "born", FieldSpec(shape=("n_graphs",), dtype="int64"), "per atom"),
        ("features", "pos", FieldSpec(shape=("n_nodes", 3), dtype="float"), "engine"),
        ("features", "tag", FieldSpec(shape=("n_edges",), dtype="float"), "per atom"),
        ("features", "forces", FieldSpec(shape=("n_nodes",), dtype="float"), "stores"),
        ("features", "tag", FieldSpec(shape=("n_nodes", 0), dtype="float"), "positive"),
    ],
)
def test_a_layout_read_back_is_held_to_the_same_declarations(kind, name, spec, message):
    """A layout recorded with a run and loaded again skips the builders, so
    the layout itself refuses what they would."""
    record = json.loads(LAYOUT.model_dump_json())
    record[kind][name] = spec.model_dump()
    with pytest.raises(ValueError, match=message):
        GraphLayout.model_validate_json(json.dumps(record))


def test_a_declared_observable_is_batched_with_no_new_code():
    layout = graph_layout(ELEMENTS, cutoff=CUTOFF, targets=[_born_charges()])
    charges = np.arange(27.0).reshape(3, 9)
    graph, targets, info = _batch(
        _water(layout, born_charges=charges), _crystal(layout), layout=layout
    )
    assert "born_charges" not in graph
    assert targets["born_charges"].shape == (5, 9)
    np.testing.assert_array_equal(targets["born_charges"][:3], charges)
    assert not targets["born_charges"][3:].any()
    assert targets["born_charges_present"].tolist() == [True] * 3 + [False] * 2
    assert targets["born_charges_weight"].tolist() == [1.0, 0.0]
    check_layout(graph, layout, info)


def test_a_declared_input_is_batched_with_no_new_code():
    site_field = InputSpec(name="site_field", irreps="1o", per_atom=True, units="V/Å")
    layout = graph_layout(ELEMENTS, cutoff=CUTOFF, inputs=[site_field])
    values = np.full((3, 3), 0.5)
    graph, _, info = _batch(
        _water(layout, site_field=values), _crystal(layout), layout=layout
    )
    np.testing.assert_array_equal(graph["site_field"][:3], values)
    assert not graph["site_field"][3:].any()
    assert info.structure_sizes == (3, 2)


# --- from a Configuration to a sample -------------------------------------------------


def test_a_sample_is_a_graph_and_its_targets_and_the_graph_holds_no_label():
    graph, targets = _sample(_crystal(energy=-2.0, forces=np.ones((2, 3))))
    assert set(graph) == (set(GRAPH_CORE) - {"batch", "ptr", "displacement"}) | set(
        LAYOUT.inputs
    )
    assert set(targets) == set(LAYOUT.target_fields())
    assert targets["energy"] == -2.0
    assert targets["weight"] == 1.0


def test_the_head_is_its_index_in_the_models_head_list():
    layout = graph_layout(ELEMENTS, cutoff=CUTOFF, heads=("Default", "pbe"))
    graph, _ = _sample(_atom(layout=layout, head="pbe"), layout)
    assert graph["head"] == 1


def test_an_unknown_head_is_an_error_rather_than_a_fallback():
    layout = graph_layout(ELEMENTS, cutoff=CUTOFF, heads=("Default", "pbe"))
    with pytest.raises(SampleError, match="r2scan"):
        _sample(_atom(layout=layout, head="r2scan"), layout)


def test_an_element_outside_the_table_is_refused_and_named_by_its_source():
    with pytest.raises(SampleError, match=r"train\.xyz.*elements \[26\]"):
        _sample(_atom(26), edges=False, source="train.xyz")


def test_a_non_finite_position_is_refused():
    structure = _water()
    structure.positions[1, 0] = np.nan
    with pytest.raises(SampleError, match="non-finite position"):
        _sample(structure, edges=False)


def test_a_non_finite_cell_is_refused():
    structure = _crystal()
    structure.cell[0, 0] = np.inf
    with pytest.raises(SampleError, match="non-finite"):
        _sample(structure, edges=False)


def test_a_pbc_that_is_not_three_booleans_is_refused():
    structure = _crystal()
    structure.pbc = (True, 1, False)
    with pytest.raises(SampleError, match="three booleans"):
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


@pytest.mark.parametrize("cell", [None, np.zeros((3, 3))])
def test_a_periodic_structure_without_a_cell_is_refused_rather_than_given_one(cell):
    with pytest.raises(SampleError, match="axis 0 is periodic"):
        _sample(_atom(cell=cell, pbc=(True, False, False)), edges=False)


def test_a_stress_label_on_a_slab_of_zero_volume_is_refused():
    open_slab = _slab(
        [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]], stress=np.eye(3)
    )
    with pytest.raises(SampleError, match="zero volume"):
        _sample(open_slab, edges=False)


def test_a_stress_label_on_a_periodic_cell_of_dependent_rows_is_refused():
    structure = _crystal(stress=np.eye(3))
    structure.cell[2] = structure.cell[0] + structure.cell[1]
    with pytest.raises(SampleError, match="three independent rows"):
        _sample(structure, edges=False)


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
    carried rather than refused. Its cell is zero, the parsed none."""
    structure = _water(stress=np.eye(3))
    graph, targets = _sample(structure, edges=False)
    assert targets["stress_present"]
    np.testing.assert_array_equal(graph["cell"], np.zeros((3, 3)))
    assert structure.cell is None


def test_an_aperiodic_structure_carries_its_parsed_cell():
    parsed = np.diag([10.0, 11.0, 12.0])
    structure = _water()
    structure.cell = parsed
    structure.pbc = (False, False, False)
    graph, _ = _sample(structure)
    np.testing.assert_array_equal(graph["cell"], parsed)
    assert not graph["unit_shifts"].any()


def test_a_present_label_with_zero_weight_keeps_its_flag():
    structure = _water(energy=-1.0, forces=np.ones((3, 3)))
    structure.property_weights = {"energy": 0.0, "forces": 0.0}
    _, targets = _sample(structure, edges=False)
    assert targets["energy_present"]
    assert targets["forces_present"].tolist() == [True, True, True]
    assert targets["energy_weight"] == 0.0
    assert targets["forces_weight"] == 0.0


def test_an_absent_label_is_zero_flagged_absent_and_weighted_zero():
    _, targets = _sample(_water(), edges=False)
    assert not targets["energy_present"]
    assert targets["energy"] == 0.0
    assert targets["energy_weight"] == 0.0
    assert targets["stress"].shape == (3, 3)
    assert not targets["stress"].any()
    assert not targets["dipole_present"]
    assert targets["dipole_weight"].tolist() == [0.0, 0.0, 0.0]
    assert targets["polarizability_weight"].shape == (3, 3)


def test_a_property_weight_is_carried_for_a_present_label():
    structure = _water(dipole=np.ones(3))
    structure.property_weights = {"dipole": 2.5}
    structure.weight = 0.5
    _, targets = _sample(structure, edges=False)
    assert targets["dipole_weight"].tolist() == [2.5, 2.5, 2.5]
    assert targets["weight"] == 0.5


def test_an_absent_total_spin_is_one_and_other_inputs_are_zero():
    graph, _ = _sample(_water(), edges=False)
    assert graph["total_spin"] == 1.0
    assert graph["total_charge"] == 0.0
    assert graph["magmom"].shape == (3, 3)
    assert not graph["magmom"].any()


@pytest.mark.parametrize(
    "layout",
    [
        graph_layout(
            ELEMENTS,
            cutoff=CUTOFF,
            inputs=[
                InputSpec(name="total_charg", irreps="0e", per_atom=False, units="e")
            ],
        ),
        graph_layout(
            ELEMENTS,
            cutoff=CUTOFF,
            targets=[
                ObservableSpec(name="energ", irreps="0e", per_atom=False, units="eV")
            ],
        ),
    ],
    ids=["input", "target"],
)
def test_a_name_the_structure_does_not_declare_is_an_error_not_an_absence(layout):
    """A misspelling in the layout, or a key specification that does not
    resolve a declared name, must not read as an unlabelled structure."""
    with pytest.raises(SampleError, match=r"the layout declares '(total_charg|energ)'"):
        _sample(_water(), layout, edges=False)


def test_a_label_of_the_wrong_shape_is_refused():
    layout = graph_layout(ELEMENTS, cutoff=CUTOFF, targets=[_born_charges()])
    with pytest.raises(SampleError, match="label 'born_charges' of shape"):
        _sample(_water(layout, born_charges=np.zeros((3, 3, 3))), layout, edges=False)


def test_an_input_of_the_wrong_shape_is_refused():
    site_field = InputSpec(name="site_field", irreps="1o", per_atom=True, units="V/Å")
    layout = graph_layout(ELEMENTS, cutoff=CUTOFF, inputs=[site_field])
    with pytest.raises(SampleError, match="input 'site_field' of shape"):
        _sample(_water(layout, site_field=np.zeros((3, 2))), layout, edges=False)


def _featured(features):
    return graph_layout(ELEMENTS, cutoff=CUTOFF, features=features)


def test_a_declared_atom_feature_with_the_wrong_rows_is_not_reshaped():
    layout = _featured({"spin_state": ("atom", ())})
    with pytest.raises(SampleError, match="not reshaped"):
        _sample(_water(layout, spin_state=np.array([0, 1])), layout, edges=False)


def test_a_declared_feature_that_is_not_a_real_number_is_refused():
    layout = _featured({"spin_state": ("atom", ())})
    with pytest.raises(SampleError, match="real number"):
        _sample(
            _water(layout, spin_state=np.array(["up", "down", "up"])),
            layout,
            edges=False,
        )


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_a_non_finite_feature_is_refused(value):
    layout = _featured({"spin_state": ("atom", ())})
    spin_state = np.array([0.0, 1.0, value])
    with pytest.raises(
        SampleError, match="non-finite value in embedding feature 'spin_state'"
    ):
        _sample(_water(layout, spin_state=spin_state), layout, edges=False)


def test_a_declared_feature_the_structure_lacks_is_refused():
    layout = _featured({"spin_state": ("atom", ())})
    with pytest.raises(SampleError, match="cannot be made up"):
        _sample(_water(layout), layout, edges=False)


def test_a_declared_feature_is_batched_by_its_storage():
    layout = _featured({"spin_state": ("atom", ()), "phase": ("graph", ())})
    graph, _, _ = _batch(
        _water(layout, spin_state=np.array([0, 1, 1]), phase=3.0),
        _crystal(layout, spin_state=np.array([2, 2]), phase=4.0),
        layout=layout,
    )
    assert graph["spin_state"].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0]
    assert graph["phase"].tolist() == [3.0, 4.0]


@pytest.mark.parametrize(
    ("storage", "value_shape", "value"),
    [
        ("atom", (4,), np.zeros((3, 5))),
        ("atom", (), np.zeros((3, 1))),
        ("graph", (), np.zeros(2)),
        ("graph", (2,), np.zeros(3)),
    ],
)
def test_a_feature_of_another_width_than_declared_is_refused_per_structure(
    storage, value_shape, value
):
    layout = _featured({"spin_state": (storage, value_shape)})
    with pytest.raises(SampleError, match="layout declares"):
        _sample(_water(layout, spin_state=value), layout, edges=False)


def test_a_sample_without_a_neighbour_list_has_no_edge_keys():
    graph, _ = _sample(_crystal(), edges=False)
    assert not set(graph) & EDGE_KEYS


def test_a_neighbour_list_searched_with_another_cutoff_is_refused():
    structure = _crystal()
    with pytest.raises(SampleError, match="cutoff"):
        graph_sample(structure, _search(structure, CUTOFF + 0.5), LAYOUT)


def _stored(edge_index, unit_shifts, shifts=None):
    unit_shifts = np.asarray(unit_shifts)
    return NeighborList(
        edge_index=np.asarray(edge_index),
        unit_shifts=unit_shifts,
        shifts=np.zeros(unit_shifts.shape) if shifts is None else shifts,
        cutoff=CUTOFF,
    )


@pytest.mark.parametrize(
    ("neighbors", "message"),
    [
        (_stored([[0], [5]], np.zeros((1, 3), dtype=np.int64)), "edge to atom 5"),
        (_stored([[0], [1]], [[0, 0, 1]]), "aperiodic"),
        (_stored([[0, 1, 2]], np.zeros((3, 3), dtype=np.int64)), "edge_index of shape"),
        (_stored([[0], [1]], np.zeros((5, 3), dtype=np.int64)), "unit_shifts of shape"),
        (_stored([[0], [1]], [[0.5, 0.0, 0.0]]), "Expected integers"),
    ],
    ids=["outside", "open-axis", "edge-shape", "shift-rows", "half-shift"],
)
def test_a_malformed_stored_neighbour_list_is_refused_and_named(neighbors, message):
    with pytest.raises(
        SampleError, match=rf"a structure .*its neighbour list.*{message}"
    ):
        graph_sample(_water(), neighbors, LAYOUT)


def test_a_stored_neighbour_list_with_a_true_self_edge_is_refused():
    stored = _stored([[0, 0, 1], [0, 1, 0]], np.zeros((3, 3), dtype=np.int64))
    with pytest.raises(SampleError, match="edge to itself with a zero shift"):
        graph_sample(_water(), stored, LAYOUT)


def test_an_empty_stored_neighbour_list_of_any_rank_is_no_edges():
    graph, _ = graph_sample(_water(), _stored(np.zeros(0), np.zeros(0)), LAYOUT)
    assert graph["edge_index"].shape == (2, 0)
    assert graph["shifts"].shape == (0, 3)


def test_a_stored_neighbour_list_has_its_shifts_recomputed_in_this_cell():
    structure = _crystal()
    searched = _search(structure)
    stored = _stored(
        searched.edge_index,
        searched.unit_shifts,
        shifts=np.full(searched.shifts.shape, 99.0),
    )
    graph, _ = graph_sample(structure, stored, LAYOUT)
    np.testing.assert_array_equal(graph["edge_index"], searched.edge_index)
    np.testing.assert_array_equal(graph["unit_shifts"], searched.unit_shifts)
    np.testing.assert_array_equal(graph["shifts"], searched.shifts)


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


def test_the_floats_take_the_layouts_floating_dtype():
    layout = _with(LAYOUT, float_dtype="float32")
    graph, targets, info = _batch(_water(), _crystal(), layout=layout)
    assert graph["positions"].dtype == np.float32
    assert graph["total_spin"].dtype == np.float32
    assert graph["edge_index"].dtype == np.int64
    assert targets["energy"].dtype == np.float32
    assert targets["weight"].dtype == np.float32
    assert targets["energy_present"].dtype == np.bool_
    with pytest.raises(GraphValidationError, match="'cell' has dtype float32"):
        check_layout(graph, LAYOUT, info)


def test_numpy_collation_refuses_a_dtype_numpy_lacks():
    layout = _with(LAYOUT, float_dtype="bfloat16")
    sample, _ = _sample(_water())
    with pytest.raises(ValueError, match="numpy has no bfloat16"):
        collate([sample], layout)


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


def test_check_false_leaves_the_layout_check_to_the_caller(batch):
    water, _ = _sample(_water())
    water["magmom"] = np.zeros((3, 2))
    graph, info = collate([water], LAYOUT, check=False)
    with pytest.raises(GraphValidationError, match="'magmom' has shape"):
        check_layout(graph, LAYOUT, info)


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


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ptr", np.zeros((3, 1), dtype=np.int64)),
        ("positions", np.zeros(15)),
        ("edge_index", np.zeros(4, dtype=np.int64)),
    ],
)
def test_a_key_that_binds_the_sizes_with_the_wrong_rank_is_named(batch, key, value):
    graph, _, info = batch
    graph[key] = value
    with pytest.raises(GraphValidationError, match=f"'{key}' has shape"):
        check_layout(graph, LAYOUT, info)


@pytest.mark.parametrize("key", ["my_annotation", "rcell", "energy"])
def test_an_undeclared_key_is_named(batch, key):
    graph, _, info = batch
    graph[key] = np.zeros(2)
    with pytest.raises(GraphValidationError, match=f"'{key}' is not declared"):
        check_layout(graph, LAYOUT, info)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("magmom", np.zeros((5, 2))),
        ("magmom", np.zeros((4, 3))),
        ("magmom", np.zeros((5, 3, 1))),
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


def test_the_optional_core_keys_may_be_absent_or_present(batch):
    graph, _, info = batch
    del graph["head"]
    check_layout(graph, LAYOUT, info)
    graph["displacement"] = np.zeros((2, 3, 3))
    check_layout(graph, LAYOUT, info)


def test_an_embedding_feature_is_checked_on_its_declared_shape():
    layout = _featured({"spin_state": ("atom", (4,))})
    graph, _, info = _batch(
        _water(layout, spin_state=np.zeros((3, 4))),
        _crystal(layout, spin_state=np.zeros((2, 4))),
        layout=layout,
    )
    assert graph["spin_state"].shape == (5, 4)
    check_layout(graph, layout, info)
    for shape in [(4, 4), (5, 3)]:
        graph["spin_state"] = np.zeros(shape)
        with pytest.raises(GraphValidationError, match="'spin_state' has shape"):
            check_layout(graph, layout, info)


# --- batch-requirement fields ---------------------------------------------------------

REQUIRED = {
    "pair_index": FieldSpec(shape=(2, "n_pairs"), dtype="int64"),
    "pair_length": FieldSpec(shape=("n_pairs",), dtype="float"),
}
WITH_REQUIREMENTS = _with(LAYOUT, requirement_fields=REQUIRED)


def test_a_batch_before_its_requirement_fields_are_built_passes(batch):
    graph, _, info = batch
    check_layout(graph, WITH_REQUIREMENTS, info)


def test_a_batch_with_its_requirement_fields_built_passes(batch):
    graph, _, info = batch
    graph["pair_index"] = np.zeros((2, 4), dtype=np.int64)
    graph["pair_length"] = np.zeros(4)
    check_layout(graph, WITH_REQUIREMENTS, info)


def test_a_batch_with_its_requirement_fields_partly_built_is_refused(batch):
    graph, _, info = batch
    graph["pair_index"] = np.zeros((2, 4), dtype=np.int64)
    with pytest.raises(GraphValidationError, match="pair_length"):
        check_layout(graph, WITH_REQUIREMENTS, info)


def test_a_requirement_dimension_bound_twice_with_different_sizes_is_refused(batch):
    graph, _, info = batch
    graph["pair_index"] = np.zeros((2, 4), dtype=np.int64)
    graph["pair_length"] = np.zeros(5)
    with pytest.raises(GraphValidationError, match="'pair_length' has shape"):
        check_layout(graph, WITH_REQUIREMENTS, info)


def test_a_requirement_field_of_the_wrong_rank_is_named(batch):
    graph, _, info = batch
    graph["pair_index"] = np.zeros((2, 4, 2), dtype=np.int64)
    graph["pair_length"] = np.zeros(4)
    with pytest.raises(GraphValidationError, match="'pair_index' has shape"):
        check_layout(graph, WITH_REQUIREMENTS, info)


@pytest.mark.parametrize(
    ("key", "value"),
    [("pair_length", np.zeros(2)), ("node_mark", np.zeros(3)), ("graph_mark", 1.0)],
)
def test_a_sample_carrying_a_requirement_field_is_refused_by_collation(key, value):
    """Requirement fields are built on the collated batch, so collation
    refuses one a sample already carries rather than stacking it."""
    layout = _with(
        LAYOUT,
        requirement_fields={
            **REQUIRED,
            "node_mark": FieldSpec(shape=("n_nodes",), dtype="float"),
            "graph_mark": FieldSpec(shape=("n_graphs",), dtype="float"),
        },
    )
    sample, _ = _sample(_water(), layout)
    sample[key] = value
    with pytest.raises(ValueError, match=f"carries '{key}', a batch-requirement"):
        collate([sample], layout)


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


def test_graph_info_refuses_sizes_that_disagree_with_its_counts():
    with pytest.raises(ValueError, match="entries"):
        GraphInfo(num_graphs=2, num_nodes=5, num_edges=0, structure_sizes=(5,))
    with pytest.raises(ValueError, match="sums to"):
        GraphInfo(num_graphs=2, num_nodes=5, num_edges=0, structure_sizes=(3, 3))


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
