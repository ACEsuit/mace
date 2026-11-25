"""The output layer: readouts built from declarations, their per-head layout,
and the typed output. Its numbers against the frozen tree are in
tests/parity/test_output_layer_parity.py."""

import re
from typing import Any

import pytest
import torch
from conftest import (
    ELEMENTS,
    assert_close,
    cluster,
    clusters,
    precision,
    rotation_matrix,
    transform_features,
)
from mace_core.config import ModelConfig, ReadoutConfig
from mace_core.elements import ResolvedE0s
from mace_core.observables import (
    DEFAULT_CATALOGUE,
    ObservableCatalogue,
    ObservableSpec,
)
from mace_core.outputs import MACEOutput
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.kernels import canonical_state, initialize_model_weights, segment_sum
from mace_torch.models import EnergyOutputHead, MACEOutputs, ObservableReadout
from mace_torch.models.readouts import head_columns
from mace_torch.nn.backbone import MACEBackbone

#: The comparisons read values; the head tests ask for their gradients.
pytestmark = pytest.mark.usefixtures("no_gradients")

HEADS = ("pbe", "mp2")
BACKEND = ReferenceBackend()

#: A per-atom rank-2 observable declared in configuration only.
TENSOR = ObservableSpec(name="site_tensor", irreps="0e+2e", per_atom=True, units="1")
#: A per-structure vector.
VECTOR = ObservableSpec(name="moment", irreps="1o", per_atom=False, units="1")


def _energy_head(num_heads=1, model_precision=None):
    heads = HEADS[:num_heads]
    return EnergyOutputHead(
        ResolvedE0s(
            values={head: {1: -13.6, 6: -1030.0, 8: -2041.3} for head in heads},
            provenance=dict.fromkeys(heads, "table"),
        ),
        heads,
        ELEMENTS,
        [1.7, 0.6][:num_heads],
        [0.3, -0.4][:num_heads],
        zbl_in_scale_shift=True,
        precision=model_precision or precision(),
    )


def _model(
    observables=DEFAULT_CATALOGUE,
    num_heads=1,
    readout=None,
    seed=0,
    **config,
):
    options: dict[str, Any] = {
        "r_max": 3.0,
        "num_interactions": 2,
        "num_channels": 4,
        "max_L": 2,
        "max_ell": 3,
        "correlation": 2,
        "radial_mlp": (8,),
    } | config
    backbone = MACEBackbone(
        BACKEND,
        ModelConfig(**options),
        ELEMENTS,
        avg_num_neighbors=3.0,
        precision=precision(),
    )
    names = [spec.name for spec in observables.observables]
    outputs = MACEOutputs(
        BACKEND,
        observables,
        readout or ReadoutConfig(mlp_irreps="8x0e"),
        backbone.layer_irreps,
        num_heads,
        _energy_head(num_heads) if "energy" in names else None,
        precision=precision(),
    )
    initialize_model_weights(backbone, seed)
    initialize_model_weights(outputs, seed + 1)
    return backbone, outputs


def _run(model, graph):
    backbone, outputs = model
    return outputs(graph, backbone(graph))


def _two_structures(heads=(0, 0)):
    graph = clusters(0, 1)
    graph["head"] = torch.tensor(heads)
    return graph


def _per_structure(values, graph):
    return segment_sum(values, graph["batch"], graph["ptr"].shape[0] - 1)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_each_head_owns_its_copy_of_every_term():
    """``0e+1o`` for two heads is ``2x0e+2x1o``: head 1 owns column 1 and the
    second three of the vector block."""
    assert head_columns("0e+1o", 2).tolist() == [[0, 2, 3, 4], [1, 5, 6, 7]]
    assert head_columns("0e", 1).tolist() == [[0]]


# ---------------------------------------------------------------------------
# Building readouts from declarations
# ---------------------------------------------------------------------------


def test_every_layer_is_read_out_and_the_last_through_the_gate():
    _, outputs = _model()
    readout = outputs.readouts["energy"]
    assert readout.layers == (0, 1)
    assert [hasattr(m, "linear_in") for m in readout.maps] == [False, True]


def test_last_only_reads_the_last_layer():
    _, outputs = _model(readout=ReadoutConfig(mlp_irreps="8x0e", last_only=True))
    assert outputs.readouts["energy"].layers == (1,)


def test_a_one_layer_model_reads_out_linearly():
    _, outputs = _model(num_interactions=1)
    readout = outputs.readouts["energy"]
    assert readout.layers == (0,)
    assert not hasattr(readout.maps[0], "linear_in")


def test_a_rank_two_observable_declared_in_configuration_gets_a_readout():
    """Nothing but the declaration: the readout appears, per atom, with the
    six components of ``0e+2e``, in `extras` under its own name."""
    catalogue = ObservableCatalogue(
        inputs=DEFAULT_CATALOGUE.inputs,
        observables=(*DEFAULT_CATALOGUE.observables, TENSOR),
    )
    output = _run(_model(catalogue), cluster())
    assert isinstance(output, MACEOutput)
    assert output.extras["site_tensor"].shape == (6, 6)
    assert output.extras["site_tensor"].abs().max() > 0


def test_an_irrep_no_layer_carries_is_refused_at_construction():
    """Scalar node features cannot give a vector: the readout would be zero
    there, and a loss on it could never go down."""
    catalogue = ObservableCatalogue(
        inputs=DEFAULT_CATALOGUE.inputs, observables=(VECTOR,)
    )
    with pytest.raises(ValueError, match=r"'moment' declares '1o'.*\['1o'\]"):
        _model(catalogue, max_L=0)


def test_the_last_layer_alone_cannot_give_what_only_earlier_layers_carry():
    catalogue = ObservableCatalogue(
        inputs=DEFAULT_CATALOGUE.inputs, observables=(VECTOR,)
    )
    _, outputs = _model(catalogue)
    assert outputs.readouts["moment"].layers == (0,)
    with pytest.raises(ValueError, match="through the gated middle '8x0e'"):
        _model(catalogue, readout=ReadoutConfig(mlp_irreps="8x0e", last_only=True))


@pytest.mark.parametrize("given", [True, False])
def test_the_energy_head_is_given_exactly_when_the_energy_is_declared(given):
    backbone, _ = _model()
    vector_only = ObservableCatalogue(
        inputs=DEFAULT_CATALOGUE.inputs, observables=(VECTOR,)
    )
    observables = vector_only if given else DEFAULT_CATALOGUE
    with pytest.raises(ValueError, match="energy head was"):
        MACEOutputs(
            BACKEND,
            observables,
            ReadoutConfig(),
            backbone.layer_irreps,
            1,
            _energy_head() if given else None,
            precision=precision(),
        )


def test_the_energy_head_has_a_row_per_head():
    backbone, _ = _model()
    with pytest.raises(ValueError, match="for 1 head"):
        MACEOutputs(
            BACKEND,
            DEFAULT_CATALOGUE,
            ReadoutConfig(),
            backbone.layer_irreps,
            2,
            _energy_head(1),
            precision=precision(),
        )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            ObservableSpec(
                name="node_energies", irreps="0e", per_atom=True, units="eV"
            ),
            "a name the energy fills",
        ),
        (
            ObservableSpec(
                name="interaction_energy", irreps="0e", per_atom=False, units="eV"
            ),
            "a name the energy fills",
        ),
        (
            ObservableSpec(name="dipole", irreps="1o", per_atom=True, units="e*A"),
            "declared '1o', per atom",
        ),
        (
            ObservableSpec(name="dipole", irreps="1e", per_atom=False, units="e*A"),
            "declared '1e', per structure",
        ),
    ],
)
def test_an_observable_that_would_overwrite_another_value_is_refused(spec, message):
    """In either order of the declarations: the result would otherwise depend
    on which one is written last."""
    for observables in (
        (*DEFAULT_CATALOGUE.observables, spec),
        (spec, *DEFAULT_CATALOGUE.observables),
    ):
        with pytest.raises(ValueError, match=re.escape(message)):
            _model(
                ObservableCatalogue(
                    inputs=DEFAULT_CATALOGUE.inputs, observables=observables
                )
            )


def test_an_observable_filling_a_derivatives_field_is_refused():
    """With the energy declared the catalogue already refuses ``forces``;
    without it, the output layer does."""
    forces = ObservableSpec(name="forces", irreps="1o", per_atom=True, units="eV/A")
    with pytest.raises(ValueError, match="'forces' field"):
        _model(ObservableCatalogue(observables=(forces,)))


@pytest.mark.parametrize("irreps", ["1o", "1x1o"])
def test_a_structure_dipole_fills_the_dipole_field(irreps):
    """However the vector is spelled."""
    dipole = ObservableSpec(name="dipole", irreps=irreps, per_atom=False, units="e*A")
    output = _run(_model(ObservableCatalogue(observables=(dipole,))), _two_structures())
    assert output.dipole is not None and output.dipole.shape == (2, 3)
    assert "dipole" not in output.extras


def test_a_model_has_at_least_one_head():
    backbone, _ = _model()
    with pytest.raises(ValueError, match="at least one head"):
        MACEOutputs(
            BACKEND,
            ObservableCatalogue(observables=(VECTOR,)),
            ReadoutConfig(),
            backbone.layer_irreps,
            0,
            None,
            precision=precision(),
        )


def test_a_readout_of_several_heads_needs_each_nodes_head():
    backbone, outputs = _model(num_heads=2)
    layers = backbone(cluster()).layers
    with pytest.raises(ValueError, match="2 head"):
        outputs.readouts["energy"](layers, None)


# ---------------------------------------------------------------------------
# The output
# ---------------------------------------------------------------------------


def test_the_energy_fills_its_fields_and_the_rest_its_extras():
    catalogue = ObservableCatalogue(
        inputs=DEFAULT_CATALOGUE.inputs,
        observables=(*DEFAULT_CATALOGUE.observables, VECTOR),
    )
    output = _run(_model(catalogue), _two_structures())
    assert output.total_energy is not None and output.total_energy.shape == (2,)
    assert output.node_energies is not None and output.node_energies.shape == (12,)
    assert set(output.extras) == {"interaction_energy", "moment"}
    assert output.extras["moment"].shape == (2, 3)


def test_a_structure_level_observable_is_the_sum_of_its_atoms():
    per_atom = ObservableSpec(name="moment", irreps="1o", per_atom=True, units="1")
    graph = _two_structures()
    summed = _run(_model(ObservableCatalogue(observables=(VECTOR,))), graph)
    atoms = _run(_model(ObservableCatalogue(observables=(per_atom,))), graph)
    assert_close(
        summed.extras["moment"], _per_structure(atoms.extras["moment"], graph), "sum"
    )


def test_an_absent_head_is_head_zero():
    model = _model(num_heads=2)
    graph = _two_structures((0, 0))
    with_head = _run(model, graph)
    del graph["head"]
    assert torch.equal(_run(model, graph).total_energy, with_head.total_energy)


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


def test_the_energy_is_invariant_and_a_vector_rotates():
    catalogue = ObservableCatalogue(
        inputs=DEFAULT_CATALOGUE.inputs,
        observables=(*DEFAULT_CATALOGUE.observables, VECTOR, TENSOR),
    )
    model = _model(catalogue)
    graph = cluster(4)
    rotation = rotation_matrix(5)
    dtype = graph["positions"].dtype
    moved = dict(graph)
    moved["positions"] = graph["positions"] @ rotation.T.to(dtype) + torch.tensor(
        [1.3, -0.7, 2.2], dtype=dtype
    )
    before, after = _run(model, graph), _run(model, moved)
    assert_close(after.total_energy, before.total_energy, "energy")
    assert_close(after.node_energies, before.node_energies, "node energies")
    assert_close(
        after.extras["moment"],
        transform_features(before.extras["moment"], "1o", rotation, parity=1),
        "vector",
    )
    assert_close(
        after.extras["site_tensor"],
        transform_features(before.extras["site_tensor"], "0e+2e", rotation, parity=1),
        "rank two",
    )


def test_an_inverted_structure_flips_the_vector_and_not_the_energy():
    catalogue = ObservableCatalogue(
        inputs=DEFAULT_CATALOGUE.inputs,
        observables=(*DEFAULT_CATALOGUE.observables, VECTOR),
    )
    model = _model(catalogue)
    graph = cluster(6)
    inverted = dict(graph)
    inverted["positions"] = -graph["positions"]
    before, after = _run(model, graph), _run(model, inverted)
    assert_close(after.total_energy, before.total_energy, "energy")
    assert_close(after.extras["moment"], -before.extras["moment"], "vector")


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


def _reached(model, graph):
    """The readout weights a batch sends gradient to, by name and index."""
    backbone, outputs = model
    readout = outputs.readouts["energy"]
    readout.zero_grad()
    with torch.enable_grad():
        outputs(graph, backbone(graph)).total_energy.sum().backward()
    return {
        (name, index)
        for name, parameter in readout.named_parameters()
        for index in map(tuple, torch.nonzero(parameter.grad).tolist())
    }


def test_each_head_reaches_its_own_readout_weights():
    """Disjoint sets, each as large as a one-head model's, so a structure
    trains its own head's readout and no other."""
    model = _model(num_heads=2)
    first = _reached(model, _two_structures((0, 0)))
    second = _reached(model, _two_structures((1, 1)))
    assert first and not first & second
    alone = _reached(_model(num_heads=1), _two_structures((0, 0)))
    assert len(first) == len(second) == len(alone)


# ---------------------------------------------------------------------------
# One head's readout on its own
# ---------------------------------------------------------------------------

LAYERS = ("4x0e+4x1o+4x2e", "4x0e+4x1o")
#: Scalars and a vector: the gated last layer reads out the scalars, the
#: first layer both.
MIXED = ObservableSpec(name="mixed", irreps="0e+1o", per_atom=True, units="1")


def _readout(num_heads, seed):
    readout = ObservableReadout(
        BACKEND, MIXED, LAYERS, num_heads, "3x0e", precision=precision()
    )
    initialize_model_weights(readout, seed)
    return readout


def _features(seed=0):
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.get_default_dtype()
    return [
        torch.randn(5, 36, generator=generator, dtype=dtype),
        torch.randn(5, 16, generator=generator, dtype=dtype),
    ]


def _on_head(readout, features, head):
    node_head = torch.full((5,), head) if readout.num_heads > 1 else None
    return torch.stack(readout(features, node_head))


def test_a_one_head_readout_written_into_a_head_reads_out_the_same():
    """The second map of the gated readout reads every head's middle, so its
    stored weights carry the ratio of the two fan-ins; written in and read
    back, the head is the one-head readout."""
    one, three = _readout(1, seed=1), _readout(3, seed=2)
    other_heads = {head: three.head_state(head) for head in (0, 2)}
    three.load_head_state(1, canonical_state(one))
    features = _features()
    assert_close(_on_head(three, features, 1), _on_head(one, features, 0), "head 1")
    for path, tensors in three.head_state(1).items():
        assert_close(tensors["weight"], canonical_state(one)[path]["weight"], path)
    for head, state in other_heads.items():
        for path, tensors in three.head_state(head).items():
            assert torch.equal(tensors["weight"], state[path]["weight"]), (head, path)


def test_a_head_started_from_another_reads_out_what_that_one_does():
    readout = _readout(3, seed=3)
    readout.load_head_state(2, readout.head_state(0))
    features = _features(1)
    assert_close(_on_head(readout, features, 2), _on_head(readout, features, 0), "copy")


def test_a_head_the_readout_does_not_have_is_refused():
    readout = _readout(2, seed=4)
    with pytest.raises(IndexError, match="head 2"):
        readout.head_state(2)
    with pytest.raises(IndexError, match="head -1"):
        readout.load_head_state(-1, readout.head_state(0))


def test_a_state_of_another_shape_writes_nothing():
    readout = _readout(2, seed=5)
    before = canonical_state(readout)
    wrong = ObservableReadout(
        BACKEND, MIXED, ("8x0e+8x1o", "8x0e"), 1, "3x0e", precision=precision()
    )
    with pytest.raises(ValueError, match="one head of this readout holds"):
        readout.load_head_state(1, canonical_state(wrong))
    scalar = ObservableSpec(name="scalar", irreps="0e", per_atom=True, units="1")
    last_only = ObservableReadout(
        BACKEND, scalar, LAYERS, 1, "3x0e", last_only=True, precision=precision()
    )
    with pytest.raises(ValueError, match="the state holds the ops"):
        readout.load_head_state(1, canonical_state(last_only))
    for path, tensors in canonical_state(readout).items():
        assert torch.equal(tensors["weight"], before[path]["weight"]), path


def test_a_one_head_readout_given_each_nodes_head_is_refused():
    readout = _readout(1, seed=6)
    with pytest.raises(ValueError, match="was given each node's head"):
        readout(_features(), torch.zeros(5, dtype=torch.int64))


@pytest.mark.parametrize("name", ["values", "type", "update"])
def test_an_observable_named_like_an_attribute_of_the_container_is_refused(name):
    spec = ObservableSpec(name=name, irreps="0e", per_atom=True, units="1")
    with pytest.raises(ValueError, match=f"observable '{name}' is named like"):
        _model(
            ObservableCatalogue(inputs=DEFAULT_CATALOGUE.inputs, observables=(spec,))
        )


def test_an_energy_head_of_another_precision_is_refused():
    other = "float32" if precision() == "float64" else "float64"
    backbone, _ = _model()
    with pytest.raises(ValueError, match="build both at the model's precision"):
        MACEOutputs(
            BACKEND,
            DEFAULT_CATALOGUE,
            ReadoutConfig(mlp_irreps="8x0e"),
            backbone.layer_irreps,
            1,
            _energy_head(1, other),
            precision=precision(),
        )
