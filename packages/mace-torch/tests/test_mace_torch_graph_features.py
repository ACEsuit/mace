"""The graph-feature embedding and the energy read out of the node embedding.
Their numbers against the frozen tree are in
tests/parity/test_graph_features_parity.py."""

import re

import pytest
import torch
from conftest import assert_close, cluster, precision, rotation_matrix
from mace_core.config import (
    CategoricalFeatureConfig,
    ContinuousFeatureConfig,
    ReadoutConfig,
)
from mace_core.observables import DEFAULT_CATALOGUE, InputSpec, ObservableCatalogue
from mace_torch.kernels import (
    canonical_state,
    initialize_model_weights,
    load_canonical_state,
)
from mace_torch.nn.graph_features import GraphFeatureEmbedding
from test_mace_torch_output_layer import _model, _run, _two_structures

CHARGE = CategoricalFeatureConfig(
    name="total_charge", num_classes=5, offset=2, embedding_dim=3
)
TEMPERATURE = ContinuousFeatureConfig(name="elec_temp", embedding_dim=3)
LABEL = CategoricalFeatureConfig(
    name="label", per="atom", num_classes=2, embedding_dim=3
)
VECTOR = ContinuousFeatureConfig(name="field", per="atom", input_dim=2, embedding_dim=3)


pytestmark = pytest.mark.usefixtures("no_gradients")


def _embedding(features, seed=0):
    embedding = GraphFeatureEmbedding(features, 4, precision())
    initialize_model_weights(embedding, seed)
    return embedding


def test_a_label_is_offset_into_its_table():
    """A charge of -2 with an offset of 2 is row 0."""
    embedding = _embedding([CHARGE])
    table = embedding.embeddings[0].table
    graph = {"total_charge": torch.tensor([-2.0, 1.0]), "ptr": torch.tensor([0, 2, 3])}
    batch = torch.tensor([0, 0, 1])
    expected = torch.nn.functional.silu(
        table[torch.tensor([0, 0, 3])] @ embedding.projection.T
    )
    assert_close(embedding(graph, batch), expected, "offset")


def test_a_structure_feature_of_several_components_is_each_of_its_atoms():
    """Such as an external field, ``[n_graphs, 3]``."""
    per_structure = ContinuousFeatureConfig(name="field", input_dim=3, embedding_dim=3)
    per_atom = ContinuousFeatureConfig(
        name="field", per="atom", input_dim=3, embedding_dim=3
    )
    structure, atoms = _embedding([per_structure]), _embedding([per_atom], seed=1)
    load_canonical_state(atoms, canonical_state(structure))
    value = torch.tensor([[0.3, -0.1, 0.8], [1.2, 0.4, -0.5]])
    batch = torch.tensor([0, 0, 1, 1, 1])
    ptr = torch.tensor([0, 2, 5])
    assert_close(
        structure({"field": value, "ptr": ptr}, batch),
        atoms({"field": value[batch], "ptr": ptr}, batch),
        "broadcast",
    )
    with pytest.raises(ValueError, match=re.escape("holds it (3, 2)")):
        structure({"field": value.T, "ptr": ptr}, batch)


def test_a_structure_feature_is_each_of_its_atoms_feature():
    per_structure = ContinuousFeatureConfig(name="field", embedding_dim=3)
    per_atom = ContinuousFeatureConfig(name="field", per="atom", embedding_dim=3)
    structure, atoms = _embedding([per_structure]), _embedding([per_atom], seed=1)
    load_canonical_state(atoms, canonical_state(structure))
    value = torch.tensor([0.3, 1.2])
    batch = torch.tensor([0, 0, 1, 1, 1])
    ptr = torch.tensor([0, 2, 5])
    assert_close(
        structure({"field": value, "ptr": ptr}, batch),
        atoms({"field": value[batch], "ptr": ptr}, batch),
        "broadcast",
    )


@pytest.mark.parametrize(
    ("feature", "value", "found"),
    [
        (ContinuousFeatureConfig(name="field", embedding_dim=3), None, "absent"),
        (
            ContinuousFeatureConfig(name="field", embedding_dim=3),
            torch.arange(5.0),
            "(5,)",
        ),
        (
            ContinuousFeatureConfig(name="field", per="atom", embedding_dim=3),
            torch.arange(2.0),
            "(2,)",
        ),
        (
            ContinuousFeatureConfig(
                name="field", per="atom", input_dim=2, embedding_dim=3
            ),
            torch.arange(5.0),
            "(5,)",
        ),
    ],
)
def test_a_feature_of_the_wrong_shape_names_itself(feature, value, found):
    """Given per atom when declared per structure, a value would otherwise be
    indexed by structure and embedded without a word."""
    graph = {"ptr": torch.tensor([0, 2, 5])}
    if value is not None:
        graph["field"] = value
    with pytest.raises(ValueError, match=re.escape(f"holds it {found}")):
        _embedding([feature])(graph, torch.tensor([0, 0, 1, 1, 1]))


def test_a_scalar_without_a_structure_axis_names_the_feature():
    graph = {"field": torch.tensor(0.3), "ptr": torch.tensor([0, 2])}
    embedding = _embedding([ContinuousFeatureConfig(name="field", embedding_dim=3)])
    with pytest.raises(ValueError, match=re.escape("holds it ()")):
        embedding(graph, torch.tensor([0, 0]))


def test_an_embedding_of_no_feature_is_refused():
    with pytest.raises(ValueError, match="at least one feature"):
        GraphFeatureEmbedding([], 4)


def test_a_feature_may_be_named_like_an_attribute_of_a_module():
    """The embeddings are held in order, not by name."""
    feature = ContinuousFeatureConfig(name="type", embedding_dim=3)
    embedding = _embedding([feature])
    graph = {"type": torch.tensor([0.5]), "ptr": torch.tensor([0, 2])}
    assert embedding(graph, torch.tensor([0, 0])).shape == (2, 4)


def test_it_saves_every_table_map_and_bias_and_draws_them_from_its_seed():
    embedding = _embedding([CHARGE, TEMPERATURE, LABEL, VECTOR], seed=3)
    state = canonical_state(embedding)
    continuous = {"weight_in", "bias_in", "weight_out", "bias_out"}
    assert {path: set(tensors) for path, tensors in state.items()} == {
        "": {"projection"},
        "embeddings.0": {"table"},
        "embeddings.1": continuous,
        "embeddings.2": {"table"},
        "embeddings.3": continuous,
    }
    assert state["embeddings.3"]["weight_in"].shape == (3, 2)
    assert state[""]["projection"].shape == (4, 12)
    again = canonical_state(_embedding([CHARGE, TEMPERATURE, LABEL, VECTOR], seed=3))
    for path, tensors in again.items():
        for name, tensor in tensors.items():
            assert torch.equal(tensor, state[path][name]), (path, name)
            assert tensor.abs().max() > 0, (path, name)


def test_a_map_without_bias_saves_none():
    unbiased = ContinuousFeatureConfig(name="elec_temp", embedding_dim=3, bias=False)
    state = canonical_state(_embedding([unbiased]))
    assert set(state["embeddings.0"]) == {"weight_in", "weight_out"}


#: The structure inputs the features read, declared as the data carries them.
STRUCTURE_INPUTS = tuple(
    InputSpec(name=name, irreps="0e", per_atom=False, units="1")
    for name in ("total_charge", "elec_temp")
)


def _featured_model(from_embedding=False, num_heads=1):
    return _model(
        ObservableCatalogue(
            inputs=DEFAULT_CATALOGUE.inputs + STRUCTURE_INPUTS,
            observables=DEFAULT_CATALOGUE.observables,
        ),
        num_heads=num_heads,
        graph_features=(CHARGE, TEMPERATURE),
        readout=ReadoutConfig(mlp_irreps="8x0e", from_embedding=from_embedding),
    )


def _with_features(graph, charges=(1.0, -1.0), temperatures=(0.2, 0.7)):
    graph = dict(graph)
    graph["total_charge"] = torch.tensor(charges)
    graph["elec_temp"] = torch.tensor(temperatures)
    return graph


def test_the_features_are_added_to_what_the_first_layer_reads():
    backbone, _ = _featured_model()
    graph = _with_features(_two_structures())
    features = backbone(graph)
    element = backbone.element_embedding(graph["node_atomic_numbers"]).features
    assert backbone.graph_feature_embedding is not None
    added = backbone.graph_feature_embedding(graph, graph["batch"])
    assert_close(features.embedding, element + added, "embedding")
    changed = backbone(_with_features(graph, charges=(0.0, 2.0)))
    assert not torch.allclose(changed.layers[-1], features.layers[-1])


def test_a_model_without_features_reads_the_element_embedding():
    backbone, _ = _model()
    graph = _two_structures()
    element = backbone.element_embedding(graph["node_atomic_numbers"]).features
    assert_close(backbone(graph).embedding, element, "embedding")


def test_the_energy_read_out_of_the_embedding_is_in_the_atoms_and_the_total():
    """The frozen tree leaves it out of the per-atom energies; here they sum
    to the total."""
    model = _featured_model(from_embedding=True)
    output = _run(model, _with_features(_two_structures()))
    energies = output.node_energies
    total = output.total_energy
    assert energies is not None and total is not None
    assert_close(torch.stack([energies[:6].sum(), energies[6:].sum()]), total, "sum")
    plain = _run(_featured_model(), _with_features(_two_structures()))
    assert plain.total_energy is not None
    assert not torch.allclose(total, plain.total_energy)


def test_each_structure_reads_its_heads_embedding_energy():
    model = _featured_model(from_embedding=True, num_heads=2)
    graph = _with_features(_two_structures((0, 1)))
    _, outputs = model
    before = _run(model, graph).total_energy
    assert outputs.embedding_readout is not None
    state = dict(outputs.embedding_readout.to_canonical())
    state["weight"] = state["weight"].clone()
    state["weight"].view(2, -1)[1] += 1.0
    outputs.embedding_readout.load_canonical(state)
    after = _run(model, graph).total_energy
    assert before is not None and after is not None
    assert after[0] == before[0]
    assert after[1] != before[1]


def test_the_energy_stays_put_under_a_rotation_with_features():
    model = _featured_model(from_embedding=True)
    graph = _with_features(cluster(9), charges=(1.0,), temperatures=(0.4,))
    rotation = rotation_matrix(10).to(graph["positions"].dtype)
    moved = dict(graph)
    moved["positions"] = graph["positions"] @ rotation.T
    assert_close(_run(model, moved).total_energy, _run(model, graph).total_energy, "E")
