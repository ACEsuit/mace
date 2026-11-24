"""The graph-feature embedding and the embedding readout against the frozen
tree, in one process.

The feature declarations are the two the frozen tree's embedding trainings
pin (tests/workflows/test_embedding_train.py): a structure's electronic
temperature, total charge and total spin, and three features of an atom, a
label and two real values. The joint embedding is compared on its own with
its weights carried, and then inside whole models, `MACE` and
`ScaleShiftMACE`, reading an energy out of the embedding as well.

One difference is deliberate: the frozen tree adds that energy to the total
and not to the per-atom energies, so its per-atom energies do not sum to the
total. Here they do, and the comparison adds it to the frozen tree's.
"""

import pytest
import torch
from mace_core.config import (
    CategoricalFeatureConfig,
    ContinuousFeatureConfig,
    ModelConfig,
    ReadoutConfig,
)
from mace_core.observables import DEFAULT_CATALOGUE, InputSpec, ObservableCatalogue
from mace_torch.nn.graph_features import GraphFeatureEmbedding

from mace import modules
from mace.modules.embeddings import GenericJointEmbedding
from tests.golden import anchors, harness
from tests.parity.legacy_models import (
    ANCHOR_CONFIG,
    ELEMENTS,
    V1Model,
    forces,
    live_model,
    perturb,
)
from tests.parity.legacy_weights import carry_joint_embedding

CLOSED_FORM = harness.tolerance("closed_form_fp64")
FIXTURES = ("water_cluster", "triclinic_bulk", "dimer_short", "isolated_atom")

#: The frozen tree's spelling of the structure features it pins.
GRAPH_SPECS = {
    "elec_temp": {"type": "continuous", "per": "graph", "in_dim": 1, "emb_dim": 8},
    "total_charge": {
        "type": "categorical",
        "per": "graph",
        "emb_dim": 8,
        "num_classes": 200,
        "offset": 100,
    },
    "total_spin": {
        "type": "categorical",
        "per": "graph",
        "emb_dim": 8,
        "num_classes": 100,
    },
}

#: And of the atom features.
ATOM_SPECS = {
    "cat": {"type": "categorical", "per": "atom", "emb_dim": 8, "num_classes": 2},
    "quant": {"type": "continuous", "per": "atom", "in_dim": 1, "emb_dim": 8},
    "quant_v": {"type": "continuous", "per": "atom", "in_dim": 2, "emb_dim": 8},
}


def _v1_features(specs):
    """The same declarations as model configuration sections."""
    features = []
    for name, spec in specs.items():
        common = {"name": name, "per": spec["per"], "embedding_dim": spec["emb_dim"]}
        if spec["type"] == "categorical":
            features.append(
                CategoricalFeatureConfig(
                    num_classes=spec["num_classes"],
                    offset=spec.get("offset", 0),
                    **common,
                )
            )
        else:
            features.append(ContinuousFeatureConfig(input_dim=spec["in_dim"], **common))
    return tuple(features)


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return harness.load_fixtures(names=FIXTURES, elements=ELEMENTS)


@pytest.mark.parametrize("specs", [GRAPH_SPECS, ATOM_SPECS], ids=["graph", "atom"])
def test_the_joint_embedding_is_reproduced(fp64, specs):
    torch.manual_seed(41)
    legacy = GenericJointEmbedding(base_dim=16, embedding_specs=specs).double()
    ours = GraphFeatureEmbedding(_v1_features(specs), 16)
    carry_joint_embedding(legacy, ours)

    generator = torch.Generator().manual_seed(42)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    values = {
        "elec_temp": torch.rand(3, generator=generator, dtype=torch.float64),
        "total_charge": torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64),
        "total_spin": torch.tensor([1.0, 3.0, 2.0], dtype=torch.float64),
        "cat": torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
        "quant": torch.randn(9, generator=generator, dtype=torch.float64),
        "quant_v": torch.randn(9, 2, generator=generator, dtype=torch.float64),
    }
    graph = {name: values[name] for name in specs}
    graph["ptr"] = torch.tensor([0, 3, 5, 9])
    torch.testing.assert_close(
        ours(graph, batch),
        legacy(batch, graph),
        atol=CLOSED_FORM.atol,
        rtol=CLOSED_FORM.rtol,
    )


def _v1(legacy) -> V1Model:
    config = ModelConfig(
        **ANCHOR_CONFIG | {"correlation": 2},
        interaction_first="plain",
        interaction="residual",
        observables=ObservableCatalogue(
            inputs=DEFAULT_CATALOGUE.inputs
            + tuple(
                InputSpec(name=name, irreps="0e", per_atom=False, units="1")
                for name in GRAPH_SPECS
            ),
            observables=DEFAULT_CATALOGUE.observables,
        ),
        graph_features=_v1_features(GRAPH_SPECS),
        readout=ReadoutConfig(mlp_irreps="8x0e", from_embedding=True),
    )
    return V1Model(legacy, config)


@pytest.mark.parametrize(
    ("model_class", "extra"),
    [
        (modules.MACE, {}),
        (
            modules.ScaleShiftMACE,
            {"atomic_inter_scale": 1.7, "atomic_inter_shift": 0.3},
        ),
    ],
    ids=["MACE", "ScaleShiftMACE"],
)
def test_a_model_embedding_graph_features_is_reproduced(
    fp64, fixtures, model_class, extra
):
    legacy = perturb(
        live_model(
            model_class,
            seed=43,
            hidden_irreps="16x0e + 16x1o",
            use_reduced_cg=False,
            embedding_specs=GRAPH_SPECS,
            use_embedding_readout=True,
            **extra,
        ),
        seed=44,
    )
    v1 = _v1(legacy)
    graph = dict(
        anchors.anchor_batch(
            legacy, [fixtures[name] for name in FIXTURES], torch.float64
        ).to_dict()
    )
    graph["total_charge"] = torch.tensor([1.0, -2.0, 0.0, 3.0], dtype=torch.float64)
    graph["total_spin"] = torch.tensor([2.0, 1.0, 4.0, 1.0], dtype=torch.float64)
    graph["elec_temp"] = torch.tensor([0.1, 0.5, 0.0, 0.3], dtype=torch.float64)

    captured = []
    handle = legacy.embedding_readout.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output[:, 0])
    )
    try:
        expected = legacy(dict(graph), compute_force=True)
    finally:
        handle.remove()

    output, positions = v1(graph)
    for name, value, reference in (
        ("energy", output.total_energy, expected["energy"]),
        ("forces", forces(output.total_energy, positions), expected["forces"]),
        ("node energies", output.node_energies, expected["node_energy"] + captured[0]),
    ):
        torch.testing.assert_close(
            value,
            reference,
            atol=CLOSED_FORM.atol,
            rtol=CLOSED_FORM.rtol,
            msg=lambda message, name=name: f"{name}: {message}",
        )
    assert captured[0].abs().max() > 1e-3
