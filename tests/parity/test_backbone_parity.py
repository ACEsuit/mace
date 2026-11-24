"""The v1 backbone's node features against the frozen tree's, in one process.

The legacy model's ``node_feats`` is every layer's node features side by side,
grouped by irrep, which is the layout :class:`MACEBackbone` returns per layer.
So a backbone holding the same weights must reproduce it layer by layer.

The weights are carried from the legacy modules this ticket rebuilds into the
v1 ones here, op by op, through the maps in :mod:`tests.parity.legacy_weights`,
which read each legacy module's own instruction list. The two committed tiny
anchors both have the plain first layer. The residual first layer, the legacy
command line's default, and the other interaction blocks are built live, from
the legacy constructor, with their weights perturbed so that no parameter sits
at a value that hides it (the nonlinear block's ``beta`` starts at zero, which
would make its density invisible).
"""

import numpy as np
import pytest
import torch
from ase.data import chemical_symbols
from e3nn import o3
from mace_core.config import (
    AgnesiTransformConfig,
    ModelConfig,
)
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.descriptors import aggregate_atomic_descriptors
from mace_torch.nn.backbone import MACEBackbone

from mace import modules
from mace.tools import utils
from tests.golden import anchors, harness
from tests.golden.harness import tolerance
from tests.parity.legacy_models import ANCHOR_CONFIG, v1_graph
from tests.parity.legacy_weights import LEGACY_CLASS, carry_weights

FP64 = tolerance("fp64_cpu_reference")
BACKEND = ReferenceBackend()
ELEMENTS = (1, 6, 8)
FIXTURES = ("water_cluster", "triclinic_bulk", "dimer_short")

# ---------------------------------------------------------------------------
# Building and running
# ---------------------------------------------------------------------------


def _legacy_model(config: ModelConfig, avg_num_neighbors: float, seed: int):
    """A live legacy model of ``config``'s architecture, weights perturbed."""
    torch.manual_seed(seed)
    hidden = o3.Irreps(
        " + ".join(
            f"{config.num_channels}x{degree}{'e' if degree % 2 == 0 else 'o'}"
            for degree in range(config.max_L + 1)
        )
    )
    transform = config.distance_transform.kind
    model = modules.MACE(
        r_max=config.r_max,
        num_bessel=config.radial_basis.num_basis,
        num_polynomial_cutoff=config.cutoff.polynomial_order,
        max_ell=config.max_ell,
        interaction_cls=modules.interaction_classes[LEGACY_CLASS[config.interaction]],
        interaction_cls_first=modules.interaction_classes[
            LEGACY_CLASS[config.interaction_of(0)]
        ],
        num_interactions=config.num_interactions,
        num_elements=len(ELEMENTS),
        hidden_irreps=hidden,
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=torch.zeros(len(ELEMENTS)),
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=list(ELEMENTS),
        correlation=config.correlation,
        radial_type="bessel",
        distance_transform="None" if transform == "none" else transform.capitalize(),
        apply_cutoff=config.apply_cutoff,
        radial_MLP=list(config.radial_mlp),
        edge_irreps=None
        if config.edge_irreps is None
        else o3.Irreps(config.edge_irreps),
        use_edge_irreps_first=config.use_edge_irreps_first,
        use_agnostic_product=config.use_agnostic_product,
        use_reduced_cg=False,
    )
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
    return model


def _compare(legacy, backbone: MACEBackbone, fixtures) -> None:
    for name in FIXTURES:
        graph = anchors.anchor_graph(legacy, fixtures[name], torch.float64)
        expected = legacy(dict(graph), compute_force=False)["node_feats"]
        features = backbone(v1_graph(legacy, graph))
        assert torch.equal(features.node_attrs, graph["node_attrs"]), name
        layers = features.layers
        widths = [layer.shape[-1] for layer in layers]
        for index, (ours, theirs) in enumerate(
            zip(layers, expected.split(widths, dim=-1), strict=True)
        ):
            torch.testing.assert_close(
                ours,
                theirs,
                atol=FP64.atol,
                rtol=FP64.rtol,
                msg=lambda message, name=name, index=index: (
                    f"{name}, layer {index}: {message}"
                ),
            )


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return harness.load_fixtures(names=FIXTURES, elements=ELEMENTS)


# ---------------------------------------------------------------------------
# The anchors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", ["tiny_mace", "tiny_scaleshift"])
def test_the_backbone_reproduces_an_anchor_layer_by_layer(fp64, fixtures, anchor):
    legacy = anchors.load_anchor(anchor)
    assert type(legacy.interactions[0]).__name__ == LEGACY_CLASS["plain"]
    config = ModelConfig(
        interaction_first="plain", interaction="residual", **ANCHOR_CONFIG
    )
    table = utils.AtomicNumberTable([int(z) for z in legacy.atomic_numbers])
    backbone = MACEBackbone(
        BACKEND,
        config,
        table.zs,
        avg_num_neighbors=float(legacy.interactions[0].avg_num_neighbors),
    )
    carry_weights(legacy, backbone)
    _compare(legacy, backbone, fixtures)


def test_the_two_first_layer_spellings_are_two_models(fp64):
    """The residual first layer cannot even hold the plain anchor's weights:
    one skip reads the message, the other the element embedding."""
    legacy = anchors.load_anchor("tiny_mace")
    config = ModelConfig(
        interaction_first="residual", interaction="residual", **ANCHOR_CONFIG
    )
    backbone = MACEBackbone(BACKEND, config, list(ELEMENTS))
    with pytest.raises(AssertionError):
        carry_weights(legacy, backbone)


# ---------------------------------------------------------------------------
# The other interaction blocks, live
# ---------------------------------------------------------------------------

LIVE_CASES = {
    "residual_first": {"interaction_first": "residual", "interaction": "residual"},
    "density": {"interaction_first": "density", "interaction": "density_residual"},
    "plain_then_plain": {"interaction_first": "plain", "interaction": "plain"},
    "nonlinear": {
        "interaction_first": "nonlinear",
        "interaction": "nonlinear",
        "edge_irreps": "8x0e+8x1o",
    },
    "nonlinear_edge_irreps_first": {
        "interaction_first": "nonlinear",
        "interaction": "nonlinear",
        "edge_irreps": "8x0e+8x1o",
        "use_edge_irreps_first": True,
    },
    "nonlinear_first_then_residual": {
        "interaction_first": "nonlinear",
        "interaction": "residual",
    },
    "agnostic_product": {"use_agnostic_product": True},
    "envelope_after_the_radial_networks": {
        "apply_cutoff": False,
        "interaction_first": "density",
        "interaction": "nonlinear",
    },
    "agnesi_transform": {"distance_transform": AgnesiTransformConfig()},
    "three_layers_max_L_2": {"num_interactions": 3, "max_L": 2, "num_channels": 8},
}


@pytest.mark.parametrize("case", sorted(LIVE_CASES))
def test_the_backbone_reproduces_each_kept_variant(fp64, fixtures, case):
    options = {**ANCHOR_CONFIG, "radial_mlp": (16, 16), **LIVE_CASES[case]}
    config = ModelConfig(**options)
    legacy = _legacy_model(config, avg_num_neighbors=3.0, seed=7)
    backbone = MACEBackbone(BACKEND, config, list(ELEMENTS), avg_num_neighbors=3.0)
    carry_weights(legacy, backbone)
    _compare(legacy, backbone, fixtures)


@pytest.mark.parametrize("basis", ["reduced", "full"])
def test_either_basis_reproduces_the_anchor(fp64, fixtures, basis):
    """The basis is a recorded choice of weight coordinates, not of function:
    a legacy model carries into either."""
    legacy = anchors.load_anchor("tiny_mace")
    config = ModelConfig(
        interaction_first="plain",
        interaction="residual",
        clebsch_gordan_basis=basis,
        **ANCHOR_CONFIG,
    )
    backbone = MACEBackbone(BACKEND, config, list(ELEMENTS), avg_num_neighbors=8.0)
    assert {p.contraction.kernel_spec.basis for p in backbone.products} == {basis}
    carry_weights(legacy, backbone)
    _compare(legacy, backbone, fixtures)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_layers", [-1, 1, 2])
@pytest.mark.parametrize("invariants_only", [True, False])
def test_the_descriptors_are_the_legacy_calculators(
    fp64, fixtures, num_layers, invariants_only
):
    """Every ``invariants_only`` and ``num_layers`` the legacy calculator
    accepts, including the invariant-only last layer's narrower width."""
    from mace.calculators import MACECalculator

    legacy = anchors.load_anchor("tiny_mace")
    calculator = MACECalculator(models=legacy, device="cpu", default_dtype="float64")
    config = ModelConfig(
        interaction_first="plain", interaction="residual", **ANCHOR_CONFIG
    )
    backbone = MACEBackbone(BACKEND, config, list(ELEMENTS), avg_num_neighbors=8.0)
    carry_weights(legacy, backbone)
    for name in FIXTURES:
        atoms = fixtures[name]
        expected = calculator.get_descriptors(
            atoms, invariants_only=invariants_only, num_layers=num_layers
        )
        ours = backbone.atomic_descriptors(
            backbone(
                v1_graph(legacy, anchors.anchor_graph(legacy, atoms, torch.float64))
            ),
            num_layers=num_layers,
            invariants_only=invariants_only,
        )
        np.testing.assert_allclose(
            ours.detach().numpy(),
            expected,
            atol=FP64.atol,
            rtol=FP64.rtol,
            err_msg=name,
        )


def test_the_aggregations_are_the_eval_command_lines(fp64, fixtures):
    """``mean`` and ``per_element_mean`` as the legacy eval command line
    computes them from the per-atom descriptors, one structure at a time."""
    legacy = anchors.load_anchor("tiny_mace")
    config = ModelConfig(
        interaction_first="plain", interaction="residual", **ANCHOR_CONFIG
    )
    backbone = MACEBackbone(BACKEND, config, list(ELEMENTS), avg_num_neighbors=8.0)
    carry_weights(legacy, backbone)
    structures = [fixtures[name] for name in FIXTURES]
    batch = anchors.anchor_batch(legacy, structures).to_dict()
    descriptors = backbone.atomic_descriptors(
        backbone(v1_graph(legacy, batch))
    ).detach()
    element = batch["node_attrs"].argmax(dim=-1)
    num_graphs = batch["ptr"].numel() - 1
    means = aggregate_atomic_descriptors(
        descriptors, batch["batch"], num_graphs, "mean"
    )
    per_element = aggregate_atomic_descriptors(
        descriptors,
        batch["batch"],
        num_graphs,
        "per_element_mean",
        element=element,
        atomic_numbers=ELEMENTS,
    )
    for graph, atoms in enumerate(structures):
        rows = descriptors[batch["batch"] == graph].numpy()
        # What eval_configs writes for each aggregation.
        np.testing.assert_allclose(
            means[graph].numpy(), np.mean(rows, axis=0), atol=FP64.atol
        )
        expected = {
            element_symbol: np.mean(rows[atoms.symbols == element_symbol], axis=0)
            for element_symbol in np.unique(atoms.symbols)
        }
        numbers = {symbol: chemical_symbols.index(symbol) for symbol in expected}
        assert sorted(per_element[graph]) == sorted(numbers.values())
        for symbol, value in expected.items():
            np.testing.assert_allclose(
                per_element[graph][numbers[symbol]].numpy(), value, atol=FP64.atol
            )
