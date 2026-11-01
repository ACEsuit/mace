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
    BesselBasisConfig,
    ModelConfig,
    PolynomialCutoffConfig,
)
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.nn.backbone import MACEBackbone, aggregate_descriptors

from mace import modules
from mace.tools import utils
from tests.golden import anchors, harness
from tests.golden.harness import tolerance
from tests.parity.legacy_weights import (
    legacy_contraction_blocks,
    legacy_to_reference_map,
    linear_permutation,
    skip_permutation,
)

FP64 = tolerance("fp64_cpu_reference")
BACKEND = ReferenceBackend()
ELEMENTS = (1, 6, 8)
FIXTURES = ("water_cluster", "triclinic_bulk", "dimer_short")

#: The legacy class of each interaction kind.
LEGACY_CLASS = {
    "plain": "RealAgnosticInteractionBlock",
    "residual": "RealAgnosticResidualInteractionBlock",
    "density": "RealAgnosticDensityInteractionBlock",
    "density_residual": "RealAgnosticDensityResidualInteractionBlock",
    "nonlinear": "RealAgnosticResidualNonLinearInteractionBlock",
}


# ---------------------------------------------------------------------------
# Carrying the weights
# ---------------------------------------------------------------------------


def _carry_linear(legacy: o3.Linear, op) -> None:
    assert str(legacy.irreps_in) == op.descriptor.irreps_in, (
        legacy.irreps_in,
        op.descriptor.irreps_in,
    )
    assert str(legacy.irreps_out) == op.descriptor.irreps_out, (
        legacy.irreps_out,
        op.descriptor.irreps_out,
    )
    state = op.to_canonical()
    weight = torch.zeros_like(state["weight"])
    weight[linear_permutation(legacy)] = legacy.weight.detach()
    op.load_canonical({"weight": weight, "bias": state["bias"]})


def _carry_skip(legacy: o3.FullyConnectedTensorProduct, op) -> None:
    assert str(legacy.irreps_in1) == op.descriptor.irreps_in1
    assert str(legacy.irreps_out) == op.descriptor.irreps_out
    weight = torch.zeros_like(op.to_canonical()["weight"])
    weight.view(-1)[skip_permutation(legacy)] = legacy.weight.detach()
    op.load_canonical({"weight": weight})


def _carry_fully_connected_net(legacy, op) -> None:
    layers = len(op.descriptor.channels) - 1
    op.load_canonical(
        {
            f"layer_{i}": getattr(legacy, f"layer{i}").weight.detach()
            for i in range(layers)
        }
    )


def _carry_radial_mlp(legacy, op) -> None:
    linears = [m for m in legacy.net if isinstance(m, torch.nn.Linear)]
    norms = [m for m in legacy.net if isinstance(m, torch.nn.LayerNorm)]
    state = {}
    for i, linear in enumerate(linears):
        state[f"layer_{i}"] = linear.weight.detach().T
        state[f"bias_{i}"] = linear.bias.detach()
    for i, norm in enumerate(norms):
        state[f"norm_weight_{i}"] = norm.weight.detach()
        state[f"norm_bias_{i}"] = norm.bias.detach()
    op.load_canonical(state)


def _carry_contraction(legacy, op) -> None:
    correlation = op.descriptor.correlation
    with torch.no_grad():
        for position, contraction in enumerate(legacy.contractions):
            tables = list(op.bases[position])
            for coupling, weight, order in legacy_contraction_blocks(contraction):
                if weight.numel() == 0:
                    continue
                target = op.weights[position * correlation + order - 1]
                mapping = torch.from_numpy(
                    legacy_to_reference_map(coupling, tables[order - 1], order)
                )
                target.copy_(torch.einsum("pq,zpc->zqc", mapping, weight))


def _carry_interaction(legacy, block) -> None:
    name = type(legacy).__name__
    if name == LEGACY_CLASS["nonlinear"]:
        _carry_linear(legacy.source_embedding, block.sender_embedding)
        _carry_linear(legacy.target_embedding, block.receiver_embedding)
        _carry_linear(legacy.linear_up, block.linear_up)
        _carry_linear(legacy.skip_tp, block.skip)
        _carry_linear(legacy.linear_res, block.linear_residual)
        _carry_linear(legacy.linear_1, block.linear_mix)
        _carry_linear(legacy.linear_2, block.linear_out)
        _carry_radial_mlp(legacy.conv_tp_weights, block.radial_mlp)
        _carry_radial_mlp(legacy.density_fn, block.density_mlp)
        block.load_canonical(
            {"alpha": legacy.alpha.detach(), "beta": legacy.beta.detach()}
        )
        return
    body = block.body
    _carry_linear(legacy.linear_up, body.linear_up)
    _carry_fully_connected_net(legacy.conv_tp_weights, body.radial_mlp)
    _carry_linear(legacy.linear, body.linear)
    _carry_skip(legacy.skip_tp, block.skip)
    if hasattr(legacy, "density_fn"):
        _carry_fully_connected_net(legacy.density_fn, body.density_mlp)
    with torch.no_grad():
        body.neighbours.fill_(float(legacy.avg_num_neighbors))


def carry_weights(legacy, backbone: MACEBackbone) -> None:
    """Every weight of the modules the backbone rebuilds, legacy to v1."""
    _carry_linear(legacy.node_embedding.linear, backbone.node_embedding)
    for legacy_block, block in zip(
        legacy.interactions, backbone.interactions, strict=True
    ):
        _carry_interaction(legacy_block, block)
    for legacy_product, product in zip(legacy.products, backbone.products, strict=True):
        _carry_contraction(legacy_product.symmetric_contractions, product.contraction)
        _carry_linear(legacy_product.linear, product.linear)


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
            LEGACY_CLASS[config.interaction_first]
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


def _v1_graph(legacy, atoms) -> dict[str, torch.Tensor]:
    """The legacy model's own graph, with the one key v1 adds."""
    graph = anchors.anchor_graph(legacy, atoms, torch.float64)
    table = torch.tensor([int(z) for z in legacy.atomic_numbers])
    graph["node_atomic_numbers"] = table[graph["node_attrs"].argmax(dim=-1)]
    return graph


def _compare(legacy, backbone: MACEBackbone, fixtures) -> None:
    for name in FIXTURES:
        graph = _v1_graph(legacy, fixtures[name])
        expected = legacy(dict(graph), compute_force=False)["node_feats"]
        layers = backbone(graph).layers
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

ANCHOR_CONFIG = {
    "r_max": 3.5,
    "num_interactions": 2,
    "num_channels": 16,
    "max_L": 1,
    "max_ell": 2,
    "correlation": 3,
    "radial_basis": BesselBasisConfig(num_basis=8),
    "cutoff": PolynomialCutoffConfig(polynomial_order=5),
}


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
    assert {p.contraction.descriptor.basis for p in backbone.products} == {basis}
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
        ours = backbone.descriptors(
            backbone(_v1_graph(legacy, atoms)),
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
    table = torch.tensor(list(ELEMENTS))
    batch["node_atomic_numbers"] = table[batch["node_attrs"].argmax(dim=-1)]
    descriptors = backbone.descriptors(backbone(batch)).detach()
    element = batch["node_attrs"].argmax(dim=-1)
    num_graphs = batch["ptr"].numel() - 1
    means = aggregate_descriptors(descriptors, batch["batch"], num_graphs, "mean")
    per_element = aggregate_descriptors(
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
