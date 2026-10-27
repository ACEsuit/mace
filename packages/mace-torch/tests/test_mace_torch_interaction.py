"""The interaction blocks, one at a time.

What each block asks the backend for, where its skip goes, how it
normalizes, and how the nonlinear block starts. The whole-backbone properties,
equivariance among them, are in the backbone's suite.
"""

import pytest
import torch
from conftest import assert_close
from mace_core.clebsch_gordan.irreps import Irreps
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.kernels import initialize_model_weights
from mace_torch.nn.edges import EdgeInputs
from mace_torch.nn.gate import Gate, gate_input_irreps
from mace_torch.nn.interaction import (
    InteractionBlock,
    NonLinearInteractionBlock,
    ResidualInteractionBlock,
)

BACKEND = ReferenceBackend()


@pytest.fixture(autouse=True)
def no_gradients():
    with torch.no_grad():
        yield


def _precision() -> str:
    return {torch.float64: "float64", torch.float32: "float32"}[
        torch.get_default_dtype()
    ]


class _Edges(EdgeInputs):
    """Fixed edge inputs, without the modules that would compute them."""

    def __init__(self, radial: torch.Tensor, attributes: torch.Tensor) -> None:
        super().__init__(
            torch.zeros(radial.shape[0], 3),
            torch.zeros(0),
            torch.zeros(2, 0),
            torch.nn.Identity(),
            torch.nn.Identity(),
        )
        self._radial = radial
        self._attributes = attributes
        self._envelope = torch.ones(radial.shape[0], 1)


def _inputs(num_features: int, irreps_node: str, num_elements: int = 2):
    generator = torch.Generator().manual_seed(0)
    num_nodes, num_edges = 4, 7
    node_features = torch.randn(
        num_nodes,
        num_features * Irreps.parse(irreps_node).dimension,
        generator=generator,
    )
    attributes = torch.randn(num_edges, 9, generator=generator)
    radial = torch.randn(num_edges, 5, generator=generator)
    node_attrs = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 1, 0]), num_elements
    ).to(torch.get_default_dtype())
    sender = torch.tensor([0, 1, 2, 3, 0, 2, 1])
    receiver = torch.tensor([1, 0, 3, 2, 2, 0, 3])
    return (
        node_features,
        _Edges(radial, attributes),
        node_attrs,
        sender,
        receiver,
        num_nodes,
    )


def _block(kind, **options):
    common = {
        "irreps_node": "0e+1o",
        "irreps_edge": "0e+1o+2e",
        "irreps_target": "0e+1o+2e",
        "num_radial": 5,
        "num_features": 3,
        "num_elements": 2,
        "radial_hidden": (8,),
        "precision": _precision(),
    }
    common.update(options)
    if kind is NonLinearInteractionBlock:
        common.setdefault("irreps_up", "0e+1o")
        common.setdefault("num_up_features", 2)
    block = kind(BACKEND, **common)
    initialize_model_weights(block, 1)
    return block


def test_the_plain_block_applies_its_skip_to_the_message_and_carries_none():
    block = _block(InteractionBlock)
    inputs = _inputs(3, "0e+1o")
    message, carried = block(*inputs)
    assert carried is None
    expected = block.skip(block.body(inputs[0], inputs[1], *inputs[3:]), inputs[2])
    assert_close(message, expected, "skip of the message")


def test_the_residual_block_carries_the_skip_of_its_input():
    block = _block(ResidualInteractionBlock, irreps_skip_out="0e+1o")
    inputs = _inputs(3, "0e+1o")
    message, carried = block(*inputs)
    assert_close(carried, block.skip(inputs[0], inputs[2]), "carried skip")
    assert message.shape[1] == 3 * Irreps.parse("0e+1o+2e").dimension


def test_the_average_normalization_divides_after_the_linear():
    block = _block(
        ResidualInteractionBlock, irreps_skip_out="0e", avg_num_neighbors=1.0
    )
    inputs = _inputs(3, "0e+1o")
    once, _ = block(*inputs)
    block.body.neighbours.fill_(4.0)
    quartered, _ = block(*inputs)
    assert_close(quartered, once / 4.0, "divided by the average")


def test_a_learned_density_replaces_the_average():
    block = _block(ResidualInteractionBlock, irreps_skip_out="0e", learned_density=True)
    assert block.body.density_mlp is not None
    inputs = _inputs(3, "0e+1o")
    before, _ = block(*inputs)
    block.body.neighbours.fill_(100.0)
    after, _ = block(*inputs)
    assert_close(after, before, "the average is not read")


def test_the_envelope_outside_scales_the_radial_weights():
    block = _block(
        ResidualInteractionBlock, irreps_skip_out="0e", envelope_outside=True
    )
    inputs = _inputs(3, "0e+1o")
    whole, _ = block(*inputs)
    inputs[1]._envelope = torch.zeros_like(inputs[1]._envelope)
    cut, _ = block(*inputs)
    assert_close(cut, torch.zeros_like(cut), "an envelope of zero cuts every edge")
    assert whole.abs().max() > 0


def test_the_nonlinear_block_starts_where_the_frozen_tree_does():
    block = _block(NonLinearInteractionBlock, irreps_skip_out="0e+1o")
    assert float(block.alpha) == 20.0
    assert float(block.beta) == 0.0
    for embedding in (block.sender_embedding, block.receiver_embedding):
        weight = embedding.to_canonical()["weight"]
        assert weight.abs().max() <= 1e-3
        assert weight.abs().max() > 0


def test_the_nonlinear_block_reads_both_elements():
    block = _block(NonLinearInteractionBlock, irreps_skip_out="0e+1o")
    block.load_canonical({"alpha": torch.tensor(1.0), "beta": torch.tensor(0.5)})
    inputs = list(_inputs(3, "0e+1o"))
    before, _ = block(*inputs)
    swapped = inputs[2].flip(-1)
    inputs[2] = swapped
    after, _ = block(*inputs)
    assert not torch.allclose(before, after)


def test_the_gate_scales_each_copy_by_its_own_gate():
    irreps = "2x0e+2x1o"
    assert gate_input_irreps(irreps) == "4x0e+2x1o"
    gate = Gate(irreps)
    features = torch.randn(3, 4 + 6, generator=torch.Generator().manual_seed(2))
    out = gate(features)
    assert out.shape == (3, 2 + 6)
    vectors = features[:, 4:].reshape(3, 2, 3)
    gated = out[:, 2:].reshape(3, 2, 3)
    ratio = gated / vectors
    assert_close(ratio - ratio[:, :, :1], torch.zeros_like(ratio), "one gate per copy")


def test_an_odd_scalar_cannot_be_gated():
    with pytest.raises(ValueError, match="odd scalar"):
        Gate("2x0o")
