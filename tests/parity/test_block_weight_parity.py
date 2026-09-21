"""Every weight the backbone holds, carried from a trained anchor.

Four kinds, and each is checked by copying the anchor's own numbers in and
comparing what the block computes. Nothing here compares weights: the layouts
differ on purpose, and two tensors that hold the same map are not the same
tensor.

Between them these cover the whole backbone, so a converted anchor has nowhere
left to differ except in how the blocks are wired together.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from fm00_convert import (
    contraction_weights_to_canonical,
    fully_connected_tp_weights_to_canonical,
    linear_weights_to_canonical,
)
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import FullyConnectedTPDescriptor
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.backends.reference.backend import ReferenceFullyConnectedTP
from mace_torch.nn.layout import channel_layout_index
from mace_torch.nn.product_basis import EquivariantProductBasisBlock

ANCHORS = Path(__file__).resolve().parents[1] / "golden/models"

#: One channel of the anchors' contraction input, and what each product reads
#: out. Taken from the anchors, not from their training recipes.
CHANNEL_IN = "0e+1o+2e"
PRODUCT_OUT = ["0e+1o", "0e"]
CHANNELS = 16
ELEMENTS = 3
CORRELATION = 3


def load_anchor(name: str):
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        return torch.load(ANCHORS / name, map_location="cpu", weights_only=False)
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous


@pytest.mark.parametrize("anchor", ["tiny_scaleshift.model", "tiny_mace.model"])
@pytest.mark.parametrize("layer", [0, 1])
def test_the_skip_connection_converts_exactly(fp64, anchor, layer):
    """The tensor product against the element attributes."""
    legacy = load_anchor(anchor).interactions[layer].skip_tp
    irreps_in1 = str(legacy.irreps_in1)
    irreps_in2 = str(legacy.irreps_in2)
    irreps_out = str(legacy.irreps_out)
    elements = Irreps.parse(irreps_in2).dimension

    mine = ReferenceFullyConnectedTP(
        FullyConnectedTPDescriptor(
            irreps_in1=irreps_in1, irreps_in2=irreps_in2, irreps_out=irreps_out
        )
    )
    with torch.no_grad():
        mine.weight.copy_(
            torch.tensor(
                fully_connected_tp_weights_to_canonical(
                    legacy, irreps_in1, irreps_out, elements
                ),
                dtype=torch.float64,
            )
        )

    features = torch.randn(7, Irreps.parse(irreps_in1).dimension, dtype=torch.float64)
    attributes = torch.randn(7, elements, dtype=torch.float64)
    theirs = legacy(features, attributes).detach()
    ours = mine(features, attributes).detach()

    assert float(theirs.abs().max()) > 1e-3, "the trained skip maps to nothing"
    assert float((theirs - ours).abs().max()) < 1e-13


def test_the_scale_on_the_skip_is_just_the_two_multiplicities(fp64):
    """e3nn's per-instruction factor times the scalar coupling's, which cancel.

    The factor is ``sqrt(dim_out / (mul_1 * mul_2))`` and coupling an irrep with
    a scalar carries ``1 / sqrt(dim_out)``, so the output dimension drops out
    and only the multiplicities are left. Pinned because the converter reads the
    factor off the module rather than rebuilding it from this identity, and the
    identity is what makes that reading checkable.
    """
    legacy = load_anchor("tiny_scaleshift.model").interactions[0].skip_tp
    for instruction in legacy.instructions:
        first, second, _ = instruction.path_shape
        dimension = (
            Irreps.parse(str(legacy.irreps_out)).terms[instruction.i_out][1].dimension
        )
        net = instruction.path_weight / dimension**0.5
        assert net == pytest.approx((first * second) ** -0.5, rel=1e-12)


@pytest.mark.parametrize("anchor", ["tiny_scaleshift.model", "tiny_mace.model"])
@pytest.mark.parametrize("index", [0, 1])
def test_a_whole_product_block_converts_exactly(fp64, anchor, index):
    """Contraction, projection onto the reduced basis, and the linear together.

    This is the one that carries the projection, so it is also where a wrong
    reduced basis would show up as a number rather than as a shape.
    """
    legacy = load_anchor(anchor).products[index]
    irreps_out = PRODUCT_OUT[index]
    targets = [str(irrep) for _, irrep in Irreps.parse(irreps_out).terms]

    mine = EquivariantProductBasisBlock(
        ReferenceBackend(),
        irreps_in=CHANNEL_IN,
        irreps_out=irreps_out,
        correlation=CORRELATION,
        num_elements=ELEMENTS,
        num_features=CHANNELS,
    )
    with torch.no_grad():
        for position, contraction in enumerate(
            legacy.symmetric_contractions.contractions
        ):
            carried = contraction_weights_to_canonical(
                contraction, CHANNEL_IN, targets[position], CORRELATION
            )
            for order, weights in enumerate(carried):
                mine.contraction.weights[position * CORRELATION + order].copy_(
                    torch.tensor(weights, dtype=torch.float64)
                )
        mine.linear.weight.copy_(
            torch.tensor(
                linear_weights_to_canonical(
                    legacy.linear,
                    str(legacy.linear.irreps_in),
                    str(legacy.linear.irreps_out),
                ),
                dtype=torch.float64,
            )
        )

    features = torch.randn(5, CHANNELS, 9, dtype=torch.float64)
    attributes = torch.zeros(5, ELEMENTS, dtype=torch.float64)
    attributes[:, 1] = 1.0
    element = torch.full((5,), 1, dtype=torch.long)

    theirs = legacy(features, None, attributes).detach()
    flat = features.reshape(5, -1)[:, channel_layout_index(CHANNEL_IN, CHANNELS)]
    ours = mine(flat, element, None).detach()

    assert float(theirs.abs().max()) > 1e-3, "the trained product maps to nothing"
    assert float((theirs - ours).abs().max()) < 1e-13


def test_the_orders_are_a_sum_and_the_frozen_tree_evaluates_them_as_a_cascade(fp64):
    """Different evaluation, same polynomial.

    The frozen tree folds the body orders into a Horner cascade, multiplying a
    running result by the features once per step. The rewrite sums each order
    on its own. Worth pinning, because a cascade looks like it couples the
    orders and a port that reproduced the coupling would be wrong in a way that
    still trains.
    """
    import numpy as np

    contraction = (
        load_anchor("tiny_scaleshift.model")
        .products[0]
        .symmetric_contractions.contractions[0]
    )
    features = torch.randn(5, CHANNELS, 9, dtype=torch.float64)
    attributes = torch.zeros(5, ELEMENTS, dtype=torch.float64)
    attributes[:, 0] = 1.0
    cascade = contraction(features, attributes).detach().numpy().reshape(5, CHANNELS)

    by_order = {CORRELATION: contraction.weights_max.detach().numpy()}
    for offset, weight in enumerate(contraction.weights):
        by_order[CORRELATION - 1 - offset] = weight.detach().numpy()

    values = features.numpy()
    summed = np.zeros((5, CHANNELS))
    for order in range(1, CORRELATION + 1):
        basis = np.moveaxis(getattr(contraction, f"U_matrix_{order}").numpy(), -1, 0)
        letters = "xyzuv"[:order]
        inputs = ",".join(f"nc{letter}" for letter in letters)
        contracted = np.einsum(f"p{letters},{inputs}->ncp", basis, *([values] * order))
        summed += np.einsum("ncp,pc->nc", contracted, by_order[order][0])

    assert np.abs(cascade).max() > 1e-3
    assert np.abs(cascade - summed).max() < 1e-13
