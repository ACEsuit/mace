"""The converter's projection on a trained model's own basis.

The product projection in ``mace_core`` symmetrizes each path algebraically;
the one the keystone gate was built on identifies the same map by sampling
feature vectors. They share no code past the reduced basis itself, so agreeing
on what a trained contraction computes is a check on both. Every contraction of
the anchor is covered, both output irreps of the first block and the scalar of
the second, at every body order.
"""

from __future__ import annotations

import numpy as np
import pytest
from mace_core.clebsch_gordan.conversion import full_to_reduced
from mace_core.clebsch_gordan.reduced_basis import (
    reduced_symmetric_tensor_product_basis,
)

from tests.parity.fm00_projection import as_path_first, project_weights
from tests.parity.test_fm00_training_step import load_anchor

IRREPS_IN = "0e+1o+2e"
CORRELATION = 3


def contractions():
    """``(block, target, contraction)`` for every contraction of the anchor."""
    legacy = load_anchor("tiny_scaleshift.model")
    found = []
    for block, product in enumerate(legacy.products):
        targets = [str(irrep) for _, irrep in product.linear.irreps_out]
        for target, contraction in zip(
            targets, product.symmetric_contractions.contractions, strict=True
        ):
            found.append((block, target, contraction))
    return found


def by_order(contraction):
    weights = {CORRELATION: contraction.weights_max.detach().numpy()}
    for offset, weight in enumerate(contraction.weights):
        weights[CORRELATION - 1 - offset] = weight.detach().numpy()
    return weights


def computed(basis, weights, features, order):
    letters = "xyzuv"[:order]
    inputs = ",".join(f"nc{letter}" for letter in letters)
    paths = np.einsum(f"po{letters},{inputs}->ncop", basis, *([features] * order))
    return np.einsum("ncop,pc->nco", paths, weights[0])


CASES = [
    (block, target, order)
    for block, target, _ in contractions()
    for order in range(1, CORRELATION + 1)
]


@pytest.mark.parametrize("block,target,order", CASES)
def test_the_product_projection_keeps_a_trained_contraction(block, target, order):
    contraction = next(
        found for b, t, found in contractions() if (b, t) == (block, target)
    )
    weights = by_order(contraction)[order]
    trained = as_path_first(getattr(contraction, f"U_matrix_{order}").numpy(), order, 9)
    reduced = reduced_symmetric_tensor_product_basis(IRREPS_IN, order, target)[target]
    features = np.random.default_rng(order).normal(size=(30, weights.shape[2], 9))

    carried = full_to_reduced(
        weights.astype(float), IRREPS_IN, order, target, source=trained
    )
    oracle = project_weights(
        weights.astype(float),
        IRREPS_IN,
        order,
        target,
        source=getattr(contraction, f"U_matrix_{order}").numpy(),
    )

    before = computed(trained, weights, features, order)
    assert np.abs(before).max() > 1e-6, (
        "an order that contributes nothing proves nothing"
    )
    after = computed(reduced, carried, features, order)
    sampled = computed(reduced, oracle, features, order)
    assert np.abs(before - after).max() < 1e-12
    assert np.abs(after - sampled).max() < 1e-12
