"""The full-to-reduced projection, and the rank it depends on.

The converter's one numerically dangerous step. It can be wrong while looking
entirely plausible: a reduced basis short of a direction still produces a
projection, still returns weights of the right shape, and still runs. What it
does not do is compute the same function.

So the projection is pinned three ways. Its residual is measured, never
assumed. Its rank is checked against a derivation that uses no coupling
coefficients at all. And it is applied to a real trained model's weights and
compared on what the contraction computes, never on the weights themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from fm00_projection import (
    ProjectionError,
    as_path_first,
    project_weights,
    projection_matrix,
)
from independent_basis import symmetric_multiplicity

from mace_core.clebsch_gordan.reduced_basis import (
    full_symmetric_tensor_product_basis,
    reduced_symmetric_tensor_product_basis,
)

#: One channel of the tiny anchors' contraction input, and the two irreps they
#: read out. Taken from the anchor itself, not from its training recipe: the
#: contraction sees the interaction's output, which is wider than the declared
#: hidden irreps.
IRREPS_IN = "0e+1o+2e"
TARGETS = ("0e", "1o")
ORDERS = (1, 2, 3)

#: `0e+1o+2e` as (degree, parity) pairs, for the character derivation.
TERMS = [(0, +1), (1, -1), (2, +1)]

ANCHOR = Path(__file__).resolve().parents[1] / "golden/models/tiny_scaleshift.model"


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("order", ORDERS)
def test_the_reduced_basis_reproduces_the_full_one(target, order):
    """The residual is rounding error, not a fit's leftovers."""
    full = full_symmetric_tensor_product_basis(IRREPS_IN, order, target)[target]
    reduced = reduced_symmetric_tensor_product_basis(IRREPS_IN, order, target)[target]
    matrix = projection_matrix(IRREPS_IN, order, target)

    assert matrix.shape == (reduced.shape[0], full.shape[0])

    generator = np.random.default_rng(7)
    directions = generator.normal(size=(200, full.shape[-1]))
    letters = "xyzuv"[:order]
    subscripts = f"po{letters}," + ",".join(f"s{c}" for c in letters) + "->sop"
    from_full = np.einsum(subscripts, full, *([directions] * order))
    from_reduced = np.einsum(subscripts, reduced, *([directions] * order))

    deviation = np.abs(np.einsum("rf,sor->sof", matrix, from_reduced) - from_full)
    assert deviation.max() < 1e-12, (
        f"the projected reduced basis differs from the full basis by "
        f"{deviation.max():.3e} on directions it was not identified from"
    )


@pytest.mark.parametrize("target,degree,parity", [("0e", 0, +1), ("1o", 1, -1)])
@pytest.mark.parametrize("order", ORDERS)
def test_the_rank_agrees_with_a_derivation_that_uses_no_coupling_coefficients(
    target, degree, parity, order
):
    """Counted from O(3) characters instead.

    This is what stops the gate from self-validating: the reduced basis is
    built from coupling coefficients, and if those were wrong the basis would
    still be internally consistent. The character count shares nothing with it.
    """
    reduced = reduced_symmetric_tensor_product_basis(IRREPS_IN, order, target)[target]
    counted = symmetric_multiplicity(TERMS, order, degree, parity)

    assert abs(counted - reduced.shape[0]) < 1e-2, (
        f"the reduced basis offers {reduced.shape[0]} path(s) for {target} at "
        f"order {order}, and the character integral says {counted:.4f}"
    )


def test_the_full_basis_matches_the_trained_anchor_path_counts():
    """A trained model is itself a statement about how many paths there are.

    The anchor was trained against the frozen tree's full basis, so the shapes
    of its weights record that basis's path count. Reproducing them from a
    derivation written years later and independently is a real check on both.
    """
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        model = torch.load(ANCHOR, map_location="cpu", weights_only=False)
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous

    contraction = model.products[0].symmetric_contractions.contractions[0]
    # The frozen tree stores the highest body order first and counts down, while
    # the path list runs the other way.
    trained = [contraction.weights_max.shape[1]] + [
        weight.shape[1] for weight in contraction.weights
    ]
    derived = [
        full_symmetric_tensor_product_basis(IRREPS_IN, order, "0e")["0e"].shape[0]
        for order in (3, 2, 1)
    ]
    assert trained == derived, (
        f"the anchor was trained with {trained} full paths per body order and "
        f"this basis derives {derived}"
    )


def test_a_real_model_s_weights_keep_computing_the_same_function():
    """The end-to-end claim, on a trained model, compared on outputs.

    Nothing here compares a projected weight against an original one. There is
    no sense in which they are equal: the conversion preserves what the
    contraction computes, and that is the only thing worth asserting.
    """
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        model = torch.load(ANCHOR, map_location="cpu", weights_only=False)
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous

    contraction = model.products[0].symmetric_contractions.contractions[0]
    by_order = {3: contraction.weights_max.detach().numpy()}
    for offset, weight in enumerate(contraction.weights):
        by_order[2 - offset] = weight.detach().numpy()
    # The basis those weights are written against travels with the artifact.
    # It spans the same space as this package's own full basis and is not the
    # same basis, so projecting with anything else computes another function.
    trained_basis = {
        3: contraction.U_matrix_3.numpy(),
        2: contraction.U_matrix_2.numpy(),
        1: contraction.U_matrix_1.numpy(),
    }

    generator = np.random.default_rng(11)
    features = generator.normal(size=(40, 16, 9))  # [nodes, channels, components]

    for order, weights in sorted(by_order.items()):
        full = as_path_first(trained_basis[order], order, 9)
        reduced = reduced_symmetric_tensor_product_basis(IRREPS_IN, order, "0e")["0e"]
        projected = project_weights(
            weights, IRREPS_IN, order, "0e", source=trained_basis[order]
        )

        # Input axes are named from the end of the alphabet so they cannot
        # collide with the node and channel labels.
        letters = "xyzuv"[:order]
        inputs = ",".join(f"nc{letter}" for letter in letters)
        subscripts = f"po{letters},{inputs}->ncop"
        paths_full = np.einsum(subscripts, full, *([features] * order))
        paths_reduced = np.einsum(subscripts, reduced, *([features] * order))

        # weights are [element, path, channel]; take element 0.
        before = np.einsum("ncop,pc->nco", paths_full, weights[0])
        after = np.einsum("ncop,pc->nco", paths_reduced, projected[0])

        assert np.abs(before).max() > 1e-6, (
            f"order {order} contributes nothing, so agreeing says nothing"
        )
        deviation = np.abs(before - after).max()
        assert deviation < 1e-12, (
            f"order {order} of a trained contraction changed by {deviation:.3e} "
            f"when its weights were carried onto the reduced basis"
        )


def test_weights_from_a_different_basis_are_refused():
    """Silently projecting the wrong shape would be a converted model that runs
    and computes something else."""
    weights = np.zeros((3, 7, 16))
    with pytest.raises(ProjectionError, match="trained against a different basis"):
        project_weights(weights, IRREPS_IN, 3, "0e")


def test_there_is_no_way_back():
    """Reduced to full is under-determined and is not offered.

    Recovering a full-basis weight vector means choosing one of infinitely many
    gauge representatives. Exporting for an external reimplementation is a
    separate problem with its own convention to agree on.
    """
    import fm00_projection

    assert not [name for name in dir(fm00_projection) if "reduced_to_full" in name]
    assert set(fm00_projection.__all__) == {
        "ProjectionError",
        "as_path_first",
        "project_weights",
        "projection_matrix",
    }
