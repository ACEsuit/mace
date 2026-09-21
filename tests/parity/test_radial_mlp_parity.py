"""The interaction's radial network, against the one a trained anchor carries.

This is the first block whose weights transfer one to one, so it is the first
place the rewrite can be compared to the frozen tree by loading real trained
numbers rather than by constructing something equivalent.

The comparison is done by copying the anchor's own weights in, which is the
only way to separate "computes the same function" from "was initialized the
same way".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from mace_torch.nn.radial_mlp import (
    SECOND_MOMENT_SCALE,
    RadialMLP,
    exact_second_moment_scale,
)

ANCHORS = Path(__file__).resolve().parents[1] / "golden/models"


def load_anchor(name: str):
    """The frozen tree's own pickle, with its environment flag restored after."""
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
def test_the_radial_network_reproduces_the_anchor(fp64, anchor, layer):
    """Same weights in, same numbers out."""
    model = load_anchor(anchor)
    legacy = model.interactions[layer].conv_tp_weights
    layers = [getattr(legacy, f"layer{index}") for index in range(4)]

    mine = RadialMLP(
        num_radial=layers[0].weight.shape[0],
        hidden=[weight.weight.shape[1] for weight in layers[:-1]],
        num_out=layers[-1].weight.shape[1],
    )
    with torch.no_grad():
        for parameter, source in zip(mine.weights, layers, strict=True):
            parameter.copy_(source.weight)

    lengths = torch.randn(37, layers[0].weight.shape[0], dtype=torch.float64)
    theirs = legacy(lengths)
    ours = mine(lengths)

    assert float(theirs.abs().max()) > 1e-3, "the anchor's network output is flat"
    deviation = float((theirs - ours).abs().max())
    assert deviation < 1e-12, (
        f"the radial network differs from the anchor's by {deviation:.3e}"
    )


def test_the_activation_scale_is_the_sampled_one_and_that_is_deliberate():
    """Every trained artifact carries a Monte Carlo estimate, not the exact value.

    e3nn computes the unit-second-moment multiplier for `silu` from a million
    samples under a fixed seed. The exact value by quadrature differs by 0.16
    per cent, which is enormous next to the tolerances these models are
    compared at, so the rewrite carries the estimate. This test exists so that
    the gap stays a recorded decision: if someone replaces the constant with
    the mathematically correct one, every converted model quietly changes.
    """
    exact = exact_second_moment_scale()
    gap = abs(SECOND_MOMENT_SCALE - exact) / exact

    assert 1e-3 < gap < 1e-2, (
        f"the carried constant and the exact one now differ by {gap:.2e}, "
        f"where 1.6e-3 was measured. Either the constant changed or the "
        f"quadrature did."
    )
    assert pytest.approx(1.679176792398942, abs=0.0) == SECOND_MOMENT_SCALE
