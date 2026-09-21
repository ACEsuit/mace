"""The equivariant linear, against the ones a trained anchor carries.

Four of them, covering the case where one input term feeds an output and the
case where three do, since the normalization depends on that count and getting
it wrong is a constant factor rather than a visible break.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from fm00_convert import linear_weights_to_canonical
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import LinearDescriptor
from mace_torch.backends.reference.backend import ReferenceLinear

ANCHORS = Path(__file__).resolve().parents[1] / "golden/models"


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
@pytest.mark.parametrize("which", ["linear_up", "linear"])
def test_a_trained_linear_converts_exactly(fp64, anchor, layer, which):
    model = load_anchor(anchor)
    legacy = getattr(model.interactions[layer], which)
    irreps_in, irreps_out = str(legacy.irreps_in), str(legacy.irreps_out)

    mine = ReferenceLinear(LinearDescriptor(irreps_in=irreps_in, irreps_out=irreps_out))
    with torch.no_grad():
        mine.weight.copy_(
            torch.tensor(
                linear_weights_to_canonical(legacy, irreps_in, irreps_out),
                dtype=torch.float64,
            )
        )

    features = torch.randn(9, Irreps.parse(irreps_in).dimension, dtype=torch.float64)
    theirs = legacy(features).detach()
    ours = mine(features).detach()

    assert float(theirs.abs().max()) > 1e-3, "the trained linear maps to nothing"
    deviation = float((theirs - ours).abs().max())
    assert deviation < 1e-13, (
        f"{which} of layer {layer} differs by {deviation:.3e} after conversion"
    )


def test_the_normalization_depends_on_the_fan_in(fp64):
    """Three different factors inside one module, which is why it is folded in.

    The second interaction's linear reads seven path blocks into three output
    irreps: two feed the scalars, three feed the vectors, two feed the rank
    two. Each group carries its own factor.
    """
    model = load_anchor("tiny_scaleshift.model")
    weights = sorted(
        {
            round(float(instruction.path_weight), 6)
            for instruction in model.interactions[1].linear.instructions
        }
    )
    assert weights == [
        pytest.approx(48**-0.5, abs=1e-6),
        pytest.approx(32**-0.5, abs=1e-6),
    ], f"the fan-in factors are {weights}"
