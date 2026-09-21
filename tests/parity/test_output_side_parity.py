"""The half of the model that is not the backbone, against a trained anchor.

The readouts, the short-range repulsion, and the constants that turn site
energies into a total. Between these and the block weights, every number a
trained anchor carries has a place in the rewrite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from fm00_convert import energy_constants_to_canonical, linear_weights_to_canonical
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import LinearDescriptor
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.backends.reference.backend import ReferenceLinear
from mace_torch.models.heads import _GatedReadout
from mace_torch.nn.radial import ZBLBasis

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
def test_the_linear_readout_converts_exactly(fp64, anchor):
    legacy = load_anchor(anchor).readouts[0].linear
    irreps_in, irreps_out = str(legacy.irreps_in), str(legacy.irreps_out)

    mine = ReferenceLinear(LinearDescriptor(irreps_in=irreps_in, irreps_out=irreps_out))
    with torch.no_grad():
        mine.weight.copy_(
            torch.tensor(
                linear_weights_to_canonical(legacy, irreps_in, irreps_out),
                dtype=torch.float64,
            )
        )

    features = torch.randn(6, Irreps.parse(irreps_in).dimension, dtype=torch.float64)
    assert float(legacy(features).abs().max()) > 1e-6
    assert float((legacy(features) - mine(features)).abs().max()) < 1e-13


@pytest.mark.parametrize("anchor", ["tiny_scaleshift.model", "tiny_mace.model"])
def test_the_nonlinear_readout_converts_exactly(fp64, anchor):
    """Linear, activation, linear. The activation carries the same fixed
    multiplier the radial network does, and a plain SiLU is off by 1.86."""
    legacy = load_anchor(anchor).readouts[1]
    first_in = str(legacy.linear_1.irreps_in)
    hidden = Irreps.parse(str(legacy.linear_1.irreps_out)).dimension

    mine = _GatedReadout(
        ReferenceBackend(),
        first_in,
        str(legacy.linear_2.irreps_out),
        hidden,
        "float64",
    )
    with torch.no_grad():
        mine.first.weight.copy_(
            torch.tensor(
                linear_weights_to_canonical(
                    legacy.linear_1, first_in, str(legacy.linear_1.irreps_out)
                ),
                dtype=torch.float64,
            )
        )
        mine.second.weight.copy_(
            torch.tensor(
                linear_weights_to_canonical(
                    legacy.linear_2,
                    str(legacy.linear_2.irreps_in),
                    str(legacy.linear_2.irreps_out),
                ),
                dtype=torch.float64,
            )
        )

    features = torch.randn(6, Irreps.parse(first_in).dimension, dtype=torch.float64)
    assert float(legacy(features).abs().max()) > 1e-6
    assert float((legacy(features) - mine(features)).abs().max()) < 1e-13


@pytest.mark.parametrize("anchor", ["tiny_scaleshift.model", "tiny_mace.model"])
def test_the_short_range_repulsion_matches(fp64, anchor):
    """Exactly, because it has no trained weights: only constants and a formula.

    Its envelope order is not a property of the repulsion, it comes from the
    model's cutoff setting, so it is read off the anchor rather than defaulted.
    Defaulting it costs 0.41 eV at a third of an Angstrom.
    """
    legacy = load_anchor(anchor).pair_repulsion_fn
    numbers = torch.tensor([1, 6, 8])
    nodes = 4
    attributes = torch.zeros(nodes, 3, dtype=torch.float64)
    attributes[[0, 1, 2, 3], [2, 0, 0, 1]] = 1.0
    edge_index = torch.tensor([[0, 1, 2, 3, 1], [1, 0, 3, 2, 2]])
    # Short enough that the envelope has not closed yet; at ordinary bond
    # lengths the term is exactly zero and the comparison would say nothing.
    lengths = torch.tensor([[0.35], [0.35], [0.5], [0.5], [0.75]], dtype=torch.float64)

    theirs = legacy(lengths, attributes, edge_index, numbers).detach().flatten()
    mine = (
        ZBLBasis(polynomial_order=int(legacy.p))(
            lengths, numbers[attributes.argmax(dim=1)], edge_index
        )
        .detach()
        .flatten()
    )

    assert float(theirs.abs().max()) > 1.0, "the repulsion is not switched on here"
    assert torch.equal(theirs, mine), f"off by {float((theirs - mine).abs().max()):.3e}"


def test_the_energy_constants_transfer_without_conversion(fp64):
    """Same numbers, same units. The transfer introduces no error at all."""
    model = load_anchor("tiny_scaleshift.model")
    values, scale, shift = energy_constants_to_canonical(model)

    table = model.atomic_energies_fn.atomic_energies.detach().flatten().tolist()
    numbers = [int(z) for z in model.atomic_numbers.tolist()]
    assert list(values) == ["default"]
    assert values["default"] == {z: table[i] for i, z in enumerate(numbers)}
    assert scale == (float(model.scale_shift.scale),)
    assert shift == (float(model.scale_shift.shift),)


def test_a_model_without_a_scale_shift_block_reports_the_identity(fp64):
    """The plain model has none, and the converted one must not invent numbers.

    Its interaction energy is used as it stands, which is a scale of one and a
    shift of zero rather than a missing field.
    """
    model = load_anchor("tiny_mace.model")
    _, scale, shift = energy_constants_to_canonical(model)
    if getattr(model, "scale_shift", None) is None:
        assert scale == (1.0,) and shift == (0.0,)
    else:
        assert scale == (float(model.scale_shift.scale),)
