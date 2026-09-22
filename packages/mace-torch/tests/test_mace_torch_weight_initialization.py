"""A model built to be trained needs weights that are not all zero.

This is the defect the initialisation exists to remove, and it is invisible to
every test that loads a checkpoint or overwrites the parameters first: an op
that allocates zeros computes the right answer for the weights it was given,
and the model built from it produces a constant and has no gradient anywhere.

So the first test here is the one that would have caught it, and it is
deliberately about the built op rather than about the drawing.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from mace_core.clebsch_gordan.irreps import Irrep
from mace_core.kernels.canonical import (
    fully_connected_tp_weight_scale,
    linear_weight_scale,
)
from mace_core.kernels.descriptors import (
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.protocol import INTERNAL_WEIGHT_OPS, InternalWeights
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.kernels import initialize_model_weights, op_seed

BACKEND = ReferenceBackend()
LINEAR = LinearDescriptor(irreps_in="4x0e+4x1o", irreps_out="4x0e+4x1o", has_bias=True)
SKIP = FullyConnectedTPDescriptor(
    irreps_in1="4x0e+4x1o", irreps_in2="2x0e", irreps_out="4x0e+4x1o"
)
CONTRACTION = SymmetricContractionDescriptor(
    irreps_in="0e+1o", irreps_out="0e", correlation=2, num_elements=2, num_features=4
)


def built() -> dict[str, Any]:
    return {
        "linear": BACKEND.make_linear(LINEAR),
        "fully_connected_tp": BACKEND.make_fully_connected_tp(SKIP),
        "symmetric_contraction": BACKEND.make_symmetric_contraction(CONTRACTION),
    }


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_every_op_that_holds_weights_can_draw_a_fresh_set():
    """The three named in the contract, and no fourth one quietly added."""
    for name, op in built().items():
        assert name in INTERNAL_WEIGHT_OPS
        assert isinstance(op, InternalWeights), name


def test_an_op_starts_at_zero_and_a_draw_moves_it():
    """Zeros are the right allocation and the wrong thing to train from."""
    for name, op in built().items():
        before = [p.detach().clone() for p in op.parameters()]
        assert all(float(p.abs().max()) == 0.0 for p in before), name
        op.initialize_weights(7)
        after = list(op.parameters())
        assert any(float(p.detach().abs().max()) > 0.0 for p in after), name


def test_a_draw_is_reproducible_from_its_seed():
    """A recorded seed has to rebuild the same model or it records nothing."""
    first, second = BACKEND.make_linear(LINEAR), BACKEND.make_linear(LINEAR)
    first.initialize_weights(11)
    second.initialize_weights(11)
    assert torch.equal(first.weight, second.weight)


def test_two_seeds_give_two_models():
    first, second = BACKEND.make_linear(LINEAR), BACKEND.make_linear(LINEAR)
    first.initialize_weights(11)
    second.initialize_weights(12)
    assert not torch.equal(first.weight, second.weight)


def test_the_draw_does_not_touch_the_global_generator():
    """A model whose weights depend on how many numbers were drawn before it
    is not reproducible from its seed, only from its whole program."""
    torch.manual_seed(0)
    expected = torch.randn(3)
    torch.manual_seed(0)
    BACKEND.make_linear(LINEAR).initialize_weights(5)
    assert torch.equal(torch.randn(3), expected)


# ---------------------------------------------------------------------------
# The scales, which are part of the checkpoint format
# ---------------------------------------------------------------------------


def test_a_linear_weight_is_scaled_by_its_fan_in():
    """The canonical layout carries the normalization in the number, so a draw
    that ignored it would train at a different effective rate than a converted
    checkpoint of the same architecture."""
    op = BACKEND.make_linear(LinearDescriptor(irreps_in="64x0e", irreps_out="8x0e"))
    op.initialize_weights(3)
    assert float(op.weight.detach().std()) == pytest.approx(64.0**-0.5, rel=0.1)


def test_the_two_scales_are_the_ones_the_conversion_folds_in():
    """The fan-in of a linear counts the inputs sharing the output's irrep; the
    skip connection's counts the element attributes as well, because it sees
    every one of them."""
    assert linear_weight_scale("16x0e+4x1o", Irrep(0, 1)) == pytest.approx(16.0**-0.5)
    assert linear_weight_scale("16x0e+4x1o", Irrep(1, -1)) == pytest.approx(4.0**-0.5)
    assert fully_connected_tp_weight_scale(8, 3) == pytest.approx(24.0**-0.5)


def test_an_irrep_nothing_feeds_has_no_weights_and_no_scale():
    """One rather than a division by zero: the map cannot produce that output
    at all, so there is no weight for the scale to apply to."""
    assert linear_weight_scale("16x0e", Irrep(2, 1)) == 1.0


def test_a_bias_starts_at_zero():
    """It is an offset on the energy, and a drawn one shifts the model before
    it has seen a structure."""
    op = BACKEND.make_linear(LINEAR)
    op.initialize_weights(4)
    assert float(op.bias.detach().abs().max()) == 0.0


def test_the_symmetric_contraction_is_the_unscaled_one():
    """The one weighted op the canonical layout applies no factor to."""
    op = BACKEND.make_symmetric_contraction(
        SymmetricContractionDescriptor(
            irreps_in="0e+1o",
            irreps_out="0e",
            correlation=2,
            num_elements=4,
            num_features=64,
        )
    )
    op.initialize_weights(9)
    assert float(op.weights[0].detach().std()) == pytest.approx(1.0, rel=0.2)


# ---------------------------------------------------------------------------
# The walk over a model
# ---------------------------------------------------------------------------


def test_the_walk_reaches_every_op_and_names_them():
    model = torch.nn.ModuleDict(built())
    paths = initialize_model_weights(model, seed=1)
    assert set(paths) == set(built())
    assert all(
        float(parameter.detach().abs().max()) > 0.0
        for name, parameter in model.named_parameters()
        if not name.endswith("bias")
    )


def test_two_ops_of_the_same_shape_do_not_get_the_same_weights():
    """One seed for the whole model would give them one draw, and a layer that
    is a copy of another is not a model, it is a symmetry that never breaks."""
    first, second = BACKEND.make_linear(LINEAR), BACKEND.make_linear(LINEAR)
    initialize_model_weights(torch.nn.ModuleList([first, second]), seed=2)
    assert not torch.equal(first.weight, second.weight)


def test_the_derived_seed_is_stable_across_processes():
    """Committed values: `hash` is salted per process, so a model built today
    and rebuilt tomorrow would differ."""
    assert op_seed(0, "backbone.interactions.0.body.linear") == 2033526461
    assert op_seed(1, "backbone.interactions.0.body.linear") == 2033526462


def test_a_model_with_no_weighted_ops_says_so():
    assert initialize_model_weights(torch.nn.Linear(2, 2), seed=0) == ()


# ---------------------------------------------------------------------------
# After the model has been moved
# ---------------------------------------------------------------------------

#: A non-host device to move to, if this machine has one. The failure is about
#: two tensors on different devices, so any second device shows it.
OTHER_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else None
)

#: The same three ops at float32. Metal has no float64 at all, so a float64
#: module cannot even be moved there; the device question is the same at either
#: precision.
NARROW_LINEAR = LinearDescriptor(
    irreps_in="4x0e+4x1o",
    irreps_out="4x0e+4x1o",
    has_bias=True,
    precision="float32",
)


def built_narrow() -> dict[str, Any]:
    return {
        "linear": BACKEND.make_linear(NARROW_LINEAR),
        "fully_connected_tp": BACKEND.make_fully_connected_tp(
            FullyConnectedTPDescriptor(
                irreps_in1="4x0e+4x1o",
                irreps_in2="2x0e",
                irreps_out="4x0e+4x1o",
                precision="float32",
            )
        ),
        "symmetric_contraction": BACKEND.make_symmetric_contraction(
            SymmetricContractionDescriptor(
                irreps_in="0e+1o",
                irreps_out="0e",
                correlation=2,
                num_elements=2,
                num_features=4,
                precision="float32",
            )
        ),
    }


@pytest.mark.skipif(OTHER_DEVICE is None, reason="this machine has one device")
def test_a_moved_model_still_draws_its_weights():
    """A model is built, moved, and then initialised, in that order.

    The draw happens on the host, because a device generator seeded the same
    way gives different numbers and a recorded seed has to rebuild the same
    model anywhere. So the drawn tensor has to be moved to where the parameter
    is. The contraction is the one that would hide this, since it applies no
    scale and its copy crosses devices without complaint.
    """
    for name, op in built_narrow().items():
        op.to(OTHER_DEVICE).initialize_weights(3)
        for parameter in op.parameters():
            assert parameter.device.type == OTHER_DEVICE, name


@pytest.mark.skipif(OTHER_DEVICE is None, reason="this machine has one device")
def test_the_weights_do_not_depend_on_where_the_model_was_built():
    """Same seed, two devices, same numbers. Otherwise a run is reproducible
    only on the machine that produced it."""
    host = BACKEND.make_linear(NARROW_LINEAR)
    host.initialize_weights(11)
    moved = BACKEND.make_linear(NARROW_LINEAR).to(OTHER_DEVICE)
    moved.initialize_weights(11)
    assert torch.allclose(host.weight, moved.weight.cpu())
