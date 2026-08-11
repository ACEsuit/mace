"""The one-training-step gradient golden.

This closes the training-numerics gap: the loss-decrease smoke test and the
committed final-error table both stay green while `d(loss)/d(theta)` is wrong
by an amount comparable to initialisation noise, because they only ever look
at where training ended up. A committed single-step gradient looks at the
derivative itself.

It characterizes the current implementation rather than comparing two of
them: nothing outside this stack is involved. A rewrite of the training path
has to reproduce these numbers, and a rewrite that changes them has to say
which gradient moved and why.

Read `train_step.py` for the digest design and for why the raw 37,704-element
gradient vector is not what is committed. The two properties that make the
digest trustworthy -- that it moves when a single weight moves by 1e-9, and
that it notices a permutation -- are asserted below rather than argued.

No GPU, no network, no optional dependency, no capability marker.
"""

import pytest
import torch

from tests.golden import harness
from tests.golden.train_step import (
    GRADIENT_REFERENCES,
    _digest,
    deviations,
    snapshot,
    train_step,
)

TOL = harness.FP64_CPU_REFERENCE


@pytest.fixture(name="taken", scope="module")
def fixture_taken():
    """One step per anchor, taken once for the whole module."""
    return {name: snapshot(name) for name in GRADIENT_REFERENCES}


@pytest.mark.parametrize("anchor", sorted(GRADIENT_REFERENCES))
def test_the_step_reproduces_its_committed_gradient(anchor, taken):
    reference = harness.load_reference(
        harness.REFERENCES_DIR / GRADIENT_REFERENCES[anchor]
    )
    problems = deviations(taken[anchor], reference, TOL)
    assert not problems, "\n  ".join([f"{anchor}:"] + problems)


@pytest.mark.parametrize("anchor", sorted(GRADIENT_REFERENCES))
def test_the_reference_records_what_produced_it(anchor):
    reference = harness.load_reference(
        harness.REFERENCES_DIR / GRADIENT_REFERENCES[anchor]
    )
    assert reference["kind"] == "train_step_gradient"
    assert reference["dtype"] == "float64"
    assert reference["device"] == "cpu"
    assert reference["metadata"]["loss"] == "WeightedEnergyForcesLoss"
    assert reference["metadata"]["seed"]
    assert reference["provenance"]["recipe"] == "tests/golden/train_step.py"
    assert reference["provenance"]["tolerance_row"] == TOL.name
    assert reference["n_elements"] > 1000
    assert reference["parameters_without_a_gradient"] == [], (
        "a parameter with no gradient is a parameter this golden does not "
        "pin; it has to be explained before it is committed"
    )


@pytest.mark.parametrize("anchor", sorted(GRADIENT_REFERENCES))
def test_gradcheck_and_gradgradcheck_pass_on_the_tiny_fixtures(anchor, taken):
    """The pass-flags, recomputed here and also committed in the reference.

    `gradgradcheck` is the one that matters: force training backpropagates
    through the force computation itself, so the second derivative is on the
    critical path and a first-order-correct implementation is not enough.
    """
    flags = taken[anchor]["gradcheck"]
    assert set(flags) == {
        "gradcheck_positions",
        "gradgradcheck_positions",
        "gradcheck_strain",
        "gradgradcheck_strain",
    }
    assert all(flags.values()), flags
    committed = harness.load_reference(
        harness.REFERENCES_DIR / GRADIENT_REFERENCES[anchor]
    )["gradcheck"]
    assert committed == flags


@pytest.mark.parametrize("anchor", sorted(GRADIENT_REFERENCES))
def test_the_backward_ran_through_the_second_derivative_path(anchor, taken):
    """An energy-only loss would leave the force term's graph unbuilt.

    The recorded loss is the energy+forces combination, so if the forces
    contribution ever stopped reaching the parameters the loss would still be
    finite and the gradients would still exist -- they would just be a
    different, smaller number. Checked by taking the same step with the
    forces weight removed and requiring the digest to disagree.
    """
    from tests.golden import train_step as module  # noqa: PLC0415

    energy_only = dict(module.LOSS_WEIGHTS)
    energy_only["forces_weight"] = 0.0
    saved = module.LOSS_WEIGHTS
    try:
        module.LOSS_WEIGHTS = energy_only
        without_forces = train_step(anchor)
    finally:
        module.LOSS_WEIGHTS = saved
    assert deviations(without_forces, taken[anchor], TOL)


def test_the_digest_resolves_a_weight_change_six_orders_below_the_weights():
    """Why this golden is worth its bytes, as a measured resolution.

    Perturbing one weight and watching the digest, the response is linear
    with a gain of 0.49: a change of d in a single weight moves the largest
    digest field by about 0.49 * d. Against the 1e-6 reference row that puts
    the detection floor at a weight change of roughly 2e-6, while the weights
    themselves are of order 1 -- so a gradient defect anywhere near the scale
    of initialisation noise is six orders above what this can see, which is
    the claim the Context makes and the one a final-error table cannot.

    The perturbation used here is 1e-5, comfortably above the floor and still
    far below anything an error table would report: it moves the loss by
    1.9e-7 eV.
    """
    anchor = "tiny_scaleshift"
    baseline = train_step(anchor)
    nudged = train_step(
        anchor, perturbation=("node_embedding.linear.weight", 0, 1e-5)
    )
    problems = deviations(nudged, baseline, TOL)
    assert problems, "the digest did not notice a perturbed weight"
    # and it is the gradient that moved, not merely the loss: the loss shift
    # is itself under the row, so a golden that pinned only the loss would
    # have missed this.
    assert abs(nudged["loss"] - baseline["loss"]) < TOL.atol


def test_the_projection_is_what_catches_a_permuted_gradient():
    """The design justification for the fifth number, as a measurement.

    Swapping two elements leaves the sum, the absolute sum and the sum of
    squares bit-identical -- so the first four fields cannot distinguish a
    correct gradient from one whose irreps layout is permuted, which is the
    likeliest way this codebase gets a port wrong. Only the positional
    projection moves.
    """
    gradient = torch.tensor([0.5, -1.25, 3.0, 0.125], dtype=torch.float64)
    permuted = gradient[[1, 0, 2, 3]]
    original, swapped = _digest(gradient), _digest(permuted)
    for field in ("sum", "abs_sum", "sq_sum", "count"):
        assert original[field] == swapped[field], field
    assert original["projection"] != swapped["projection"]
