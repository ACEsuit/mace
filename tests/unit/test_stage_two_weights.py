"""The stage-two loss weights are the ones the flags asked for.

`--swa_energy_weight` and its siblings only take effect after the stage-two swap,
which is why no end-to-end run pinned them: a short training either never swaps or
swaps into a loss nobody inspects. `get_swa` is where the flags become a loss, so
that is where they are checked.

The weights are deliberately not the defaults and not equal to each other, so a
loss that took the first-stage weight, or read the wrong flag, produces a
different number here.
"""

import argparse

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules, tools
from mace.tools.scripts_utils import get_swa

ENERGY, FORCES, VIRIALS, STRESS = 7.0, 11.0, 13.0, 17.0


@pytest.fixture(name="model")
def fixture_model():
    table = tools.AtomicNumberTable([1, 8])
    torch.manual_seed(0)
    return modules.MACE(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([1.0, 3.0]),
        avg_num_neighbors=3,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
    )


def settings(**overrides):
    args = argparse.Namespace(
        loss="weighted",
        start_swa=3,
        max_num_epochs=10,
        swa_lr=1e-3,
        swa_energy_weight=ENERGY,
        swa_forces_weight=FORCES,
        swa_virials_weight=VIRIALS,
        swa_stress_weight=STRESS,
        swa_dipole_weight=1.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def build(model, **overrides):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    swa, swas = get_swa(settings(**overrides), model, optimizer, [])
    return swa, swas


def test_the_energy_and_forces_weights_reach_the_stage_two_loss(model):
    swa, _ = build(model)

    assert swa.loss_fn.energy_weight == pytest.approx(ENERGY)
    assert swa.loss_fn.forces_weight == pytest.approx(FORCES)


def test_the_virials_weight_reaches_the_virials_loss(model):
    swa, _ = build(model, loss="virials")

    assert swa.loss_fn.energy_weight == pytest.approx(ENERGY)
    assert swa.loss_fn.virials_weight == pytest.approx(VIRIALS)


def test_the_stress_weight_reaches_the_stress_loss(model):
    swa, _ = build(model, loss="stress")

    assert swa.loss_fn.energy_weight == pytest.approx(ENERGY)
    assert swa.loss_fn.stress_weight == pytest.approx(STRESS)


def test_the_stage_two_weights_are_not_the_first_stage_ones(model):
    """The mistake this guards: reading `args.energy_weight` instead of
    `args.swa_energy_weight`. Those names differ by four characters."""
    args = settings()
    args.energy_weight = 1.0
    args.forces_weight = 100.0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    swa, _ = get_swa(args, model, optimizer, [])

    assert swa.loss_fn.energy_weight != pytest.approx(args.energy_weight)
    assert swa.loss_fn.forces_weight != pytest.approx(args.forces_weight)


def test_the_swap_epoch_is_the_one_requested(model):
    swa, _ = build(model, start_swa=4)

    assert swa.start == 4


def test_an_unset_swap_epoch_lands_three_quarters_in(model):
    """The documented default: `max(1, max_num_epochs // 4 * 3)`."""
    swa, _ = build(model, start_swa=None, max_num_epochs=8)

    assert swa.start == 6


def test_a_swap_after_the_last_epoch_is_refused(model):
    """`swas` is how the caller learns stage two will not happen. A swap epoch
    beyond the run is warned about and disabled rather than silently accepted."""
    _, swas = build(model, start_swa=10, max_num_epochs=10)

    assert swas[-1] is False


def test_stage_two_is_refused_with_a_forces_only_loss(model):
    with pytest.raises(ValueError, match="forces only"):
        build(model, loss="forces_only")
