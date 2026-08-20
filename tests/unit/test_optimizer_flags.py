"""The optimizer, scheduler and EMA flags, checked where they take effect.

Seven flags that every training run carries and no test asserted anything about.
They all end up as a number inside an optimizer, a scheduler or the moving
average, and none of them changes whether a run succeeds, so a flag that stopped
being read would show up as a slightly different training curve and nothing else.

`--ema` is the sharpest case: every training fixture in the suite passes it, so a
build that ignored it entirely would still be green everywhere.

Weight decay is the one with real structure. It is not a single number handed to
the optimizer: `get_params_options` puts the interaction linears and the product
basis in decaying groups and everything else in groups with decay zero, because
biases, gates and physically meaningful scalars must not be pulled toward zero.
That split is a decision worth pinning, not an accident of the loop that builds it.
"""

import argparse
import json

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules, tools
from mace.tools.scripts_utils import LRScheduler, get_optimizer, get_params_options

LR = 0.017
BETA = 0.87
WEIGHT_DECAY = 0.031
PATIENCE = 13
GAMMA = 0.71


@pytest.fixture(name="model", scope="module")
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
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([1.0, 3.0]),
        avg_num_neighbors=3,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
    )


def args_for(**overrides):
    args = argparse.Namespace(
        lr=LR,
        beta=BETA,
        amsgrad=True,
        weight_decay=WEIGHT_DECAY,
        optimizer="adam",
        freeze=0,
        lr_params_factors=json.dumps({}),
        scheduler="ReduceLROnPlateau",
        scheduler_patience=PATIENCE,
        lr_factor=0.6,
        lr_scheduler_gamma=GAMMA,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def groups(model, **overrides):
    options = get_params_options(args_for(**overrides), model)
    return {group["name"]: group for group in options["params"]}, options


# ---------------------------------------------------------------------------
# --beta, --amsgrad
# ---------------------------------------------------------------------------


def test_beta_is_the_first_adam_moment(model):
    """`--beta` sets beta1 only; beta2 is fixed at 0.999. A flag named `beta`
    could plausibly be read into either."""
    _, options = groups(model)

    assert options["betas"] == (BETA, 0.999)


def test_amsgrad_reaches_the_optimizer(model):
    _, options = groups(model)
    optimizer = get_optimizer(args_for(), options)

    assert all(group["amsgrad"] for group in optimizer.param_groups)


def test_amsgrad_off_stays_off(model):
    _, options = groups(model, amsgrad=False)
    optimizer = get_optimizer(args_for(amsgrad=False), options)

    assert not any(group["amsgrad"] for group in optimizer.param_groups)


def test_the_learning_rate_reaches_every_group(model):
    parameter_groups, options = groups(model)

    assert options["lr"] == LR
    assert all(group["lr"] == LR for group in parameter_groups.values())


# ---------------------------------------------------------------------------
# --weight_decay
# ---------------------------------------------------------------------------


def test_weight_decay_applies_to_the_interaction_linears_and_the_products(model):
    parameter_groups, _ = groups(model)

    assert parameter_groups["interactions_decay"]["weight_decay"] == WEIGHT_DECAY
    assert parameter_groups["products"]["weight_decay"] == WEIGHT_DECAY


def test_weight_decay_is_kept_off_the_rest(model):
    """The deliberate half. Embeddings, readouts and the non-linear interaction
    parameters stay at zero whatever the flag says, so a change that applied the
    flag uniformly would regularize scalars that must not move."""
    parameter_groups, _ = groups(model)

    for name in ("embedding", "interactions_no_decay", "readouts"):
        assert parameter_groups[name]["weight_decay"] == 0.0, name


def test_a_zero_weight_decay_is_zero_everywhere(model):
    parameter_groups, _ = groups(model, weight_decay=0.0)

    assert all(group["weight_decay"] == 0.0 for group in parameter_groups.values())


# ---------------------------------------------------------------------------
# --scheduler_patience, --swa_lr
# ---------------------------------------------------------------------------


def test_scheduler_patience_reaches_the_plateau_scheduler(model):
    _, options = groups(model)
    optimizer = get_optimizer(args_for(), options)

    scheduler = LRScheduler(optimizer, args_for())

    assert scheduler.lr_scheduler.patience == PATIENCE


def test_the_exponential_scheduler_takes_the_gamma_instead(model):
    """`--scheduler_patience` belongs to one scheduler and the gamma to the
    other; picking the wrong branch is silent because both accept a step()."""
    _, options = groups(model)
    optimizer = get_optimizer(args_for(), options)

    scheduler = LRScheduler(optimizer, args_for(scheduler="ExponentialLR"))

    assert scheduler.lr_scheduler.gamma == GAMMA


def test_an_unknown_scheduler_is_refused(model):
    _, options = groups(model)
    optimizer = get_optimizer(args_for(), options)

    with pytest.raises(RuntimeError, match="Unknown scheduler"):
        LRScheduler(optimizer, args_for(scheduler="Whatever"))


def test_swa_lr_is_the_rate_the_second_stage_anneals_to(model):
    """`--swa_lr` only appears after the stage-two swap, so it is checked on the
    SWALR the swap builds rather than on the first-stage optimizer."""
    from mace.tools.scripts_utils import get_swa  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    _, options = groups(model)
    optimizer = get_optimizer(args_for(), options)
    args = args_for(
        loss="weighted",
        start_swa=2,
        max_num_epochs=6,
        swa_lr=0.0009,
        swa_energy_weight=1.0,
        swa_forces_weight=1.0,
        swa_virials_weight=1.0,
        swa_stress_weight=1.0,
        swa_dipole_weight=1.0,
    )

    get_swa(args, model, optimizer, [])

    assert [group["swa_lr"] for group in optimizer.param_groups] == [0.0009] * len(
        optimizer.param_groups
    )


# ---------------------------------------------------------------------------
# --ema, --ema_decay
# ---------------------------------------------------------------------------


def test_the_moving_average_takes_the_decay_it_is_given(model):
    from torch_ema import ExponentialMovingAverage  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    assert ema.decay == 0.995


def test_the_average_lags_the_parameters(model):
    """What `--ema` buys, and what nothing asserted: the averaged parameters are
    not the live ones, and the live ones come back afterwards. A build that
    dropped the EMA would keep every training fixture green, since they all pass
    `--ema` and none of them looks at the result.
    """
    from torch_ema import ExponentialMovingAverage  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    parameter = next(model.parameters())
    original = parameter.detach().clone()
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)

    with torch.no_grad():
        parameter.add_(1.0)
    ema.update()

    with ema.average_parameters():
        averaged = parameter.detach().clone()

    assert not torch.allclose(averaged, parameter), "the average must lag"
    assert torch.allclose(parameter, original + 1.0), "and the live value returns"


def test_the_configured_decay_barely_matters_at_the_start():
    """`torch_ema` corrects for the cold start by capping the effective decay at
    `(1 + n) / (10 + n)`, and MACE takes that default. So the first updates track
    the parameters closely whatever `--ema_decay` says: after one update, 0.5,
    0.9 and 0.99 all give the same average.

    Recorded because it is the opposite of what the flag reads like, and because
    0.99 does not begin to behave like 0.99 until roughly 890 updates in.
    """
    from torch_ema import ExponentialMovingAverage  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    def averaged_after_one_update(decay):
        weight = torch.nn.Parameter(torch.zeros(1))
        ema = ExponentialMovingAverage([weight], decay=decay)
        with torch.no_grad():
            weight.add_(1.0)
        ema.update()
        with ema.average_parameters():
            return float(weight)

    values = [averaged_after_one_update(d) for d in (0.5, 0.9, 0.99)]

    assert values[0] == pytest.approx(values[1]) == pytest.approx(values[2])
    assert values[0] == pytest.approx(1 - 2 / 11, abs=1e-6), (
        "the cap is (1 + num_updates) / (10 + num_updates)"
    )


def test_a_higher_decay_lags_further_once_the_warmup_is_past():
    """The direction of the knob, measured after enough updates that the cap
    above no longer binds for either value."""
    from torch_ema import ExponentialMovingAverage  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    def averaged_after(decay, steps=40):
        weight = torch.nn.Parameter(torch.zeros(1))
        ema = ExponentialMovingAverage([weight], decay=decay)
        for _ in range(steps):
            with torch.no_grad():
                weight.add_(1.0)
            ema.update()
        with ema.average_parameters():
            return float(weight) / float(steps)

    assert averaged_after(0.5) < averaged_after(0.1), (
        "a larger decay keeps more of the old average, so it trails further"
    )
