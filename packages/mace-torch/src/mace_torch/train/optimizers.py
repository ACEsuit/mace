"""The optimizer, its parameter groups, and the learning-rate schedule.

The groups are not cosmetic. Weight decay applies to the tensors that are
linear maps between feature spaces and not to the embedding, the readouts or
the radial networks, and a run that decays all of them trains a different
model. The frozen tree selects them by substring over its own module names
(``mace/tools/scripts_utils.py:930-940``); the same distinction is expressed
here against this tree's names.

Per-group learning-rate factors come from the scheduler section, so freezing a
group is a factor of zero rather than a separate mechanism.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from mace_core.config.training import (
    AdamOptimizer,
    AdamWOptimizer,
    ConstantSchedule,
    ExponentialSchedule,
    LBFGSOptimizer,
    PlateauSchedule,
    SchedulerConfig,
    TrainingConfig,
)
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

__all__ = [
    "GROUPS",
    "GROUP_MARKERS",
    "ONE_BODY_MARKER",
    "POLAR_GROUP_MARKERS",
    "UnsupportedOptimizerError",
    "build_optimizer",
    "build_scheduler",
    "parameter_groups",
]

#: The four groups, in the order they are built, and whether weight decay
#: applies. The names are what a per-group learning-rate factor is keyed by.
GROUPS = ("embedding", "interactions", "products", "readouts")

#: What a parameter's name contains when it belongs to each group. Stated once,
#: because freezing a group and optimizing it have to agree on what it holds.
GROUP_MARKERS: dict[str, str] = {
    "embedding": "node_embedding",
    "interactions": ".interactions.",
    "products": ".products.",
    "readouts": ".readouts.",
}

#: The blocks a charge-aware model adds around its density, one group without
#: weight decay. They mix biases, gates and physically meaningful scalars that
#: must not be pulled towards zero, which is the frozen tree's reason too; it
#: gives each block a group of its own with the same settings.
POLAR_GROUP_MARKERS: tuple[str, ...] = (
    ".source_maps.",
    ".layer_mixer.",
    ".fukui_readout.",
    ".updates.",
    ".electron_energy.",
)

#: The magnetic model's one-body energy of the moment length, a group of its
#: own without weight decay, as the frozen tree has it.
ONE_BODY_MARKER = ".one_body."

#: Which interaction tensors decay: the linear maps between feature spaces. The
#: radial network and the up-projection do not, which is the frozen tree's
#: split expressed against this tree's names.
_DECAYING_INTERACTION_SUFFIXES = (".linear.weight", ".skip.weight")


class UnsupportedOptimizerError(NotImplementedError):
    """An optimizer kind this stage does not build."""


def _named(model: nn.Module, prefix: str) -> Iterator[tuple[str, nn.Parameter]]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and prefix in name:
            yield name, parameter


def parameter_groups(model: nn.Module, config: TrainingConfig) -> list[dict]:
    """The optimizer's groups, with every trainable parameter in exactly one.

    Raises:
        ValueError: If a trainable parameter belongs to no group. One that no
            group claims is never updated, and the symptom is a model that
            trains except for the part nobody looks at.
    """
    factors = config.scheduler.group_factors
    claimed: set[str] = set()
    groups: list[dict] = []

    def add(name: str, items: list[tuple[str, nn.Parameter]], decay: float) -> None:
        claimed.update(parameter for parameter, _ in items)
        groups.append(
            {
                "name": name,
                "params": [parameter for _, parameter in items],
                "weight_decay": decay,
                "lr": factors.get(name, 1.0) * config.lr,
            }
        )

    interactions = list(_named(model, GROUP_MARKERS["interactions"]))
    add("embedding", list(_named(model, GROUP_MARKERS["embedding"])), 0.0)
    add(
        "interactions",
        [
            item
            for item in interactions
            if item[0].endswith(_DECAYING_INTERACTION_SUFFIXES)
        ],
        config.weight_decay,
    )
    groups[-1]["name"] = "interactions"
    add(
        "interactions_no_decay",
        [
            item
            for item in interactions
            if not item[0].endswith(_DECAYING_INTERACTION_SUFFIXES)
        ],
        0.0,
    )
    groups[-1]["lr"] = factors.get("interactions", 1.0) * config.lr
    add("products", list(_named(model, GROUP_MARKERS["products"])), config.weight_decay)
    add("readouts", list(_named(model, GROUP_MARKERS["readouts"])), 0.0)
    add(
        "polar",
        [
            item
            for marker in POLAR_GROUP_MARKERS
            for item in _named(model, marker)
            if item[0] not in claimed
        ],
        0.0,
    )
    add("one_body", list(_named(model, ONE_BODY_MARKER)), 0.0)

    unclaimed = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name not in claimed
    )
    if unclaimed:
        raise ValueError(
            f"no parameter group claims {unclaimed}, so those weights would "
            f"never be updated. Add them to a group, or make them buffers if "
            f"they are not meant to train."
        )
    return [group for group in groups if group["params"]]


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """The optimizer the configuration names.

    Raises:
        UnsupportedOptimizerError: For a kind another ticket owns, naming it
            rather than substituting one that steps differently.
    """
    groups = parameter_groups(model, config)
    settings = config.optimizer
    # Matched on the type rather than on the tag: the settings each kind
    # carries are its own, and reading `beta` off the tag alone is only correct
    # by coincidence of which kinds happen to have one.
    if isinstance(settings, AdamOptimizer):
        return torch.optim.Adam(
            groups,
            lr=config.lr,
            betas=(settings.beta, 0.999),
            amsgrad=settings.amsgrad,
        )
    if isinstance(settings, AdamWOptimizer):
        return torch.optim.AdamW(
            groups,
            lr=config.lr,
            betas=(settings.beta, 0.999),
            amsgrad=settings.amsgrad,
        )
    if isinstance(settings, LBFGSOptimizer):
        return _lbfgs(groups, config, settings)
    raise UnsupportedOptimizerError(
        f"the {settings.kind!r} optimizer is not built here. 'adam', 'adamw' "
        f"and 'lbfgs' are; the schedule-free regime is its own work."
    )


def _lbfgs(
    groups: list[dict], config: TrainingConfig, settings: LBFGSOptimizer
) -> Optimizer:
    """L-BFGS over every parameter that trains, as one group.

    One group because torch's L-BFGS takes one: it treats the parameters as a
    single flat vector with a single step length. So a frozen group, whose
    factor is zero, is left out, and a factor between zero and one has nothing
    to scale. No weight decay, as in the frozen tree, which builds L-BFGS over
    the bare parameters.

    Raises:
        ValueError: On a learning-rate factor other than zero or one, naming
            the group.
    """
    factors = config.scheduler.group_factors
    partial = sorted(
        name for name, factor in factors.items() if factor not in (0.0, 1.0)
    )
    if partial:
        raise ValueError(
            f"training.scheduler.group_factors scales {partial}, and L-BFGS "
            f"steps every parameter with one step length. Use a factor of zero "
            f"to freeze a group, or one to train it."
        )
    parameters = [
        parameter
        for group in groups
        if factors.get(group["name"].removesuffix("_no_decay"), 1.0) != 0.0
        for parameter in group["params"]
    ]
    return torch.optim.LBFGS(
        parameters,
        lr=settings.lr,
        history_size=settings.history_size,
        max_iter=settings.max_iter,
        line_search_fn=settings.line_search_fn,
    )


def build_scheduler(
    optimizer: Optimizer, config: SchedulerConfig
) -> LRScheduler | None:
    """The learning-rate schedule, or ``None`` for a constant one."""
    schedule = config.kind
    if isinstance(schedule, ConstantSchedule):
        return None
    if isinstance(schedule, ExponentialSchedule):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=schedule.gamma)
    if isinstance(schedule, PlateauSchedule):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=schedule.factor, patience=schedule.patience
        )
    raise UnsupportedOptimizerError(
        f"{schedule.kind!r} is not a schedule this builds. A kind that falls "
        f"through here would otherwise train at a constant rate while the "
        f"configuration asked for something else."
    )
