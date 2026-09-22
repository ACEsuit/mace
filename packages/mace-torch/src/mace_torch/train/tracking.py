"""Reporting a run to an experiment tracker, or to nothing.

Two implementations of one interface, and the loop holds whichever it was
given. There is no ``if config.wandb.enabled`` in the loop: a run without
tracking holds the one that does nothing, which is the same shape as a run
with it and cannot drift from it.

**wandb is never imported unless it is enabled.** The import is inside the
tracker's constructor, so a run with tracking off does not touch the package
and does not need it installed. Legacy has the same property by the same means
and states it nowhere, which is why it is stated here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch.distributed as dist
from mace_core.config.resolved import ResolvedConfig

__all__ = [
    "NullTracker",
    "Tracker",
    "WandbTracker",
    "epoch_values",
    "open_tracker",
]

logger = logging.getLogger(__name__)


@runtime_checkable
class Tracker(Protocol):
    """Where a run's numbers go besides its log."""

    def log(self, values: Mapping[str, float], step: int) -> None:
        """Record one epoch's numbers."""

    def summary(self, values: Mapping[str, float]) -> None:
        """Record the run's final numbers, which belong to no epoch."""

    def finish(self) -> None:
        """Close the run, so a tracker that buffers flushes it."""


class NullTracker:
    """The tracker a run without one holds. Every method does nothing."""

    def log(self, values: Mapping[str, float], step: int) -> None:
        return

    def summary(self, values: Mapping[str, float]) -> None:
        return

    def finish(self) -> None:
        return


class WandbTracker:
    """Weights and Biases.

    The whole resolved configuration is logged as the run's hyperparameters,
    because it is one validated object that round-trips through JSON. Legacy
    copies eleven named fields out of an argparse namespace instead, and a run
    tuned through a twelfth records nothing about it.

    Raises:
        ImportError: Naming the extra to install, since the alternative is a
            traceback from inside a constructor that says only the module name.
    """

    def __init__(self, config: ResolvedConfig) -> None:
        try:
            # Unresolvable to the type checker on purpose: the lint job
            # installs the package's required dependencies and this is not one
            # of them, which is the property being asserted.
            import wandb  # ty: ignore[unresolved-import]
        except ImportError as missing:
            raise ImportError(
                "tracking is enabled and wandb is not installed. Install it "
                "with `pip install mace-torch[wandb]`, or set "
                "`runtime.wandb.enabled = false`."
            ) from missing
        settings = config.runtime.wandb
        self._wandb = wandb
        self._run = wandb.init(
            project=settings.project,
            entity=settings.entity,
            name=settings.name or config.runtime.name,
            dir=str(settings.directory) if settings.directory else None,
            config=config.model_dump(mode="json"),
        )

    def log(self, values: Mapping[str, float], step: int) -> None:
        self._wandb.log(dict(values), step=step)

    def summary(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            self._run.summary[name] = value

    def finish(self) -> None:
        self._run.finish()


def open_tracker(config: ResolvedConfig) -> Tracker:
    """The tracker this configuration asks for.

    A rank other than the first gets the null one whatever the configuration
    says: several processes reporting one run under one name interleave their
    epochs, and the result is a curve nobody can read.
    """
    if not config.runtime.wandb.enabled:
        return NullTracker()
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return NullTracker()
    return WandbTracker(config)


def epoch_values(
    per_head: Mapping[str, Mapping[str, float]], prefix: str = "valid"
) -> dict[str, float]:
    """One epoch's per-head metrics, flattened into the keys a tracker takes.

    ``valid_<head>_<metric>``, which is the row name the error table uses with
    the metric appended, so a chart and a table row are about the same thing.
    """
    return {
        f"{prefix}_{head}_{name}": value
        for head, metrics in per_head.items()
        for name, value in metrics.items()
    }
