"""Where a run puts its output, and how it is driven.

Nothing here reaches the model. A field that changes what is computed is not a
runtime field, however much it looks like one, and two of legacy's live in this
flag group without belonging to it: ``--default_dtype`` is a precision choice
and the three acceleration flags are a backend choice, both resolved once at
model build time and both owned by their own tickets.

**Six directory flags become one.** Legacy takes ``--work_dir``, ``--log_dir``,
``--model_dir``, ``--checkpoints_dir``, ``--results_dir`` and
``--downloads_dir``, each defaulting to ``.``, so the default run scatters six
kinds of output into the working directory and a user who sets only some of
them gets a half-organised tree. Here there is one root and a fixed layout
under it, so a run is one directory and moving it moves everything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator

from mace_core.config.section import FrozenSection
from mace_core.config.tracking import WandbConfig
from mace_core.tables import TABLE_TYPES

__all__ = ["LOG_LEVELS", "WORK_DIR_LAYOUT", "RuntimeConfig"]

#: The layout under ``work_dir``. A convention rather than six fields: the
#: names are stated once here so a reader of a run directory and the code that
#: wrote it cannot disagree.
WORK_DIR_LAYOUT: dict[str, str] = {
    "logs": "logs",
    "models": "models",
    "checkpoints": "checkpoints",
    "results": "results",
    "downloads": "downloads",
}

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

#: The accepted levels, for a caller that enumerates them.
LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")


class RuntimeConfig(FrozenSection):
    """Run identity, output layout, logging and the checkpoint retention.

    Args:
        name: The run's name. It labels the artifacts under ``work_dir``, so
            two runs sharing a work directory and a name overwrite each other.
        seed: The seed every generator in the run is derived from.
        work_dir: The root of this run's output. The layout under it is
            :data:`WORK_DIR_LAYOUT`.
        device: Where the run executes. A string rather than a framework
            object, since this package imports no framework.
        distributed: Whether the run is one process of a multi-process job.
        launcher: How a distributed run was launched, when that has to be
            recorded to reconstruct the environment.
        log_level: Verbosity of the run's own logging.
        error_table: Which error table the evaluation prints. Validated here
            rather than where it is rendered, because the render happens after
            the training it reports on.
        plot: Whether to write training curves.
        plot_frequency: Epochs between plots. ``0`` means only at the end,
            which is what makes a separate "plot at all" flag worth keeping:
            the two answer different questions.
        wandb: Reporting to an experiment tracker, off by default.
        restart_latest: Resume from the newest checkpoint in the run directory.
        keep_checkpoints: Keep every checkpoint an improving epoch wrote,
            rather than only the newest. Off, a new checkpoint replaces the
            one before it, as legacy does by default.
        save_all_checkpoints: Keep a checkpoint of every epoch, improving or
            not.
    """

    name: str = "mace"
    seed: int = 123
    work_dir: Path = Path()
    device: str = "cpu"
    distributed: bool = False
    launcher: str | None = None
    log_level: LogLevel = "INFO"
    error_table: str = "PerAtomRMSE"
    plot: bool = False
    plot_frequency: int = 0
    wandb: WandbConfig = WandbConfig()
    restart_latest: bool = False
    keep_checkpoints: bool = False
    save_all_checkpoints: bool = False

    @field_validator("error_table")
    @classmethod
    def _known_error_table(cls, value: str) -> str:
        if value not in TABLE_TYPES:
            raise ValueError(
                f"{value!r} is not an error table. They are {sorted(TABLE_TYPES)}."
            )
        return value

    def directory(self, which: str) -> Path:
        """One of the run's output directories, by its name in the layout.

        Args:
            which: A key of :data:`WORK_DIR_LAYOUT`.

        Returns:
            The path under ``work_dir``. Not created here: a config object
            that touched the filesystem could not be validated without side
            effects.

        Raises:
            KeyError: Naming the directories there are, since a typo would
                otherwise silently create one.
        """
        if which not in WORK_DIR_LAYOUT:
            raise KeyError(
                f"{which!r} is not part of the run layout. The directories are "
                f"{sorted(WORK_DIR_LAYOUT)}."
            )
        return self.work_dir / WORK_DIR_LAYOUT[which]
