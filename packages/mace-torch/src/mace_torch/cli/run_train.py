"""Training, from a command line.

The whole of it: read a configuration, run three stages, write what came out.
The frozen tree's equivalent is one function of about 1,150 lines that reads 74
attributes off an argparse namespace and assigns back to it at 43 sites, so
that what a run does depends on the order the assignments happen in.

Nothing is decided here. A default belongs to the schema, a refusal belongs to
the stage that can explain it, and this module is what connects them, which is
why it is short enough to read in one go.

The surface is the one ``mace train`` has: a configuration file and the
explicit flags of :data:`mace_torch.cli.commands.TRAIN_FLAGS`. This module is
what the legacy script name ``mace_run_train`` reaches on the v1 engine. The
faithful port of the legacy flags is its own work, and the shim that maps a
legacy namespace onto this configuration already exists beside the schema.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch.distributed as dist
from mace_core.config import read_config_file
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import DEFAULT_CATALOGUE
from mace_core.stages import TrainedModel

from mace_torch.finetune.stages import build
from mace_torch.train import (
    latest_run_checkpoint,
    run_train_stage,
    setup_logging,
)
from mace_torch.train.ddp import init_distributed

__all__ = ["NOT_MIGRATED", "main", "parse", "run", "train"]

#: What is said about a capability the v1 stack does not carry yet. The same
#: words the launcher uses about a whole script, because a caller that has to
#: skip cannot tell the two apart and should not have to.
NOT_MIGRATED = "not yet available on v1 engine"


logger = logging.getLogger(__name__)


def parse(argv: Sequence[str] | None = None) -> ResolvedConfig:
    """The configuration a command line asks for.

    Args:
        argv: The arguments after the program name. ``None`` reads
            ``sys.argv``.

    Returns:
        The resolved configuration: the file, with the flags given written
        into it, over the schema's own defaults.
    """
    from mace_torch.cli.commands import TRAIN

    parser = argparse.ArgumentParser(
        prog="mace_run_train --engine v1",
        description="Train a model from a configuration file.",
    )
    TRAIN.add_to(parser)
    known, rest = parser.parse_known_args(argv)
    options = [
        argument.split("=", 1)[0] for argument in rest if argument.startswith("--")
    ]
    dotted = [option for option in options if "." in option]
    if dotted:
        raise SystemExit(
            f"{', '.join(sorted(dotted))}: this engine takes no dotted "
            f"overrides. Set the value in the configuration file; the flags it "
            f"takes are {', '.join(flag.option for flag in TRAIN.flags)}."
        )
    if rest:
        # A flag this command does not declare is a legacy one. Saying so is
        # the difference between a suite that skips and one that fails: the
        # port of those flags is its own work, and until it lands a run that
        # asks for them has not been tried on this engine rather than tried
        # and broken.
        raise SystemExit(
            f"the legacy command-line flags are {NOT_MIGRATED}: "
            f"{', '.join(sorted(options) or rest)}. This engine takes a "
            f"configuration file and the flags "
            f"{', '.join(flag.option for flag in TRAIN.flags)}. Run it with "
            f"--engine legacy, or write the run as a configuration."
        )
    if known.config is not None:
        legacy_keys = set(read_config_file(known.config)) - set(
            ResolvedConfig.model_fields
        )
        if legacy_keys:
            # The same distinction for a file: a v1 configuration holds only
            # sections, so a top-level key that is not one is a legacy flag
            # written as YAML, which the flag port will read.
            raise SystemExit(
                f"a legacy YAML configuration is {NOT_MIGRATED}: {known.config} "
                f"sets {', '.join(sorted(legacy_keys))} at the top level, where "
                f"a v1 configuration has only the sections "
                f"{', '.join(ResolvedConfig.model_fields)}. Run it with "
                f"--engine legacy."
            )
    return cast(ResolvedConfig, TRAIN.configuration(known))


def run(config: ResolvedConfig) -> TrainedModel:
    """Read the data, build the model, train it.

    The three stages and the objects between them. A stage takes the previous
    stage's object and nothing else, so there is no state here for them to
    disagree about. A fine-tune reads its foundation model first, and hands
    the data stage what it needs of it and the model stage the rest.
    """
    processes = init_distributed(
        config.runtime.distributed, config.runtime.launcher, config.runtime.device
    )
    # The same steps a fine-tune run step by step goes through, so the two
    # cannot come to differ.
    built = build(config, DEFAULT_CATALOGUE, context=processes)
    checkpoint_path = Path(config.runtime.work_dir) / config.runtime.name
    return run_train_stage(
        config,
        built,
        device=processes.device,
        checkpoint_path=checkpoint_path,
        resume=_resumes(config, checkpoint_path),
        distributed=processes,
    )


def _resumes(config: ResolvedConfig, checkpoint_path: Path) -> bool:
    """Whether the run continues one already written.

    The frozen tree's contract for ``restart_latest``: the newest checkpoint
    for the run's name if there is one, and a fresh run that says so if there
    is none.
    """
    if not config.runtime.restart_latest:
        return False
    latest = latest_run_checkpoint(checkpoint_path.parent, checkpoint_path.name)
    if latest is None:
        logger.warning(
            "restart_latest is set and there is no run checkpoint for %r in %s, "
            "so the run starts at epoch 0",
            checkpoint_path.name,
            checkpoint_path.parent,
        )
    return latest is not None


def train(config: ResolvedConfig) -> int:
    """Train from a resolved configuration and report the outcome. Returns a
    process exit status."""
    setup_logging(config.runtime)
    try:
        trained = run(config)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if trained.best is None:
        logging.info(
            "Trained %d epoch(s); nothing was evaluated.", len(trained.history)
        )
    else:
        logging.info(
            "Best epoch %d, validation loss %.6g, written to %s",
            trained.best.epoch,
            trained.best.valid_loss,
            trained.checkpoint_path,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """The console entry point. Returns a process exit status."""
    return train(parse(argv))
