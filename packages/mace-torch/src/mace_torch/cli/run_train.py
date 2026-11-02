"""Training, from a command line.

The whole of it: read a configuration, run three stages, write what came out.
The frozen tree's equivalent is one function of about 1,150 lines that reads 74
attributes off an argparse namespace and assigns back to it at 43 sites, so
that what a run does depends on the order the assignments happen in.

Nothing is decided here. A default belongs to the schema, a refusal belongs to
the stage that can explain it, and this module is what connects them, which is
why it is short enough to read in one go.

Both ``mace train`` and ``mace_run_train --engine v1`` accept structured
configuration files and explicit flags. Legacy spellings are translated at the
CLI boundary; the stages receive only the agreed V1 configuration.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import torch.distributed as dist
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
    from mace_torch.cli.legacy_flags import add_arguments
    from mace_torch.cli.training_config import configuration

    parser = argparse.ArgumentParser(prog="mace train", allow_abbrev=False)
    add_arguments(parser)
    known, rest = parser.parse_known_args(argv)
    dotted = [
        token
        for token in rest
        if token.startswith("--") and "." in token.split("=", 1)[0]
    ]
    if dotted:
        raise SystemExit(
            f"{', '.join(dotted)}: set this value in the configuration "
            f"file; V1 takes explicit flags"
        )
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    return configuration(known)


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
    built = build(config, config.catalogue or DEFAULT_CATALOGUE, context=processes)
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
    from mace_torch.cli.reporting import print_citations

    print_citations(trained.metadata)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """The console entry point. Returns a process exit status."""
    return train(parse(argv))
