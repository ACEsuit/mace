"""Training, from a command line.

The whole of it: read a configuration, run three stages, write what came out.
The frozen tree's equivalent is one function of about 1,150 lines that reads 74
attributes off an argparse namespace and assigns back to it at 43 sites, so
that what a run does depends on the order the assignments happen in.

Nothing is decided here. A default belongs to the schema, a refusal belongs to
the stage that can explain it, and this module is what connects them, which is
why it is short enough to read in one go.

The surface is the configuration file plus dotted overrides. The faithful port
of the legacy flags is its own work, and the shim that maps a legacy namespace
onto this configuration already exists beside the schema; what is here is the
native one.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import load_default_catalogue
from mace_core.stages import TrainedModel

from mace_torch.finetune.foundation import read_foundation
from mace_torch.train import (
    latest_run_checkpoint,
    run_data_stage,
    run_model_stage,
    run_train_stage,
    setup_logging,
)

__all__ = ["NOT_MIGRATED", "main", "parse", "run"]

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
        The resolved configuration: the file, then the dotted overrides, over
        the schema's own defaults.
    """
    parser = argparse.ArgumentParser(
        prog="mace_run_train --engine v1",
        description="Train a model from a configuration file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML, YAML or JSON. Omitted, the run is the schema's defaults "
        "plus whatever the overrides set.",
    )
    known, overrides = parser.parse_known_args(argv)
    legacy = [
        argument
        for argument in overrides
        if argument.startswith("--") and "." not in argument.split("=", 1)[0]
    ]
    if legacy:
        # Every field of the configuration is inside a section, so a dotless
        # override names nothing here and is a legacy flag. Saying so is the
        # difference between a suite that skips and one that fails: the port of
        # those flags is its own work, and until it lands a run that asks for
        # them has not been tried on this engine rather than tried and broken.
        raise SystemExit(
            f"the legacy command-line flags are {NOT_MIGRATED}: "
            f"{', '.join(sorted(legacy))}. This engine takes a configuration "
            f"file and dotted overrides such as --training.lr 0.005. Run it "
            f"with --engine legacy, or write the run as a configuration."
        )
    return ResolvedConfig.load(known.config, overrides)


def run(config: ResolvedConfig) -> TrainedModel:
    """Read the data, build the model, train it.

    The three stages and the objects between them. A stage takes the previous
    stage's object and nothing else, so there is no state here for them to
    disagree about. A fine-tune reads its foundation model first, and hands
    the data stage what it needs of it and the model stage the rest.
    """
    catalogue = load_default_catalogue()
    foundation = (
        read_foundation(config.finetune.foundation_model, catalogue)
        if config.finetune.foundation_model is not None
        else None
    )
    samples_by_descriptor = any(
        head.subselect is not None and head.subselect.method == "fps"
        for head in config.data.heads.values()
    )
    data = run_data_stage(
        config,
        catalogue,
        foundation=(
            foundation.context(describe=samples_by_descriptor)
            if foundation is not None
            else None
        ),
    )
    built = run_model_stage(config, data, catalogue, foundation=foundation)
    checkpoint_path = Path(config.runtime.work_dir) / config.runtime.name
    return run_train_stage(
        config,
        built,
        device=config.runtime.device,
        checkpoint_path=checkpoint_path,
        resume=_resumes(config, checkpoint_path),
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


def main(argv: Sequence[str] | None = None) -> int:
    """The console entry point. Returns a process exit status."""
    config = parse(argv)
    setup_logging(config.runtime)
    trained = run(config)
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
