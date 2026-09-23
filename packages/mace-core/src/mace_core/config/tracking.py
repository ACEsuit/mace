"""Where a run reports itself to, beyond its own log.

One section, one experiment tracker. It is a runtime concern: nothing here
reaches the model, and a run with tracking on and one with it off compute the
same numbers.

**The allow-list is gone.** Legacy takes ``--wandb_log_hypers``, a list of
argument names to copy out of the parsed namespace, defaulting to eleven of
them (``mace/tools/scripts_utils.py:1095-1112``). It exists because the
namespace is not serialisable and nobody wanted to decide which parts were:
the same function computes the whole thing as JSON two lines later and throws
it away. The resolved configuration is one validated object that round-trips
through JSON, so the whole of it is logged and there is nothing to choose.
"""

from __future__ import annotations

from pathlib import Path

from mace_core.config.section import FrozenSection

__all__ = ["WandbConfig"]


class WandbConfig(FrozenSection):
    """Reporting to Weights and Biases.

    Args:
        enabled: Whether to report at all. The import happens only when this
            is set, so the dependency is optional in the sense that matters:
            a run without it never touches the package.
        project: The project the run is filed under.
        entity: The team or user account. ``None`` uses the logged-in one.
        name: The run's name there. ``None`` takes the run's own name, so the
            two agree unless someone deliberately separates them.
        directory: Where the client writes its local state. ``None`` leaves
            that to the client, which puts it under the working directory.
    """

    enabled: bool = False
    project: str = "mace"
    entity: str | None = None
    name: str | None = None
    directory: Path | None = None
