"""Setting up the run's own logging, once, and once per run rather than per rank.

Two things the frozen tree gets right and states nowhere, plus one it does not.

It writes a file beside the run as well as to the console, which is what makes
a finished run readable a week later, and it does that only on the first rank,
because eight processes appending to one file interleave their lines.

What it does not do is say so on the ranks that are silent. A run whose other
seven processes log nothing looks, from inside one of them, like a run that
lost its logging. So the silent ranks keep their warnings: a rank that fails is
the one whose output matters, and it is the one whose output legacy discards.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mace_core.config.runtime import RuntimeConfig

__all__ = ["LOG_FORMAT", "setup_logging"]

#: The line format, stated once so the file and the console agree.
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def setup_logging(runtime: RuntimeConfig, rank: int = 0) -> Path | None:
    """Attach the run's handlers to the root logger.

    Args:
        runtime: The run's settings. The level and the directory come from it.
        rank: Which process this is. Only the first writes the file and logs
            at the configured level; the others are held at ``WARNING``, so a
            failure still reaches the console from whichever rank had it.

    Returns:
        The log file, or ``None`` on a rank that does not write one.

    The handlers this adds are removed first if they are already there, so
    calling it twice in one process, which a test does and a notebook does,
    does not double every line.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_mace_run_log", False):
            root.removeHandler(handler)
            handler.close()

    level = runtime.log_level if rank == 0 else "WARNING"
    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console._mace_run_log = True  # ty: ignore[unresolved-attribute]
    root.addHandler(console)

    if rank != 0:
        return None

    directory = runtime.directory("logs")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{runtime.name}.log"
    to_file = logging.FileHandler(path, encoding="utf-8")
    to_file.setFormatter(formatter)
    to_file._mace_run_log = True  # ty: ignore[unresolved-attribute]
    root.addHandler(to_file)
    return path
