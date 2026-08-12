"""References for the published checkpoints that have to be downloaded.

Not part of ``--target all``: regenerating these fetches the published
releases, and a regeneration that silently reaches the network is one that
behaves differently on a machine that cannot. Ask for it by name when you
mean to refresh a downloaded artifact's reference.
"""

from __future__ import annotations

from tests.golden.targets import _foundations

ORDER = 41
HELP = "references for the published checkpoints that are downloaded"
IN_ALL = False


def run() -> None:
    _foundations.regenerate(network=True)
