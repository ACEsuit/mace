"""References for the published checkpoint that is tracked in this tree."""

from __future__ import annotations

from tests.golden.targets import _foundations

ORDER = 40
HELP = "references for the published checkpoint tracked in this repository"


def run() -> None:
    _foundations.regenerate(network=False)
