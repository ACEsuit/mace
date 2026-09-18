"""Where this directory sits, for the modules that print relative paths.

Its own module so that the target modules can import it without importing
``regenerate``, which imports them.
"""

from __future__ import annotations

from pathlib import Path

GOLDEN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = GOLDEN_ROOT.parent.parent
