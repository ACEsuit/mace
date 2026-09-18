"""The committed input structures and the tiny training set."""

from __future__ import annotations

from tests.golden.paths import REPO_ROOT

ORDER = 10
HELP = "the committed .xyz structures and their manifest"


def run() -> None:
    # Imported here, not at module scope: the targets package is imported to
    # build --help, and that has to work where ase is absent.
    from tests.golden.make_fixtures import (  # pylint: disable=import-outside-toplevel
        write_fixtures,
    )

    written = write_fixtures()
    for name, path in written.items():
        print(f"  fixture  {name:16s} -> {path.relative_to(REPO_ROOT)}")
