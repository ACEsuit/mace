"""The end-to-end training set, rebuilt from its closed-form labeller."""

from __future__ import annotations

from tests.golden.paths import REPO_ROOT

ORDER = 50
HELP = "the end-to-end regression training set"


def run() -> None:
    # Late import: see the package docstring. Nothing above this point needs
    # ase.
    from ase.io import write as ase_write  # pylint: disable=import-outside-toplevel

    from tests.golden.make_regression_set import (  # pylint: disable=import-outside-toplevel
        OUTPUT,
        build,
        self_check,
    )

    configs = build()
    worst_force = max(
        (self_check(config)[0] for config in configs if len(config) > 1), default=0.0
    )
    print(f"  self-check: worst analytic-vs-numeric force {worst_force:.3e} eV/Ang")
    ase_write(OUTPUT, configs, format="extxyz")
    print(f"  dataset  regression_train -> {OUTPUT.relative_to(REPO_ROOT)}")
