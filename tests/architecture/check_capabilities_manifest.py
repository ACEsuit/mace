#!/usr/bin/env python3
"""Validate `capabilities.toml` and report what CI derives from it.

Two things are checked that a reader cannot check by looking: that the
manifest's probe set is exactly `CAPABILITY_PROBES` from `tests/conftest.py`,
and that the generated regions of the workflow files still match. The first is
asserted against the dict itself rather than a list retyped here, because
`magnetic` and `wandb` are routinely dropped from hand-written enumerations
and each has its own CI job.

The printed summary is the point of running this by hand: it is the one place
the migration state of every capability is visible at once.

Run from anywhere:  python3 tests/architecture/check_capabilities_manifest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.architecture import capabilities, generate_ci  # noqa: E402


def main() -> int:
    problems = capabilities.problems()

    try:
        from tests.conftest import CAPABILITY_PROBES
    except ImportError as error:  # torch, ase and numpy are conftest imports
        print(f"could not import tests.conftest, so the probe set was not checked: {error}")
        return 1

    declared = set(capabilities.probes())
    for name in sorted(set(CAPABILITY_PROBES) - declared):
        problems.append(
            f"{name} is a capability probe in tests/conftest.py with no row in "
            f"the manifest, so its migration state is recorded nowhere"
        )
    for name in sorted(declared - set(CAPABILITY_PROBES)):
        problems.append(
            f"{name} is declared a probe here but is not in CAPABILITY_PROBES"
        )

    stale = generate_ci.drift()
    for path in stale:
        problems.append(
            f"{path.relative_to(capabilities.REPO_ROOT)}: its generated region "
            f"disagrees with the manifest. Run "
            f"`python3 tests/architecture/generate_ci.py --write`"
        )

    if problems:
        print(f"capabilities.toml: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    manifest = capabilities.load()
    print(f"capabilities.toml: {len(manifest)} capabilities, all valid")
    for kind, heading in (("axis", "functional axes"), ("probe", "capability probes")):
        print(f"\n  {heading}:")
        for name, cap in manifest.items():
            if cap.kind != kind:
                continue
            owner = f" via {cap.owned_by}" if cap.owned_by else ""
            print(f"    {name:14s} {cap.state:11s} {cap.burn_step:16s}{owner}")
    print("\n  generated:")
    for path in generate_ci.files():
        print(f"    in sync    {path.relative_to(capabilities.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
