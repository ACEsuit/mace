"""Which ``les`` produced a number, and whether it is the one installed now.

The LES goldens are the only ones in this directory whose value depends on a
package outside this repository. ``MACELES`` predicts latent multipoles and
then hands them to ``les.Les``, which does the Ewald sum and the Born-charge
derivative; the energy, the forces and every latent quantity that comes back
through the solver are as much a fact about that library as about the
checkpoint. So a reference that does not say which ``les`` it was taken with
is not reproducible, and this module is what makes that sayable.

It is not a hypothetical. Two comparisons in
``tests/extensions/les/test_maceles.py`` are ``xfail``ed with exactly this
diagnosis: their hardcoded reference energies were generated against an
unrecorded ``les`` and do not reproduce against the pinned one. The
information needed to tell "the model changed" from "the solver changed" was
never written down, so neither test can say which happened, and both had to be
abandoned rather than fixed.

Three facts are available and this module keeps them apart:

* the **pinned** commit -- what ``requirements/les.txt`` asks for;
* the **installed** commit -- what pip actually put in site-packages, read
  from the ``direct_url.json`` a VCS install records (PEP 610);
* the **reference** commit -- what the committed JSON says it was taken with.

A golden compares the third against the second and names both. The first is
reported alongside, because when all three disagree the useful sentence is
which of the two moved.

Deliberately free of any ``mace`` import, like the harness it sits next to:
the parity suites that will consume these references live outside the legacy
tree, and the question "which solver produced this number" is one they have to
be able to ask too.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

GOLDEN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = GOLDEN_ROOT.parent.parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements" / "les.txt"

#: A 40-hex git object name at the end of a pip VCS requirement.
_PINNED_SHA = re.compile(r"les\.git@(?P<sha>[0-9a-f]{40})\b")


def pinned_les_commit(path: Path = REQUIREMENTS_PATH) -> Optional[str]:
    """The commit ``requirements/les.txt`` asks for, or ``None``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _PINNED_SHA.search(text)
    return match.group("sha") if match else None


def installed_les_commit() -> Optional[str]:
    """The commit the installed ``les`` was built from, or ``None``.

    ``None`` means the question cannot be answered, which is a different
    outcome from a mismatch and is reported differently: a wheel from PyPI, a
    local editable checkout or a vendored copy all record no VCS provenance.
    """
    from importlib import metadata  # noqa: PLC0415

    try:
        distribution = metadata.distribution("les")
    except metadata.PackageNotFoundError:
        return None
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    commit = payload.get("vcs_info", {}).get("commit_id")
    return str(commit) if commit else None


def describe_les() -> str:
    """A one-line provenance string for a failure message."""
    installed = installed_les_commit()
    pinned = pinned_les_commit()
    return (
        f"installed les: {installed or 'unknown provenance'}; "
        f"requirements/les.txt pins: {pinned or 'nothing parseable'}"
    )


def check_les_matches(reference_commit: Optional[str]) -> Optional[str]:
    """Why this ``les`` cannot reproduce ``reference_commit``, or ``None``.

    Returns a complete failure message rather than raising, so the caller
    decides whether that is a skip or an error; in this repository it is
    always an error, because a LES golden asserted against the wrong solver
    reports a tolerance failure on eight channels and says nothing about the
    cause.
    """
    if not reference_commit:
        return (
            "this reference records no les commit, so nothing can be "
            "reproduced from it: the same numbers would 'pass' against any "
            "version of the solver. Regenerate it with "
            "tests/golden/regenerate.py, which records the commit."
        )
    installed = installed_les_commit()
    if installed is None:
        return (
            f"the reference was taken with les {reference_commit}, and the "
            f"installed les records no VCS provenance (no direct_url.json "
            f"commit), so it cannot be shown to be the same code. Install it "
            f"the way CI does -- pip install -r requirements/les.txt -- "
            f"rather than from a wheel or a local checkout. "
            f"{describe_les()}."
        )
    if installed != reference_commit:
        pinned = pinned_les_commit()
        moved = (
            "requirements/les.txt has been bumped and the golden was not "
            "regenerated"
            if pinned == installed
            else "the installed les is neither the reference's nor the pinned one"
        )
        return (
            f"les commit mismatch: this reference was generated against "
            f"{reference_commit} and the installed les is {installed} "
            f"({moved}; requirements/les.txt pins "
            f"{pinned or 'nothing parseable'}). The long-range energy, the "
            f"forces and every latent quantity that comes back through the "
            f"solver are properties of that library as much as of the "
            f"checkpoint, so comparing across versions produces a tolerance "
            f"failure that says nothing about which side moved. This is the "
            f"failure the two xfails in tests/extensions/les/test_maceles.py "
            f"document. Either install the pinned commit, or regenerate the "
            f"reference deliberately with tests/golden/regenerate.py --target "
            f"les --i-know-what-i-am-doing and review the numeric diff."
        )
    return None


__all__ = [
    "check_les_matches",
    "describe_les",
    "installed_les_commit",
    "pinned_les_commit",
]
