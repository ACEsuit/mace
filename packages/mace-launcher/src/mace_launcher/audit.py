"""Runtime guard against the v1 stack reaching into the legacy package.

Static analysis sees ``import mace`` and ``from mace import x``. It does not see
``importlib.import_module("mace." + name)``, and that is the form a port under
time pressure reaches for, so the runtime guard exists to cover exactly it.

**An audit hook alone does not cover it.** CPython raises the ``import`` audit
event from the C implementation of ``__import__``, and ``importlib`` bypasses
that path entirely. Measured on 3.11: a plain ``import mace`` raises the event,
``importlib.import_module("mace.modules")`` raises nothing at all. A guard built
only on ``sys.addaudithook`` is therefore blind to the one case it is for, and
blind silently, which is worse than absent.

So the primary guard is a ``sys.meta_path`` finder. Every import that is not
already cached reaches ``_find_and_load``, which consults ``meta_path`` however
the import was spelled, so the finder sees both forms. The audit hook stays as a
second net.

Neither mechanism sees an import satisfied from ``sys.modules``. That is not a
hole worth closing: the first crossing is a real load and it is the one that
fails, and on the v1 engine the launcher has not imported the legacy package, so
there is nothing warm in the cache to hide behind.

The guard is installed only on the v1 engine. On the legacy engine the whole
process is legacy, so a crossing is not a crossing.
"""

from __future__ import annotations

import sys
from typing import Iterable, Sequence

#: Import roots of the v1 stack. A crossing is any of these importing `mace`.
V1_ROOTS = ("mace_core", "mace_torch", "mace_jax")

#: The two allowlisted crossers. The launcher dispatches into both stacks by
#: definition, and the parity harness compares them in one process.
ALLOWED_MODULE_PREFIXES = ("mace_launcher",)
ALLOWED_PATH_FRAGMENTS = ("tests/parity/", "tests\\parity\\")

_installed = False


class LegacyReachIn(RuntimeError):
    """A v1 module imported the frozen legacy package."""


def _is_legacy(module_name: object) -> bool:
    return isinstance(module_name, str) and (
        module_name == "mace" or module_name.startswith("mace.")
    )


def _offending_caller() -> str | None:
    """Name of the nearest v1 frame responsible for the import, if any.

    Walks outward from the guard. The first frame belonging to a v1 package is
    the culprit; an allowlisted frame reached first means the crossing was
    deliberate.
    """
    frame = sys._getframe(1)  # pylint: disable=protected-access
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        if name == __name__:
            # This module's own frames sit between the guard and the importer,
            # and `mace_launcher` is allowlisted, so counting them would
            # allowlist every crossing there is.
            frame = frame.f_back
            continue
        filename = frame.f_globals.get("__file__", "") or ""
        if name.startswith(ALLOWED_MODULE_PREFIXES):
            return None
        if any(fragment in filename for fragment in ALLOWED_PATH_FRAGMENTS):
            return None
        if name.split(".", 1)[0] in V1_ROOTS:
            return name
        frame = frame.f_back
    return None


def _check(module_name: object) -> None:
    """Raise if a v1 module is importing the legacy package."""
    if not _is_legacy(module_name):
        return
    caller = _offending_caller()
    if caller is None:
        return
    raise LegacyReachIn(
        f"{caller} imported the legacy package '{module_name}'. The v1 stack "
        f"never imports mace/: it reproduces the numbers, it does not reuse the "
        f"code. Only mace_launcher and tests/parity/ may import both."
    )


class LegacyReachInFinder:
    """A meta_path finder that inspects, never resolves.

    ``find_spec`` returning ``None`` means "not mine", so the real finders
    behind it load the module exactly as they would have. It sits at position 0
    only so that it sees the name before anything claims it.
    """

    # The signature and the explicit `return None` are both the finder
    # protocol's, not this class's choice: `path` and `target` are passed by the
    # import system whether or not a finder reads them, and returning None is
    # how a finder says "not mine" and hands the import on.
    @staticmethod
    def find_spec(  # pylint: disable=unused-argument,useless-return
        fullname: str, path: Sequence[str] | None = None, target=None
    ):
        # Runs for every uncached import in the process, so the cheap string
        # test comes before anything that touches frames.
        if _is_legacy(fullname):
            _check(fullname)
        return None


def _hook(event: str, args: Iterable[object]) -> None:
    if event != "import":
        return
    _check(next(iter(args), ""))


def install() -> None:
    """Register both guards. Idempotent, and irreversible for the process.

    An audit hook cannot be removed once added, which is what makes this a
    guard rather than a convention.
    """
    global _installed  # pylint: disable=global-statement
    if _installed:
        return
    sys.meta_path.insert(0, LegacyReachInFinder())
    sys.addaudithook(_hook)
    _installed = True
