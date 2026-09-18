"""Advance notice for the surfaces MACE v1.0 removes or replaces.

MACE v1.0 is a rewrite. Its feature inventory carries a KEEP / MERGE / DROP
disposition for every enumerable surface of this package, and the non-KEEP rows
of it are reproduced in :mod:`mace.tools.deprecation_table`. This module turns
those rows into warnings, so that a 0.3.x user hears about a removal from the
release that still has the feature rather than from the one that dropped it.

Two things about the wording are deliberate.

**No message names a v1 command.** The last 0.3.x release predates the v1 CLI,
so telling a reader to "use ``mace train``" would point at a binary they cannot
run. A message says what replaces the feature in kind, and leaves the new
spelling to the v1.0 migration guide.

**A warning fires only for something the caller chose.** A flag is warned about
when it appears in ``argv``, not when it merely has a default; a command is
warned about when it is run. Some rows have no such moment, and
:data:`NEVER_WARNED` below says which and why. Those rows stay in the table and
are printed by::

    python -m mace.tools.deprecation

which is the one place the whole disposition list is visible.

Each warning goes out twice on purpose. :func:`warnings.warn` raises a
``FutureWarning``, which is what ``-W error`` and library consumers can see and
filter; ``logging.warning`` puts the same text in the run log, which is what a
CLI user actually reads. Both are deduplicated per identifier per process, so a
flag repeated across a resumed run does not repeat its warning.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from .deprecation_table import DISPOSITIONS, DROP, MERGE

#: A named logger, deliberately not the root one. The module-level
#: logging.warning() installs a StreamHandler on the root logger when root has
#: no handlers yet, via basicConfig. These warnings fire while the CLI is still
#: parsing, before setup_logger runs, so that handler would be installed at
#: NOTSET and would then receive every record the run logs afterwards. The whole
#: training log would be duplicated onto stderr. A named logger reaches the same
#: reader through logging's lastResort handler without touching root.
_log = logging.getLogger(__name__)

#: The version that removes or replaces everything in the table.
REMOVED_IN = "1.0"

_GUIDE = "See the MACE v1.0 migration guide."

#: Surfaces with no emission site, by decision rather than by oversight. MACE
#: builds these itself on every ordinary run: a model class, an interaction
#: block, a loss class, a symmetric-contraction helper, a data transform, a
#: model or calculator output key, the foundation-model roster, and the
#: unsafe-pickle escape hatch this package sets for itself. Warning on them
#: would tell every user about a decision none of them made. What a user does
#: choose is the option that selects them, and those options warn: --model,
#: --interaction, --loss. tests/unit/test_deprecation.py holds this list to
#: having no call site, so a warning added here has to be a deliberate change.
NEVER_WARNED = frozenset(
    {
        "block",
        "contraction",
        "fm",
        "loss",
        "model",
        "out",
        "stdenv",
        "transform",
    }
)


@dataclass(frozen=True)
class Deprecation:
    """One non-KEEP row of the v1 feature inventory."""

    id: str
    kind: str
    what: str
    why: str

    def message(
        self, context: Optional[str] = None, as_written: Optional[str] = None
    ) -> str:
        verb = "removes" if self.kind == DROP else "replaces"
        where = f" ({context})" if context else ""
        return (
            f"MACE v{REMOVED_IN} {verb} {as_written or self.what}{where}: "
            f"{self.why}. {_GUIDE}"
        )


DEPRECATIONS: Dict[str, Deprecation] = {
    row[0]: Deprecation(*row) for row in DISPOSITIONS
}

_warned: Set[str] = set()


def reset_warned() -> None:
    """Forget what has already been warned about. For tests."""
    _warned.clear()


def warn(
    dep_id: str,
    context: Optional[str] = None,
    as_written: Optional[str] = None,
    stacklevel: int = 2,
) -> bool:
    """Warn once about ``dep_id``. Returns whether this call was the one to warn.

    An unknown identifier is a programming error here rather than at import
    time, because the emission sites are spread over the tree and a typo in one
    of them must not be able to silence it.
    """
    try:
        dep = DEPRECATIONS[dep_id]
    except KeyError:
        raise KeyError(
            f"{dep_id!r} is not a row of the v1 disposition table; "
            f"add it to mace/tools/deprecation_table.py or fix the call site"
        ) from None
    if dep_id in _warned:
        return False
    _warned.add(dep_id)
    message = dep.message(context, as_written)
    warnings.warn(message, FutureWarning, stacklevel=stacklevel + 1)
    _log.warning(message)
    return True


def warn_env(*names: str) -> List[str]:
    """Warn for each named environment variable that is actually set."""
    fired = []
    for name in names:
        dep_id = f"env.{name}"
        if os.environ.get(name) is not None and warn(dep_id, stacklevel=3):
            fired.append(dep_id)
    return fired


def _option_dests(parser: argparse.ArgumentParser) -> Dict[str, str]:
    """Map every option string of ``parser`` to the dest it writes."""
    # argparse exposes no public accessor for this, and matching on the dest
    # name alone would miss the aliases: --swa_lr and --stage_two_lr are one
    # dest, and only one of the two spellings is deprecated.
    mapping = {}
    for action in parser._actions:  # pylint: disable=protected-access
        for option in action.option_strings:
            mapping[option] = action.dest
    return mapping


def _dests_from_config_file(parser: argparse.ArgumentParser) -> Dict[str, str]:
    """The dests a config file or environment variable supplied, if any.

    The training and preprocessing parsers are ``configargparse`` parsers with
    ``is_config_file=True`` on ``--config``, so a YAML config sets values on
    the namespace without putting anything in ``argv``. Reading ``argv`` alone
    therefore misses every option supplied the way the README documents. This
    asks configargparse where each setting came from, which is exact, rather
    than guessing from values that differ from their default.

    Returns an empty mapping under plain ``argparse``, where there is no config
    file to read, and after a parse that recorded no provenance.
    """
    describe = getattr(parser, "get_source_to_settings_dict", None)
    if describe is None:
        return {}
    try:
        sources = describe().items()
    except AttributeError:
        # configargparse records provenance during parse_args and raises here
        # if it has not run on this parser. A caller inspecting a parser it did
        # not parse with should get "nothing from a config file", not a crash.
        return {}
    found: Dict[str, str] = {}
    for source, settings in sources:
        # command_line is handled from argv, where the spelling is visible;
        # defaults are not something the caller chose.
        if source == "command_line" or source.startswith("defaults"):
            continue
        for action, _value in settings.values():
            if action is None or not action.option_strings:
                continue
            found.setdefault(action.dest, action.option_strings[0])
    return found


def explicit_options(
    parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None
) -> Dict[str, str]:
    """Map each dest the caller supplied to how it was spelled.

    Defaults are not included: a flag nobody wrote is not a flag anybody chose,
    and warning about it would fire on every run. The spelling matters because
    deprecated aliases share a dest, and a message should quote the option the
    reader actually used.

    Both routes count. A flag on the command line is read from ``argv``, and one
    supplied through ``--config`` is read from configargparse's own record of
    where each setting came from.
    """
    if argv is None:
        argv = sys.argv[1:]
    options = _option_dests(parser)
    found: Dict[str, str] = {}
    for token in argv:
        if not token.startswith("-") or token in ("-", "--"):
            continue
        name = token.split("=", 1)[0]
        dest = options.get(name)
        if dest is None and name.startswith("--"):
            # argparse accepts any unambiguous prefix of a long option.
            matches = {d for opt, d in options.items() if opt.startswith(name)}
            dest = matches.pop() if len(matches) == 1 else None
        if dest is not None and dest not in found:
            found[dest] = name
    for dest, spelling in _dests_from_config_file(parser).items():
        found.setdefault(dest, spelling)
    return found


def explicit_dests(
    parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None
) -> List[str]:
    """The dests the caller named on the command line, in the order given."""
    return list(explicit_options(parser, argv))


def warn_args(
    prefix: str,
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
) -> List[str]:
    """Warn for every deprecated option the caller passed to ``parser``.

    ``prefix`` is the inventory namespace of the parser: ``"train"`` for the
    training parser, ``"prep"`` for the preprocessing one, ``"cli.<module>"``
    for a per-command parser.
    """
    fired = []
    for dest, as_written in explicit_options(parser, argv).items():
        dep_id = f"{prefix}.{dest}"
        if dep_id in DEPRECATIONS and warn(dep_id, as_written=as_written, stacklevel=3):
            fired.append(dep_id)
    return fired


def warn_choice(namespace: str, value: Optional[str]) -> Optional[str]:
    """Warn for a deprecated value of a string-to-class choice.

    ``--model ScaleShiftMACE`` and ``--interaction`` name a class through a
    registry, so the deprecated thing is the value, not the option.
    """
    if value is None:
        return None
    dep_id = f"{namespace}.{value}"
    if dep_id in DEPRECATIONS and warn(dep_id, stacklevel=3):
        return dep_id
    return None


def rows(kind: Optional[str] = None) -> List[Deprecation]:
    """The table, optionally narrowed to ``DROP`` or ``MERGE``, in file order."""
    return [d for d in DEPRECATIONS.values() if kind is None or d.kind == kind]


def report(stream=None) -> None:
    """Print the whole disposition table, grouped by surface."""
    out = stream if stream is not None else sys.stdout
    groups: Dict[str, List[Deprecation]] = {}
    for dep in DEPRECATIONS.values():
        groups.setdefault(dep.id.split(".", 1)[0], []).append(dep)
    dropped = len(rows(DROP))
    print(
        f"MACE v{REMOVED_IN} removes {dropped} surfaces and replaces "
        f"{len(rows(MERGE))} more. {len(DEPRECATIONS)} rows.\n",
        file=out,
    )
    for surface in sorted(groups):
        print(f"[{surface}]", file=out)
        for dep in groups[surface]:
            print(f"  {dep.kind:5}  {dep.what}", file=out)
            print(f"         {dep.why}.", file=out)
        print(file=out)


if __name__ == "__main__":
    report()
