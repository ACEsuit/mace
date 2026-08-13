"""Regeneration targets: one module per family of goldens.

``regenerate.py`` does not know what can be regenerated. It asks this package,
which discovers every module beside this one and reads four names off it:

``ORDER``
    an integer. ``--target all`` runs the targets in ascending order, ties
    broken by name, because the families depend on each other: fixtures are
    inputs to the anchors, and the anchors are inputs to the references.
``HELP``
    one line, shown in ``--help``.
``run()``
    rewrites that family's artifacts. It may import the framework; this
    package must not, so that ``--help`` works in an environment that cannot
    load torch.
``IN_ALL``
    optional, defaults to ``True``. Set it to ``False`` for a family whose
    regeneration needs something ``all`` cannot assume -- a download, an
    optional dependency, a particular GPU. ``regenerate.py`` names what it
    skipped rather than leaving those references silently stale.
``DOC``
    optional, defaults to the module's own name. The stem of the page in
    ``docs/`` that describes this target. Set it when one family is split
    across two targets because they differ in what running them requires --
    the two foundation tiers are one story and one page.

A module may also define ``ANCHORS``, mapping anchor name to
``(checkpoint_path, reference_filename, description)``. ``all_anchors()``
merges them, so a target that reads another family's checkpoint -- the
gradient digests do -- can find it without importing that family.

The discovery is the point. A new family of goldens is a new file here, so two
families added on two branches never edit the same line and merge without a
conflict; an enumeration in the driver would have to be edited by both.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Dict, NamedTuple


class Target(NamedTuple):
    name: str
    order: int
    help: str
    in_all: bool
    doc: str
    run: Callable[[], None]


def discover() -> Dict[str, Target]:
    """Every target module in this package, in the order ``all`` runs them."""
    found = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        missing = [
            attr for attr in ("ORDER", "HELP", "run") if not hasattr(module, attr)
        ]
        if missing:
            raise AttributeError(
                f"{module.__name__} is in the targets package but defines none of "
                f"{missing}. A target module must declare ORDER, HELP and run(); "
                "a module that is not a target does not belong here."
            )
        name = info.name.replace("_", "-")
        found.append(
            Target(
                name=name,
                order=module.ORDER,
                help=module.HELP,
                in_all=getattr(module, "IN_ALL", True),
                doc=getattr(module, "DOC", info.name),
                run=module.run,
            )
        )
    return {t.name: t for t in sorted(found, key=lambda t: (t.order, t.name))}


def all_anchors() -> Dict[str, tuple]:
    """Anchor name -> (checkpoint, reference filename, what it is for)."""
    anchors: Dict[str, tuple] = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        for name, spec in getattr(module, "ANCHORS", {}).items():
            if name in anchors:
                raise ValueError(
                    f"two target modules both claim the anchor {name!r}. An "
                    "anchor is owned by exactly one family, so that "
                    "regenerating that family rewrites it exactly once."
                )
            anchors[name] = spec
    return anchors
