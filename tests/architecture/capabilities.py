"""Read `capabilities.toml`, the migration state of every MACE capability.

This module is the only reader. The manifest exists so that the migration
state of a capability is written down once and everything else is derived from
it: the paths-filter entries of the extension jobs, the per-file coverage
floors, and the xfail markers of the per-debt fitness tests. Deriving them is
the whole point -- a second copy of any of those three would go stale in the
direction nobody checks, which is how a retired capability keeps a CI job that
watches nothing.

Vocabulary, and why there are two kinds in one file:

* a **probe** is an optional requirement of a test (a device, an optional
  dependency, a binary, network access). The set is exactly the keys of
  ``CAPABILITY_PROBES`` in ``tests/conftest.py``.
* an **axis** is a functional slice of the product. An axis is what
  ``--engine`` selects, so an axis has a default to flip and a probe does not.

Both migrate, so both are here, and the meta-tests hold each to its own rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

# tomllib is 3.11+. pytest declares tomli on 3.10, and every environment that
# runs this file runs it under pytest or beside it, so the fallback always
# resolves; a bare `import tomllib` would break collection on the 3.10 leg of
# the matrix.
try:  # pragma: no cover - one branch per interpreter
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("capabilities.toml")

#: The migration states, in order. A capability moves forwards through them and
#: never backwards; the order is what lets a test ask "at least v1-default".
STATES: Tuple[str, ...] = ("legacy", "v1-optin", "v1-default", "retired")

#: The states in which the v1 stack is what a user gets by default. A
#: capability outside this set still owes a default flip, and that debt is
#: what its fitness test xfails on.
V1_BY_DEFAULT = frozenset({"v1-default", "retired"})

KINDS: Tuple[str, ...] = ("probe", "axis")


@dataclass(frozen=True)
class CoverageFloor:
    """One per-file line-coverage floor, and the capability that owns it.

    The owner is the load-bearing part. A floor protects a *behaviour*, not a
    legacy file, so when the capability ports, the floor has to move to the
    corresponding module under ``packages/``. Recording the owner here is what
    turns that from something to remember into a manifest edit.
    """

    path: str
    floor: int
    owner: str


@dataclass(frozen=True)
class CiFilter:
    """A paths-filter entry: the extension job this capability starts."""

    name: str
    paths: Tuple[str, ...]
    path_notes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Capability:
    name: str
    kind: str
    state: str
    summary: str
    burn_step: str
    owned_by: Optional[str] = None
    legacy_surface: Tuple[str, ...] = ()
    ci_filter: Optional[CiFilter] = None
    coverage_floors: Tuple[CoverageFloor, ...] = ()

    @property
    def is_probe(self) -> bool:
        return self.kind == "probe"

    @property
    def is_axis(self) -> bool:
        return self.kind == "axis"

    @property
    def defaults_to_v1(self) -> bool:
        return self.state in V1_BY_DEFAULT


def _load_document() -> dict:
    return tomllib.loads(MANIFEST.read_text(encoding="utf-8"))


def load() -> Dict[str, Capability]:
    """Every capability, keyed by name, in the order the manifest declares.

    Declaration order is preserved deliberately: the generated CI fragments are
    compared as text, so an arbitrary ordering would make the check fail on a
    dictionary reshuffle rather than on a real change.
    """
    document = _load_document()
    result: Dict[str, Capability] = {}
    for name, entry in document["capabilities"].items():
        ci_entry = entry.get("ci")
        ci_filter = None
        if ci_entry is not None:
            ci_filter = CiFilter(
                name=ci_entry["filter"],
                paths=tuple(ci_entry.get("paths", ())),
                path_notes=dict(ci_entry.get("path_notes", {})),
            )
        result[name] = Capability(
            name=name,
            kind=entry["kind"],
            state=entry["state"],
            summary=entry["summary"],
            burn_step=entry["burn_step"],
            owned_by=entry.get("owned_by"),
            legacy_surface=tuple(entry.get("legacy_surface", ())),
            ci_filter=ci_filter,
            coverage_floors=tuple(
                CoverageFloor(path=row["path"], floor=int(row["floor"]), owner=name)
                for row in entry.get("coverage_floors", ())
            ),
        )
    return result


def declared_states() -> Tuple[str, ...]:
    """The state vocabulary as the manifest spells it, for cross-checking."""
    return tuple(_load_document()["meta"]["states"])


def shared_ci_paths() -> Tuple[str, ...]:
    return tuple(_load_document()["shared_ci_paths"]["paths"])


def probes() -> Dict[str, Capability]:
    return {name: cap for name, cap in load().items() if cap.is_probe}


def axes() -> Dict[str, Capability]:
    return {name: cap for name, cap in load().items() if cap.is_axis}


def ci_filters() -> List[CiFilter]:
    """The paths-filter entries, in manifest order."""
    return [cap.ci_filter for cap in load().values() if cap.ci_filter is not None]


def coverage_floors() -> List[CoverageFloor]:
    """Every per-file coverage floor, in manifest order."""
    return [floor for cap in load().values() for floor in cap.coverage_floors]


# ---------------------------------------------------------------------------
# Validation
#
# One function, returning a list of problems rather than raising, so that the
# runnable check script and the meta-test report the same set and a caller can
# see every problem at once instead of the first one.
# ---------------------------------------------------------------------------


def problems() -> List[str]:
    """Everything wrong with the manifest, as sentences."""
    found: List[str] = []
    document = _load_document()
    capabilities = load()

    if declared_states() != STATES:
        found.append(
            f"[meta].states is {list(declared_states())} but the code knows "
            f"{list(STATES)}; the vocabulary lives in both places so that a "
            f"new state cannot be introduced by a manifest edit alone"
        )

    for name, cap in capabilities.items():
        if cap.kind not in KINDS:
            found.append(f"{name}: kind {cap.kind!r} is not one of {list(KINDS)}")
        if cap.state not in STATES:
            found.append(f"{name}: state {cap.state!r} is not one of {list(STATES)}")
        if not cap.summary.strip():
            found.append(f"{name}: no summary")
        if not cap.burn_step.strip():
            found.append(
                f"{name}: no burn_step. Every capability names the ticket that "
                f"retires its legacy half, or nothing ever finishes it"
            )
        if cap.is_probe and not cap.owned_by:
            found.append(
                f"{name}: a probe must name the axis whose retirement carries "
                f"it (owned_by), so that no optional dependency sits here "
                f"unclaimed by a ticket"
            )
        if cap.is_axis and cap.owned_by:
            found.append(
                f"{name}: an axis is owned by nobody; owned_by is for probes"
            )
        if cap.owned_by and cap.owned_by not in capabilities:
            found.append(f"{name}: owned_by names {cap.owned_by!r}, not a capability")
        elif cap.owned_by and not capabilities[cap.owned_by].is_axis:
            found.append(f"{name}: owned_by names {cap.owned_by!r}, which is not an axis")

    for floor in coverage_floors():
        if not 0 < floor.floor <= 100:
            found.append(f"{floor.path}: {floor.floor} is not a coverage percentage")
        if not (REPO_ROOT / floor.path).is_file():
            found.append(
                f"{floor.path}: a coverage floor on a file that does not exist. "
                f"`coverage report --include` takes an fnmatch pattern, so a "
                f"stale path selects nothing rather than failing. If the "
                f"behaviour moved to packages/, move the floor with it"
            )

    seen_floor_paths: Dict[str, str] = {}
    for floor in coverage_floors():
        if floor.path in seen_floor_paths:
            found.append(
                f"{floor.path}: two floors, owned by {seen_floor_paths[floor.path]} "
                f"and {floor.owner}. One file, one floor"
            )
        seen_floor_paths[floor.path] = floor.owner

    seen_filters: Dict[str, str] = {}
    for cap in capabilities.values():
        if cap.ci_filter is None:
            continue
        if cap.ci_filter.name in seen_filters:
            found.append(
                f"filter {cap.ci_filter.name!r} is claimed by both "
                f"{seen_filters[cap.ci_filter.name]} and {cap.name}"
            )
        seen_filters[cap.ci_filter.name] = cap.name
        for path in cap.ci_filter.path_notes:
            if path not in cap.ci_filter.paths:
                found.append(
                    f"{cap.name}: a path_note for {path!r}, which is not one of "
                    f"its paths"
                )

    unexpected = set(document) - {"meta", "shared_ci_paths", "capabilities"}
    if unexpected:
        found.append(f"unknown top-level tables: {sorted(unexpected)}")

    return found
