"""Finding electrostatics solvers, apart from the kernel backends.

A registry of its own, one entry point group per framework, beside the kernel
backends' rather than inside it: the message-passing op contract stays closed,
and a solver is not a kernel backend with one more op.

**Discovery records failures; resolution raises.** A solver whose import fails
on this machine, because its optional dependency is not installed, is listed
with the reason. Asking for it by name raises with that reason, and nothing
else is handed back in its place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

__all__ = [
    "ENTRY_POINT_GROUPS",
    "DiscoveredSolver",
    "SolverNotAvailableError",
    "available_solvers",
    "get_solver",
]

#: One group per framework, parallel to the kernel backends' groups.
ENTRY_POINT_GROUPS: dict[str, str] = {
    "torch": "mace.electrostatics_backends.torch",
    "jax": "mace.electrostatics_backends.jax",
}


class SolverNotAvailableError(RuntimeError):
    """A solver was asked for by name and could not be delivered."""


@dataclass(frozen=True)
class DiscoveredSolver:
    """One entry point, and whether it loaded.

    Attributes:
        name: The registered name.
        framework: Which framework's group it came from.
        loaded: Whether importing it succeeded here.
        reason: Why it did not, when it did not.
    """

    name: str
    framework: str
    loaded: bool
    reason: str = ""
    factory: Any = field(default=None, repr=False, compare=False)


def _discover(framework: str) -> list[DiscoveredSolver]:
    group = ENTRY_POINT_GROUPS.get(framework)
    if group is None:
        raise ValueError(
            f"{framework!r} is not a framework this registry knows. The "
            f"frameworks are {sorted(ENTRY_POINT_GROUPS)}."
        )
    found = []
    for entry in entry_points(group=group):
        try:
            factory = entry.load()
        except Exception as failure:
            found.append(DiscoveredSolver(entry.name, framework, False, repr(failure)))
        else:
            found.append(DiscoveredSolver(entry.name, framework, True, "", factory))
    return sorted(found, key=lambda solver: solver.name)


def available_solvers(framework: str = "torch") -> list[DiscoveredSolver]:
    """Every registered solver for a framework, loaded or not."""
    return _discover(framework)


def get_solver(name: str, framework: str = "torch") -> Any:
    """The solver registered under ``name``, built.

    Raises:
        SolverNotAvailableError: If no solver has that name, or one does and
            it did not import. The two are told apart, and the import failure
            is quoted, since they call for different fixes.
    """
    by_name = {solver.name: solver for solver in _discover(framework)}
    if name not in by_name:
        raise SolverNotAvailableError(
            f"no {framework} electrostatics solver is registered as {name!r}. "
            f"The registered names are {sorted(by_name)}, from the entry point "
            f"group {ENTRY_POINT_GROUPS[framework]!r}."
        )
    solver = by_name[name]
    if not solver.loaded:
        raise SolverNotAvailableError(
            f"the {framework} electrostatics solver {name!r} is registered but "
            f"did not import here: {solver.reason}. Nothing is substituted for "
            f"it, because another solver is another set of numbers."
        )
    return solver.factory()
