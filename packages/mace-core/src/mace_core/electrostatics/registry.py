"""Discover electrostatics solvers in separate groups from kernel backends.

Listings retain import failures; named resolution raises instead of substituting
a solver. Conflicting registrations raise ``DuplicateSolverError``; identical
registrations coalesce. Checkpoint substitution follows ``solver_to_load``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from mace_core._entry_points import (
    load_entry_point,
    resolve_entry_point,
    unique_entries,
)

__all__ = [
    "ENTRY_POINT_GROUPS",
    "DiscoveredSolver",
    "DuplicateSolverError",
    "SolverNotAvailableError",
    "SolverSubstitutionError",
    "available_solvers",
    "get_solver",
    "solver_to_load",
]

#: One group per framework, parallel to the kernel backends' groups.
ENTRY_POINT_GROUPS: dict[str, str] = {
    "torch": "mace.electrostatics_backends.torch",
    "jax": "mace.electrostatics_backends.jax",
}


class SolverNotAvailableError(RuntimeError):
    """A solver was asked for by name and could not be delivered."""


class SolverSubstitutionError(RuntimeError):
    """A model was asked to load with a solver other than its own, and the
    swap would change its numbers.
    """


class DuplicateSolverError(RuntimeError):
    """Different electrostatics solvers registered under one name."""


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


def _entries(framework: str) -> dict[str, EntryPoint]:
    group = ENTRY_POINT_GROUPS.get(framework)
    if group is None:
        raise ValueError(
            f"{framework!r} is not a framework this registry knows. The "
            f"frameworks are {sorted(ENTRY_POINT_GROUPS)}."
        )
    return unique_entries(entry_points(group=group), group, DuplicateSolverError)


def _discover(framework: str) -> list[DiscoveredSolver]:
    found = []
    for name, entry in sorted(_entries(framework).items()):
        factory, reason = load_entry_point(entry)
        found.append(DiscoveredSolver(name, framework, not reason, reason, factory))
    return found


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
    entries = _entries(framework)
    factory = resolve_entry_point(
        entries,
        name,
        group=ENTRY_POINT_GROUPS[framework],
        description=f"{framework} electrostatics solver",
        unavailable=SolverNotAvailableError,
    )
    return factory()


def solver_to_load(
    recorded: str,
    recorded_bit_parity: bool,
    requested: str | None = None,
    framework: str = "torch",
) -> str:
    """Which solver a trained model is rebuilt with.

    The one it recorded, unless another is asked for. A swap is allowed only
    between two solvers that both reproduce the reference bit for bit, since
    then no number moves. Any other swap changes the model's predictions, in
    either direction, and is refused rather than made quietly.

    Args:
        recorded: The solver the checkpoint says the model was trained with.
        recorded_bit_parity: Whether that solver declared bit parity with the
            reference. Recorded rather than looked up, so the rule holds on a
            machine where the recorded solver is not installed.
        requested: The solver the caller asks for, or ``None`` for the
            recorded one.
        framework: Which framework's registry to look the requested one up in.

    Raises:
        SolverSubstitutionError: Naming both solvers and why the swap would
            change the model.
        SolverNotAvailableError: If the requested solver cannot be delivered.
    """
    if requested is None or requested == recorded:
        return recorded
    parity = bool(get_solver(requested, framework).capabilities.bit_parity)
    if recorded_bit_parity and parity:
        return requested
    moved = recorded if not recorded_bit_parity else requested
    raise SolverSubstitutionError(
        f"the model was trained with the electrostatics solver {recorded!r} and "
        f"is being loaded with {requested!r}. {moved!r} does not reproduce the "
        f"reference solver bit for bit, so the swap changes the model's "
        f"predictions. Load it with {recorded!r}, or retrain with the solver "
        f"you want."
    )
