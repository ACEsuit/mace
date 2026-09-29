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
    swap would change its numbers."""


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
