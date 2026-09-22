"""Turning a list of names in a configuration into what a model computes.

A configuration asks for ``["energy", "forces"]``, which is two names of two
different kinds: one is an observable and the other is a derivative of it. The
distinction matters because they are produced differently, by a head and by the
derivative engine, and because a model that declares ``energy`` alone still
knows how a force would be named.

Resolving the pair is a property of the catalogue, so it is here rather than in
either framework. The alternative is each of them doing it, which is how the
torch and the jax stacks end up disagreeing about whether ``stress`` was asked
for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mace_core.observables.spec import ObservableCatalogue, ObservableSpec

__all__ = ["RequestedOutputs", "UnknownObservableError", "resolve_requested"]


class UnknownObservableError(KeyError):
    """A configured name that is neither an observable nor a derivative."""


@dataclass(frozen=True)
class RequestedOutputs:
    """What the configuration asked for, split by how it is produced.

    Attributes:
        observables: The declarations the output layer builds heads from, in
            the order the configuration named them.
        derivatives: The derivative names the engine is asked to compute, such
            as ``forces``. Asking for none is legitimate: an inference run
            wants the energy alone.
    """

    observables: tuple[ObservableSpec, ...] = ()
    derivatives: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Every quantity the run produces, observables first."""
        return tuple(spec.name for spec in self.observables) + self.derivatives


def resolve_requested(
    names: Sequence[str], catalogue: ObservableCatalogue
) -> RequestedOutputs:
    """Split configured names into observables and derivatives.

    A derivative name implies its observable, so ``["forces"]`` alone resolves
    to the energy observable with the force derivative. Naming both is the
    usual spelling and means the same thing.

    Raises:
        UnknownObservableError: Naming the value and listing what the catalogue
            does declare. A misspelled observable otherwise trains a model that
            is quietly missing an output.
        ValueError: If a name is asked for twice, since two heads under one
            name is not a thing a model can have.
    """
    observables: list[ObservableSpec] = []
    derivatives: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(
                f"{name!r} is requested twice. Each name is one output, so a "
                f"repeat is either a typo or two things meant to be different."
            )
        seen.add(name)
        owner, derivative = _find(name, catalogue)
        if derivative is None:
            if owner not in observables:
                observables.append(owner)
            continue
        if owner not in observables:
            observables.append(owner)
        derivatives.append(derivative)
    return RequestedOutputs(tuple(observables), tuple(derivatives))


def _find(
    name: str, catalogue: ObservableCatalogue
) -> tuple[ObservableSpec, str | None]:
    """The observable a name belongs to, and the derivative name if it is one."""
    for spec in catalogue.observables:
        if spec.name == name:
            return spec, None
    for spec in catalogue.observables:
        for request in spec.derivatives:
            if spec.derivative_name(request.wrt) == name:
                return spec, name
    known = sorted(
        {spec.name for spec in catalogue.observables}
        | {
            spec.derivative_name(request.wrt)
            for spec in catalogue.observables
            for request in spec.derivatives
        }
    )
    raise UnknownObservableError(
        f"{name!r} is neither a declared observable nor a derivative of one. "
        f"The catalogue offers {known}."
    )
